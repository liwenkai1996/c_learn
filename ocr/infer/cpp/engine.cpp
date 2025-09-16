// engine.cpp - Shared TensorRT helpers for OCR modules
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include "../hpp/engine.hpp"

namespace ocrEngine {

static size_t bytes_for_input(int max_batch,
                              int channels,
                              int height,
                              int width) {
    return static_cast<size_t>(max_batch) * channels * height * width * sizeof(float);
}

bool load_engine(const std::string& engine_path,
                 nvinfer1::ILogger& logger,
                 std::unique_ptr<nvinfer1::IRuntime>& runtime,
                 std::unique_ptr<nvinfer1::ICudaEngine>& engine,
                 std::unique_ptr<nvinfer1::IExecutionContext>& context,
                 EngineIOInfo& io,
                 int& input_height,
                 int& input_width) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening engine file: " << engine_path << std::endl;
        return false;
    }
    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), static_cast<std::streamsize>(size));
    file.close();

    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    engine.reset(runtime->deserializeCudaEngine(data.data(), size));
    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }
    context.reset(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }
    // IO names
    io.input_name.clear();
    io.output_name.clear();
    int num_tensors = engine->getNbIOTensors();
    for (int i = 0; i < num_tensors; ++i) {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) io.input_name = name;
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT) io.output_name = name;
    }
    if (io.input_name.empty() || io.output_name.empty()) {
        std::cerr << "Failed to get input/output tensor names" << std::endl;
        return false;
    }
    // Input H/W if static
    nvinfer1::Dims in_dims = engine->getTensorShape(io.input_name.c_str());
    if (in_dims.nbDims == 4) {
        if (in_dims.d[2] > 0) input_height = in_dims.d[2];
        if (in_dims.d[3] > 0) input_width = in_dims.d[3];
    }
    return true;
}

bool allocate_buffers(nvinfer1::ICudaEngine* engine,
                      const std::string& input_name,
                      const std::string& output_name,
                      int max_batch,
                      int input_channels,
                      int input_height,
                      int input_width,
                      void*& d_input,
                      void*& d_output,
                      cudaStream_t& stream,
                      size_t fallback_output_elements) {
    if (stream == nullptr) {
        cudaStreamCreate(&stream);
    }
    size_t input_bytes = bytes_for_input(max_batch, input_channels, input_height, input_width);
    nvinfer1::Dims out_dims = engine->getTensorShape(output_name.c_str());
    bool dynamic_output = false;
    size_t output_bytes = 0;
    if (out_dims.nbDims > 0) {
        output_bytes = sizeof(float);
        for (int i = 0; i < out_dims.nbDims; ++i) {
            if (out_dims.d[i] < 0) dynamic_output = true;
            int di = out_dims.d[i] > 0 ? out_dims.d[i] : 1;
            output_bytes *= static_cast<size_t>(di);
        }
        if (dynamic_output) {
            output_bytes = fallback_output_elements * sizeof(float);
        }
    } else {
        output_bytes = fallback_output_elements * sizeof(float);
    }
    if (d_input == nullptr) cudaMalloc(&d_input, input_bytes);
    if (d_output == nullptr) cudaMalloc(&d_output, output_bytes);
    bool ok = (d_input != nullptr) && (d_output != nullptr);
    if (!ok) std::cerr << "CUDA memory allocation failed" << std::endl;
    return ok;
}

void free_buffers(void*& d_input, void*& d_output, cudaStream_t& stream) {
    if (d_input) { cudaFree(d_input); d_input = nullptr; }
    if (d_output) { cudaFree(d_output); d_output = nullptr; }
    if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
}

bool set_input_shape(nvinfer1::IExecutionContext* context,
                     const std::string& input_name,
                     int n,
                     int c,
                     int h,
                     int w) {
    if (!context) return false;
    nvinfer1::Dims dims = context->getEngine().getTensorShape(input_name.c_str());
    if (dims.nbDims != 4) return true; // nothing to set
    dims.d[0] = n;
    dims.d[1] = c;
    dims.d[2] = h;
    dims.d[3] = w;
    return context->setInputShape(input_name.c_str(), dims);
}

bool bind_io(nvinfer1::IExecutionContext* context,
             const std::string& input_name,
             const std::string& output_name,
             void* d_input,
             void* d_output) {
    if (!context) return false;
    if (!context->setTensorAddress(input_name.c_str(), d_input)) return false;
    if (!context->setTensorAddress(output_name.c_str(), d_output)) return false;
    return true;
}

bool enqueue(nvinfer1::IExecutionContext* context, cudaStream_t stream) {
    if (!context) return false;
    return context->enqueueV3(stream);
}

nvinfer1::Dims get_output_dims(nvinfer1::IExecutionContext* context,
                               const std::string& output_name) {
    if (!context) return nvinfer1::Dims{};
    return context->getTensorShape(output_name.c_str());
}

bool infer(nvinfer1::IExecutionContext* context,
           const std::string& input_name,
           const std::string& output_name,
           const float* h_input,
           size_t num_input_elements,
           void* d_input,
           void* d_output,
           cudaStream_t stream,
           std::vector<float>& h_output,
           nvinfer1::Dims& out_dims) {
    if (!context || !h_input || !d_input || !d_output || stream == nullptr) return false;
    size_t input_bytes = num_input_elements * sizeof(float);
    cudaError_t err = cudaMemcpyAsync(d_input, h_input, input_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return false;
    if (!bind_io(context, input_name, output_name, d_input, d_output)) return false;
    if (!enqueue(context, stream)) return false;
    out_dims = get_output_dims(context, output_name);
    int output_size = 1;
    for (int i = 0; i < out_dims.nbDims; ++i) {
        int di = out_dims.d[i] > 0 ? out_dims.d[i] : 1;
        output_size *= di;
    }
    h_output.resize(static_cast<size_t>(output_size));
    err = cudaMemcpyAsync(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return false;
    cudaStreamSynchronize(stream);
    return true;
}

} // namespace ocrEngine

