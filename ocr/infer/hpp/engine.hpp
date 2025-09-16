// engine.hpp - Shared TensorRT helpers for OCR modules
#ifndef OCR_ENGINE_HPP
#define OCR_ENGINE_HPP

#include <memory>
#include <string>
#include <utility>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

namespace ocrEngine {

struct EngineIOInfo {
    std::string input_name;
    std::string output_name;
};

// Load TensorRT engine from serialized file and extract IO info and static H/W if present.
bool load_engine(const std::string& engine_path,
                 nvinfer1::ILogger& logger,
                 std::unique_ptr<nvinfer1::IRuntime>& runtime,
                 std::unique_ptr<nvinfer1::ICudaEngine>& engine,
                 std::unique_ptr<nvinfer1::IExecutionContext>& context,
                 EngineIOInfo& io,
                 int& input_height,
                 int& input_width);

// Allocate device buffers and CUDA stream. If output dimensions are dynamic,
// use fallback_elements (number of float elements) to size output buffer.
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
                      size_t fallback_output_elements);

// Free device buffers and stream
void free_buffers(void*& d_input, void*& d_output, cudaStream_t& stream);

// Set input shape for dynamic inputs (NCHW)
bool set_input_shape(nvinfer1::IExecutionContext* context,
                     const std::string& input_name,
                     int n,
                     int c,
                     int h,
                     int w);

// Bind device pointers to named IO
bool bind_io(nvinfer1::IExecutionContext* context,
             const std::string& input_name,
             const std::string& output_name,
             void* d_input,
             void* d_output);

// Enqueue inference on stream
bool enqueue(nvinfer1::IExecutionContext* context, cudaStream_t stream);

// Query output shape at runtime from context
nvinfer1::Dims get_output_dims(nvinfer1::IExecutionContext* context,
                               const std::string& output_name);

// High-level helper to run inference end-to-end on a single input:
// - H2D copy of host input
// - bind IO
// - enqueue
// - query output dims, D2H copy to host output, synchronize
// Returns true on success and fills out_dims and h_output
bool infer(nvinfer1::IExecutionContext* context,
           const std::string& input_name,
           const std::string& output_name,
           const float* h_input,
           size_t num_input_elements,
           void* d_input,
           void* d_output,
           cudaStream_t stream,
           std::vector<float>& h_output,
           nvinfer1::Dims& out_dims);

} // namespace ocrEngine

#endif // OCR_ENGINE_HPP




