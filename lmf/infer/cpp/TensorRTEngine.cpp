#include "../hpp/TensorRTEngine.hpp"
#include "../hpp/ImagePreprocessor.hpp"
#include "../hpp/LoggerV10.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>

// 前向声明
class LoggerV10;

// ==================== TensorRTEngine 实现 ====================
TensorRTEngine::TensorRTEngine() 
    : inputWidth(0), inputHeight(0), inputChannels(0), 
      outputSize(0), outputSize2(0), boxesBindingIndex(-1), classesBindingIndex(-1) {
    
    // 初始化CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device");
    }
}

TensorRTEngine::~TensorRTEngine() {
    cleanup();
}

bool TensorRTEngine::loadModel(const std::string& modelPath) {
    try {
        // 读取engine文件
        std::ifstream file(modelPath, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Error opening engine file: " << modelPath << std::endl;
            return false;
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();

        // 创建TensorRT runtime
        runtime.reset(nvinfer1::createInferRuntime(LoggerV10::getInstance()));
        if (!runtime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }

        // 从序列化数据创建engine
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
        if (!engine) {
            std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
            return false;
        }

        // 创建执行上下文
        context.reset(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Failed to create TensorRT execution context" << std::endl;
            return false;
        }

        // 获取输入输出维度
        inferOutputDimensions();
        
        // 分配内存
        if (!allocateMemory()) {
            return false;
        }

        std::cout << "Model loaded successfully!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception during model loading: " << e.what() << std::endl;
        return false;
    }
}

InferenceResult TensorRTEngine::executeInference(const PreprocessedImage& preprocessedImage) {
    if (!isModelLoaded()) {
        throw std::runtime_error("Model not loaded");
    }

    // 分配GPU缓冲区
    void *buffers[3];
    cudaMalloc(&buffers[0], preprocessedImage.size() * sizeof(float));
    cudaMalloc(&buffers[1], outputSize * sizeof(float));
    cudaMalloc(&buffers[2], outputSize2 * sizeof(float));

    // 分配CPU输出缓冲区
    std::vector<float> output1(outputSize);
    std::vector<float> output2(outputSize2);

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 设置张量地址
    setupTensorAddresses(buffers[0], buffers[1], buffers[2]);

    // 拷贝输入数据到GPU
    cudaMemcpyAsync(buffers[0], preprocessedImage.data.data(), 
                    preprocessedImage.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // 执行推理
    if (context->enqueueV3(stream)) {
        std::cout << "Inference executed successfully" << std::endl;
    } else {
        std::cout << "Inference execution failed" << std::endl;
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to execute TensorRT inference");
    }

    // 拷贝输出数据到CPU
    cudaMemcpyAsync(output1.data(), buffers[1], outputSize * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(output2.data(), buffers[2], outputSize2 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // 清理GPU内存
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaStreamDestroy(stream);

    // 返回推理结果
    return {output1, output2, outputSize, outputSize2};
}

void TensorRTEngine::inferOutputDimensions() {
    // 获取输入维度
    auto inputTensorName = engine->getIOTensorName(0);
    auto inputShape = engine->getTensorShape(inputTensorName);
    
    if (inputShape.nbDims == 4) {
        inputWidth = inputShape.d[3];
        inputHeight = inputShape.d[2];
        inputChannels = inputShape.d[1];
    } else if (inputShape.nbDims == 3) {
        inputWidth = inputShape.d[2];
        inputHeight = inputShape.d[1];
        inputChannels = inputShape.d[0];
    } else {
        throw std::runtime_error("Unexpected input dimensions");
    }

    // 获取输出维度
    auto output1TensorName = engine->getIOTensorName(1);
    auto output1Shape = engine->getTensorShape(output1TensorName);
    
    auto output2TensorName = engine->getIOTensorName(2);
    auto output2Shape = engine->getTensorShape(output2TensorName);

    // 计算输出大小
    outputSize = 1;
    for (int i = 0; i < output1Shape.nbDims; ++i) {
        outputSize *= output1Shape.d[i];
    }

    outputSize2 = 1;
    for (int i = 0; i < output2Shape.nbDims; ++i) {
        outputSize2 *= output2Shape.d[i];
    }

    // 自动识别boxes和classes输出
    auto inferBoxesIndex = [&](const nvinfer1::Dims& dims, int bindingIndex) {
        if (dims.nbDims >= 2 && dims.d[dims.nbDims - 1] == 4) {
            return bindingIndex;
        }
        return -1;
    };

    boxesBindingIndex = inferBoxesIndex(output1Shape, 1);
    if (boxesBindingIndex == -1) {
        boxesBindingIndex = inferBoxesIndex(output2Shape, 2);
    }

    classesBindingIndex = (boxesBindingIndex == 1) ? 2 : 1;

    std::cout << "Model dimensions: " << inputWidth << "x" << inputHeight << "x" << inputChannels << std::endl;
    std::cout << "Output sizes: " << outputSize << ", " << outputSize2 << std::endl;
    std::cout << "Boxes binding: " << boxesBindingIndex << ", Classes binding: " << classesBindingIndex << std::endl;
}

bool TensorRTEngine::allocateMemory() {
    // 在TensorRT 10.x中，内存分配在推理时动态进行
    return true;
}

void TensorRTEngine::setupTensorAddresses(void* inputBuffer, void* outputBuffer1, void* outputBuffer2) {
    context->setTensorAddress(engine->getIOTensorName(0), inputBuffer);
    context->setTensorAddress(engine->getIOTensorName(1), outputBuffer1);
    context->setTensorAddress(engine->getIOTensorName(2), outputBuffer2);
}

void TensorRTEngine::cleanup() {
    context.reset();
    engine.reset();
    runtime.reset();
}

void TensorRTEngine::freeMemory() {
    // 在TensorRT 10.x中，内存释放在推理时动态进行
}
