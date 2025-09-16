#ifndef TENSORRT_ENGINE_HPP
#define TENSORRT_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>

// 前向声明
struct PreprocessedImage;
struct InferenceResult;

// TensorRT推理引擎模块 - 负责模型加载、推理执行等
class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();
    
    // 模型管理
    bool loadModel(const std::string& modelPath);
    bool isModelLoaded() const { return engine != nullptr; }
    
    // 推理执行
    InferenceResult executeInference(const PreprocessedImage& preprocessedImage);
    
    // 获取模型信息
    int getInputWidth() const { return inputWidth; }
    int getInputHeight() const { return inputHeight; }
    int getInputChannels() const { return inputChannels; }
    int getOutputSize() const { return outputSize; }
    int getOutputSize2() const { return outputSize2; }

private:
    // TensorRT组件
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // 模型参数
    int inputWidth;
    int inputHeight;
    int inputChannels;
    int outputSize;
    int outputSize2;
    
    // 输出绑定索引
    int boxesBindingIndex;
    int classesBindingIndex;
    
    // 内部方法
    void cleanup();
    bool allocateMemory();
    void freeMemory();
    void inferOutputDimensions();
    void setupTensorAddresses(void* inputBuffer, void* outputBuffer1, void* outputBuffer2);
};

// 数据结构定义
struct InferenceResult {
    std::vector<float> output1;  // 通常是boxes
    std::vector<float> output2;  // 通常是classes
    int output1Size;
    int output2Size;
};

#endif // TENSORRT_ENGINE_HPP
