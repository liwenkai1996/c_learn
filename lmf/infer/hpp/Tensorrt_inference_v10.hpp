#ifndef TENSORRT_INFERENCE_V10_HPP
#define TENSORRT_INFERENCE_V10_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// 包含分离的类头文件
#include "ImagePreprocessor.hpp"
#include "TensorRTEngine.hpp"
#include "DetectionPostprocessor.hpp"
#include "LoggerV10.hpp"

// 主推理类 - 协调各个模块
class TensorRTInferenceV10 {
public:
    TensorRTInferenceV10();
    ~TensorRTInferenceV10();

    // 主要接口
    bool loadModel(const std::string& modelPath);
    std::vector<DetectionBox> infer(const cv::Mat& inputImage, float confidenceThreshold = 0.5f);
    
    // 获取模型信息
    int getInputWidth() const { return preprocessor.getTargetWidth(); }
    int getInputHeight() const { return preprocessor.getTargetHeight(); }
    int getInputChannels() const { return preprocessor.getTargetChannels(); }
    int getOutputSize() const { return engine.getOutputSize(); }
    int getOutputSize2() const { return engine.getOutputSize2(); }
    
    // 检查模型是否已加载
    bool isModelLoaded() const { return engine.isModelLoaded(); }

private:
    ImagePreprocessor preprocessor;
    TensorRTEngine engine;
    DetectionPostprocessor postprocessor;
};

#endif // TENSORRT_INFERENCE_V10_HPP
