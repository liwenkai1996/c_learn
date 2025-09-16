#include "../hpp/Tensorrt_inference_v10.hpp"
#include <iostream>
#include <stdexcept>
#include <chrono>

// ==================== TensorRTInferenceV10 主类实现 ====================
TensorRTInferenceV10::TensorRTInferenceV10() {
    // 构造函数中不需要特殊初始化，成员对象会自动构造
}

TensorRTInferenceV10::~TensorRTInferenceV10() {
    // 析构函数中不需要特殊清理，成员对象会自动析构
}

bool TensorRTInferenceV10::loadModel(const std::string& modelPath) {
    // 委托给TensorRT引擎加载模型
    bool success = engine.loadModel(modelPath);
    
    if (success) {
        // 模型加载成功后，更新预处理器和后处理器的参数
        preprocessor.setTargetDimensions(engine.getInputWidth(), 
                                       engine.getInputHeight(), 
                                       engine.getInputChannels());
        
        postprocessor.setInputDimensions(engine.getInputWidth(), 
                                       engine.getInputHeight());
        
        postprocessor.setNumDetections(300);  // 默认值，实际应该从模型获取
        postprocessor.setNumClasses(80);      // 默认值，实际应该从模型获取
        postprocessor.setBboxDim(4);          // 边界框维度
    }
    
    return success;
}

std::vector<DetectionBox> TensorRTInferenceV10::infer(const cv::Mat& inputImage, float confidenceThreshold) {
    using Clock = std::chrono::high_resolution_clock;
    auto t_start = Clock::now();

    // 1. 图像预处理
    std::cout << "=== Starting Image Preprocessing ===" << std::endl;
    PreprocessedImage preprocessedImage = preprocessor.preprocess(inputImage);
    // 将原图尺寸与缩放比例传递给后处理器
    postprocessor.setOriginalScale(preprocessedImage.originalWidth,
                                   preprocessedImage.originalHeight,
                                   preprocessedImage.scale);
    auto t_after_pre = Clock::now();
    auto preprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_pre - t_start).count();
    std::cout << "Preprocess time: " << preprocess_ms << " ms" << std::endl;
    
    // 2. TensorRT推理
    std::cout << "=== Starting TensorRT Inference ===" << std::endl;
    InferenceResult inferenceResult = engine.executeInference(preprocessedImage);
    auto t_after_infer = Clock::now();
    auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_infer - t_after_pre).count();
    std::cout << "Inference time: " << infer_ms << " ms" << std::endl;
    
    // 3. 后处理检测结果
    std::cout << "=== Starting Detection Postprocessing ===" << std::endl;
    // 在后处理内部已做百分比->输入像素->原图像素的换算与裁剪
    std::vector<DetectionBox> detections = postprocessor.postprocess(inferenceResult, confidenceThreshold);
    auto t_after_post = Clock::now();
    auto post_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_post - t_after_infer).count();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_post - t_start).count();
    std::cout << "Postprocess time: " << post_ms << " ms" << std::endl;
    
    std::cout << "=== Complete Inference Pipeline ===" << std::endl;
    std::cout << "Total detections: " << detections.size() << std::endl;
    std::cout << "Total time: " << total_ms << " ms" << std::endl;
    
    return detections;
}

