#ifndef INFERENCE_ENGINE_HPP
#define INFERENCE_ENGINE_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <chrono>
#include "../../infer/hpp/DetectionPostprocessor.hpp"


// DetectionBox 已在 DetectionPostprocessor.hpp 中定义

// 推理结果结构（避免与 infer 模块的 InferenceResult 同名冲突）
struct EngineResult {
    std::vector<DetectionBox> detections;
    double inference_time_ms;
    double preprocess_time_ms;
    double postprocess_time_ms;
    int status_code;
    std::string status_message;
    
    EngineResult() : inference_time_ms(0.0), preprocess_time_ms(0.0), 
                        postprocess_time_ms(0.0), status_code(0) {}
};

// 推理引擎类 - 封装 infer 模块
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    
    // 初始化引擎
    bool initialize(const std::string& model_path, int device_id = 0);
    
    // 执行推理
    EngineResult infer(const cv::Mat& input_image, float confidence_threshold = 0.5f);
    
    // 批量推理
    std::vector<EngineResult> batch_infer(const std::vector<cv::Mat>& images, 
                                            float confidence_threshold = 0.5f);
    
    // 检查引擎状态
    bool is_ready() const;
    
    // 获取模型信息
    int get_input_width() const;
    int get_input_height() const;
    int get_input_channels() const;
    
    // 获取性能统计
    double get_average_inference_time() const;
    int get_total_inference_count() const;
    
    // 清理资源
    void cleanup();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // 性能统计
    mutable std::chrono::high_resolution_clock::time_point last_inference_start;
    std::vector<double> inference_times;
    int total_inference_count;
};

#endif // INFERENCE_ENGINE_HPP
