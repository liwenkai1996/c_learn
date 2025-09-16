#include "../hpp/InferenceEngine.hpp"
#include "../../infer/hpp/Tensorrt_inference_v10.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>


// PIMPL 实现类
class InferenceEngine::Impl {
public:
    Impl() : engine_initialized(false), device_id(0), total_inference_count(0) {}
    
    ~Impl() = default;
    
    // 核心推理引擎
    std::unique_ptr<TensorRTInferenceV10> inference_engine;
    
    // 状态管理
    bool engine_initialized;
    int device_id;
    
    // 性能统计
    std::vector<double> inference_times;
    int total_inference_count;
    
    // 模型信息
    int input_width;
    int input_height;
    int input_channels;
};

// 推理引擎实现
InferenceEngine::InferenceEngine() 
    : pImpl(std::make_unique<Impl>()), total_inference_count(0) {
}

InferenceEngine::~InferenceEngine() = default;

bool InferenceEngine::initialize(const std::string& model_path, int device_id) {
    try {
        std::cout << "Initializing inference engine..." << std::endl;
        std::cout << "Model path: " << model_path << std::endl;
        std::cout << "Device ID: " << device_id << std::endl;
        
        // 设置 CUDA 设备
        if (device_id >= 0) {
            cudaSetDevice(device_id);
            pImpl->device_id = device_id;
        }
        
        // 创建推理引擎
        pImpl->inference_engine = std::make_unique<TensorRTInferenceV10>();
        
        // 加载模型
        if (!pImpl->inference_engine->loadModel(model_path)) {
            std::cerr << "Failed to load model from " << model_path << std::endl;
            pImpl->inference_engine.reset();
            return false;
        }
        
        // 获取模型信息
        pImpl->input_width = pImpl->inference_engine->getInputWidth();
        pImpl->input_height = pImpl->inference_engine->getInputHeight();
        pImpl->input_channels = pImpl->inference_engine->getInputChannels();
        
        pImpl->engine_initialized = true;
        
        std::cout << "Inference engine initialized successfully!" << std::endl;
        std::cout << "Input dimensions: " << pImpl->input_width << "x" 
                  << pImpl->input_height << "x" << pImpl->input_channels << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing inference engine: " << e.what() << std::endl;
        return false;
    }
}

EngineResult InferenceEngine::infer(const cv::Mat& input_image, float confidence_threshold) {
    EngineResult result;
    
    if (!pImpl->engine_initialized) {
        result.status_code = -1;
        result.status_message = "Engine not initialized";
        return result;
    }
    
    if (input_image.empty()) {
        result.status_code = -2;
        result.status_message = "Input image is empty";
        return result;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 执行推理
        std::vector<DetectionBox> detections = pImpl->inference_engine->infer(input_image, confidence_threshold);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // 转换检测结果
        // 直接沿用 infer 模块的 DetectionBox（rect/classId/confidence）
        result.detections = detections;
        
        // 更新性能统计
        result.inference_time_ms = total_time.count() / 1000.0;
        pImpl->inference_times.push_back(result.inference_time_ms);
        pImpl->total_inference_count++;
        total_inference_count++;
        
        result.status_code = 0;
        result.status_message = "Success";
        
        std::cout << "Inference completed in " << result.inference_time_ms << " ms" << std::endl;
        std::cout << "Detections found: " << result.detections.size() << std::endl;
        
    } catch (const std::exception& e) {
        result.status_code = -3;
        result.status_message = std::string("Inference error: ") + e.what();
        std::cerr << "Error during inference: " << e.what() << std::endl;
    }
    return result;
}

std::vector<EngineResult> InferenceEngine::batch_infer(const std::vector<cv::Mat>& images, 
                                                          float confidence_threshold) {
    std::vector<EngineResult> results;
    
    if (!pImpl->engine_initialized) {
        EngineResult error_result;
        error_result.status_code = -1;
        error_result.status_message = "Engine not initialized";
        results.resize(images.size(), error_result);
        return results;
    }
    
    results.reserve(images.size());
    
    for (const auto& image : images) {
        results.push_back(infer(image, confidence_threshold));
    }
    
    return results;
}

bool InferenceEngine::is_ready() const {
    return pImpl->engine_initialized;
}

int InferenceEngine::get_input_width() const {
    return pImpl->input_width;
}

int InferenceEngine::get_input_height() const {
    return pImpl->input_height;
}

int InferenceEngine::get_input_channels() const {
    return pImpl->input_channels;
}

double InferenceEngine::get_average_inference_time() const {
    if (pImpl->inference_times.empty()) {
        return 0.0;
    }
    
    double sum = std::accumulate(pImpl->inference_times.begin(), pImpl->inference_times.end(), 0.0);
    return sum / pImpl->inference_times.size();
}

int InferenceEngine::get_total_inference_count() const {
    return pImpl->total_inference_count;
}

void InferenceEngine::cleanup() {
    try {
        if (pImpl->inference_engine) {
            pImpl->inference_engine.reset();
        }
        pImpl->engine_initialized = false;
        
        // 清空性能统计
        pImpl->inference_times.clear();
        pImpl->total_inference_count = 0;
        total_inference_count = 0;
        
        std::cout << "Inference engine cleaned up" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during cleanup: " << e.what() << std::endl;
    }
}
