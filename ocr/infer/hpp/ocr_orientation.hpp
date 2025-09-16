#ifndef OCR_ORIENTATION_HPP
#define OCR_ORIENTATION_HPP

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "../hpp/utils.hpp"

// 独立的Logger，避免与检测模块的Logger重复定义
class OrientationLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// 文档方向结果
struct OrientationResult {
    int label;                    // 方向类别：0/1 等
    float confidence;             // 置信度
    std::string label_name;       // 标签名称：如 "0_degree", "180_degree" 等
};

// 文档方向分类器
class OCROrientation {
public:
    OCROrientation();
    ~OCROrientation();

    // 初始化（加载TensorRT引擎并分配内存）
    bool initialize(const std::string& model_path);

    // 推理（完整流程：预处理->推理->后处理）
    OrientationResult infer(const cv::Mat& image);

    // 预处理：与 OCRDetector 完全一致，返回CHW float
    std::vector<float> preprocess(const cv::Mat& image);

    // 后处理：占位
    OrientationResult postprocess(const float* output,
                                  const nvinfer1::Dims& output_dims);

private:
    // TensorRT相关
    std::unique_ptr<OrientationLogger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // 输入规格
    int input_height_;
    int input_width_;
    int input_channels_;
    int max_batch_size_;

    // CUDA & IO 名称
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    std::string input_name_;
    std::string output_name_;
    cudaStream_t stream_ = nullptr;
    
    // 预处理缩放模式
    ocrUtils::ResizeMode resize_mode_ = ocrUtils::ResizeMode::Stretch;
    
    // 类别ID到标签名称的映射
    std::map<int, std::string> class_id_map_;

    // 内部方法
    bool loadEngine(const std::string& model_path);
    bool allocateMemory();
    void freeMemory();
    void initializeClassIdMap();
};

#endif // OCR_ORIENTATION_HPP


