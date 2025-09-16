#ifndef OCR_DETECTOR_HPP
#define OCR_DETECTOR_HPP

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "../hpp/engine.hpp"
#include "../hpp/utils.hpp"
#include "../../../third_part/clipper/clipper.hpp"

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// 检测结果结构体
struct DetectionResult {
    std::vector<cv::Point2f> box;  // 检测框的四个顶点
    float confidence;              // 置信度
};

// OCR检测器类
class OCRDetector {
public:
    OCRDetector();
    ~OCRDetector();
    
    // 初始化检测器
    bool initialize(const std::string& model_path);
    
    // 执行检测
    std::vector<DetectionResult> detect(const cv::Mat& image);
    
    // 预处理图像，返回按CHW布局的float数据
    std::vector<float> preprocess(const cv::Mat& image);
    
    // 后处理检测结果（基于概率图阈值、轮廓、最小外接矩形，映射回原图）
    std::vector<DetectionResult> postprocess(const float* output,
                                           const nvinfer1::Dims& output_dims,
                                           const cv::Size& original_size);

private:
    // Helpers (ported from PaddleX boxes_from_bitmap - quad branch)
    static void get_mini_boxes(const std::vector<cv::Point>& contour,
                               std::vector<cv::Point2f>& box,
                               float& short_side);

    static float box_score_fast(const cv::Mat& bitmap,
                                std::vector<cv::Point2f> box);

    static std::vector<cv::Point2f> unclip_quad(const std::vector<cv::Point2f>& quad,
                                                float unclip_ratio);

    static void boxes_from_bitmap(const cv::Mat& pred, const cv::Mat& binMask,
                                  float box_thresh, float unclip_ratio,
                                  std::vector<DetectionResult> &results);

    // TensorRT相关
    std::unique_ptr<Logger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    // 模型参数
    int input_height_;
    int input_width_;
    int input_channels_;
    int max_batch_size_;
    
    // CUDA与命名IO
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    std::string input_name_;
    std::string output_name_;
    cudaStream_t stream_ = nullptr;

    // 预处理缩放模式
    ocrUtils::ResizeMode resize_mode_ = ocrUtils::ResizeMode::KeepAndPad;
    
    // 初始化TensorRT引擎
    bool loadEngine(const std::string& model_path);
    
    // 分配GPU内存
    bool allocateMemory();
    
    // 释放GPU内存
    void freeMemory();
};

#endif // OCR_DETECTOR_HPP
