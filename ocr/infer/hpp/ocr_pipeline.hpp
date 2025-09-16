#ifndef OCR_PIPELINE_HPP
#define OCR_PIPELINE_HPP

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ocr_detector.hpp"
#include "ocr_recognizer.hpp"
#include "ocr_orientation.hpp"

// 最终OCR结果结构体
struct OCRResult {
    std::vector<cv::Point2f> box;  // 检测框
    std::string text;              // 识别的文本
    float detection_confidence;    // 检测置信度
    float recognition_confidence;  // 识别置信度
    float overall_confidence;      // 综合置信度
};

// OCR流水线类
class OCRPipeline {
public:
    OCRPipeline();
    ~OCRPipeline();
    
    // 初始化OCR流水线
    bool initialize(const std::string& det_model_path,
                   const std::string& ori_model_path,
                   const std::string& rec_model_path,
                   const std::string& dict_path,
                   float rec_threshold = 0.5f);
    
    // 执行完整的OCR识别
    std::vector<OCRResult> run(const cv::Mat& image);
    
    // 设置参数
    void setDetectionThreshold(float threshold);
    void setDetectionNMSThreshold(float threshold);
    void setRecognitionThreshold(float threshold);
    
    // 获取检测器和识别器
    OCRDetector* getDetector() { return detector_.get(); }
    OCRRecognizer* getRecognizer() { return recognizer_.get(); }

private:
    std::unique_ptr<OCRDetector> detector_;
    std::unique_ptr<OCRRecognizer> recognizer_;
    std::unique_ptr<OCROrientation> orientation_;
    
    // 从检测框提取文本区域
    cv::Mat extractTextRegion(const cv::Mat& image, 
                             const std::vector<cv::Point2f>& box);
    
    // 完全按照Python版本的get_rotate_crop_image实现
    cv::Mat getRotateCropImage(const cv::Mat& img, const std::vector<cv::Point2f>& points);
    
    
    // 检查检测框是否为矩形
    bool isRectangular(const std::vector<cv::Point2f>& box);
    
    // 透视变换
    cv::Mat perspectiveTransform(const cv::Mat& image,
                               const std::vector<cv::Point2f>& src_points,
                               const cv::Size& dst_size);
    
    // 计算综合置信度
    float calculateOverallConfidence(float det_conf, float rec_conf);
    
    // 图像旋转函数（对应Python的rotate_image）
    cv::Mat rotateImage(const cv::Mat& image, int angle);
};

#endif // OCR_PIPELINE_HPP

