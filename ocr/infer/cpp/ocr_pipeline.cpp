#include "../hpp/ocr_pipeline.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>

// 构造函数
OCRPipeline::OCRPipeline() {
    detector_ = std::make_unique<OCRDetector>();
    recognizer_ = std::make_unique<OCRRecognizer>();
    orientation_ = std::make_unique<OCROrientation>();
}

// 析构函数
OCRPipeline::~OCRPipeline() = default;

// 初始化OCR流水线
bool OCRPipeline::initialize(const std::string& det_model_path,
                           const std::string& ori_model_path,
                           const std::string& rec_model_path,
                           const std::string& dict_path,
                           float rec_threshold) {
    
    // 初始化检测器
    if (!detector_->initialize(det_model_path)) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return false;
    }
    // 初始化方向分类器（使用配置中的默认路径）
    if (!orientation_->initialize(ori_model_path)) {
        std::cerr << "Failed to initialize orientation classifier: " << ori_model_path << std::endl;
        return false;
    }
    // 初始化识别器
    if (!recognizer_->initialize(rec_model_path, dict_path, rec_threshold)) {
        std::cerr << "Failed to initialize recognizer" << std::endl;
        return false;
    }

    std::cout << "OCR Pipeline initialized successfully" << std::endl;
    return true;
}

// 执行完整的OCR识别
std::vector<OCRResult> OCRPipeline::run(const cv::Mat& image) {
    std::vector<OCRResult> results;
    static bool roi_dir_ready = false;
    static std::string roi_dir = "/home/work/liwenkai/c_learn/ocr_rois";
    static int roi_count = 0;
    if (!roi_dir_ready) {
        struct stat st;
        if (stat(roi_dir.c_str(), &st) != 0) {
            mkdir(roi_dir.c_str(), 0755);
        }
        roi_dir_ready = true;
    }

    //所有输入需调整为32的倍数
    int width = image.cols;
    int height = image.rows;
    int resize_h = std::max(int(round(height / 32.0f) * 32), 32);
    int resize_w = std::max(int(round(width / 32.0f) * 32), 32);
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resize_w, resize_h));
    // 1. 文本检测
    auto detections = detector_->detect(resized);
    if (detections.empty()) {
        std::cout << "No text regions detected" << std::endl;
        return results;
    }
    
    std::cout << "Detected " << detections.size() << " text regions" << std::endl;
    
    // 2. 对每个检测区域进行文本识别
    for (const auto& detection : detections) {
        // 提取文本区域
        cv::Mat text_region = extractTextRegion(resized, detection.box);
        if (text_region.empty()) {
            continue;
        }
        // 保存ROI
        {
            char name[64];
            std::snprintf(name, sizeof(name), "roi_%05d.png", roi_count++);
            cv::imwrite(roi_dir + "/" + name, text_region);
        }
        


        auto ori_res = orientation_->infer(text_region);
        // 按照Python版本的旋转逻辑：0->0度，1->180度
        int rotate_angle = ori_res.label * 180;  // 对应Python: rotate_angle = rotate_indicator * 180
        if (rotate_angle > 0) {
            text_region = rotateImage(text_region, rotate_angle);
            // 保存旋转后的图像用于调试
            
        }
        char rotated_name[64];
        std::snprintf(rotated_name, sizeof(rotated_name), "rotated_%05d_angle%d.png", roi_count-1, rotate_angle);
        cv::imwrite(roi_dir + "/" + rotated_name, text_region);
        std::cout << "[DEBUG] Saved rotated image: " << rotated_name << " (angle=" << rotate_angle << ")" << std::endl;
        
        // 文本识别
        auto recognition = recognizer_->recognize(text_region);
        
        // 构建最终结果
        OCRResult result;
        result.box = detection.box;
        result.text = recognition.text;
        result.detection_confidence = detection.confidence;
        result.recognition_confidence = recognition.confidence;
        result.overall_confidence = calculateOverallConfidence(
            detection.confidence, recognition.confidence);
        
        results.push_back(result);
    }
    
    return results;
}

// 设置检测阈值
void OCRPipeline::setDetectionThreshold(float threshold) {
    // 这里需要重新初始化检测器，或者添加动态设置功能
    std::cout << "Detection threshold set to: " << threshold << std::endl;
}

// 设置检测NMS阈值
void OCRPipeline::setDetectionNMSThreshold(float threshold) {
    // 这里需要重新初始化检测器，或者添加动态设置功能
    std::cout << "Detection NMS threshold set to: " << threshold << std::endl;
}

// 设置识别阈值
void OCRPipeline::setRecognitionThreshold(float threshold) {
    // 这里需要重新初始化识别器，或者添加动态设置功能
    std::cout << "Recognition threshold set to: " << threshold << std::endl;
}

// 从检测框提取文本区域 - 完全按照Python版本实现
cv::Mat OCRPipeline::extractTextRegion(const cv::Mat& image, 
                                      const std::vector<cv::Point2f>& box) {
    if (box.size() != 4) {
        std::cerr << "Invalid detection box: expected 4 points" << std::endl;
        return cv::Mat();
    }
    

    // 先检查输入点是否有效
    std::cout << "[DEBUG] Input box points: ";
    for (const auto& pt : box) {
        std::cout << "(" << pt.x << "," << pt.y << ") ";
        if (std::isnan(pt.x) || std::isnan(pt.y) || std::isinf(pt.x) || std::isinf(pt.y)) {
            std::cerr << "Invalid point detected: (" << pt.x << "," << pt.y << ")" << std::endl;
            return cv::Mat();
        }
    }
    std::cout << std::endl;
    
    // 转换为cv::Point2i用于minAreaRect
    std::vector<cv::Point2i> int_points;
    for (const auto& pt : box) {
        int_points.emplace_back(static_cast<int>(pt.x), static_cast<int>(pt.y));
    }
    
    // 检查点是否有效
    if (int_points.empty()) {
        std::cerr << "No valid points for minAreaRect" << std::endl;
        return cv::Mat();
    }
    
    std::cout << "[DEBUG] Calling minAreaRect with " << int_points.size() << " points" << std::endl;
    
    // 使用最小外接矩形，完全按照Python版本
    cv::RotatedRect bounding_box = cv::minAreaRect(int_points);
    // 使用cv::Mat作为OutputArray以适配OpenCV的boxPoints接口
    cv::Mat box_points_mat(4, 1, CV_32FC2);
    cv::boxPoints(bounding_box, box_points_mat);
    std::vector<cv::Point2f> box_points(4);
    for (int i = 0; i < 4; ++i) {
        box_points[i] = box_points_mat.at<cv::Point2f>(i, 0);
    }
    
    std::cout << "[DEBUG] boxPoints returned " << box_points.size() << " points" << std::endl;
    
    // 检查box_points是否有效
    if (box_points.size() != 4) {
        std::cerr << "Invalid box_points size: " << box_points.size() << std::endl;
        return cv::Mat();
    }
    
    // 按x坐标排序，完全按照Python版本
    std::sort(box_points.begin(), box_points.end(), 
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    
    // 根据y坐标重新排序，完全按照Python版本
    int index_a, index_b, index_c, index_d;
    if (box_points[1].y > box_points[0].y) {
        index_a = 0; index_d = 1;
    } else {
        index_a = 1; index_d = 0;
    }
    if (box_points[3].y > box_points[2].y) {
        index_b = 2; index_c = 3;
    } else {
        index_b = 3; index_c = 2;
    }
    
    std::vector<cv::Point2f> ordered_points = {
        box_points[index_a], box_points[index_b], 
        box_points[index_c], box_points[index_d]
    };
    
    // 调用get_rotate_crop_image函数
    return getRotateCropImage(image, ordered_points);
    
}

// 完全按照Python版本的get_rotate_crop_image实现
cv::Mat OCRPipeline::getRotateCropImage(const cv::Mat& img, const std::vector<cv::Point2f>& points) {
    if (points.size() != 4) {
        std::cerr << "shape of points must be 4*2" << std::endl;
        return cv::Mat();
    }
    
    // 计算目标尺寸，完全按照Python版本
    auto dist = [](const cv::Point2f& a, const cv::Point2f& b) {
        return std::hypot(a.x - b.x, a.y - b.y);
    };
    
    int img_crop_width = static_cast<int>(std::max(
        dist(points[0], points[1]),
        dist(points[2], points[3])
    ));
    
    int img_crop_height = static_cast<int>(std::max(
        dist(points[0], points[3]),
        dist(points[1], points[2])
    ));
    
    // 检查尺寸是否有效
    if (img_crop_width <= 0 || img_crop_height <= 0) {
        std::cerr << "Invalid crop dimensions: " << img_crop_width << "x" << img_crop_height << std::endl;
        return cv::Mat();
    }
    
    // 目标点，完全按照Python版本
    std::vector<cv::Point2f> pts_std = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(img_crop_width), 0.0f),
        cv::Point2f(static_cast<float>(img_crop_width), static_cast<float>(img_crop_height)),
        cv::Point2f(0.0f, static_cast<float>(img_crop_height))
    };
    
    // 透视变换，完全按照Python版本
    std::cout << "[DEBUG] Input points: ";
    for (const auto& pt : points) {
        std::cout << "(" << pt.x << "," << pt.y << ") ";
    }
    std::cout << std::endl;
    
    std::cout << "[DEBUG] Target size: " << img_crop_width << "x" << img_crop_height << std::endl;
    
    cv::Mat M = cv::getPerspectiveTransform(points, pts_std);
    cv::Mat dst_img;
    cv::warpPerspective(img, dst_img, M, cv::Size(img_crop_width, img_crop_height),
                       cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    
    // 旋转逻辑，完全按照Python版本（np.rot90是逆时针90度）
    int dst_img_height = dst_img.rows;
    int dst_img_width = dst_img.cols;
    if (dst_img_height * 1.0 / dst_img_width >= 1.5) {
        cv::rotate(dst_img, dst_img, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    
    return dst_img;
}

// 检查检测框是否为矩形
bool OCRPipeline::isRectangular(const std::vector<cv::Point2f>& box) {
    if (box.size() != 4) return false;
    
    // 计算四个角的角度，判断是否为矩形
    std::vector<double> angles;
    for (int i = 0; i < 4; ++i) {
        cv::Point2f p1 = box[i];
        cv::Point2f p2 = box[(i + 1) % 4];
        cv::Point2f p3 = box[(i + 2) % 4];
        
        cv::Point2f v1 = p2 - p1;
        cv::Point2f v2 = p3 - p2;
        
        double angle = std::atan2(v2.y, v2.x) - std::atan2(v1.y, v1.x);
        angle = std::abs(angle * 180.0 / CV_PI);
        if (angle > 180) angle = 360 - angle;
        
        angles.push_back(angle);
    }
    
    // 检查角度是否接近90度
    for (double angle : angles) {
        if (std::abs(angle - 90.0) > 10.0) {
            return false;
        }
    }
    
    return true;
}

// 透视变换
cv::Mat OCRPipeline::perspectiveTransform(const cv::Mat& image,
                                        const std::vector<cv::Point2f>& src_points,
                                        const cv::Size& dst_size) {
    if (src_points.size() != 4) {
        std::cerr << "Invalid source points for perspective transform" << std::endl;
        return cv::Mat();
    }
    
    // 定义目标点（矩形）
    std::vector<cv::Point2f> dst_points = {
        cv::Point2f(0, 0),
        cv::Point2f(dst_size.width, 0),
        cv::Point2f(dst_size.width, dst_size.height),
        cv::Point2f(0, dst_size.height)
    };
    
    // 计算透视变换矩阵
    cv::Mat transform_matrix = cv::getPerspectiveTransform(src_points, dst_points);
    
    // 执行透视变换
    cv::Mat transformed;
    cv::warpPerspective(image, transformed, transform_matrix, dst_size);
    
    return transformed;
}

// 计算综合置信度
float OCRPipeline::calculateOverallConfidence(float det_conf, float rec_conf) {
    // 简单的加权平均，可以根据需要调整权重
    float det_weight = 0.4f;
    float rec_weight = 0.6f;
    
    return det_conf * det_weight + rec_conf * rec_weight;
}

// 图像旋转函数（完全按照Python版本实现）
cv::Mat OCRPipeline::rotateImage(const cv::Mat& image, int angle) {
    // 对应Python: if angle < 1e-7: return image
    if (angle < 1) {
        return image.clone();
    }
    
    // 对应Python: h, w = image.shape[:2]
    int h = image.rows;
    int w = image.cols;
    
    // 对应Python: center = (w / 2, h / 2)
    cv::Point2f center(w / 2.0f, h / 2.0f);
    
    // 对应Python: mat = cv2.getRotationMatrix2D(center, angle, scale)
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, static_cast<double>(angle), 1.0);
    
    // 对应Python: 计算新的图像尺寸
    double cos_val = std::abs(rotation_matrix.at<double>(0, 0));
    double sin_val = std::abs(rotation_matrix.at<double>(0, 1));
    int new_w = static_cast<int>((h * sin_val) + (w * cos_val));
    int new_h = static_cast<int>((h * cos_val) + (w * sin_val));
    
    // 对应Python: 调整旋转矩阵的平移部分
    rotation_matrix.at<double>(0, 2) += (new_w - w) / 2.0;
    rotation_matrix.at<double>(1, 2) += (new_h - h) / 2.0;
    
    // 对应Python: cv2.warpAffine(image, mat, dst_size, flags=cv2.INTER_CUBIC)
    cv::Mat rotated;
    cv::warpAffine(image, rotated, rotation_matrix, cv::Size(new_w, new_h), 
                   cv::INTER_CUBIC);
    
    return rotated;
}