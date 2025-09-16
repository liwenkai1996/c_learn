
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <vector>
#include <cmath>
#include "../config.h"
#include <filesystem>
#include "../hpp/ocr_detector.hpp"

// Logger实现
void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity != Severity::kINFO) {
        std::cout << msg << std::endl;
    }
}

// 构造函数
OCRDetector::OCRDetector() 
    : input_height_(640), input_width_(640), input_channels_(3), 
      max_batch_size_(1) {
    logger_ = std::make_unique<Logger>();
}

// 析构函数
OCRDetector::~OCRDetector() {
    freeMemory();
}

// 初始化检测器
bool OCRDetector::initialize(const std::string& model_path) {
    
    if (!loadEngine(model_path)) {
        std::cerr << "Failed to load detection engine: " << model_path << std::endl;
        return false;
    }
    
    if (!allocateMemory()) {
        std::cerr << "Failed to allocate GPU memory" << std::endl;
        return false;
    }
    
    return true;
}

// 加载TensorRT引擎
bool OCRDetector::loadEngine(const std::string& model_path) {
    ocrEngine::EngineIOInfo io;
    bool ok = ocrEngine::load_engine(model_path, *logger_, runtime_, engine_, context_, io, input_height_, input_width_);
    if (!ok) return false;
    input_name_ = io.input_name;
    output_name_ = io.output_name;
    std::cout << "[TRT][Det] Input tensor name: " << input_name_ << std::endl;
    std::cout << "[TRT][Det] Output tensor name: " << output_name_ << std::endl;
    nvinfer1::Dims in_dims = engine_->getTensorShape(input_name_.c_str());
    std::cout << "[TRT][Det] Engine input dims (NCHW): ["
              << (in_dims.nbDims > 0 ? in_dims.d[0] : -1) << ", "
              << (in_dims.nbDims > 1 ? in_dims.d[1] : -1) << ", "
              << (in_dims.nbDims > 2 ? in_dims.d[2] : -1) << ", "
              << (in_dims.nbDims > 3 ? in_dims.d[3] : -1) << "]" << std::endl;
    std::cout << "[TRT][Det] Using input size HxW: " << input_height_ << "x" << input_width_ << std::endl;
    return true;
}

// 分配GPU内存
bool OCRDetector::allocateMemory() {
    // Fallback elements: for DB-like text detection, output ~ [N,1,H,W]
    size_t fallback_elems = static_cast<size_t>(max_batch_size_) * input_height_ * input_width_;
    return ocrEngine::allocate_buffers(
        engine_.get(), input_name_, output_name_,
        max_batch_size_, input_channels_, input_height_, input_width_,
        d_input_, d_output_, stream_, fallback_elems);
}

// 释放GPU内存
void OCRDetector::freeMemory() {
    ocrEngine::free_buffers(d_input_, d_output_, stream_);
}
// 预处理图像
std::vector<float> OCRDetector::preprocess(const cv::Mat& image) {
    cv::Mat src;
    if (image.channels() == 3) src = image; else cv::cvtColor(image, src, cv::COLOR_GRAY2BGR);
    cv::Mat resized = ocrUtils::resize_image(src, input_width_, input_height_, resize_mode_, cv::Scalar(0,0,0));
    try {
        const std::string dbg_dir = "/home/work/liwenkai/c_learn/det_debug";
        std::filesystem::create_directories(dbg_dir);
        cv::imwrite(dbg_dir + "/padded_bgr.jpg", resized);
    } catch (...) {}
    
    // Debug: print first 10 values of resized image
    {
        const uint8_t* p = resized.ptr<uint8_t>();
        int total = resized.rows * resized.cols * resized.channels();
        int n = std::min(10, total);
        std::cout << "[Det][Resized] first values:";
        for (int i = 0; i < n; ++i) {
            std::cout << " " << static_cast<int>(p[i]);
        }
        std::cout << std::endl;
    }
    //这里图像的通道是bgr,但是为了跟paddlex保持一致，还是用rgb规范化方式，
    cv::Mat normalized = ocrUtils::normalize_image_rgb(resized);
    

    // Debug: print first 10 values after normalization (interleaved BGR)
    {
        const float* p = reinterpret_cast<const float*>(normalized.data);
        int total = normalized.rows * normalized.cols * normalized.channels();
        int n = std::min(10, total);
        std::cout << "[Det][Norm] first values:";
        for (int i = 0; i < n; ++i) {
            std::cout << " " << p[i];
        }
        std::cout << std::endl;
    }
    std::vector<float> chw = ocrUtils::hwc_to_chw(normalized);
    return chw;
}

// 执行检测
std::vector<DetectionResult> OCRDetector::detect(const cv::Mat& image) {
    std::vector<float> chw = preprocess(image);
    // 如果输入为动态，设置当前shape
    ocrEngine::set_input_shape(context_.get(), input_name_, 1, input_channels_, input_height_, input_width_);
    std::vector<float> output;
    nvinfer1::Dims output_dims;
    if (!ocrEngine::infer(context_.get(), input_name_, output_name_,
                          chw.data(), chw.size(), d_input_, d_output_, stream_,
                          output, output_dims)) {
        std::cerr << "[Det] infer failed" << std::endl;
        return {};
    }
    
    // 后处理
    auto results = postprocess(output.data(), output_dims, image.size());

    // 将检测到的框体叠加到原图并保存
    try {
        const std::string dbg_dir = "/home/work/liwenkai/c_learn/det_debug";
        std::filesystem::create_directories(dbg_dir);
        cv::Mat vis = image.clone();
        for (const auto& det : results) {
            if (det.box.size() < 4) continue;
            std::vector<cv::Point> poly;
            poly.reserve(4);
            for (int i = 0; i < 4; ++i) {
                int x = std::max(0, std::min(static_cast<int>(std::round(det.box[i].x)), image.cols - 1));
                int y = std::max(0, std::min(static_cast<int>(std::round(det.box[i].y)), image.rows - 1));
                poly.emplace_back(x, y);
            }
            const std::vector<std::vector<cv::Point>> polys{poly};
            cv::polylines(vis, polys, true, cv::Scalar(0, 0, 255), 2);
        }
        cv::imwrite(dbg_dir + "/cpp_boxes_on_orig.jpg", vis);
        std::cout << "[Det][Draw] wrote cpp_boxes_on_orig.jpg" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Det][Draw] exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[Det][Draw] unknown exception while drawing boxes" << std::endl;
    }

    return results;
}

// 后处理检测结果
std::vector<DetectionResult> OCRDetector::postprocess(const float* output,
                                                    const nvinfer1::Dims& output_dims,
                                                    const cv::Size& original_size) {
    std::vector<DetectionResult> results;
    // 固定输出格式为 [N, 1, H, W]
    if (output_dims.nbDims != 4 || output_dims.d[1] != 1) {
        return results;
    }
    int out_h = output_dims.d[2];
    int out_w = output_dims.d[3];
    if (out_h <= 0 || out_w <= 0) return results;
    
    // 构造概率图
    cv::Mat prob(out_h, out_w, CV_32FC1);
    {
        // 取第一张 n=0 的通道
        const float* base = output; // 假设 batch=1
        for (int y = 0; y < out_h; ++y) {
            std::memcpy(prob.ptr<float>(y), base + y * out_w, out_w * sizeof(float));
        }
    }
    
    // 阈值化
    cv::Mat binMask;
    cv::threshold(prob, binMask, OCRConfig::DET_THRESHOLD, 1.0, cv::THRESH_BINARY);
    binMask.convertTo(binMask, CV_8UC1, 255.0);
    
    if (original_size.width > 0 && original_size.height > 0) {
        cv::Rect valid = ocrUtils::compute_valid_region(
            original_size.width, original_size.height,
            binMask.cols, binMask.rows,
            resize_mode_);
        ocrUtils::zero_out_padding(binMask, valid);
    }
    
    // DEBUG: 保存二值化后的概率图
    try {
        const std::string dbg_dir = "/home/work/liwenkai/c_learn/det_debug";
        std::filesystem::create_directories(dbg_dir);
        cv::imwrite(dbg_dir + "/bin_mask.jpg", binMask);
    } catch (...) {}
    
    try {
        boxes_from_bitmap(prob, binMask, OCRConfig::DET_BOX_THRESHOLD,
            OCRConfig::UNCLIP_RATIO, results);
    } catch (...) {
        // 如果处理失败，返回空结果
        results.clear();
    }
    
    return results;
}


void OCRDetector::get_mini_boxes(const std::vector<cv::Point>& contour,
    std::vector<cv::Point2f>& box,
    float& short_side) {
    cv::RotatedRect rr = cv::minAreaRect(contour);
    short_side = std::min(rr.size.width, rr.size.height);
    cv::Point2f pts[4];
    rr.points(pts);
    std::vector<cv::Point2f> points(pts, pts + 4);
    std::sort(points.begin(), points.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

    int index_1 = 0, index_2 = 1, index_3 = 2, index_4 = 3;
    if (points[1].y > points[0].y) { index_1 = 0; index_4 = 1; }
    else { index_1 = 1; index_4 = 0; }
    if (points[3].y > points[2].y) { index_2 = 2; index_3 = 3; }
    else { index_2 = 3; index_3 = 2; }

    box.clear();
    box.push_back(points[index_1]);
    box.push_back(points[index_2]);
    box.push_back(points[index_3]);
    box.push_back(points[index_4]);
}

// box_score_fast: mean score in the quad region on bitmap (float 0~1)
float OCRDetector::box_score_fast(const cv::Mat& bitmap, std::vector<cv::Point2f> box) {
    const int h = bitmap.rows, w = bitmap.cols;
    float xmin = std::max(0.f, std::min(std::floor(box[0].x), (float)w - 1));
    float xmax = std::max(0.f, std::min(std::ceil (box[0].x), (float)w - 1));
    float ymin = std::max(0.f, std::min(std::floor(box[0].y), (float)h - 1));
    float ymax = std::max(0.f, std::min(std::ceil (box[0].y), (float)h - 1));
    for (int i = 1; i < 4; ++i) {
        xmin = std::max(0.f, std::min(std::floor(std::min(xmin, box[i].x)), (float)w - 1));
        xmax = std::max(0.f, std::min(std::ceil (std::max(xmax, box[i].x)), (float)w - 1));
        ymin = std::max(0.f, std::min(std::floor(std::min(ymin, box[i].y)), (float)h - 1));
        ymax = std::max(0.f, std::min(std::ceil (std::max(ymax, box[i].y)), (float)h - 1));
    }
    int ixmin = (int)xmin, ixmax = (int)xmax, iymin = (int)ymin, iymax = (int)ymax;
    if (ixmax < ixmin || iymax < iymin) return 0.f;
    cv::Mat mask = cv::Mat::zeros(iymax - iymin + 1, ixmax - ixmin + 1, CV_8UC1);
    for (auto& p : box) { p.x -= ixmin; p.y -= iymin; }
    std::vector<std::vector<cv::Point>> polys(1);
    polys[0].reserve(4);
    for (auto& p : box) polys[0].emplace_back((int)std::round(p.x), (int)std::round(p.y));
    cv::fillPoly(mask, polys, cv::Scalar(255));
    cv::Mat roi = bitmap(cv::Rect(ixmin, iymin, mask.cols, mask.rows));
    cv::Scalar m = cv::mean(roi, mask);
    return (float)m[0];
}

// Python-like unclip via raster dilation: distance = area * unclip_ratio / length
std::vector<cv::Point2f> OCRDetector::unclip_quad(const std::vector<cv::Point2f>& quad, float unclip_ratio) {
    if (quad.size() < 4) return quad;
    // Compute area and perimeter (arc length)
    std::vector<cv::Point2f> poly = quad;
    std::vector<cv::Point> ipoly; ipoly.reserve(poly.size());
    for (auto &p : poly) ipoly.emplace_back((int)std::round(p.x), (int)std::round(p.y));
    double area = std::fabs(cv::contourArea(ipoly));
    double length = cv::arcLength(ipoly, true);
    double distance = length > 1e-6 ? (area * std::max(0.f, unclip_ratio)) / length : 0.0;

    // Prefer exact polygon offset if Clipper is available
    
    const double scale = 100.0;
    ClipperLib::Path path; path.reserve(ipoly.size());
    for (const auto &pt : ipoly) path.emplace_back((ClipperLib::cInt)std::llround(pt.x * scale), (ClipperLib::cInt)std::llround(pt.y * scale));
    ClipperLib::Paths sol; ClipperLib::ClipperOffset co;
    co.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
    co.Execute(sol, distance * scale);
    if (!sol.empty()) {
        size_t best = 0; double bestArea = 0.0;
        for (size_t i = 0; i < sol.size(); ++i) {
            double a = std::fabs(ClipperLib::Area(sol[i]));
            if (a > bestArea) { bestArea = a; best = i; }
        }
        std::vector<cv::Point> expanded; expanded.reserve(sol[best].size());
        for (const auto &pt : sol[best]) expanded.emplace_back((int)std::lround(pt.X / scale), (int)std::lround(pt.Y / scale));
        if (!expanded.empty()) {
            cv::RotatedRect rr = cv::minAreaRect(expanded);
            cv::Point2f pts[4]; rr.points(pts);
            return std::vector<cv::Point2f>(pts, pts + 4);
        }
    }
    std::cout << "[DEBUG] quad get error: "<< std::endl;
    return quad;
}


// boxes_from_bitmap (quad): returns quads in bitmap coordinate (no scaling to dest)
void OCRDetector::boxes_from_bitmap(const cv::Mat& pred, const cv::Mat& binMask,
       float box_thresh, float unclip_ratio,
       std::vector<DetectionResult> &results) {
    results.clear();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binMask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    
    std::cout << "[DEBUG] Found " << contours.size() << " contours" << std::endl;
    // DEBUG: 保存轮廓点与可视化
    try {
        const std::string dbg_dir = "/home/work/liwenkai/c_learn/det_debug";
        std::filesystem::create_directories(dbg_dir);
        // 画轮廓
        cv::Mat vis;
        if (binMask.channels() == 1) {
            cv::cvtColor(binMask, vis, cv::COLOR_GRAY2BGR);
        } else {
            vis = binMask.clone();
        }
        cv::drawContours(vis, contours, -1, cv::Scalar(0, 255, 0), 1);
        cv::imwrite(dbg_dir + "/cpp_contours_vis.jpg", vis);
        // 写txt
        std::ofstream ofs(dbg_dir + "/cpp_contours.txt");
        for (size_t i = 0; i < contours.size(); ++i) {
            ofs << i << ":";
            const auto &c = contours[i];
            for (const auto &pt : c) {
                ofs << " " << pt.x << "," << pt.y;
            }
            ofs << "\n";
        }
        ofs.close();
    } catch (...) {}
    
    const int max_candidates = 1000;
    int num = std::min((int)contours.size(), max_candidates);
    std::cout << "[DEBUG] Processing " << num << " contours (max_candidates=" << max_candidates << ")" << std::endl;
    
    for (int i = 0; i < num; ++i) {
        const auto& contour = contours[i];
        std::vector<cv::Point2f> quad;
        float sside = 0.f;
        get_mini_boxes(contour, quad, sside);
        if (sside < 3.f) continue;
        float score = box_score_fast(pred, quad);
        if (score < box_thresh) continue;
        // unclip & refine to quad again
        auto expanded = unclip_quad(quad, unclip_ratio);
        std::vector<cv::Point> exp_contour;
        exp_contour.reserve(4);
        for (auto& p : expanded) exp_contour.emplace_back((int)std::round(p.x), (int)std::round(p.y));
        // 重新计算quad和sside，与Python版本保持一致
        std::vector<cv::Point2f> refined_quad;
        float refined_sside = 0.f;
        get_mini_boxes(exp_contour, refined_quad, refined_sside);
        if (refined_sside < 5.f) continue;  // 使用更严格的阈值，与Python版本一致
        
        // 使用refined_quad而不是原来的quad
        for (auto& p : refined_quad) {
            p.x = std::max(0.f, std::min(p.x, (float)binMask.cols - 1));
            p.y = std::max(0.f, std::min(p.y, (float)binMask.rows - 1));
        }
        DetectionResult det;
        det.box = refined_quad;  // 使用refined_quad而不是原来的quad
        det.confidence = score;
        results.push_back(det);
    }
}
