#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include "../hpp/ocr_orientation.hpp"
#include "../hpp/engine.hpp"
#include "../hpp/utils.hpp"

void OrientationLogger::log(Severity severity, const char* msg) noexcept {
    if (severity != Severity::kINFO) {
        std::cout << msg << std::endl;
    }
}

OCROrientation::OCROrientation()
    : input_height_(80), input_width_(160), input_channels_(3), max_batch_size_(1) {
    logger_ = std::make_unique<OrientationLogger>();
    initializeClassIdMap();
}

OCROrientation::~OCROrientation() {
    freeMemory();
}

bool OCROrientation::initialize(const std::string& model_path) {
    if (!loadEngine(model_path)) {
        std::cerr << "Failed to load orientation engine: " << model_path << std::endl;
        return false;
    }
    if (!allocateMemory()) {
        std::cerr << "Failed to allocate GPU memory for orientation" << std::endl;
        return false;
    }
    return true;
}

bool OCROrientation::loadEngine(const std::string& model_path) {
    ocrEngine::EngineIOInfo io;
    bool ok = ocrEngine::load_engine(model_path, *logger_, runtime_, engine_, context_, io, input_height_, input_width_);
    if (!ok) return false;
    input_name_ = io.input_name;
    output_name_ = io.output_name;
    std::cout << "[TRT][Ori] Input tensor name: " << input_name_ << std::endl;
    std::cout << "[TRT][Ori] Output tensor name: " << output_name_ << std::endl;
    nvinfer1::Dims in_dims = engine_->getTensorShape(input_name_.c_str());
    std::cout << "[TRT][Ori] Engine input dims (NCHW): ["
              << (in_dims.nbDims > 0 ? in_dims.d[0] : -1) << ", "
              << (in_dims.nbDims > 1 ? in_dims.d[1] : -1) << ", "
              << (in_dims.nbDims > 2 ? in_dims.d[2] : -1) << ", "
              << (in_dims.nbDims > 3 ? in_dims.d[3] : -1) << "]" << std::endl;
    std::cout << "[TRT][Ori] Using input size HxW: " << input_height_ << "x" << input_width_ << std::endl;
    return true;
}

bool OCROrientation::allocateMemory() {
    // 分类通常输出 [N, C] 或 [N, C, 1, 1]，给个小的回退buffer
    size_t fallback_elems = static_cast<size_t>(max_batch_size_) * 16;
    return ocrEngine::allocate_buffers(
        engine_.get(), input_name_, output_name_,
        max_batch_size_, input_channels_, input_height_, input_width_,
        d_input_, d_output_, stream_, fallback_elems);
}

void OCROrientation::freeMemory() {
    ocrEngine::free_buffers(d_input_, d_output_, stream_);
}

std::vector<float> OCROrientation::preprocess(const cv::Mat& image) {
    cv::Mat src;
    if (image.channels() == 3) src = image; else cv::cvtColor(image, src, cv::COLOR_GRAY2BGR);
    cv::Mat resized = ocrUtils::resize_image(src, input_width_, input_height_, resize_mode_, cv::Scalar(0,0,0));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    cv::Mat normalized = ocrUtils::normalize_image_rgb(resized);
    return ocrUtils::hwc_to_chw(normalized);
}

OrientationResult OCROrientation::postprocess(const float* output, const nvinfer1::Dims& output_dims) {
    // 计算总元素数量
    int total = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        total *= output_dims.d[i] > 0 ? output_dims.d[i] : 1;
    }
    int num_classes = total;
    
    // 打印输出信息
    std::cout << "[TRT][Ori] Output dims: [";
    for (int i = 0; i < output_dims.nbDims; ++i) {
        std::cout << output_dims.d[i];
        if (i < output_dims.nbDims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "[TRT][Ori] Raw output values: [";
    for (int i = 0; i < num_classes; ++i) {
        std::cout << output[i];
        if (i < num_classes - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // 按照Python Topk逻辑：argsort + 取topk=1（即最大值）
    // Python: indexes = preds[0].argsort(axis=1)[:, -topk:][:, ::-1]
    // 这里我们实现topk=1的情况，即找最大值的索引
    
    // 创建索引-值对进行排序
    std::vector<std::pair<float, int>> indexed_scores;
    for (int i = 0; i < num_classes; ++i) {
        indexed_scores.emplace_back(output[i], i);
    }
    
    // 按分数降序排序（对应Python的[::-1]）
    std::sort(indexed_scores.begin(), indexed_scores.end(), 
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });
    
    // 取top1结果
    int best_class = indexed_scores[0].second;
    float best_score = indexed_scores[0].first;
    
    // 四舍五入到5位小数（对应Python的np.around(decimals=5)）
    best_score = std::round(best_score * 100000.0f) / 100000.0f;
    
    // 获取标签名称（对应Python的self.class_id_map[i]）
    std::string label_name = "unknown";
    if (class_id_map_.find(best_class) != class_id_map_.end()) {
        label_name = class_id_map_[best_class];
    }
    
    std::cout << "[TRT][Ori] Detected orientation: class=" << best_class 
              << ", confidence=" << best_score 
              << ", label_name=" << label_name << std::endl;
    
    OrientationResult r{best_class, best_score, label_name};
    return r;
}

OrientationResult OCROrientation::infer(const cv::Mat& image) {
    std::vector<float> chw = preprocess(image);
    ocrEngine::set_input_shape(context_.get(), input_name_, 1, input_channels_, input_height_, input_width_);
    std::vector<float> output;
    nvinfer1::Dims output_dims;
    if (!ocrEngine::infer(context_.get(), input_name_, output_name_,
                          chw.data(), chw.size(), d_input_, d_output_, stream_,
                          output, output_dims)) {
        std::cerr << "[Ori] infer failed" << std::endl;
        return OrientationResult{0, 0.0f, "unknown"};
    }
    return postprocess(output.data(), output_dims);
}

void OCROrientation::initializeClassIdMap() {
    class_id_map_[0] = "0_degree";
    class_id_map_[1] = "180_degree";
}


