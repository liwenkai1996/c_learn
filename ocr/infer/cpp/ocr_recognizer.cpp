#include "../hpp/ocr_recognizer.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <map>
#include <numeric>
#include <iterator>
#include "../hpp/engine.hpp"
#include "../hpp/utils.hpp"
#include "../config.h"

// CTCLabelDecode 实现 - 完全对应Python的CTCLabelDecode
CTCLabelDecode::CTCLabelDecode(const std::vector<std::string>& character_list, bool use_space_char) {
    reverse_ = false;
    
    std::vector<std::string> character_list_copy = character_list;
    if (use_space_char) {
        character_list_copy.push_back(" ");
    }
    
    // 检查是否已经包含blank
    if (!character_list_copy.empty() && character_list_copy[0] == "blank") {
        // 字典已经包含blank，直接使用
        character_ = character_list_copy;
    } else {
        // 字典不包含blank，需要添加
        character_ = add_special_char(character_list_copy);
    }
    
    // 构建字典映射
    for (size_t i = 0; i < character_.size(); ++i) {
        dict_[character_[i]] = static_cast<int>(i);
    }
}

std::vector<std::string> CTCLabelDecode::add_special_char(const std::vector<std::string>& character_list) {
    std::vector<std::string> result = {"blank"};
    result.insert(result.end(), character_list.begin(), character_list.end());
    return result;
}

std::vector<int> CTCLabelDecode::get_ignored_tokens() {
    return {0};  // for ctc blank
}

std::pair<std::string, float> CTCLabelDecode::decode(
    const std::vector<int>& text_index, 
    const std::vector<float>& text_prob,
    bool is_remove_duplicate) {
    
    std::vector<int> ignored_tokens = get_ignored_tokens();
    std::vector<bool> selection(text_index.size(), true);
    
    // 去重逻辑 - 对应Python的is_remove_duplicate
    if (is_remove_duplicate) {
        for (size_t i = 1; i < text_index.size(); ++i) {
            selection[i] = (text_index[i] != text_index[i-1]);
        }
    }
    
    // 忽略blank token
    for (int ignored_token : ignored_tokens) {
        for (size_t i = 0; i < text_index.size(); ++i) {
            if (text_index[i] == ignored_token) {
                selection[i] = false;
            }
        }
    }
    
    // 构建字符列表
    std::string text;
    std::vector<float> conf_list;
    
    for (size_t i = 0; i < text_index.size(); ++i) {
        if (selection[i]) {
            int text_id = text_index[i];
            if (text_id >= 0 && text_id < static_cast<int>(character_.size())) {
                text += character_[text_id];
                
                if (!text_prob.empty() && i < text_prob.size()) {
                    conf_list.push_back(text_prob[i]);
                } else {
                    conf_list.push_back(1.0f);
                }
            }
        }
    }
    
    if (conf_list.empty()) {
        conf_list.push_back(0.0f);
    }
    
    // 计算平均置信度
    float avg_conf = std::accumulate(conf_list.begin(), conf_list.end(), 0.0f) / conf_list.size();
    
    return {text, avg_conf};
}

std::pair<std::string, float> CTCLabelDecode::operator()(const std::vector<std::vector<float>>& pred) {
    // 对应Python的__call__方法 - 简化版本，单个序列
    if (pred.empty()) {
        return {"", 0.0f};
    }
    
    // 计算argmax和max
    std::vector<int> indices;
    std::vector<float> probs;
    
    for (const std::vector<float>& time_pred : pred) {
        int max_idx = 0;
        float max_prob = time_pred[0];
        
        for (size_t i = 1; i < time_pred.size(); ++i) {
            if (time_pred[i] > max_prob) {
                max_prob = time_pred[i];
                max_idx = static_cast<int>(i);
            }
        }
        
        indices.push_back(max_idx);
        probs.push_back(max_prob);
    }
    
    // 调用decode方法
    return decode(indices, probs, true);
}

// 简单Logger类实现
// 使用调用方提供的 ILogger（通过 trt::load_engine 支持），此处不再定义全局logger

// 构造函数
OCRRecognizer::OCRRecognizer() 
    : input_height_(48), input_width_(320), input_channels_(3), 
      max_batch_size_(1), dict_size_(0), threshold_(0.5f),
      d_input_(nullptr), d_output_(nullptr), ctc_decoder_(nullptr) {}

// 析构函数
OCRRecognizer::~OCRRecognizer() {
    freeMemory();
}

// 初始化识别器
bool OCRRecognizer::initialize(const std::string& model_path, 
                              const std::string& dict_path,
                              float threshold) {
    threshold_ = threshold;
    
    if (!loadEngine(model_path)) {
        std::cerr << "Failed to load recognition engine: " << model_path << std::endl;
        return false;
    }
    
    // 使用配置文件中的字典路径
    if (!loadDict(OCRConfig::DICT_PATH)) {
        std::cerr << "Failed to load dictionary: " << OCRConfig::DICT_PATH << std::endl;
        return false;
    }
    
    // 初始化CTC解码器 - 对应Python的build_postprocess
    // 注意：final_dict.txt已经包含了blank，所以不需要额外添加
    ctc_decoder_ = std::make_unique<CTCLabelDecode>(dict_, false);
    
    if (!allocateMemory()) {
        std::cerr << "Failed to allocate GPU memory" << std::endl;
        return false;
    }
    
    return true;
}

// 加载TensorRT引擎
bool OCRRecognizer::loadEngine(const std::string& model_path) {
    // Reuse detector's logger if available; otherwise, use TensorRT default logger through a minimal adapter.
    class LocalLogger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity != Severity::kINFO) std::cout << msg << std::endl;
        }
    } localLogger;
    ocrEngine::EngineIOInfo io;
    bool ok = ocrEngine::load_engine(model_path, localLogger, runtime_, engine_, context_, io, input_height_, input_width_);
    if (!ok) return false;
    input_name_ = io.input_name;
    output_name_ = io.output_name;
    nvinfer1::Dims in_dims = engine_->getTensorShape(input_name_.c_str());
    std::cout << "[TRT] Input tensor name: " << input_name_ << std::endl;
    std::cout << "[TRT] Output tensor name: " << output_name_ << std::endl;
    std::cout << "[TRT] Engine input dims (NCHW): ["
              << (in_dims.nbDims > 0 ? in_dims.d[0] : -1) << ", "
              << (in_dims.nbDims > 1 ? in_dims.d[1] : -1) << ", "
              << (in_dims.nbDims > 2 ? in_dims.d[2] : -1) << ", "
              << (in_dims.nbDims > 3 ? in_dims.d[3] : -1) << "]" << std::endl;
    std::cout << "[TRT] Using input size HxW: " << input_height_ << "x" << input_width_ << std::endl;
    return true;
}

// 加载字典
bool OCRRecognizer::loadDict(const std::string& dict_path) {
    std::ifstream file(dict_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open dictionary file: " << dict_path << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            dict_.push_back(line);
        }
    }
    
    dict_size_ = dict_.size();
    std::cout << "Loaded dictionary with " << dict_size_ << " characters" << std::endl;
    
    return true;
}

// 分配GPU内存
bool OCRRecognizer::allocateMemory() {
    // Fallback: seq_len=25, dict_size_=loaded char count
    size_t fallback_elems = static_cast<size_t>(max_batch_size_) * std::max(25, 25) * std::max(1, dict_size_);
    return ocrEngine::allocate_buffers(
        engine_.get(), input_name_, output_name_,
        max_batch_size_, input_channels_, input_height_, input_width_,
        d_input_, d_output_, stream_, fallback_elems);
}

// 释放GPU内存
void OCRRecognizer::freeMemory() {
    ocrEngine::free_buffers(d_input_, d_output_, stream_);
}

// 预处理图像
std::vector<float> OCRRecognizer::preprocess(const cv::Mat& image) {
    cv::Mat src;
    if (image.channels() == 3) src = image; else cv::cvtColor(image, src, cv::COLOR_GRAY2BGR);
    //短边必须是48，跟模型贴合，长边可以往右边扩展
    cv::Mat padded = ocrUtils::resize_image(src, input_width_, input_height_, ocrUtils::ResizeMode::ResizeAndPad, cv::Scalar(0,0,0));
    
    // Save padded image before normalization for debugging
    // try {
    //     static int debug_counter = 0;
    //     std::string debug_dir = "/home/work/liwenkai/c_learn/debug_ocr";
    //     std::string filename = debug_dir + "/padded_before_norm_" + std::to_string(debug_counter++) + ".png";
    //     cv::imwrite(filename, padded);
    //     std::cout << "Saved padded image before normalization: " << filename << std::endl;
    // } catch (const std::exception& e) {
    //     std::cerr << "Failed to save padded image: " << e.what() << std::endl;
    // }
    
    cv::Mat normalized = ocrUtils::normalize_image_standard(padded);
    return ocrUtils::hwc_to_chw(normalized);
}

// 执行识别
RecognitionResult OCRRecognizer::recognize(const cv::Mat& image) {
    // 若输入是动态尺寸，设置为当前目标尺寸
    ocrEngine::set_input_shape(context_.get(), input_name_, 1, input_channels_, input_height_, input_width_);
    auto input_host = preprocess(image);
    std::vector<float> output;
    nvinfer1::Dims out_dims;
    if (!ocrEngine::infer(context_.get(), input_name_, output_name_,
                          input_host.data(), input_host.size(),
                          d_input_, d_output_, stream_,
                          output, out_dims)) {
        std::cerr << "Failed to execute inference" << std::endl;
        return {"", 0.0f};
    }
    
    // 使用CTCLabelDecode进行后处理 - 对应Python的post_op(batch_preds)
    return postprocess(output.data(), out_dims);
}



// 后处理识别结果 - 对应Python的postprocess方法
RecognitionResult OCRRecognizer::postprocess(const float* output, const nvinfer1::Dims& output_dims) {
    if (!ctc_decoder_) {
        std::cerr << "CTC decoder not initialized" << std::endl;
        return {"", 0.0f};
    }
    
    // 将TensorRT输出转换为CTCLabelDecode期望的格式
    // 假设输出格式为 [batch_size, seq_len, num_classes] 或 [batch_size, num_classes, seq_len]
    if (output_dims.nbDims != 3) {
        return {"", 0.0f};
    }
    
    int batch_size = output_dims.d[0] > 0 ? output_dims.d[0] : 1;
    int dim1 = output_dims.d[1] > 0 ? output_dims.d[1] : 1;
    int dim2 = output_dims.d[2] > 0 ? output_dims.d[2] : 1;
    
    // 判断布局：[N, T, C] 或 [N, C, T]
    bool is_nct = (dim1 == static_cast<int>(dict_.size()) + 1); // +1 for blank
    int seq_len = is_nct ? dim2 : dim1;
    int num_classes = is_nct ? dim1 : dim2;
    
    // 构建预测结果 - 简化为单个序列
    std::vector<std::vector<float>> batch_pred;
    
    for (int t = 0; t < seq_len; ++t) {
        std::vector<float> time_pred;
        for (int c = 0; c < num_classes; ++c) {
            int idx = is_nct ? (c * seq_len + t) : (t * num_classes + c);
            time_pred.push_back(output[idx]);
        }
        batch_pred.push_back(time_pred);
    }
    
    // 直接调用CTCLabelDecode的operator()
    auto result = (*ctc_decoder_)(batch_pred);
    return {result.first, result.second};
}
