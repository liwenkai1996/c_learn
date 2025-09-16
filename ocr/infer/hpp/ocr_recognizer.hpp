#ifndef OCR_RECOGNIZER_HPP
#define OCR_RECOGNIZER_HPP

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

// CTC标签解码器类 - 完全对应Python的CTCLabelDecode
class CTCLabelDecode {
public:
    CTCLabelDecode(const std::vector<std::string>& character_list, bool use_space_char = true);
    
    // 对应Python的decode方法 - 简化版本，单个序列
    std::pair<std::string, float> decode(const std::vector<int>& text_index, 
                                         const std::vector<float>& text_prob = {},
                                         bool is_remove_duplicate = false);
    
    // 对应Python的__call__方法 - 简化版本，单个序列
    std::pair<std::string, float> operator()(const std::vector<std::vector<float>>& pred);

private:
    std::vector<std::string> character_;  // 对应Python的self.character
    std::map<std::string, int> dict_;     // 对应Python的self.dict
    bool reverse_;                        // 对应Python的self.reverse
    
    // 对应Python的get_ignored_tokens方法
    std::vector<int> get_ignored_tokens();
    
    // 对应Python的add_special_char方法
    std::vector<std::string> add_special_char(const std::vector<std::string>& character_list);
};

// 识别结果结构体
struct RecognitionResult {
    std::string text;     // 识别的文本
    float confidence;     // 置信度
};

// OCR识别器类
class OCRRecognizer {
public:
    OCRRecognizer();
    ~OCRRecognizer();
    
    // 初始化识别器
    bool initialize(const std::string& model_path, 
                   const std::string& dict_path,
                   float threshold = 0.5f);
    
    // 执行识别
    RecognitionResult recognize(const cv::Mat& image);
    
    
    // 预处理图像，返回按CHW布局的float数据
    std::vector<float> preprocess(const cv::Mat& image);
    
    // 后处理识别结果（基于实际输出维度进行CTC解码）
    RecognitionResult postprocess(const float* output, const nvinfer1::Dims& output_dims);

private:
    // TensorRT相关
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    // 模型参数
    int input_height_;
    int input_width_;
    int input_channels_;
    int max_batch_size_;
    int dict_size_;
    
    // 推理参数
    float threshold_;
    
    // 字典和CTC解码器
    std::vector<std::string> dict_;
    std::unique_ptr<CTCLabelDecode> ctc_decoder_;
    
    // CUDA内存
    void* d_input_;
    void* d_output_;
    std::string input_name_;
    std::string output_name_;
    cudaStream_t stream_ = nullptr;
    
    // 初始化TensorRT引擎
    bool loadEngine(const std::string& model_path);
    
    // 加载字典
    bool loadDict(const std::string& dict_path);
    
    // 分配GPU内存
    bool allocateMemory();
    
    // 释放GPU内存
    void freeMemory();
};

#endif // OCR_RECOGNIZER_HPP
