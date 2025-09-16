#include "../hpp/ocr_pipeline.hpp"
#include "../config.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>

void printResults(const std::vector<OCRResult>& results) {
    std::cout << "\nOCR Results:" << std::endl;
    std::cout << "=============" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        std::cout << "Text " << (i + 1) << ": " << result.text << std::endl;
        std::cout << "  Detection confidence: " << result.detection_confidence << std::endl;
        std::cout << "  Recognition confidence: " << result.recognition_confidence << std::endl;
        std::cout << "  Overall confidence: " << result.overall_confidence << std::endl;
        std::cout << "  Box coordinates: ";
        for (const auto& point : result.box) {
            std::cout << "(" << point.x << ", " << point.y << ") ";
        }
        std::cout << std::endl << std::endl;
    }
}

int main() {
    // 从配置文件读取路径
    std::string det_model_path = OCRConfig::DET_MODEL_PATH;
    std::string ori_model_path = OCRConfig::ORI_MODEL_PATH;
    std::string rec_model_path = OCRConfig::REC_MODEL_PATH;
    std::string dict_path = OCRConfig::DICT_PATH;
    std::string image_path = OCRConfig::IMAGE_PATH;
    
    std::cout << "=== OCR Configuration ===" << std::endl;
    std::cout << "Detection model: " << det_model_path << std::endl;
    std::cout << "Recognition model: " << rec_model_path << std::endl;
    std::cout << "Orientation model: " << ori_model_path << std::endl;
    std::cout << "Dictionary: " << dict_path << std::endl;
    std::cout << "Image path: " << image_path << std::endl;
    std::cout << "========================" << std::endl;
    
    // 检查模型文件是否存在
    std::ifstream det_file(det_model_path);
    if (!det_file.good()) {
        std::cerr << "Error: Detection model not found: " << det_model_path << std::endl;
        std::cerr << "Please check the model path or download the model files." << std::endl;
        return -1;
    }
    det_file.close();
    
    std::ifstream ori_file(ori_model_path);
    if (!ori_file.good()) {
        std::cerr << "Error: Detection model not found: " << rec_model_path << std::endl;
        std::cerr << "Please check the model path or download the model files." << std::endl;
        return -1;
    }
    det_file.close();

    std::ifstream rec_file(rec_model_path);
    if (!rec_file.good()) {
        std::cerr << "Error: Recognition model not found: " << rec_model_path << std::endl;
        std::cerr << "Please check the model path or download the model files." << std::endl;
        return -1;
    }
    rec_file.close();
    
    std::ifstream dict_file(dict_path);
    if (!dict_file.good()) {
        std::cerr << "Error: Dictionary file not found: " << dict_path << std::endl;
        std::cerr << "Please check the dictionary path." << std::endl;
        return -1;
    }
    dict_file.close();
    
    // 创建OCR流水线
    OCRPipeline ocr_pipeline;
    
    // 初始化OCR流水线
    std::cout << "Initializing OCR pipeline..." << std::endl;
    if (!ocr_pipeline.initialize(det_model_path, ori_model_path, rec_model_path, dict_path, OCRConfig::REC_THRESHOLD)) {
        std::cerr << "Failed to initialize OCR pipeline" << std::endl;
        return -1;
    }
    
    // 处理图像
    std::cout << "Processing image: " << image_path << std::endl;
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    
    // 执行OCR识别
    auto start_time = std::chrono::high_resolution_clock::now();
    auto results = ocr_pipeline.run(image);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
    
    // 输出识别结果
    printResults(results);
    
    // 保存结果图像（可选）
    if (OCRConfig::SAVE_RESULT) {
        cv::imwrite(OCRConfig::OUTPUT_PATH, image);
        std::cout << "Original image saved to: " << OCRConfig::OUTPUT_PATH << std::endl;
    }
    
    return 0;
}
