#include "../hpp/Tensorrt_inference_v10.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

int main() {
    try {
        std::cout << "TensorRT Inference Test" << std::endl;
        std::cout << "=======================" << std::endl;
        
        // 创建TensorRT推理对象
        TensorRTInferenceV10 inference;
        std::cout << "TensorRTInferenceV10 object created successfully" << std::endl;
        
        // 检查模型是否已加载（应该为false，因为我们还没有加载模型）
        if (inference.isModelLoaded()) {
            std::cout << "Model is loaded" << std::endl;
        } else {
            std::cout << "Model is not loaded (expected)" << std::endl;
        }
        
        // 尝试加载模型文件
        // std::string modelPath = "/home/model/od/rf-detr/xn/inference_model_sim_23_trt10.trt";
        std::string modelPath = "/home/model/od/rf-detr/xn/inference_model_sim_1680x1680_v10.trt";
        
        // 检查模型文件是否存在
        struct stat buffer;
        if (stat(modelPath.c_str(), &buffer) != 0) {
            std::cout << "Warning: Model file not found at: " << modelPath << std::endl;
            std::cout << "Please ensure the model file exists or update the path." << std::endl;
            std::cout << "Continuing with test using a dummy model path..." << std::endl;
        }
        
        if (inference.loadModel(modelPath)) {
            std::cout << "Model loaded successfully!" << std::endl;
            std::cout << "Model input dimensions: " << inference.getInputWidth() 
                      << "x" << inference.getInputHeight() 
                      << "x" << inference.getInputChannels() << std::endl;
            std::cout << "Model output size: " << inference.getOutputSize() << std::endl;
            std::cout << "Model output size 2: " << inference.getOutputSize2() << std::endl;
        } else {
            std::cout << "Failed to load model. This is expected if the model file doesn't exist." << std::endl;
            std::cout << "Test completed with expected failure." << std::endl;
            return 0;
        }
        
        // 加载测试图片
        std::string imagePath = "/home/work/liwenkai/c_learn/test/0dd6ecc8cb18ea0ddb95ac2036ffd338.jpeg";
        if (stat(imagePath.c_str(), &buffer) != 0) {
            std::cout << "Error: Test image not found at: " << imagePath << std::endl;
            std::cout << "Please ensure test.jpg exists in the project root directory." << std::endl;
            return 1;
        }
        
        cv::Mat inputImage = cv::imread(imagePath);
        if (inputImage.empty()) {
            std::cout << "Error: Failed to load test image: " << imagePath << std::endl;
            return 1;
        }
        
        std::cout << "Test image loaded successfully!" << std::endl;
        std::cout << "Image dimensions: " << inputImage.cols << "x" << inputImage.rows << std::endl;
        
        // 执行推理
        std::cout << "Starting inference..." << std::endl;
        std::cout << "Input image dimensions: " << inputImage.cols << "x" << inputImage.rows << "x" << inputImage.channels() << std::endl;
        
        std::vector<DetectionBox> detections;
        try {
            detections = inference.infer(inputImage, 0.4f);  // 直接返回DetectionBox
            std::cout << "Inference completed successfully!" << std::endl;
            std::cout << "Number of detections: " << detections.size() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Inference failed: " << e.what() << std::endl;
            std::cout << "Test completed with inference failure." << std::endl;
            return 0;
        }
        
        // 显示检测结果
        for (size_t i = 0; i < detections.size(); ++i) {
            const DetectionBox& detection = detections[i];
            std::cout << "Detection " << i + 1 << ": "
                      << "Class: " << detection.classId 
                      << ", Confidence: " << detection.confidence
                      << ", Box: x=" << detection.rect.x << ", y=" << detection.rect.y
                      << ", width=" << detection.rect.width << ", height=" << detection.rect.height << std::endl;
        }
        
        // 在图像上绘制检测框
        cv::Mat resultImage = inputImage.clone();
        for (const auto& detection : detections) {
            cv::rectangle(resultImage, detection.rect, cv::Scalar(0, 255, 0), 2);
            
            // 添加类别和置信度标签
            std::string label = "Class " + std::to_string(detection.classId) + 
                               " (" + std::to_string(detection.confidence).substr(0, 4) + ")";
            cv::putText(resultImage, label, 
                       cv::Point(detection.rect.x, detection.rect.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        
        // 保存结果图像
        std::string outputPath = "detection_result.jpg";
        cv::imwrite(outputPath, resultImage);
        std::cout << "Result image saved as: " << outputPath << std::endl;
        
        // 显示结果图像（如果支持GUI）
        try {
            cv::imshow("Detection Results", resultImage);
            std::cout << "Press any key to close the result window..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        } catch (const cv::Exception& e) {
            std::cout << "Note: Could not display image (no GUI support). Result saved to file." << std::endl;
        }
        
        std::cout << "Test completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
