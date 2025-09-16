#include "../hpp/DetectionPostprocessor.hpp"
#include "../hpp/TensorRTEngine.hpp"
#include <iostream>
#include <algorithm>

// ==================== DetectionPostprocessor 实现 ====================
DetectionPostprocessor::DetectionPostprocessor(int inputWidth, int inputHeight)
    : inputWidth(inputWidth), inputHeight(inputHeight), 
      numDetections(0), numClasses(0), bboxDim(4) {
}

void DetectionPostprocessor::setInputDimensions(int width, int height) {
    inputWidth = width;
    inputHeight = height;
}

std::vector<DetectionBox> DetectionPostprocessor::postprocess(
    const InferenceResult& inferenceResult, float confidenceThreshold) {
    
    std::vector<DetectionBox> detections;
    
    // 解析输出数据
    const std::vector<float>& boxes = inferenceResult.output1;
    const std::vector<float>& classes = inferenceResult.output2;
    
    // 计算检测数量和类别数量
    int totalDetections = boxes.size() / bboxDim;
    int totalClasses = classes.size() / totalDetections;
    
    // 设置后处理参数
    setNumDetections(totalDetections);
    setNumClasses(totalClasses);
    
    // 对每个检测进行处理
    for (int i = 0; i < totalDetections; i++) {
        // 找到最高置信度的类别
        int classStart = i * totalClasses;
        auto maxElement = std::max_element(classes.begin() + classStart, 
                                         classes.begin() + classStart + totalClasses);
        int classIndex = std::distance(classes.begin() + classStart, maxElement);
        float confidence = *maxElement;
        
        // 应用sigmoid激活函数到置信度
        confidence = sigmoid(confidence);
        
        // 检查置信度阈值
        if (confidence < confidenceThreshold) {
            continue;
        }
        
        // 创建检测框 - 传递正确的类别索引和置信度
        int boxStart = i * bboxDim;
        DetectionBox detection = createDetectionBox(boxes, boxStart, classIndex, confidence);
        detections.push_back(detection);
    }
    
    std::cout << "Postprocessing completed. Found " << detections.size() << " detections." << std::endl;
    return detections;
}

DetectionBox DetectionPostprocessor::createDetectionBox(const std::vector<float>& boxes, int boxIndex, 
                                                       int classIndex, float confidence) {
    // 获取边界框坐标 (cx, cy, w, h格式)
    float cx = boxes[boxIndex + 0];
    float cy = boxes[boxIndex + 1];
    float w = boxes[boxIndex + 2];
    float h = boxes[boxIndex + 3];
    
    // 调试输出：显示原始坐标值
    if (boxIndex < 10) {  // 只显示前10个检测框的调试信息
        std::cout << "Debug Box " << boxIndex << ": cx=" << cx << ", cy=" << cy 
                  << ", w=" << w << ", h=" << h << std::endl;
    }
    
    // 转换为图像坐标
    cv::Rect rect = convertToImageCoordinates(cx, cy, w, h);
    
    // 调试输出：显示转换后的坐标
    if (boxIndex < 10) {
        std::cout << "Debug Rect " << boxIndex << ": x=" << rect.x << ", y=" << rect.y 
                  << ", w=" << rect.width << ", h=" << rect.height << std::endl;
    }
    
    // 类别ID就是传入的classIndex，置信度已经处理过了
    int classId = classIndex;
    
    return DetectionBox(rect, classId, confidence);
}

cv::Rect DetectionPostprocessor::convertToImageCoordinates(float cx, float cy, float w, float h) {

    // 转换为左上角坐标 (cxcywh -> xyxy)
    float x1 = cx - w / 2.0f;
    float y1 = cy - h / 2.0f;
    float x2 = cx + w / 2.0f;
    float y2 = cy + h / 2.0f;
    
    // 百分比(0-1) -> 输入张量像素（左上(0,0)，右/下有补齐）
    int x = static_cast<int>(std::lround(x1 * inputWidth));
    int y = static_cast<int>(std::lround(y1 * inputHeight));
    int width = static_cast<int>(std::lround((x2 - x1) * inputWidth));
    int height = static_cast<int>(std::lround((y2 - y1) * inputHeight));

    // 直接一次性从模型输入像素映射回原图像素：考虑 letterbox 仅右/下补齐，无需偏移，只需除以 scale
    if (scale > 0.0f) {
        const float invScale = 1.0f / scale;
        x      = static_cast<int>(std::lround(x * invScale));
        y      = static_cast<int>(std::lround(y * invScale));
        width  = static_cast<int>(std::lround(width * invScale));
        height = static_cast<int>(std::lround(height * invScale));
    }

    // 约束到原图范围内
    const int ow = (originalWidth > 0 ? originalWidth : inputWidth);
    const int oh = (originalHeight > 0 ? originalHeight : inputHeight);
    x = std::max(0, std::min(x, ow - 1));
    y = std::max(0, std::min(y, oh - 1));
    width = std::max(1, std::min(width, ow - x));
    height = std::max(1, std::min(height, oh - y));

    return cv::Rect(x, y, width, height);
}

float DetectionPostprocessor::sigmoid(float x) const {
    return 1.0f / (1.0f + std::exp(-x));
}
