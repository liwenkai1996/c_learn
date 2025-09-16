#ifndef DETECTION_POSTPROCESSOR_HPP
#define DETECTION_POSTPROCESSOR_HPP

#include <vector>
#include <opencv2/core.hpp>

// 前向声明
struct InferenceResult;
struct DetectionBox;

// 后处理检测模块 - 负责解析推理结果、生成检测框等
class DetectionPostprocessor {
public:
    DetectionPostprocessor(int inputWidth = 0, int inputHeight = 0);
    
    // 设置输入尺寸
    void setInputDimensions(int width, int height);
    // 设置原图尺寸与缩放比例（预处理阶段的 scale）
    void setOriginalScale(int origW, int origH, float s) { originalWidth = origW; originalHeight = origH; scale = s; }
    
    // 主要后处理函数
    std::vector<DetectionBox> postprocess(const InferenceResult& inferenceResult, 
                                         float confidenceThreshold = 0.5f);
    
    // 设置后处理参数
    void setNumDetections(int num) { numDetections = num; }
    void setNumClasses(int num) { numClasses = num; }
    void setBboxDim(int dim) { bboxDim = dim; }

private:
    int inputWidth;
    int inputHeight;
    int numDetections;
    int numClasses;
    int bboxDim;
    int originalWidth{0};
    int originalHeight{0};
    float scale{1.0f};
    
    // 内部后处理方法
    DetectionBox createDetectionBox(const std::vector<float>& boxes, int boxIndex, 
                                   int classIndex, float confidence);
    cv::Rect convertToImageCoordinates(float cx, float cy, float w, float h);
    
    // Sigmoid激活函数
    float sigmoid(float x) const;
};

// 数据结构定义
struct DetectionBox {
    cv::Rect rect;
    int classId;
    float confidence;
    
    DetectionBox(const cv::Rect& r, int c, float conf) 
        : rect(r), classId(c), confidence(conf) {}
};

#endif // DETECTION_POSTPROCESSOR_HPP
