#ifndef IMAGE_PREPROCESSOR_HPP
#define IMAGE_PREPROCESSOR_HPP

#include <vector>
#include <opencv2/core.hpp>

// 前向声明
struct PreprocessedImage;

// 图像预处理模块 - 负责图像标准化、格式转换等
class ImagePreprocessor {
public:
    explicit ImagePreprocessor(int targetWidth = 0, int targetHeight = 0, int targetChannels = 0);
    
    // 设置目标尺寸
    void setTargetDimensions(int width, int height, int channels);
    
    // 主要预处理函数
    PreprocessedImage preprocess(const cv::Mat& inputImage);
    
    // 获取预处理参数
    int getTargetWidth() const { return targetWidth; }
    int getTargetHeight() const { return targetHeight; }
    int getTargetChannels() const { return targetChannels; }

private:
    int targetWidth;
    int targetHeight;
    int targetChannels;
    
    // ImageNet标准归一化参数
    std::vector<float> means;
    std::vector<float> stds;
    cv::Scalar meanScalar;
    cv::Scalar invStdScalar;
    
    // 预分配与复用的中间缓冲
    cv::Mat resizedBuf;
    cv::Mat rgbBuf;
    cv::Mat f32Buf;
    std::vector<float> flattenedDataBuf;
    std::vector<cv::Mat> dstPlanes;

    // 内部预处理步骤（向量化管线，不再提供旧的逐像素接口）
    cv::Mat resizeImage(const cv::Mat& inputImage);
    cv::Mat convertBGRtoRGB(const cv::Mat& inputImage);

    // 缓冲区准备与复用
    void ensureDestinationBuffers();
};

// 数据结构定义
struct PreprocessedImage {
    std::vector<float> data;
    int width;
    int height;
    int channels;
    int originalWidth{0};
    int originalHeight{0};
    float scale{1.0f};
    size_t size() const { return data.size(); }
};

#endif // IMAGE_PREPROCESSOR_HPP
