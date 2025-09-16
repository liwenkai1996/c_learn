#include "../hpp/ImagePreprocessor.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <opencv2/imgproc.hpp>

// ==================== ImagePreprocessor 实现 ====================
ImagePreprocessor::ImagePreprocessor(int targetWidth, int targetHeight, int targetChannels)
    : targetWidth(targetWidth), targetHeight(targetHeight), targetChannels(targetChannels) {
    // 初始化ImageNet标准归一化参数
    means = {0.485f, 0.456f, 0.406f};
    stds = {0.229f, 0.224f, 0.225f};
    meanScalar  = cv::Scalar(means[0], means[1], means[2]);
    invStdScalar= cv::Scalar(1.0f/stds[0], 1.0f/stds[1], 1.0f/stds[2]);
}

void ImagePreprocessor::setTargetDimensions(int width, int height, int channels) {
    targetWidth = width;
    targetHeight = height;
    targetChannels = channels;
}

void ImagePreprocessor::ensureDestinationBuffers() {
    const int planeSize = targetWidth * targetHeight;
    // 预分配输出向量
    if (static_cast<int>(flattenedDataBuf.size()) != planeSize * targetChannels) {
        flattenedDataBuf.assign(static_cast<size_t>(planeSize) * targetChannels, 0.0f);
        // 为 mixChannels 构造三个平面视图
        dstPlanes.clear();
        dstPlanes.reserve(targetChannels);
        for (int c = 0; c < targetChannels; ++c) {
            dstPlanes.emplace_back(targetHeight, targetWidth, CV_32F,
                                   flattenedDataBuf.data() + static_cast<size_t>(c) * planeSize);
        }
    }
}

PreprocessedImage ImagePreprocessor::preprocess(const cv::Mat& inputImage) {
    // 0) 记录原图尺寸
    const int ow = inputImage.cols;
    const int oh = inputImage.rows;

    // 1) 按长边等比缩放，短边补(114,114,114)
    // 计算缩放比例和填充
    float scale = std::min(static_cast<float>(targetWidth) / std::max(1, ow),
                           static_cast<float>(targetHeight) / std::max(1, oh));
    int newW = static_cast<int>(std::round(ow * scale));
    int newH = static_cast<int>(std::round(oh * scale));
    if (newW <= 0) newW = 1;
    if (newH <= 0) newH = 1;

    cv::Mat resized;
    cv::resize(inputImage, resized, cv::Size(newW, newH));
    const int padXTotal = targetWidth  - newW;
    const int padYTotal = targetHeight - newH;
    // 只在右边和下边补齐，左/上不填充
    const int padLeft = 0;
    const int padTop = 0;
    const int padRight = padXTotal;
    const int padBottom = padYTotal;
    cv::Mat letterboxed;
    cv::copyMakeBorder(resized, letterboxed, padTop, padBottom, padLeft, padRight,
                       cv::BORDER_CONSTANT, cv::Scalar(114,114,114));

    // 2) 直接在 BGR 上进行向量化归一化（div(255) -> 减均值 -> 除标准差），保持为 HWC 的 CV_32FC3
    //    为避免额外的通道交换，这里先在 BGR 空间做归一化，随后在 mixChannels 中一次性完成到 RGB-CHW 的重排。
    // 预分配并复用 f32 缓冲（注意：使用 letterboxed）
    f32Buf.create(targetHeight, targetWidth, CV_32FC3);
    letterboxed.convertTo(f32Buf, CV_32FC3, 1.0 / 255.0);
    // 注意：f32Buf 当前为 BGR 顺序，ImageNet 的 means/stds 通常以 RGB 给出。
    //       这里用 BGR 顺序的标量，以保证后续 mixChannels 重排为 RGB-CHW 后数值正确。
    const cv::Scalar bgrMean(means[2], means[1], means[0]);
    const cv::Scalar bgrInvStd(1.0f/stds[2], 1.0f/stds[1], 1.0f/stds[0]);
    cv::subtract(f32Buf, bgrMean, f32Buf);
    cv::multiply(f32Buf, bgrInvStd, f32Buf);

    // 4) HWC -> CHW：使用 mixChannels 直接写入预分配的平面
    ensureDestinationBuffers();
    const cv::Mat srcs[1] = { f32Buf };
    // 通道映射（BGR -> CHW RGB）：
    // f32Buf 通道顺序为 B(0), G(1), R(2)
    // 目标平面顺序为 R(plane0), G(plane1), B(plane2)
    const int fromTo[6] = { 2,0, 1,1, 0,2 };
    cv::mixChannels(srcs, 1, dstPlanes.data(), targetChannels, fromTo, 3);

    PreprocessedImage out;
    out.data = flattenedDataBuf;
    out.width = targetWidth;
    out.height = targetHeight;
    out.channels = targetChannels;
    out.originalWidth = ow;
    out.originalHeight = oh;
    out.scale = scale;
    return out;
}

cv::Mat ImagePreprocessor::resizeImage(const cv::Mat& inputImage) {
    cv::Mat resized;
    cv::resize(inputImage, resized, cv::Size(targetWidth, targetHeight));
    
    // 调试输出
    std::cout << "Resized image to: " << targetWidth << "x" << targetHeight << std::endl;
    return resized;
}

cv::Mat ImagePreprocessor::convertBGRtoRGB(const cv::Mat& inputImage) {
    cv::Mat rgbImage;
    cv::cvtColor(inputImage, rgbImage, cv::COLOR_BGR2RGB);
    return rgbImage;
}

// 旧的逐像素/三重循环版本已移除，使用向量化方案替代
