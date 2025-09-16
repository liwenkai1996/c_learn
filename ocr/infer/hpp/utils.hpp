// utils.hpp - Common image preprocessing helpers
#ifndef OCR_UTILS_HPP
#define OCR_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ocrUtils {

enum class ResizeMode {
    KeepAndPad, //  原始图像尺寸不动，在右边和下边填补
    Stretch,     // 直接缩放到目标尺寸
    ResizeAndPad  //将图像缩放到不超过目标尺寸，贴左上角，剩余填充
};

// Resize with mode and fill value. Returns resized/padded BGR image with size (target_w, target_h)
cv::Mat resize_image(const cv::Mat& src,
                     int target_w,
                     int target_h,
                     ResizeMode mode,
                     const cv::Scalar& pad_value = cv::Scalar(0, 0, 0));

// Normalize HWC float image.
cv::Mat normalize_image_rgb(const cv::Mat& src);
cv::Mat normalize_image_bgr(const cv::Mat& src);
cv::Mat normalize_image_standard(const cv::Mat& src);

// 2) Flexible variant: only mean/std, no color conversion
cv::Mat normalize_image(const cv::Mat& src,
                        const std::vector<float>& mean,
                        const std::vector<float>& std);
// Convert HWC float image to CHW contiguous float vector
std::vector<float> hwc_to_chw(const cv::Mat& float_hwc);

// 计算按给定resize模式，映射到目标尺寸中的有效内容区域（左上对齐）
cv::Rect compute_valid_region(int src_w,
                              int src_h,
                              int target_w,
                              int target_h,
                              ResizeMode mode);

// 将有效区域之外（右侧/下侧）的区域置零（单通道掩码）
void zero_out_padding(cv::Mat& single_channel_mask, const cv::Rect& valid_region);

} // namespace ocrUtils

#endif // OCR_UTILS_HPP
