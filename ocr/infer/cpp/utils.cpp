#include "../hpp/utils.hpp"
#include <algorithm>

namespace ocrUtils {

cv::Mat resize_image(const cv::Mat& src,
                     int target_w,
                     int target_h,
                     ResizeMode mode,
                     const cv::Scalar& pad_value) {
    if (src.empty()) return cv::Mat();
    if (mode == ResizeMode::Stretch) {
        cv::Mat dst;
        cv::resize(src, dst, cv::Size(target_w, target_h));
        return dst;
    } else if (mode == ResizeMode::KeepAndPad){
        // KeepAndPad
        int copy_w = std::min(src.cols, target_w);
        int copy_h = std::min(src.rows, target_h);
        cv::Mat padded(target_h, target_w, src.type(), pad_value);
        src(cv::Rect(0, 0, copy_w, copy_h)).copyTo(padded(cv::Rect(0, 0, copy_w, copy_h)));
        return padded;
    }
    //ResizeAndPad
    float scale = std::min(target_w / static_cast<float>(src.cols),
    target_h / static_cast<float>(src.rows));
    int new_w = std::max(1, static_cast<int>(std::round(src.cols * scale)));
    int new_h = std::max(1, static_cast<int>(std::round(src.rows * scale)));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    cv::Mat padded(target_h, target_w, src.type(), pad_value);
    resized.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));
    return padded;

}

cv::Mat normalize_image_rgb(const cv::Mat& src) {
    return normalize_image(src,
                           std::vector<float>{0.485f, 0.456f, 0.406f},
                           std::vector<float>{0.229f, 0.224f, 0.225f});
}

cv::Mat normalize_image_bgr(const cv::Mat& src) {
    // Reverse of RGB mean/std mapped onto BGR channel order
    return normalize_image(src,
                           std::vector<float>{0.406f, 0.456f, 0.485f},
                           std::vector<float>{0.225f, 0.224f, 0.229f});
}

cv::Mat normalize_image_standard(const cv::Mat& src) {
    // Reverse of RGB mean/std mapped onto BGR channel order
    return normalize_image(src,
                           std::vector<float>{0.5f, 0.5f, 0.5f},
                           std::vector<float>{0.5f, 0.5f, 0.5f});
}

cv::Mat normalize_image(const cv::Mat& src,
                        const std::vector<float>& mean,
                        const std::vector<float>& std) {
    if (src.empty()) return cv::Mat();
    cv::Mat img = src;
    if (img.depth() != CV_32F) {
        img.convertTo(img, CV_32F, 1.0 / 255.0);
    }
    std::vector<cv::Mat> ch;
    cv::split(img, ch);
    for (int c = 0; c < static_cast<int>(ch.size()); ++c) {
        float m = mean[c < static_cast<int>(mean.size()) ? c : 0];
        float s = std[c < static_cast<int>(std.size()) ? c : 0];
        if (s == 0.0f) s = 1.0f;
        ch[c] = (ch[c] - m) / s;
    }
    cv::Mat normalized;
    cv::merge(ch, normalized);
    return normalized;
}

std::vector<float> hwc_to_chw(const cv::Mat& float_hwc) {
    CV_Assert(float_hwc.type() == CV_32FC1 || float_hwc.type() == CV_32FC3);
    int channels = float_hwc.channels();
    int height = float_hwc.rows;
    int width = float_hwc.cols;
    size_t plane = static_cast<size_t>(height) * width;
    std::vector<float> chw(static_cast<size_t>(channels) * plane);
    std::vector<cv::Mat> ch;
    cv::split(float_hwc, ch);
    for (int c = 0; c < channels; ++c) {
        std::memcpy(chw.data() + c * plane, ch[c].ptr<float>(), plane * sizeof(float));
    }
    return chw;
}

cv::Rect compute_valid_region(int src_w,
                              int src_h,
                              int target_w,
                              int target_h,
                              ResizeMode mode) {
    if (src_w <= 0 || src_h <= 0 || target_w <= 0 || target_h <= 0) {
        return cv::Rect(0, 0, 0, 0);
    }
    if (mode == ResizeMode::Stretch) {
        // 全部有效，无需置零
        return cv::Rect(0, 0, target_w, target_h);
    }
    if (mode == ResizeMode::KeepAndPad) {
        int valid_w = std::min(src_w, target_w);
        int valid_h = std::min(src_h, target_h);
        return cv::Rect(0, 0, valid_w, valid_h);
    }
    // ResizeAndPad: 按短边缩放，左上对齐，剩余填充
    float scale = std::min(target_w / static_cast<float>(src_w),
                           target_h / static_cast<float>(src_h));
    int new_w = std::max(1, static_cast<int>(std::round(src_w * scale)));
    int new_h = std::max(1, static_cast<int>(std::round(src_h * scale)));
    return cv::Rect(0, 0, std::min(new_w, target_w), std::min(new_h, target_h));
}

void zero_out_padding(cv::Mat& single_channel_mask, const cv::Rect& valid_region) {
    if (single_channel_mask.empty() || single_channel_mask.channels() != 1) return;
    // 右侧
    if (valid_region.width < single_channel_mask.cols) {
        cv::Rect right(valid_region.width, 0,
                       single_channel_mask.cols - valid_region.width,
                       single_channel_mask.rows);
        single_channel_mask(right).setTo(0);
    }
    // 下侧
    if (valid_region.height < single_channel_mask.rows) {
        cv::Rect bottom(0, valid_region.height,
                        single_channel_mask.cols,
                        single_channel_mask.rows - valid_region.height);
        single_channel_mask(bottom).setTo(0);
    }
}

} // namespace ocrUtils
