# TensorRT 推理示例（TensorRT 10 / CUDA 12）

这是一个使用 TensorRT 10 对目标检测模型进行推理的 C++ 最小工程。工程采用模块化设计，包含图像预处理、TensorRT 引擎封装、后处理与日志模块，并提供可运行的测试程序。

## 项目结构

```
lmf/infer/
├── hpp/                         # 头文件
│   ├── ImagePreprocessor.hpp    # 预处理
│   ├── TensorRTEngine.hpp       # TensorRT 引擎封装
│   ├── DetectionPostprocessor.hpp # 后处理（解析输出，生成检测框）
│   ├── LoggerV10.hpp            # TensorRT 10 日志器
│   └── Tensorrt_inference_v10.hpp # 对外主接口
├── cpp/                         # 源码
│   ├── ImagePreprocessor.cpp
│   ├── TensorRTEngine.cpp
│   ├── DetectionPostprocessor.cpp
│   ├── LoggerV10.cpp
│   └── Tensorrt_inference_v10.cpp
├── test/
│   └── tensorrt_inference_test.cpp # 集成测试，可直接运行
├── CMakeLists.txt               # 构建脚本
└── README.md
```

## 功能特性

- **模块化设计**：预处理、引擎推理、后处理、日志分层实现
- **TensorRT 10 适配**：使用独立 `LoggerV10`，适配 TRT v10 API
- **易用的高层接口**：`TensorRTInferenceV10::loadModel` 与 `infer`
- **OpenCV 集成**：输入输出均为 `cv::Mat` 与 `cv::Rect`

## 依赖要求

### 系统要求
- Linux (Ubuntu 18.04+)
- CUDA 12.0+
- TensorRT 10.0+
- OpenCV 4.0+

### 必需依赖
- CMake 3.10+
- C++ 编译器：GCC 7.0+ 或 Clang 6.0+
- OpenCV 开发包（含头文件与库）
- CUDA Toolkit 与 TensorRT SDK（包含 `nvinfer`, `nvonnxparser`, `nvinfer_plugin` 等）

## 编译构建

本工程的 `CMakeLists.txt` 直接链接 `${OpenCV_LIBS}` 与 TensorRT/CUDA 库。若本机未全局配置 OpenCV/TensorRT，请在配置时显式提供路径。

```bash
cd lmf/infer
mkdir build && cd build

# 若已全局安装 OpenCV/TensorRT，直接：
cmake .. -DCMAKE_BUILD_TYPE=Release

# 若需要指定路径，可在命令前导出环境变量或使用 CMAKE_PREFIX_PATH：
# export OpenCV_DIR=/path/to/opencv/share/opencv4
# export TensorRT_ROOT=/path/to/TensorRT
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${OpenCV_DIR}

make -j$(nproc)
```

构建成功后，将生成可执行文件：
- `tensorrt_inference_test_v10`（输出目录：`build/`）

## 运行示例

测试程序默认在代码中写有示例路径：
- 模型：`/home/model/od/rf-detr/xn/inference_model_sim_1680x1680_v10.trt`
- 图片：`/home/work/liwenkai/c_learn/test/0dd6ecc8cb18ea0ddb95ac2036ffd338.jpeg`

请根据实际情况修改 `test/tensorrt_inference_test.cpp` 中的路径，或在运行前创建软链接/复制到正确位置。

```bash
cd lmf/infer/build
./tensorrt_inference_test_v10 | cat
```

程序将：
- 加载 TensorRT 引擎（.trt）
- 读取测试图片
- 执行推理并打印检测结果（类别、置信度与边框）
- 在图片上绘制框并保存为 `detection_result.jpg`

## 模型准备（可选）

若你从 ONNX 开始，可用 `trtexec` 将 ONNX 转为 TensorRT 引擎（示例）：

```bash
trtexec \
  --onnx=/path/to/model.onnx \
  --saveEngine=/path/to/model.trt \
  --fp16 \
  --minShapes=x:1x3x1680x1680 \
  --optShapes=x:1x3x1680x1680 \
  --maxShapes=x:1x3x1680x1680
```

实际输入尺寸请以你的模型为准（本工程后处理以通用结构为例，已封装在 `DetectionPostprocessor` 中）。

## 对外 API（核心类）

头文件：`hpp/Tensorrt_inference_v10.hpp`

```cpp
class TensorRTInferenceV10 {
public:
    TensorRTInferenceV10();
    ~TensorRTInferenceV10();

    // 加载 TensorRT 引擎（.trt）
    bool loadModel(const std::string& modelPath);

    // 推理：输入 BGR 图像，返回检测框
    std::vector<DetectionBox> infer(const cv::Mat& inputImage,
                                    float confidenceThreshold = 0.5f);

    // 模型信息
    int getInputWidth() const;
    int getInputHeight() const;
    int getInputChannels() const;
    int getOutputSize() const;
    int getOutputSize2() const;

    bool isModelLoaded() const;
};
```

检测框数据结构（定义于 `DetectionPostprocessor.hpp`）：

```cpp
struct DetectionBox {
    cv::Rect rect;     // 检测框
    int classId;       // 类别ID
    float confidence;  // 置信度
};
```

## 最小使用示例

```cpp
#include "hpp/Tensorrt_inference_v10.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    TensorRTInferenceV10 inference;
    if (!inference.loadModel("/path/to/model.trt")) {
        std::cerr << "Failed to load TensorRT engine" << std::endl;
        return 1;
    }

    cv::Mat img = cv::imread("/path/to/image.jpg");
    auto boxes = inference.infer(img, 0.4f);

    for (const auto& b : boxes) {
        std::cout << "class=" << b.classId
                  << ", conf=" << b.confidence
                  << ", box=" << b.rect << std::endl;
    }
}
```

## 常见问题

- 找不到 OpenCV 库：请确保安装 `libopencv-dev` 或配置 `OpenCV_DIR`
- 运行时报找不到 TensorRT 库：设置 `LD_LIBRARY_PATH` 包含 TensorRT 与 CUDA `lib` 目录
- 模型尺寸不匹配：使用与你的引擎一致的输入尺寸导出/构建 TRT 引擎

## 许可证

本目录示例代码可在你的项目中自由使用与修改。


