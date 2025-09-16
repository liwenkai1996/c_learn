# PaddleOCR C++ 工程化实现

这是一个基于PaddleOCR v5的C++工程化实现，使用TensorRT进行GPU加速推理。

## 项目结构

```
ocr/infer/
├── hpp/                    # 头文件目录
│   ├── ocr_detector.hpp    # 文本检测器头文件
│   ├── ocr_recognizer.hpp  # 文本识别器头文件
│   └── ocr_pipeline.hpp    # OCR流水线头文件
├── cpp/                    # 源文件目录
│   ├── ocr_detector.cpp    # 文本检测器实现
│   ├── ocr_recognizer.cpp  # 文本识别器实现
│   ├── ocr_pipeline.cpp    # OCR流水线实现
│   └── main.cpp           # 主程序
├── CMakeLists.txt         # CMake构建文件
├── build.sh              # 构建脚本
└── README.md             # 项目说明文档
```

## 功能特性

- **文本检测**: 基于PP-OCRv5检测模型，支持任意形状文本检测
- **文本识别**: 基于PP-OCRv5识别模型，支持中英文混合识别
- **GPU加速**: 使用TensorRT进行GPU加速推理
- **实时处理**: 支持摄像头实时OCR识别
- **批量处理**: 支持批量图像处理
- **可视化**: 支持检测结果可视化显示

## 依赖要求

### 系统要求
- Linux (Ubuntu 18.04+)
- CUDA 11.0+
- TensorRT 8.0+
- OpenCV 4.0+

### 必需依赖
- **CMake**: 3.10+
- **OpenCV**: 4.0+
- **CUDA**: 11.0+
- **TensorRT**: 8.0+
- **C++编译器**: GCC 7.0+ 或 Clang 6.0+

## 安装依赖

### 1. 安装OpenCV
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libopencv-dev

# 或者从源码编译
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

### 2. 安装CUDA
```bash
# 下载并安装CUDA Toolkit
# 参考: https://developer.nvidia.com/cuda-downloads
```

### 3. 安装TensorRT
```bash
# 下载TensorRT并解压到 /usr/local/tensorrt
# 参考: https://developer.nvidia.com/tensorrt
```

## 编译构建

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd ocr/infer
```

### 2. 运行构建脚本
```bash
# 给构建脚本执行权限
chmod +x build.sh

# 构建Release版本
./build.sh release

# 构建Debug版本
./build.sh debug
```

### 3. 手动构建
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 使用方法

### 1. 准备模型文件
确保你有以下文件：
- `PP-OCRv5_server_det.trt`: 检测模型（TensorRT格式）
- `PP-OCRv5_server_rec.trt`: 识别模型（TensorRT格式）
- `dict.txt`: 字典文件

### 2. 运行OCR识别

#### 处理单张图像
```bash
cd build_release
./run_ocr.sh PP-OCRv5_server_det.trt PP-OCRv5_server_rec.trt dict.txt test.jpg
```

#### 使用摄像头实时识别
```bash
cd build_release
./run_ocr.sh PP-OCRv5_server_det.trt PP-OCRv5_server_rec.trt dict.txt
```

#### 直接运行可执行文件
```bash
cd build_release
./ocr_demo PP-OCRv5_server_det.trt PP-OCRv5_server_rec.trt dict.txt test.jpg
```

### 3. 程序参数说明
```
Usage: ocr_demo <det_model_path> <rec_model_path> <dict_path> [image_path]

参数说明:
- det_model_path: 检测模型文件路径 (.trt格式)
- rec_model_path: 识别模型文件路径 (.trt格式)
- dict_path: 字典文件路径 (.txt格式)
- image_path: 图像文件路径 (可选，不提供则使用摄像头)
```

## API使用示例

### 基本使用
```cpp
#include "hpp/ocr_pipeline.hpp"

int main() {
    // 创建OCR流水线
    OCRPipeline ocr_pipeline;
    
    // 初始化
    if (!ocr_pipeline.initialize("det_model.trt", "rec_model.trt", "dict.txt")) {
        return -1;
    }
    
    // 读取图像
    cv::Mat image = cv::imread("test.jpg");
    
    // 执行OCR识别
    auto results = ocr_pipeline.run(image);
    
    // 处理结果
    for (const auto& result : results) {
        std::cout << "Text: " << result.text << std::endl;
        std::cout << "Confidence: " << result.overall_confidence << std::endl;
    }
    
    return 0;
}
```

### 单独使用检测器
```cpp
#include "hpp/ocr_detector.hpp"

int main() {
    OCRDetector detector;
    detector.initialize("det_model.trt");
    
    cv::Mat image = cv::imread("test.jpg");
    auto detections = detector.detect(image);
    
    for (const auto& det : detections) {
        std::cout << "Detection confidence: " << det.confidence << std::endl;
    }
    
    return 0;
}
```

### 单独使用识别器
```cpp
#include "hpp/ocr_recognizer.hpp"

int main() {
    OCRRecognizer recognizer;
    recognizer.initialize("rec_model.trt", "dict.txt");
    
    cv::Mat text_region = cv::imread("text.jpg");
    auto result = recognizer.recognize(text_region);
    
    std::cout << "Recognized text: " << result.text << std::endl;
    std::cout << "Confidence: " << result.confidence << std::endl;
    
    return 0;
}
```

## 性能优化

### 1. 模型优化
- 使用TensorRT进行模型优化
- 调整批处理大小
- 使用FP16精度推理

### 2. 内存优化
- 复用GPU内存
- 使用内存池
- 及时释放不需要的内存

### 3. 并行处理
- 多线程预处理
- 异步GPU推理
- 流水线并行

## 常见问题

### 1. TensorRT模型加载失败
- 检查模型文件路径
- 确认TensorRT版本兼容性
- 检查GPU内存是否足够

### 2. CUDA内存不足
- 减少批处理大小
- 使用更小的输入尺寸
- 检查GPU内存使用情况

### 3. 识别准确率低
- 检查预处理参数
- 调整置信度阈值
- 优化后处理算法

## 扩展开发

### 1. 添加新的预处理方法
在 `ocr_detector.cpp` 或 `ocr_recognizer.cpp` 中修改 `preprocess` 函数。

### 2. 自定义后处理
在 `postprocess` 函数中添加自定义的后处理逻辑。

### 3. 支持新的模型格式
修改 `loadEngine` 函数以支持其他模型格式。

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 联系方式

如有问题，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：[your-email@example.com]
