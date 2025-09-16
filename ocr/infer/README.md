# PaddleOCR C++ 工程化实现

这是一个基于PaddleOCR v5的C++工程化实现，使用TensorRT进行GPU加速推理。

## 项目结构

```
ocr/infer/
├── hpp/                    # 头文件目录
│   ├── ocr_detector.hpp    # 文本检测器头文件
│   ├── ocr_recognizer.hpp  # 文本识别器头文件
│   ├── ocr_orientation.hpp # 文档方向检测器头文件
│   ├── ocr_pipeline.hpp    # OCR流水线头文件
│   ├── engine.hpp          # TensorRT引擎封装头文件
│   └── utils.hpp           # 工具函数头文件
├── cpp/                    # 源文件目录
│   ├── ocr_detector.cpp    # 文本检测器实现
│   ├── ocr_recognizer.cpp  # 文本识别器实现
│   ├── ocr_orientation.cpp # 文档方向检测器实现
│   ├── ocr_pipeline.cpp    # OCR流水线实现
│   ├── engine.cpp          # TensorRT引擎封装实现
│   ├── utils.cpp           # 工具函数实现
│   └── main.cpp           # 主程序
├── build_release/          # 构建输出目录
├── CMakeLists.txt         # CMake构建文件
├── config.h               # 配置文件
├── export.sh              # 模型导出脚本
├── final_dict.txt         # 字典文件
└── README.md              # 项目说明文档
```

## 功能特性

- **文本检测**: 基于PP-OCRv5检测模型，支持任意形状文本检测
- **文本识别**: 基于PP-OCRv5识别模型，支持中英文混合识别
- **文档方向检测**: 基于PP-LCNet_x1_0_textline_ori模型，自动检测文档方向
- **GPU加速**: 使用TensorRT进行GPU加速推理
- **配置化运行**: 通过config.h文件进行参数配置
- **模块化设计**: 独立的检测器、识别器和方向检测器模块
- **可视化**: 支持检测结果可视化显示
- **性能监控**: 内置推理时间统计

## 依赖要求

### 系统要求
- Linux (Ubuntu 18.04+)
- CUDA 12.0+
- TensorRT 10.0+
- OpenCV 4.0+

### 必需依赖
- **CMake**: 3.10+
- **OpenCV**: 4.0+
- **CUDA**: 12.0+
- **TensorRT**: 10.0+
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

### 1. 环境准备
确保以下路径正确配置（在CMakeLists.txt中）：
- OpenCV路径: `/home/opt/local/opencv-4.8.0/release`
- CUDA路径: `/usr/local/cuda`
- TensorRT路径: `/home/opt/local/tensorrt`

### 2. 手动构建
```bash
cd ocr/infer
mkdir build_release && cd build_release
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 3. 构建输出
构建完成后，可执行文件 `ocr_demo` 将生成在 `build_release` 目录中。

## 使用方法

### 1. 配置参数
编辑 `config.h` 文件，设置模型路径和参数：

```cpp
namespace OCRConfig {
    // 模型文件路径配置
    const std::string DET_MODEL_PATH = "/path/to/PP-OCRv5_server_det.trt";
    const std::string REC_MODEL_PATH = "/path/to/PP-OCRv5_server_rec.trt";
    const std::string ORI_MODEL_PATH = "/path/to/PP-OCRv5_server_ori.trt";
    const std::string DICT_PATH = "/path/to/final_dict.txt";
    
    // 图像路径配置
    const std::string IMAGE_PATH = "/path/to/your/image.jpg";
    
    // 输出配置
    const std::string OUTPUT_PATH = "/path/to/output_result.jpg";
    
    // 推理参数配置
    const float DET_THRESHOLD = 0.3f;
    const float DET_BOX_THRESHOLD = 0.6f;
    const float UNCLIP_RATIO = 1.5f; 
    const float REC_THRESHOLD = 0.5f;
    
    // 保存配置
    const bool SAVE_RESULT = true;
}
```

### 2. 运行OCR识别
```bash
cd build_release
./ocr_demo
```

### 3. 程序输出
程序将输出：
- 模型加载状态
- 图像处理时间
- 识别结果（文本内容、置信度、坐标）
- 结果图像保存路径（如果启用）

## API使用示例

### 基本使用
```cpp
#include "hpp/ocr_pipeline.hpp"

int main() {
    // 创建OCR流水线
    OCRPipeline ocr_pipeline;
    
    // 初始化（包含方向检测）
    if (!ocr_pipeline.initialize("det_model.trt", "ori_model.trt", "rec_model.trt", "dict.txt", 0.5f)) {
        return -1;
    }
    
    // 读取图像
    cv::Mat image = cv::imread("test.jpg");
    
    // 执行OCR识别
    auto results = ocr_pipeline.run(image);
    
    // 处理结果
    for (const auto& result : results) {
        std::cout << "Text: " << result.text << std::endl;
        std::cout << "Detection confidence: " << result.detection_confidence << std::endl;
        std::cout << "Recognition confidence: " << result.recognition_confidence << std::endl;
        std::cout << "Overall confidence: " << result.overall_confidence << std::endl;
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

### 使用方向检测器
```cpp
#include "hpp/ocr_orientation.hpp"

int main() {
    OCROrientation orientation;
    orientation.initialize("ori_model.trt");
    
    cv::Mat image = cv::imread("test.jpg");
    auto result = orientation.infer(image);
    
    std::cout << "Orientation label: " << result.label << std::endl;
    std::cout << "Label name: " << result.label_name << std::endl;
    std::cout << "Confidence: " << result.confidence << std::endl;
    
    return 0;
}
```

## 模型导出和转换

### 1. PaddlePaddle模型转ONNX
使用 `export.sh` 脚本中的命令将PaddlePaddle模型转换为ONNX格式：

```bash
# 检测模型转换
paddle2onnx \
  --model_dir /path/to/PP-OCRv5_server_det \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file /path/to/PP-OCRv5_server_det.onnx \
  --opset_version 18 \
  --enable_onnx_checker True

# 方向检测模型转换
paddle2onnx \
  --model_dir /path/to/PP-LCNet_x1_0_textline_ori \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file /path/to/PP-OCRv5_server_ori.onnx \
  --opset_version 18 \
  --enable_onnx_checker True

# 识别模型转换
paddle2onnx \
  --model_dir /path/to/PP-OCRv5_server_rec \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file /path/to/PP-OCRv5_server_rec.onnx \
  --opset_version 18 \
  --enable_onnx_checker True
```

### 2. ONNX模型转TensorRT
使用TensorRT的trtexec工具将ONNX模型转换为TensorRT引擎：

```bash
# 检测模型转换
trtexec \
  --onnx=/path/to/PP-OCRv5_server_det.onnx \
  --saveEngine=/path/to/PP-OCRv5_server_det.trt \
  --fp16 \
  --minShapes=x:1x3x1920x1920 \
  --optShapes=x:1x3x1920x1920 \
  --maxShapes=x:1x3x1920x1920

# 方向检测模型转换
trtexec \
  --onnx=/path/to/PP-OCRv5_server_ori.onnx \
  --saveEngine=/path/to/PP-OCRv5_server_ori.trt \
  --fp16 \
  --minShapes=x:1x3x80x160 \
  --optShapes=x:12x3x80x160 \
  --maxShapes=x:24x3x80x160

# 识别模型转换
trtexec \
  --onnx=/path/to/PP-OCRv5_server_rec.onnx \
  --saveEngine=/path/to/PP-OCRv5_server_rec.trt \
  --fp16 \
  --minShapes=x:1x3x48x1920 \
  --optShapes=x:1x3x48x1920 \
  --maxShapes=x:1x3x48x1920
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
在 `ocr_detector.cpp`、`ocr_recognizer.cpp` 或 `ocr_orientation.cpp` 中修改 `preprocess` 函数。

### 2. 自定义后处理
在 `postprocess` 函数中添加自定义的后处理逻辑。

### 3. 支持新的模型格式
修改 `engine.cpp` 中的 `loadEngine` 函数以支持其他模型格式。

### 4. 配置参数调整
在 `config.h` 中调整各种阈值和参数：
- `DET_THRESHOLD`: 检测置信度阈值
- `DET_BOX_THRESHOLD`: 检测框NMS阈值
- `REC_THRESHOLD`: 识别置信度阈值
- `UNCLIP_RATIO`: 检测框扩展比例

### 5. 添加新的工具函数
在 `utils.cpp` 中添加通用的工具函数。




