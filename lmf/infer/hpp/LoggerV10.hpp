#ifndef LOGGER_V10_HPP
#define LOGGER_V10_HPP

#include <NvInfer.h>

// TensorRT Logger类
class LoggerV10 : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
    static LoggerV10& getInstance();
private:
    LoggerV10() = default;
};

#endif // LOGGER_V10_HPP
