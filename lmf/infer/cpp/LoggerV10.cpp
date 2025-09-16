#include "../hpp/LoggerV10.hpp"
#include <iostream>

// ==================== LoggerV10 实现 ====================
void LoggerV10::log(Severity severity, const char* msg) noexcept {
    if (severity != Severity::kINFO) {
        std::cout << msg << std::endl;
    }
}

LoggerV10& LoggerV10::getInstance() {
    static LoggerV10 instance;
    return instance;
}
