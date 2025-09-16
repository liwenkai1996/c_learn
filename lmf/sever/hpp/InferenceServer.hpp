#ifndef INFERENCE_SERVER_HPP
#define INFERENCE_SERVER_HPP

#include "InferenceEngine.hpp"
#include <drogon/drogon.h>
#include <string>
#include <memory>
#include <atomic>
#include <functional>

// 推理服务器类 - 提供 HTTP 接口
class InferenceServer {
public:
    InferenceServer();
    ~InferenceServer();
    
    // 初始化服务器
    bool initialize(const std::string& model_path, 
                   int device_id = 0,
                   int port = 8080,
                   int num_threads = 4);
    
    // 启动服务器
    bool start();
    
    // 停止服务器
    void stop();
    
    // 检查服务器状态
    bool is_running() const;
    
    // 获取服务器信息
    int get_port() const;
    std::string get_status() const;
    
    // 设置回调函数
    void set_request_callback(std::function<void(const std::string&)> callback);
    void set_error_callback(std::function<void(const std::string&)> callback);
    
    // 执行推理（供内部使用）
    EngineResult infer(const cv::Mat& input_image, float confidence_threshold = 0.5f);
    
    // 获取统计信息
    int get_total_requests() const;
    double get_average_inference_time() const;
    
    // 创建响应
    std::string create_response(const EngineResult& result);
    std::string create_error_response(int status_code, const std::string& message);
    std::string create_batch_response(const std::vector<EngineResult>& batch_results);

private:
    // 设置路由
    void setup_routes();
    
    // HTTP 请求处理函数
    void handle_inference_request(const drogon::HttpRequestPtr& req, bool is_batch,
                                std::function<void(const drogon::HttpResponsePtr&)>&& callback);
    
    void handle_health_check(const drogon::HttpRequestPtr& req,
                           std::function<void(const drogon::HttpResponsePtr&)>&& callback);
    
    void handle_status_request(const drogon::HttpRequestPtr& req,
                             std::function<void(const drogon::HttpResponsePtr&)>&& callback);
    
    // 新增：视频推理处理函数
    void handle_video_inference_request(const drogon::HttpRequestPtr& req,
                                      std::function<void(const drogon::HttpResponsePtr&)>&& callback);
    
    // 新增：视频推理核心函数
    bool process_video_inference(const std::string& input_video_path,
                                const std::string& output_video_path,
                                float confidence_threshold,
                                int frame_interval = 1);
    
    // 新增：在帧上绘制检测结果
    void draw_detections_on_frame(cv::Mat& frame, const std::vector<DetectionBox>& detections);
    
    // 成员变量
    std::unique_ptr<InferenceEngine> inference_engine;
    
    // 服务器配置
    int server_port;
    int num_worker_threads;
    std::string model_path;
    int device_id;
    
    // 服务器状态
    std::atomic<bool> server_running;
    
    // 回调函数
    std::function<void(const std::string&)> request_callback;
    std::function<void(const std::string&)> error_callback;
    
    // 统计信息
    std::atomic<int> total_requests;
    std::atomic<int> successful_requests;
    std::atomic<int> failed_requests;
};

// 服务器配置结构
struct ServerConfig {
    std::string model_path;
    int device_id = 0;
    int port = 8080;
    int num_threads = 4;
    int max_request_size = 10 * 1024 * 1024;  // 10MB
    int request_timeout = 30;  // 30 seconds
    bool enable_cors = true;
    std::string log_level = "info";
};

#endif // INFERENCE_SERVER_HPP
