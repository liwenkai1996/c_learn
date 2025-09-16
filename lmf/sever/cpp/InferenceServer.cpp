#include "../hpp/InferenceServer.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>  // 新增：视频IO支持
#include <filesystem>           // 新增：文件系统操作
#include <fstream>              // 新增：文件流操作
#include <sstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <drogon/MultiPart.h>

using json = nlohmann::json;

// InferenceServer 实现
InferenceServer::InferenceServer() 
    : server_port(8080), num_worker_threads(4), server_running(false), 
      total_requests(0), successful_requests(0), failed_requests(0) {
}

InferenceServer::~InferenceServer() {
    stop();
}

bool InferenceServer::initialize(const std::string& model_path, int device_id, int port, int num_threads) {
    try {
        this->model_path = model_path;
        this->device_id = device_id;
        this->server_port = port;
        this->num_worker_threads = num_threads;
        
        // 初始化推理引擎
        inference_engine = std::make_unique<InferenceEngine>();
        if (!inference_engine->initialize(model_path, device_id)) {
            return false;
        }
        
        // 配置 Drogon
        drogon::app().setLogLevel(trantor::Logger::kInfo);
        drogon::app().setThreadNum(num_threads);
        drogon::app().setClientMaxBodySize(500 * 1024 * 1024);
        
        // 设置路由
        setup_routes();
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

void InferenceServer::setup_routes() {
    // 推理接口
    drogon::app().registerHandler("/api/detr/inference", 
        [this](const drogon::HttpRequestPtr& req,
               std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            this->handle_inference_request(req, false,std::move(callback));
        },
        {drogon::Post});

    drogon::app().registerHandler("/api/detr/batch_inference", 
        [this](const drogon::HttpRequestPtr& req,
                std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            this->handle_inference_request(req, true, std::move(callback));
        },
        {drogon::Post});
    
    // 健康检查
    drogon::app().registerHandler("/api/detr/health",
        [this](const drogon::HttpRequestPtr& req,
               std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            this->handle_health_check(req, std::move(callback));
        },
        {drogon::Get});
    
    // 状态查询
    drogon::app().registerHandler("/api/detr/status",
        [this](const drogon::HttpRequestPtr& req,
               std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            this->handle_status_request(req, std::move(callback));
        },
        {drogon::Get});
    
    // 新增：视频推理接口
    drogon::app().registerHandler("/api/detr/video_inference", 
        [this](const drogon::HttpRequestPtr& req,
                std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            this->handle_video_inference_request(req, std::move(callback));
        },
        {drogon::Post});
}

bool InferenceServer::start() {
    if (!inference_engine || !inference_engine->is_ready()) {
        return false;
    }
    
    try {
        std::cout << "Starting HTTP server on port " << server_port << std::endl;
        
        server_running = true;
        
        // 启动 Drogon 服务器
        drogon::app().addListener("0.0.0.0", server_port);
        drogon::app().run();
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

void InferenceServer::stop() {
    server_running = false;
    drogon::app().quit();
}

bool InferenceServer::is_running() const {
    return server_running;
}

int InferenceServer::get_port() const {
    return server_port;
}

std::string InferenceServer::get_status() const {
    std::ostringstream oss;
    oss << "Server running on port " << server_port 
        << ", Total requests: " << total_requests;
    return oss.str();
}

void InferenceServer::set_request_callback(std::function<void(const std::string&)> callback) {
    request_callback = callback;
}

void InferenceServer::set_error_callback(std::function<void(const std::string&)> callback) {
    error_callback = callback;
}

EngineResult InferenceServer::infer(const cv::Mat& input_image, float confidence_threshold) {
    total_requests++;
    
    try {
        EngineResult result = inference_engine->infer(input_image, confidence_threshold);
        
        if (result.status_code == 0) {
            successful_requests++;
        } else {
            failed_requests++;
        }
        
        return result;
        
    } catch (const std::exception& e) {
        failed_requests++;
        EngineResult error_result;
        error_result.status_code = -1;
        error_result.status_message = std::string("Exception: ") + e.what();
        return error_result;
    }
}

int InferenceServer::get_total_requests() const {
    return total_requests;
}

double InferenceServer::get_average_inference_time() const {
    if (inference_engine) {
        return inference_engine->get_average_inference_time();
    }
    return 0.0;
}

void InferenceServer::handle_inference_request(const drogon::HttpRequestPtr& req, bool is_batch,
                                             std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    try {
        // 解析表单并获取上传的文件
        drogon::MultiPartParser parser;
        if (parser.parse(req) != 0) {
            auto resp = drogon::HttpResponse::newHttpJsonResponse(
                create_error_response(400, "Failed to parse multipart/form-data"));
            resp->setStatusCode(drogon::k400BadRequest);
            callback(resp);
            return;
        }
        const auto& files = parser.getFiles();
        if (files.empty()) {
            auto resp = drogon::HttpResponse::newHttpJsonResponse(
                create_error_response(400, "No image file uploaded"));
            resp->setStatusCode(drogon::k400BadRequest);
            callback(resp);
            return;
        }
        
        // 获取置信度阈值
        float confidence_threshold = 0.5f;
        auto threshold_param = req->getParameter("confidence_threshold");
        if (!threshold_param.empty()) {
            try {
                confidence_threshold = std::stof(threshold_param);
            } catch (...) {
                // 使用默认值
            }
        }
        
        if (is_batch) {
            // 批量推理：处理所有上传的文件
            std::vector<cv::Mat> images;
            images.reserve(files.size());
            
            // 解码所有图片
            for (const auto& file : files) {
                auto contentView = file.fileContent();
                std::vector<uchar> buffer(contentView.begin(), contentView.end());
                cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
                
                if (image.empty()) {
                    auto resp = drogon::HttpResponse::newHttpJsonResponse(
                        create_error_response(400, "Invalid image data in file: " + file.getFileName()));
                    resp->setStatusCode(drogon::k400BadRequest);
                    callback(resp);
                    return;
                }
                images.push_back(image);
            }
            
            // 执行批量推理
            std::vector<EngineResult> batch_results = inference_engine->batch_infer(images, confidence_threshold);
            
            // 创建批量响应
            auto resp = drogon::HttpResponse::newHttpJsonResponse(create_batch_response(batch_results));
            resp->setStatusCode(drogon::k200OK);
            callback(resp);
            
        } else {
            // 单张图片推理：处理第一个上传的文件
            const auto& file = files.front();
            auto contentView = file.fileContent();
            std::vector<uchar> buffer(contentView.begin(), contentView.end());
            cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
            
            if (image.empty()) {
                auto resp = drogon::HttpResponse::newHttpJsonResponse(
                    create_error_response(400, "Invalid image data"));
                resp->setStatusCode(drogon::k400BadRequest);
                callback(resp);
                return;
            }
            
            // 执行单张推理
            EngineResult result = infer(image, confidence_threshold);
            
            // 创建响应
            auto resp = drogon::HttpResponse::newHttpJsonResponse(create_response(result));
            resp->setStatusCode(drogon::k200OK);
            callback(resp);
        }
        
    } catch (const std::exception& e) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            create_error_response(500, "Internal server error"));
        resp->setStatusCode(drogon::k500InternalServerError);
        callback(resp);
    }
}

void InferenceServer::handle_health_check(const drogon::HttpRequestPtr& /*req*/,
                                        std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    json response;
    response["status"] = "healthy";
    response["timestamp"] = std::time(nullptr);
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response.dump());
    resp->setStatusCode(drogon::k200OK);
    
    callback(resp);
}

void InferenceServer::handle_status_request(const drogon::HttpRequestPtr& /*req*/,
                                          std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    if (!inference_engine) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            create_error_response(500, "Server not initialized"));
        resp->setStatusCode(drogon::k500InternalServerError);
        callback(resp);
        return;
    }
    
    json response;
    response["status"] = "running";
    response["timestamp"] = std::time(nullptr);
    response["model_loaded"] = inference_engine->is_ready();
    response["total_requests"] = total_requests.load();
    
    if (inference_engine->is_ready()) {
        response["model_info"]["input_width"] = inference_engine->get_input_width();
        response["model_info"]["input_height"] = inference_engine->get_input_height();
        response["model_info"]["input_channels"] = inference_engine->get_input_channels();
    }
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response.dump());
    resp->setStatusCode(drogon::k200OK);
    
    callback(resp);
}

std::string InferenceServer::create_response(const EngineResult& result) {
    json response;
    
    response["status_code"] = result.status_code;
    response["status_message"] = result.status_message;
    response["inference_time_ms"] = result.inference_time_ms;
    
    if (result.status_code == 0) {
        json detections_array = json::array();
        
        for (const auto& det : result.detections) {
            json detection;
            detection["class_id"] = det.classId;
            detection["confidence"] = det.confidence;
            detection["x"] = det.rect.x;
            detection["y"] = det.rect.y;
            detection["width"] = det.rect.width;
            detection["height"] = det.rect.height;
            
            detections_array.push_back(detection);
        }
        
        response["detections"] = detections_array;
        response["detection_count"] = result.detections.size();
    }
    
    return response.dump(2);
}

std::string InferenceServer::create_error_response(int status_code, const std::string& message) {
    json response;
    response["status_code"] = status_code;
    response["status_message"] = message;
    response["timestamp"] = std::time(nullptr);
    
    return response.dump();
}

std::string InferenceServer::create_batch_response(const std::vector<EngineResult>& results) {
    json response;
    response["status_code"] = 0;
    response["status_message"] = "Batch inference completed";
    response["timestamp"] = std::time(nullptr);
    response["total_images"] = results.size();
    json results_array = json::array();
    int total_detections = 0;
    double total_inference_time = 0.0;
    int success_count = 0;

    for(size_t index=0; index<results.size(); index++){
        const auto&result = results[index];
        json detections_result;
        detections_result["status_code"] = result.status_code;
        detections_result["status_message"] = result.status_message;
        detections_result["inference_time_ms"] =  result.inference_time_ms;
        detections_result["image_index"] = index;
        if (result.status_code == 0) {
            json detections_array = json::array();
            for(const auto& det : result.detections){
                json detection;
                detection["class_id"] = det.classId;
                detection["confidence"] = det.confidence;
                detection["x"] = det.rect.x;
                detection["y"] = det.rect.y;
                detection["width"] = det.rect.width;
                detection["height"] = det.rect.height;
                detections_array.push_back(detection);
            }
            detections_result["detections"] = detections_array;
            detections_result["detection_count"] = result.detections.size();
            total_detections += result.detections.size();
            total_inference_time += result.inference_time_ms;
            success_count++;
        }
        results_array.push_back(detections_result);
    }
    response["results"] = results_array;
    response["total_detections"] = total_detections;
    response["success_count"] = success_count;
    response["failed_count"] = results.size() - success_count;
    response["average_inference_time_ms"] = success_count > 0 ? total_inference_time / success_count : 0.0;
    return response.dump(2);
}

// 新增：视频推理处理函数
void InferenceServer::handle_video_inference_request(const drogon::HttpRequestPtr& req,
                                                   std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    try {
        // 解析表单并获取上传的视频文件
        drogon::MultiPartParser parser;
        if (parser.parse(req) != 0) {
            auto resp = drogon::HttpResponse::newHttpJsonResponse(
                create_error_response(400, "Failed to parse multipart/form-data"));
            resp->setStatusCode(drogon::k400BadRequest);
            callback(resp);
            return;
        }
        
        const auto& files = parser.getFiles();
        if (files.empty()) {
            auto resp = drogon::HttpResponse::newHttpJsonResponse(
                create_error_response(400, "No video file uploaded"));
            resp->setStatusCode(drogon::k400BadRequest);
            callback(resp);
            return;
        }
        
        // 获取置信度阈值和帧间隔
        float confidence_threshold = 0.5f;
        int frame_interval = 1;
        
        auto threshold_param = req->getParameter("confidence_threshold");
        if (!threshold_param.empty()) {
            try {
                confidence_threshold = std::stof(threshold_param);
            } catch (...) {
                // 使用默认值
            }
        }
        
        auto interval_param = req->getParameter("frame_interval");
        if (!interval_param.empty()) {
            try {
                frame_interval = std::stoi(interval_param);
            } catch (...) {
                // 使用默认值
            }
        }
        
        // 保存上传的视频文件到临时目录
        std::string temp_dir = "/tmp/video_inference/";
        std::filesystem::create_directories(temp_dir);
        
        std::string input_video_path = temp_dir + "input_" + std::to_string(std::time(nullptr)) + ".mp4";
        std::string output_video_path = temp_dir + "output_" + std::to_string(std::time(nullptr)) + ".mp4";
        
        // 复制上传的文件
        const auto& file = files.front();
        auto contentView = file.fileContent();
        std::vector<uchar> buffer(contentView.begin(), contentView.end());
        
        // 将文件内容写入临时文件
        std::ofstream output_file(input_video_path, std::ios::binary);
        if (!output_file.is_open()) {
            auto resp = drogon::HttpResponse::newHttpJsonResponse(
                create_error_response(500, "Failed to create temporary video file"));
            callback(resp);
            return;
        }
        output_file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
        output_file.close();
        
        // 执行视频推理
        bool success = process_video_inference(input_video_path, output_video_path, confidence_threshold, frame_interval);
        
        if (success) {
            // 读取输出视频文件并返回
            std::ifstream output_file(output_video_path, std::ios::binary);
            if (output_file.is_open()) {
                std::vector<char> buffer((std::istreambuf_iterator<char>(output_file)), 
                                       std::istreambuf_iterator<char>());
                
                auto resp = drogon::HttpResponse::newHttpResponse();
                resp->setBody(std::string(buffer.begin(), buffer.end()));
                resp->setContentTypeCode(drogon::CT_VIDEO_MP4);
                resp->addHeader("Content-Disposition", "attachment; filename=output_video.mp4");
                callback(resp);
                
                // 清理临时文件
                std::filesystem::remove(input_video_path);
                std::filesystem::remove(output_video_path);
            } else {
                auto resp = drogon::HttpResponse::newHttpJsonResponse(
                    create_error_response(500, "Failed to read output video file"));
                callback(resp);
            }
        } else {
            auto resp = drogon::HttpResponse::newHttpJsonResponse(
                create_error_response(500, "Video inference failed"));
            callback(resp);
        }
        
    } catch (const std::exception& e) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            create_error_response(500, std::string("Video inference error: ") + e.what()));
        callback(resp);
    }
}

// 新增：视频推理核心函数
bool InferenceServer::process_video_inference(const std::string& input_video_path,
                                            const std::string& output_video_path,
                                            float confidence_threshold,
                                            int frame_interval) {
    try {
        // 打开输入视频
        cv::VideoCapture input_cap(input_video_path);
        if (!input_cap.isOpened()) {
            std::cerr << "Error: Could not open input video: " << input_video_path << std::endl;
            return false;
        }
        
        // 获取视频属性
        int frame_width = static_cast<int>(input_cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(input_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = input_cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(input_cap.get(cv::CAP_PROP_FRAME_COUNT));
        
        std::cout << "Processing video: " << frame_width << "x" << frame_height 
                  << " @ " << fps << "fps, " << total_frames << " frames" << std::endl;
        
        // 创建视频写入器
        cv::VideoWriter output_writer(output_video_path, 
                                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                    fps, cv::Size(frame_width, frame_height));
        
        if (!output_writer.isOpened()) {
            std::cerr << "Error: Could not create output video writer" << std::endl;
            input_cap.release();
            return false;
        }
        
        cv::Mat frame;
        int processed_frames = 0;
        int current_frame = 0;
        
        while (input_cap.read(frame)) {
            if (current_frame % frame_interval == 0) {
                // 对当前帧进行推理
                EngineResult result = inference_engine->infer(frame, confidence_threshold);
                
                if (result.status_code == 0) {
                    // 在帧上绘制检测结果
                    draw_detections_on_frame(frame, result.detections);
                }
                
                processed_frames++;
                
                // 显示进度
                if (processed_frames % 10 == 0) {
                    std::cout << "Processed " << processed_frames << " frames..." << std::endl;
                }
            }
            
            // 写入帧到输出视频
            output_writer.write(frame);
            current_frame++;
        }
        
        // 清理资源
        input_cap.release();
        output_writer.release();
        
        std::cout << "Video processing completed. Processed " << processed_frames 
                  << " frames, output saved to: " << output_video_path << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in video inference: " << e.what() << std::endl;
        return false;
    }
}

// 新增：在帧上绘制检测结果 - 借鉴你的绘制逻辑
void InferenceServer::draw_detections_on_frame(cv::Mat& frame, const std::vector<DetectionBox>& detections) {
    for (const auto& detection : detections) {
        // 绘制边界框 - 使用绿色，线宽2
        cv::rectangle(frame, detection.rect, cv::Scalar(0, 255, 0), 2);
        
        // 添加类别和置信度标签 - 借鉴你的逻辑
        std::string label = "Class " + std::to_string(detection.classId) + 
                           " (" + std::to_string(detection.confidence).substr(0, 4) + ")";
        
        // 在边界框上方绘制标签
        cv::putText(frame, label, 
                   cv::Point(detection.rect.x, detection.rect.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}