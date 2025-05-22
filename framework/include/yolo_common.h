#pragma once

#include <string>
#include <vector>
#include <sstream>

#include "log.hpp"

#define YOLO_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define YOLO_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))

inline std::string concatArgs() { return ""; }

template <typename T, typename... Args>
inline std::string concatArgs(const T& arg, const Args&... args) {
  std::stringstream ss;
  ss << std::string(arg);
  return ss.str() + concatArgs(args...);
}

#define YOLO_CHECK(cond, ...)                                               \
  if (YOLO_UNLIKELY(!(cond))) {                                             \
    std::string msg = concatArgs(__VA_ARGS__);                                \
    std::string error_msg =                                                   \
        "Expected [ " #cond " ] to be true, but got false. The reason is [ " + (msg) +" ]";             \
    YOLO_CRITICAL("YOLO_CHECK failed: {}", error_msg);                        \
    exit(1);                                                                  \
  }

// yolo config
enum struct yoloType {
    YOLOV5,
    YOLOV6,
    YOLOV7,
    YOLOV8
};

enum class algorithmType {
    DETECT,
    SEGMENTATION,
    LANDMARK,
    CLASSIFICATION,
    POSE_ESTIMATION,
    UNKNOWN
};

enum class deviceType {
    SOPHGO,
    NVIDIA,
    RKNN,
    JETSON,
    CPU
};

enum class resizeType {
    RESIZE,
    RESIZE_CENTER_PAD,
    RSIZE_LEFT_TOP_PAD
};

// default is YOLOv5 config
struct YOLOConfig {
    std::vector<float> mean = {1/255.0, 1/255.0, 1/255.0};
    std::vector<float> std = {1.0, 1.0, 1.0};
    bool bgr2rgb = true;
    resizeType resize_type = resizeType::RESIZE_CENTER_PAD;
    int padValue = 114;
    std::vector<std::vector<std::vector<int>>> anchors = {{{10, 13}, {16, 30}, {33, 23}},
                                                          {{30, 61}, {62, 45}, {59, 119}},
                                                          {{116, 90}, {156, 198}, {373, 326}}};
};

struct algorithmInfo {
    yoloType yolo_type;
    algorithmType algorithm_type;
    deviceType device_type;
    std::vector<std::vector<int>> input_shape;
    std::vector<std::vector<int>> output_shape;
    int batch_size;
};


// transform config
inline YOLOConfig getYOLOConfig(yoloType type) {
    YOLOConfig config;
    switch (type) {
        case yoloType::YOLOV5:
            break;
        default:
            break;
    }
    return config;
}

enum class stateType {

    SUCCESS = 0,
    ERROR = 5001,
    INFERENCE_ERROR,
    UNMATCH_YOLO_TYPE_ERROR
};



struct detectBox {
    // 左上角坐标
    int left,top;

    // 右下角坐标
    int right,bottom;

    // 宽，高
    int width, height;

    // 置信度
    float score;

    // 类别id
    int classId;

    // 类别名称
    std::string className="";
};
using detectBoxes = std::vector<detectBox>;


// 输出的数据格式
enum class INPUTTYPE{
    JPG_IMAGE,
    IMAGE,
    IMAGE_DIR,
    VIDEO,
    RTSP_OR_RTMP,
    DEVICE,
    UNKNOW
};

// 不能拷贝的类
class NoCopyable {
    protected:
      NoCopyable() =default;
      ~NoCopyable() = default;
      NoCopyable(const NoCopyable&) = delete;
      NoCopyable& operator=(const NoCopyable& rhs)= delete;
  };
  