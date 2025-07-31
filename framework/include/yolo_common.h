#pragma once

#include <string>
#include <vector>
#include <sstream>

#include "log.h"
#include "magic_enum/magic_enum.hpp"

#define YOLO_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define YOLO_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))

#define enumName(expr) magic_enum::enum_name(expr) 

namespace cv {
    class Mat;
}


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
enum class yoloType {
    YOLOV5,
    YOLOV6,
    YOLOV7,
    YOLOV8,
    YOLOV9,
    YOLOV10,
    YOLOV11,
    YOLOV12,
    UNKNOWN
};
#define enumYoloType(expr) magic_enum::enum_cast<yoloType>(expr).value_or(yoloType::UNKNOWN)

enum class algorithmType {
    DETECT,
    SEGMENT,
    CLASSIFIY,
    POSE,
    UNKNOWN
};

enum class deviceType {
    SOPHGO,
    TENSORRT,
    RKNN,
    CPU,
    UNKNOWN
};

enum class resizeType {
    RESIZE,
    RESIZE_CENTER_PAD,
    RSIZE_LEFT_TOP_PAD
};

// default is YOLOv5 config
struct YOLOConfig {
    std::vector<float> mean = {0.0f, 0.0f, 0.0f};
    std::vector<float> std = {1.0f/255, 1.0f/255, 1.0f/255};
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
        case yoloType::YOLOV6:
            break;
        case yoloType::YOLOV7:
            config.anchors = {
                {{12,16}, {19,36}, {40,28}},
                {{36,75}, {76,55}, {72,146}},
                {{142,110}, {192,243}, {459,401}}
            };
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

template <>
struct magic_enum::customize::enum_range<stateType> {
    static constexpr int min = 0;
    static constexpr int max = 6000;
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

struct segmentBox {
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

    // 分割掩码
    std::vector<float> mask;
    std::shared_ptr<cv::Mat> maskImg;
};
using segmentBoxes = std::vector<segmentBox>;

struct keypoint {
    int x, y; // 关键点坐标
    float visibility; // 可见性，0表示不可见，1表示可见
};

struct poseBox {
    // 左上角坐标
    int left, top;

    // 右下角坐标
    int right, bottom;

    // 宽，高
    int width, height;

    // 置信度
    float score;

    // 兼容性
    int classId = 0;

    // 关键点
    std::vector<keypoint> keypoints;

};
using poseBoxes = std::vector<poseBox>;


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

// 单例模式
template <class objectType>
class Sinlgeton : public NoCopyable {
public:
    // 这里要返回引用或者智能指针
    static objectType& getInstance() {
        static objectType m_obj;
        return m_obj;
    }
};

