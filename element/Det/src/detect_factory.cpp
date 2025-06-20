#include "detect_factory.h"
#include <sophgo_detect.h>
#include <trt_detect.h>

// sophgo_detect_factory implementation
sophgo_detect_factory::sophgo_detect_factory() {
    ;
}

sophgo_detect_factory::~sophgo_detect_factory() {
    ;
}

std::shared_ptr<detect> sophgo_detect_factory::getInstance(std::string modelPath, yoloType type, int devId) {
    return std::make_shared<sophgo_detect>(modelPath, type, devId);
}

// trt_detect_factory implementation
trt_detect_factory::trt_detect_factory() {
    ;
}

trt_detect_factory::~trt_detect_factory() {
    ;
}

std::shared_ptr<detect> trt_detect_factory::getInstance(std::string modelPath, yoloType type, int devId) {
    return std::make_shared<trt_detect>(modelPath, type, devId);
}