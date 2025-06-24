#include "detect_factory.h"


#ifdef sophgo
    #include <sophgo_detect.h>

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
#endif

#ifdef tensorrt
    #include <trt_detect.h>
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
#endif // tensorrt