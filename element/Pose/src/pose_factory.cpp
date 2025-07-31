#include "pose_factory.h"


#ifdef sophgo
    #include <sophgo_pose.h>

    // sophgo_pose_factory implementation
    sophgo_pose_factory::sophgo_pose_factory() {
        ;
    }

    sophgo_pose_factory::~sophgo_pose_factory() {
        ;
    }

    std::shared_ptr<pose> sophgo_pose_factory::getInstance(std::string modelPath, yoloType type, int devId) {
        return std::make_shared<sophgo_pose>(modelPath, type, devId);
    }
#endif

#ifdef tensorrt
    #include <trt_pose.h>
    // trt_pose_factory implementation
    trt_pose_factory::trt_pose_factory() {
        ;
    }

    trt_pose_factory::~trt_pose_factory() {
        ;
    }

    std::shared_ptr<pose> trt_pose_factory::getInstance(std::string modelPath, yoloType type, int devId) {
        return std::make_shared<trt_pose>(modelPath, type, devId);
    }
#endif // tensorrt