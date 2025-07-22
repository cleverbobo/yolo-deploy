#include "segment_factory.h"

#ifdef sophgo

#include <sophgo_segment.h>
std::shared_ptr<segment> sophgo_segment_factory::getInstance(std::string modelPath, yoloType type, int devId) {
    return std::make_shared<sophgo_segment>(modelPath, type, devId);
}

#endif // sophgo

