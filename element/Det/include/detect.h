#pragma once

#include <memory>

#include "yolo_common.h"



class detect : public NoCopyable{
public:
    virtual ~detect() = default;

    virtual std::vector<detectBoxes> process(void* inputImage, int num) = 0;
    virtual void printAlgorithmInfo() = 0;
    virtual stateType resetAnchor(std::vector<std::vector<std::vector<int>>>) = 0;

protected:
    yoloType m_yoloType;
    deviceType m_deviceType;
    
};



