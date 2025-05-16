#pragma once
#include <vector>

#include "yolo_common.h"
#include "json.hpp"

// math functions
float sigmoid(float x);

// algorithm functions
void NMS(detectBoxes& inputBox, detectBoxes& outputBox, float nmsThreshold);
void getAspectParam(int src_w, int src_h, int dst_w, int dst_h, 
                    int& dx, int& dy, float& scale_x, float& scale_y, resizeType type);


// draw functions only for debug
void drawBox(detectBoxes& boxes, const std::string& inputPath,std::string outputPath = "");
// void drawSegmentation();
// void drawLandmark();

// json functions
void box2json(std::string imgPath, detectBoxes& boxes, nlohmann::ordered_json& jsonObj);
// void segmentation2json();
// void landmark2json();
void jsonDump(std::string jsonPath, nlohmann::ordered_json& jsonObj);

