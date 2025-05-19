#pragma once
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

#include "yolo_common.h"
#include "json.hpp"

namespace cv {
    class Mat;
}

// math functions
float sigmoid(float x);

// algorithm functions
void NMS(detectBoxes& inputBox, detectBoxes& outputBox, float nmsThreshold);
void getAspectParam(int src_w, int src_h, int dst_w, int dst_h, 
                    int& dx, int& dy, float& scale_x, float& scale_y, resizeType type);


// draw functions only for debug
void drawBox(detectBoxes& boxes, cv::Mat& img, std::string outputName, std::string outputDirPath = "./detect_result");
// void drawSegmentation();
// void drawLandmark();

// json functions
nlohmann::ordered_json box2json(const std::string imgPath, const detectBoxes& boxes);
void boxVec2json(const std::string imgPath, const detectBoxes& boxes, nlohmann::ordered_json& jsonObj);
// void segmentation2json();
// void landmark2json();
void jsonDump(std::string jsonPath, nlohmann::ordered_json& jsonObj);
void jsonDump(std::string jsonPath, std::vector<nlohmann::ordered_json>& jsonObjVec);

// other function
void isFileExist(const std::string& filePath);
std::string getFileName(const std::string& path);

INPUTTYPE classifyInput(const std::string& str);
std::vector<std::string> getJpgFiles(const std::string& dirPath);
