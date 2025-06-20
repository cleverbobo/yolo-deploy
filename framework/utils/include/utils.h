#pragma once
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

#include "yolo_common.h"
#include "json.hpp"

namespace cv {
    class Mat;
    
    template<typename _Tp> class Size_;
    typedef Size_<int> Size;

    template<typename _Tp> class Scalar_;
    typedef Scalar_<double> Scalar;
}

// math functions
float sigmoid(float x);

// algorithm functions
void NMS(detectBoxes& inputBox, detectBoxes& outputBox, float nmsThreshold);
void getAspectParam(int src_w, int src_h, int dst_w, int dst_h, 
                    int& dx, int& dy, float& scale_x, float& scale_y, resizeType type);
cv::Mat letterbox(const cv::Mat& src, const cv::Size& dst_shape,const cv::Scalar& color);

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

// 以二进制方式加载文件
std::vector<char> loadFile(const std::string& filePath);
