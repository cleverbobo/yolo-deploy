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
int argmax(const float* data, int length);

// algorithm functions
template <typename T = detectBoxes>
void NMS(T& inputBox, T& outputBox, float nmsThreshold) {
    if (inputBox.empty()) {
        return;
    }
  
    size_t size = inputBox.size();
    // 对检测框按照置信度从高到低排序
    T sorted_dets = inputBox;
    std::sort(sorted_dets.begin(), sorted_dets.end(), [](const typename T::value_type &l, const typename T::value_type &r) {
        return l.score > r.score;
    });
  
    std::vector<char> pInvalidIndexes(size, 0); // 使用 char 节省内存
    std::vector<float> pData(5 * size);       // 存储 [x1, y1, x2, y2, area]
  
    // 预计算每个检测框的坐标和面积
    for (size_t i = 0; i < size; ++i) {
        pData[i * 5 + 0] = sorted_dets[i].left;
        pData[i * 5 + 1] = sorted_dets[i].top;
        pData[i * 5 + 2] = sorted_dets[i].right;
        pData[i * 5 + 3] = sorted_dets[i].bottom;
        pData[i * 5 + 4] = sorted_dets[i].width * sorted_dets[i].height;
    }
  
    // 开始进行 NMS 处理
    auto pValidIndexes_1 = pInvalidIndexes.begin();
    auto sorted_dets_iter  = sorted_dets.begin();
    const float *lbox = pData.data();
  
    for (size_t m = 0; m < size; ++m, pValidIndexes_1++, sorted_dets_iter++, lbox+=5) {
        if (*pValidIndexes_1) {
            continue;
        }
        outputBox.push_back(*sorted_dets_iter);
        
        auto pValidIndexes_2 = pValidIndexes_1 + 1;
        auto rbox  = lbox + 5;
        for (size_t n = m + 1; n < size; ++n, pValidIndexes_2++, rbox+=5) {
            if (*pValidIndexes_2) {
                continue;
            }
  
            // 计算交叉区域
            const float inter_x1 = std::max(lbox[0], rbox[0]);
            const float inter_y1 = std::max(lbox[1], rbox[1]);
            const float inter_x2 = std::min(lbox[2], rbox[2]);
            const float inter_y2 = std::min(lbox[3], rbox[3]);
  
            if (inter_x1 >= inter_x2 || inter_y1 >= inter_y2) {
                continue; // 没有重叠区域
            }
  
            const float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
            const float union_area = lbox[4] + rbox[4] - inter_area;
            *pValidIndexes_2 = inter_area > nmsThreshold * union_area;
  
        }
    }
};

void getAspectParam(int src_w, int src_h, int dst_w, int dst_h, 
                    int& dx, int& dy, float& scale_x, float& scale_y, resizeType type);
cv::Mat letterbox(const cv::Mat& src, const cv::Size& dst_shape,const cv::Scalar& color);

template <typename T = detectBoxes>
void restrictBox(T& box, const int img_w, const int img_h){
    box.left = std::min(std::max(0, box.left),img_w - 1);
    box.top = std::min(std::max(0, box.top),img_h - 1);
    box.right = std::min(std::max(0, box.right),img_w);
    box.bottom = std::min(std::max(0, box.bottom),img_h);

    box.width = std::max(box.right - box.left,0);
    box.height = std::max(box.bottom - box.top,0);
};

// draw functions only for debug
cv::Scalar getColor(int classId);
void drawBox(detectBoxes& boxes, cv::Mat& img, std::string outputName, std::string outputDirPath = "./detect_result");
void drawSegmentation(const segmentBoxes& boxes, cv::Mat& img, std::string outputName, std::string outputDirPath = "./segment_result");
// void drawLandmark();

// json functions

template <typename T = detectBoxes>
nlohmann::ordered_json box2json(const std::string imgPath, const T& boxes){
    nlohmann::ordered_json jsonObj;
    jsonObj["image_path"] = imgPath;
    jsonObj["boxes"] = nlohmann::ordered_json::array();
    for (const auto& box : boxes) {
        nlohmann::ordered_json boxJson;
        boxJson["left"] = box.left;
        boxJson["top"] = box.top;
        boxJson["right"] = box.right;
        boxJson["bottom"] = box.bottom;
        boxJson["width"] = box.width;
        boxJson["height"] = box.height;
        boxJson["score"] = box.score;
        boxJson["classId"] = box.classId;
        jsonObj["boxes"].push_back(boxJson);
        if constexpr (std::is_same<T, segmentBoxes>::value) {
            std::vector<int> maskImg_data(box.maskImg->rows * box.maskImg->cols);
            std::memcpy(maskImg_data.data(), box.maskImg->data, maskImg_data.size() * sizeof(int));
            boxJson["maskImg"] = maskImg_data;
            boxJson["mask"] = box.mask;
        }
    }
    return jsonObj;
};

template <typename T = detectBoxes>
void boxVec2json(const std::string imgPath, const T& boxes, nlohmann::ordered_json& jsonObj){
    auto num = imgPath.size();
    
    for (int i = 0; i < num; ++i) {
        jsonObj.push_back(box2json(imgPath[i], boxes[i]));
    }
};



void jsonDump(std::string jsonPath, nlohmann::ordered_json& jsonObj);
void jsonDump(std::string jsonPath, std::vector<nlohmann::ordered_json>& jsonObjVec);

// other function
void isFileExist(const std::string& filePath);
std::string getFileName(const std::string& path);

INPUTTYPE classifyInput(const std::string& str);
std::vector<std::string> getJpgFiles(const std::string& dirPath);

// 以二进制方式加载文件
std::vector<char> loadFile(const std::string& filePath);
