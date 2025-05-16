#include <fstream>
#include <cmath>
#include <unistd.h>
#include <sys/stat.h>

#include "utils.h"
#include "opencv2/opencv.hpp"


// math functions
// sigmoid 
float sigmoid(float x) {
    return 1.0 / (1 + expf(-x));
}

// algorithm functions
void NMS(detectBoxes& inputBox, detectBoxes& outputBox, float nmsThreshold) {
    if (inputBox.empty()) {
        return;
    }
  
    size_t size = inputBox.size();
    // 对检测框按照置信度从高到低排序
    detectBoxes sorted_dets = inputBox;
    std::sort(sorted_dets.begin(), sorted_dets.end(), [](const detectBox &l, const detectBox &r) {
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
}


void getAspectParam(int src_w, int src_h, int dst_w, int dst_h, 
                    int& dx, int& dy, float& scale_x, float& scale_y, resizeType type) {
    
    scale_x = static_cast<float>(dst_w) / src_w;
    scale_y = static_cast<float>(dst_h) / src_h;
    switch (type) {
        case resizeType::RESIZE:
            dx = dy = 0;
            break;
        case resizeType::RESIZE_CENTER_PAD:
            if (scale_x > scale_y) {
                scale_x = scale_y;
                dx = (dst_w - src_w * scale_x) / 2;
                dy = 0;
            } else {
                scale_y = scale_x;
                dx = 0;
                dy = (dst_h - src_h * scale_y) / 2;
            }
            break;
        case resizeType::RSIZE_LEFT_TOP_PAD:
            dx = dy = 0;
            scale_x = scale_y = std::min(scale_x, scale_y);
            break;
        default:
            break;
    }
}



// draw functions only for debug
// 绘制检测框
void drawBox(detectBoxes& boxes, const std::string& inputPath, std::string outputPath = "") {
    // 读取图片
    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) {
        std::cerr << "无法读取图片: " << inputPath << std::endl;
        return;
    }

    // 遍历所有检测框
    for (const auto& box : boxes) {
        // 绘制矩形框
        cv::rectangle(img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2);

        // 准备标签内容
        char label[100];
        std::string className = box.className.empty() ? std::to_string(box.classId) : box.className;
        snprintf(label, sizeof(label), "%s: %.2f", className.c_str(), box.score);

        // 计算文本尺寸
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // 绘制标签背景
        cv::rectangle(img,
                      cv::Point(box.left, box.top - textSize.height - baseline),
                      cv::Point(box.left + textSize.width, box.top),
                      cv::Scalar(0, 255, 0), cv::FILLED);

        // 绘制标签文本
        cv::putText(img, label, cv::Point(box.left, box.top - baseline),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    // 保存或显示图片
    if (outputPath.empty()) {
        if (access("./detect_result", 0) != F_OK)
            mkdir("./detect_result", S_IRWXU);
        outputPath = "./detect_result/" + inputPath.substr(inputPath.find_last_of("/") + 1);
    }
    if (!cv::imwrite(outputPath, img)) {
        std::cerr << "无法保存图片到: " << outputPath << std::endl;
    }
    
}

// json functions
void box2json(std::string imgPath, detectBoxes& boxes, nlohmann::ordered_json& jsonObj){
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
    }
}

void jsonDump(std::string jsonPath, nlohmann::ordered_json& jsonObj){
    std::ofstream ofs(jsonPath);
    ofs << jsonObj.dump(4);
    ofs.close();
}