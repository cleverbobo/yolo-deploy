#include <fstream>
#include <cmath>
#include <algorithm>
#include <dirent.h>

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

cv::Mat letterbox(const cv::Mat& src, const cv::Size& dst_shape, 
                  cv::Scalar color = cv::Scalar(114, 114, 114)) {
    int src_w = src.cols, src_h = src.rows;
    int dst_w = dst_shape.width, dst_h = dst_shape.height;
    float r = std::min((float)dst_w / src_w, (float)dst_h / src_h);
    int new_unpad_w = int(round(src_w * r));
    int new_unpad_h = int(round(src_h * r));
    int pad_w = dst_w - new_unpad_w;
    int pad_h = dst_h - new_unpad_h;
    int pad_left = pad_w / 2;
    int pad_top = pad_h / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_unpad_w, new_unpad_h));

    cv::Mat out;
    cv::copyMakeBorder(resized, out, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left, 
                       cv::BORDER_CONSTANT, color);
    return out;
}

// draw functions only for debug
// 绘制检测框
void drawBox(detectBoxes& boxes, cv::Mat& img, std::string outputName, std::string outputDirPath = "./detect_result") {

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
    if (access(outputDirPath.c_str(), 0) != F_OK)
        mkdir(outputDirPath.c_str(), S_IRWXU);
    auto outputPath = outputDirPath + "/" + outputName;

    if (!cv::imwrite(outputPath, img)) {
        std::cerr << "无法保存图片到: " << outputPath << std::endl;
    }
    
}

// json functions
nlohmann::ordered_json box2json(const std::string imgPath, const detectBoxes& boxes){
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
    }
    return jsonObj;
}

void boxVec2json(const std::vector<std::string>& imgPath, const std::vector<detectBoxes>& boxes, std::vector<nlohmann::ordered_json>& jsonObj) {
    auto num = imgPath.size();
    
    for (int i = 0; i < num; ++i) {
        jsonObj.push_back(box2json(imgPath[i], boxes[i]));
    }
}

void jsonDump(std::string jsonPath, nlohmann::ordered_json& jsonObj){
    std::ofstream ofs(jsonPath);
    ofs << jsonObj.dump(4);
    ofs.close();
}

void jsonDump(std::string jsonPath, std::vector<nlohmann::ordered_json>& jsonObjVec){
    std::ofstream ofs(jsonPath);
    ofs << std::setw(4) << jsonObjVec;
    ofs.close();
}

// other functions
void isFileExist(const std::string& filePath) {
    struct stat buffer;
    if (stat(filePath.c_str(), &buffer) != 0){
        std::cerr << "File does not exist: " << filePath << std::endl;
        exit(-1);
    };
}

bool startsWith(const std::string& str, const std::string& prefix) {
    if (str.length() < prefix.length()) return false;
    return str.compare(0, prefix.length(), prefix) == 0;
}

std::string getFileName(const std::string& path) {
    // 查找最后一个路径分隔符
    size_t last_slash = path.find_last_of("/\\");
    std::string filename = (last_slash != std::string::npos) ? 
                          path.substr(last_slash + 1) : path;

    // 查找最后一个点号
    size_t last_dot = filename.find_last_of('.');
    if (last_dot != std::string::npos) {
        return filename.substr(0, last_dot);
    }
    return filename; // 没有扩展名的情况
}

bool endsWithCaseInsensitive(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) return false;
    std::string str_suffix = str.substr(str.length() - suffix.length());
    std::transform(str_suffix.begin(), str_suffix.end(), str_suffix.begin(), ::tolower);
    std::string lower_suffix = suffix;
    std::transform(lower_suffix.begin(), lower_suffix.end(), lower_suffix.begin(), ::tolower);
    return str_suffix == lower_suffix;
}

INPUTTYPE classifyInput(const std::string& str) {
    // 类型1: rtsp:// 或 rtmp:// 开头
    if (startsWith(str, "rtsp://") || startsWith(str, "rtmp://")) {
        return INPUTTYPE::RTSP_OR_RTMP;
    }
    // 类型2: /dev/ 开头
    if (startsWith(str, "/dev/")) {
        return INPUTTYPE::DEVICE;
    }
    // 类型3: .jpg 结尾（不区分大小写）
    if (endsWithCaseInsensitive(str, ".jpg")) {
        return INPUTTYPE::JPG_IMAGE;
    }
    // 类型4: 视频扩展名（常见格式）
    std::vector<std::string> video_extensions = {".avi", ".mp4", ".mkv", ".mov", ".wmv", ".flv", ".m4v", ".3gp",".h264", ".h265"};
    for (const auto& ext : video_extensions) {
        if (endsWithCaseInsensitive(str, ext)) {
            return INPUTTYPE::VIDEO;
        }
    }

    // 类型4: 图片文件夹
    struct stat info;
    if (!stat(str.c_str(), &info) && info.st_mode & S_IFDIR) {
        return INPUTTYPE::IMAGE_DIR;
    }
    // 类型5: 其他不支持的类型
    return INPUTTYPE::UNKNOW;
}

std::vector<std::string> getJpgFiles(const std::string& dirPath) {
    std::vector<std::string> jpgFiles;
    DIR* dir = opendir(dirPath.c_str());
    if (dir == nullptr) {
        std::cerr << "无法打开目录: " << dirPath << std::endl;
        return jpgFiles;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string fileName(entry->d_name);
        if (endsWithCaseInsensitive(fileName, ".jpg")) {
            jpgFiles.push_back(dirPath + "/" + fileName);
        }
    }
    closedir(dir);
    return jpgFiles;
}

// 以二进制方式加载文件
std::vector<char> loadFile(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return {};
    }
    
    file.seekg(0, std::ifstream::end);
    auto size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    return buffer;
}