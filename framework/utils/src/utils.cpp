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

int argmax(const float* data, int length) {
    if (length <= 0 || data == nullptr) return -1; // 错误处理
    int max_idx = 0;
    float max_val = data[0];
    for (int i = 1; i < length; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}


// algorithm functions
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
                  const cv::Scalar& color = cv::Scalar(114, 114, 114)) {
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
// class id不超过100
// 每种颜色要相对深一点
cv::Scalar getColor(int classId) {
    // 限制classId在0~99
    classId += 1; 

    // HSV空间分布色相，每种类别不同色相
    int h = (classId * 179 / 100); // OpenCV HSV色相范围[0,179]
    int s = 200;                   // 饱和度高一点，颜色鲜明
    int v = 120;                   // 明度低一点，颜色深一点

    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, s, v));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    // 返回BGR颜色
    cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(color[0], color[1], color[2]);
}

// 绘制检测框
void drawBox(detectBoxes& boxes, cv::Mat& img, std::string outputName, std::string outputDirPath) {

    // 遍历所有检测框
    for (const auto& box : boxes) {
        auto color = getColor(box.classId);
        // 绘制矩形框
        cv::rectangle(img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2);

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
                      color, cv::FILLED);

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

// draw segmentation mask
void drawSegmentation(const segmentBoxes& boxes, cv::Mat& img, std::string outputName, std::string outputDirPath) {
    // 遍历所有检测框
    for (const auto& box : boxes) {
        auto color = getColor(box.classId);
        // 绘制矩形框
        cv::rectangle(img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2);

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
                      color, cv::FILLED);

        // 绘制标签文本
        cv::putText(img, label, cv::Point(box.left, box.top - baseline),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        // 绘制分割掩码
        if (box.maskImg && !box.maskImg->empty()) {
            cv::Mat mask = *box.maskImg; // 单通道，0/255
            cv::Rect roi(box.left, box.top, box.width, box.height);
            cv::Mat image_roi = img(roi);

            cv::Mat color_img(mask.size(), image_roi.type(), color);

            float alpha = 0.4f; // 透明度

            // 转为float做混合
            cv::Mat image_roi_f, color_img_f, blend_f;
            image_roi.convertTo(image_roi_f, CV_32FC3);
            color_img.convertTo(color_img_f, CV_32FC3);

            // 混合
            blend_f = image_roi_f * (1.0f - alpha) + color_img_f * alpha;
            blend_f.convertTo(blend_f, image_roi.type());

            // 只在mask为1（255）的地方拷贝混合结果
            // mask 必须是单通道，0/255
            blend_f.copyTo(image_roi, mask);

            // 可选：调试输出
            // cv::imwrite("debug_mask.jpg", mask);
            // cv::imwrite("debug_color_img.jpg", color_img);
            // cv::imwrite("debug_blend.jpg", blend_f);
            // cv::imwrite("debug_image_roi.jpg", image_roi);
        }
    }

    // 保存或显示图片    
    if (access(outputDirPath.c_str(), 0) != F_OK)
        mkdir(outputDirPath.c_str(), S_IRWXU);
    auto outputPath = outputDirPath + "/" + outputName;

    if (!cv::imwrite(outputPath, img)) {
        std::cerr << "无法保存图片到: " << outputPath << std::endl;
    }
}


void drawPoseBox(poseBoxes& boxes, cv::Mat& img, std::string outputName, std::string outputDirPath) {
    std::vector<std::pair<int, int>> skeleton = {
                                                    {0,1}, {0,2}, {1,3}, {2,4}, {0,5}, {0,6},
                                                    {5,6}, {5,7}, {7,9}, {6,8}, {8,10}, {5,11}, 
                                                    {6,12}, {11,12}, {11,13}, {13,15}, {12,14}, {14,16}
                                                };
    // 遍历所有检测框
    for (const auto& box : boxes) {
        auto color = getColor(0);
        // 绘制矩形框
        cv::rectangle(img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2);

        // 准备标签内容
        char label[100];
        std::string className = "person"; // 假设只有一个类别
        snprintf(label, sizeof(label), "%s: %.2f", className.c_str(), box.score);

        // 计算文本尺寸
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // 绘制标签背景
        cv::rectangle(img,
                      cv::Point(box.left, box.top - textSize.height - baseline),
                      cv::Point(box.left + textSize.width, box.top),
                      color, cv::FILLED);

        // 绘制标签文本
        cv::putText(img, label, cv::Point(box.left, box.top - baseline),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        // 绘制关键点
        std::vector<int> keyPointIdx;
        for (int i = 0; i < box.keypoints.size(); ++i) {
            auto& kp = box.keypoints[i];
            if (kp.visibility > 0) { // 只绘制可见的关键点
                cv::circle(img, cv::Point(kp.x, kp.y), 3, color, -1); // 绘制实心圆
                keyPointIdx.push_back(i); // 记录可见关键点的索引
            }
        }
        // 根据可见关键点的索引绘制连线
        for (const auto& pair : skeleton) {
            int idx1 = pair.first;
            int idx2 = pair.second;
            // 检查这两个关键点是否都可见
            if (std::find(keyPointIdx.begin(), keyPointIdx.end(), idx1) != keyPointIdx.end() &&
                std::find(keyPointIdx.begin(), keyPointIdx.end(), idx2) != keyPointIdx.end()) {
                cv::line(img, 
                         cv::Point(box.keypoints[idx1].x, box.keypoints[idx1].y), 
                         cv::Point(box.keypoints[idx2].x, box.keypoints[idx2].y), 
                         color, 2);
            }
        }
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