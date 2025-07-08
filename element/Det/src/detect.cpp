#include "detect.h"

detect::detect(const std::string& modelPath, const yoloType& type, const int devId)
              :m_devId(devId), m_model_path(modelPath), m_yoloType(type),m_fpsCounter(std::string(enumName(type)) + "_Detect", 100, 1000.0f)  {

    // set algorithm preprocess config
    YOLOConfig yoloConfig = getYOLOConfig(type);
    m_mean = yoloConfig.mean;
    m_std = yoloConfig.std;
    m_bgr2rgb = yoloConfig.bgr2rgb;
    m_padValue = yoloConfig.padValue;
    m_anchors = yoloConfig.anchors;
    m_resizeType = yoloConfig.resize_type;

}

detect::~detect() {
    auto avgFps = m_fpsCounter.getAvgFps();
    std::cout << enumName(m_yoloType) << " detection is finished. Average FPS: " << avgFps << std::endl;
}

algorithmInfo detect::getAlgorithmInfo() {
    return m_algorithmInfo;
}

void detect::printAlgorithmInfo() {
    std::cout << "----------------AlgorithmInfo-----------------" << std::endl;
    std::cout << "[1] Model Basic Information" << std::endl;
    std::cout << "Model Path: " << m_model_path << std::endl;
    std::cout << "YOLO Type: " << enumName(m_algorithmInfo.yolo_type) << std::endl;
    std::cout << "Device Type: " << enumName(m_algorithmInfo.device_type) << std::endl;
    std::cout << "Device ID: " << m_devId << std::endl;
    std::cout << "Algorithm Type: " << enumName(m_algorithmInfo.algorithm_type) << std::endl;
    std::cout << "Input Shape: ";
    for (const auto& shape : m_algorithmInfo.input_shape) {
        std::cout << "[";
        for (const auto& dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << "] ";
    }
    std::cout << std::endl;

    std::cout << "Output Shape: ";
    for (const auto& shape : m_algorithmInfo.output_shape) {
        std::cout << "[";
        for (const auto& dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << "] ";
    }
    std::cout << std::endl;
    std::cout << "Batch Size: " << m_algorithmInfo.batch_size << std::endl <<std::endl;;
    // preprocess config
    std::cout << "[2] Preprocess Config" << std::endl;
    std::cout << "Preprocess Mean: ";
    for (const auto& val : m_mean) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Preprocess Std: ";
    for (const auto& val : m_std) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << "BGR to RGB: " << (m_bgr2rgb ? "true" : "false") << std::endl;
    std::cout << "Pad Value: " << m_padValue << std::endl;
    std::cout << "Resize Type: " << enumName(m_resizeType) << std::endl << std::endl;

    // postprocess config
    std::cout << "[3] Postprocess Config" << std::endl;
    std::cout << "Confidence Threshold: " << m_confThreshold << std::endl;
    std::cout << "NMS Threshold: " << m_nmsThreshold << std::endl;
    if (!m_anchors.empty()) {
        std::cout << "Anchors: " << std::endl;
        for (const auto& anchor_set : m_anchors) {
            std::cout << "[";
            for (const auto& anchor : anchor_set) {
                std::cout << "[";
                for (const auto& dim : anchor) {
                    std::cout << dim << " ";
                }
                std::cout << "] ";
            }
            std::cout << "]" << std::endl;
        }
    } else {
        std::cout << "Anchors: Not set" << std::endl;
    }
    std::cout << "----------------AlgorithmInfo-----------------" << std::endl;
}

void detect::resetAnchor(const std::vector<std::vector<std::vector<int>>>& anchors) {
    m_anchors = anchors;
 }

void detect::resetConf(const float detConf, const float nmsConf) {
    m_confThreshold = detConf;
    m_nmsThreshold = nmsConf;
}

void detect::resetPreprocessConfig(const std::vector<float>& mean, const std::vector<float>& std, 
                                   const bool bgr2rgb, const int padValue, const resizeType& resizeType) {
    if (mean.size() != 3 || std.size() != 3) {
        YOLO_WARN("Mean and std vectors should have 3 elements each. Using default values.");
        YOLO_WARN("Your mean's size is {}. Your std's size is {}.", mean.size(), std.size());
        YOLO_WARN("skipping resetPreprocessConfig.");
        return;
    }
    m_mean = mean;
    m_std = std;
    m_bgr2rgb = bgr2rgb;

    if ( padValue < 0 || padValue > 255) {
        YOLO_WARN("Pad value should be non-negative and less than or equal to 255. Using default value 114.");
        m_padValue = 114;
    } else {
        m_padValue = padValue;
    }
    m_resizeType = resizeType;
}

void detect::resetClassNames(const std::vector<std::string>& class_names) {
    m_class_names = class_names;
    if(m_class_num && m_class_num != class_names.size()) {
        YOLO_WARN("Class names are already set. Old num [{}] vs new num [{}]. Resetting class names will overwrite the previous ones. ",
                  m_class_num, class_names.size());
        YOLO_WARN("It may cause unexpected behavior if the model is not compatible with the new class number.");
        m_class_num = class_names.size();
    }
    
}

