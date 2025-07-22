#pragma once

#include <memory>
#include "yolo_common.h"
#include "fps_counter.h"



class segment : public NoCopyable {
public:
    segment(const std::string& modelPath, const yoloType& type, const int devId = 0);
    virtual ~segment();

    virtual std::vector<segmentBoxes> process(void* inputImage, const int num) = 0;

    virtual algorithmInfo getAlgorithmInfo();
    virtual void printAlgorithmInfo();
    virtual void resetAnchor(const std::vector<std::vector<std::vector<int>>>&);
    virtual void resetConf(const float detConf, const float nmsConf);
    virtual void resetPreprocessConfig(const std::vector<float>& mean = {0.0f, 0.0f, 0.0f}, const std::vector<float>& std = {1.0f/255, 1.0f/255, 1.0f/255}, 
                                       const bool bgr2rgb = true, const int padValue = 114, const resizeType& resizeType = resizeType::RESIZE_CENTER_PAD);
    virtual void resetClassNames(const std::vector<std::string>& class_names);

protected:
    std::string m_model_path;
    algorithmInfo m_algorithmInfo;
    std::vector<std::vector<std::vector<int>>> m_anchors;
    int m_devId;
    yoloType m_yoloType;

    // network config
    std::vector<std::string> m_class_names;
    int m_class_num = 0;
    int m_net_h, m_net_w;
    int m_max_batch;

    // preprocess config
    std::vector<float> m_mean;
    std::vector<float> m_std;
    bool m_bgr2rgb;
    int m_padValue;
    resizeType m_resizeType;

    // postprocess config
    float m_confThreshold = 0.5f;
    float m_nmsThreshold = 0.5f;
    int m_output_num;
    int m_output_det_dim;
    int m_output_seg_dim;
    int m_seg_feature_size;
    int m_nout;

    // fps counter
    fpsCounter m_fpsCounter;

    
};