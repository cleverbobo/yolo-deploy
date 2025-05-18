#pragma once
#include "detect.h"

class BMNNContext;
class BMNNNetwork;
class BMNNHandle;
class bm_image;
typedef struct bmcv_padding_atrr_s bmcv_padding_atrr_t;



class sophgo_detect : public detect {
    public: 
        sophgo_detect(std::string modelPath, yoloType type, int devId = 0);
        ~sophgo_detect() override;
    
        std::vector<detectBoxes> process(void* inputImage, int num) override;
        algorithmInfo getAlgorithmInfo() override;
        void printAlgorithmInfo() override;
        stateType resetAnchor(std::vector<std::vector<std::vector<int>>>) override;
    
    private:
        // algorithm config
        std::shared_ptr<BMNNContext> m_bmContext;
        std::shared_ptr<BMNNNetwork> m_bmNetwork;
        std::vector<bm_image> m_preprocess_images;
        std::shared_ptr<BMNNHandle> m_handle;
        algorithmInfo m_algorithmInfo;
        int m_devId = 0;
        yoloType m_yoloType;

        // preprocess config
        std::vector<float> m_mean;
        std::vector<float> m_std;
        bool m_bgr2rgb;
        int m_padValue;
        resizeType m_resizeType;

        std::vector<bmcv_padding_atrr_t> m_padding_attr;

        // network config
        std::vector<std::string> m_class_names;
        int m_class_num;
        int m_net_h, m_net_w;
        int m_max_batch;

        // postprocess config
        float m_confThreshold= 0.5;
        float m_nmsThreshold = 0.5;
        std::vector<std::vector<std::vector<int>>> m_anchors;
        int m_output_num, m_output_dim, m_nout;

        stateType preProcess(bm_image* inputImages, int num);
        stateType inference();
        stateType postProcess(bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, int num);
        
        stateType resizeBox(bm_image* inputImages, detectBoxes& outputBoxes);
        stateType yolov5Post(bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, int num);
    };
    