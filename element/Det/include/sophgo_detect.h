#pragma once
#include "detect.h"

class BMNNContext;
class BMNNNetwork;
class BMNNHandle;
class bm_image;
typedef struct bmcv_padding_atrr_s bmcv_padding_atrr_t;



class sophgo_detect : public detect {
    public: 
        sophgo_detect(const std::string& modelPath, const yoloType& type, const int devId = 0);
        ~sophgo_detect() override;
    
        std::vector<detectBoxes> process(void* inputImage, const int num) override;

    
    private:
        // algorithm config
        std::shared_ptr<BMNNContext> m_bmContext;
        std::shared_ptr<BMNNNetwork> m_bmNetwork;
        std::vector<bm_image> m_preprocess_images;
        std::shared_ptr<BMNNHandle> m_handle;
        std::vector<bmcv_padding_atrr_t> m_padding_attr;

        int m_output_num, m_output_dim, m_nout;

        stateType preProcess(bm_image* inputImages, const int num);
        stateType inference();
        stateType postProcess(const bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, const int num);
        
        stateType resizeBox(const bm_image* inputImages, detectBoxes& outputBoxes);
        stateType yolov5Post(const bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov6Post(const bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov7Post(const bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov8Post(const bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov9Post(const bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov10Post(const bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov11Post(const bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov12Post(const bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, const int num);

    };
    