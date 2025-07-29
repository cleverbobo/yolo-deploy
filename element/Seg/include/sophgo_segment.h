#pragma once
#include "segment.h"

class BMNNContext;
class BMNNNetwork;
class BMNNHandle;
class BMNNTensor;
class bm_image;
typedef struct bmcv_padding_atrr_s bmcv_padding_atrr_t;
typedef struct bm_shape_s bm_shape_t;


class sophgo_segment : public segment {
public:
    sophgo_segment(const std::string& modelPath, const yoloType& type, const int devId = 0);
    ~sophgo_segment();

    std::vector<segmentBoxes> process(void* inputImage, const int num) override;

private:
    // algorithm config
    std::shared_ptr<BMNNContext> m_bmContext;
    std::shared_ptr<BMNNNetwork> m_bmNetwork;
    std::vector<bm_image> m_preprocess_images;
    std::shared_ptr<BMNNHandle> m_handle;
    std::vector<bmcv_padding_atrr_t> m_padding_attr;

    stateType preProcess(bm_image* inputImages, const int num);
    stateType inference();
    stateType postProcess(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num);
    
    stateType getSegmentBox(const bm_image* inputImages, segmentBoxes& outputBoxes, float* proto_data, const bm_shape_t* proto_shape);
    stateType yolov5Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num);
    stateType yolov6Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num);
    stateType yolov7Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num);
    stateType yolov8Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num);
    stateType yolov9Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num);
    stateType yolov11Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num);
    stateType yolov12Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num);
};