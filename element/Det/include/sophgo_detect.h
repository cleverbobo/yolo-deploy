#pragma once
#include "detect.h"
#include "safe_queue.hpp"
#include "threadpool.hpp"

class BMNNContext;
class BMNNNetwork;
class BMNNHandle;
class bm_image;
typedef struct bmcv_padding_atrr_s bmcv_padding_atrr_t;




class sophgo_detect : public detect {
    public: 
        sophgo_detect(const std::string& modelPath, const yoloType& type, const int devId = 0);
        ~sophgo_detect();
    
        // 同步接口
        std::vector<detectBoxes> process(void* inputImage, const int num) override;

        // 异步接口
        stateType processAsync(void* inputImage, const int num) override;
        stateType getDetResult(detectBoxes& outputBoxes, const int idx = 0) override;

    
    private:
        // algorithm config
        std::shared_ptr<BMNNContext> m_bmContext;
        std::shared_ptr<BMNNNetwork> m_bmNetwork;
        std::shared_ptr<BMNNHandle> m_handle;

        int m_output_num, m_output_dim, m_nout;

        stateType preProcess(bm_image* inputImages, bm_image* outputImages, const int num);
        stateType PreProcessAsync();

        stateType inference(bm_image* inputImage, std::vector<BMNNTensor>& outputTensors, int num);
        stateType inferenceAsync();

        stateType postProcess(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType postProcessAsync();

        // 线程池
        ThreadPool<bm_image*> m_thead_preprocess;
        ThreadPool<BMNNTensor> m_thead_inference;
        ThreadPool<std::vector<BMNNTensor>> m_thead_postprocess;
        void worker();

        int inputNum, completeNum;
        std::queue<bm_image*> m_inputImages;
        std::queue<BMNNTensor> m_inputTensors;
        std::queue<std::vector<BMNNTensor>> m_outputTensors;
        std::queue<detectBoxes> m_outputBoxes;
        
        stateType resizeBox(const bm_image* inputImages, detectBoxes& outputBoxes);
        stateType yolov5Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov6Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov7Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov8Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov9Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov10Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov11Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num);
        stateType yolov12Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num);

    };
    