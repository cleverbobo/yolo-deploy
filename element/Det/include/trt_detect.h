#pragma once

#include "detect.h"
#include "NvInfer.h"
#include "opencv2/opencv.hpp"

using nvinfer1::IRuntime;
using nvinfer1::ICudaEngine;
using nvinfer1::ILogger;
using cv::Mat;

class trt_logger : public ILogger {
    public:
        void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;
};

class trt_detect : public detect {
    
    public:
        trt_detect(std::string modelPath, yoloType type, int devId = 0);
        ~trt_detect();

        std::vector<detectBoxes> process(void* inputImage, int num) override;

        algorithmInfo getAlgorithmInfo() override;
        void printAlgorithmInfo() override;
        stateType resetAnchor(std::vector<std::vector<std::vector<int>>>) override;
    
    private:
        // algorithm config
        trt_logger m_logger;
        std::unique_ptr<IRuntime> m_trtRuntime;
        std::shared_ptr<ICudaEngine> m_trtEngine;
        cudaStream_t m_stream;
        std::vector<Mat> m_preprocess_images;

        // input and output config
        std::vector<char*> m_inputNames;
        std::vector<void*> m_inputMem;
        std::vector<int> m_inputSize;

        std::vector<char*> m_outputNames;
        std::vector<void*> m_outputMem;
        std::vector<int> m_outputSize;
        std::vector<std::unique_ptr<float[]>> m_outputCpuMem;

  
        algorithmInfo m_algorithmInfo;
        int m_devId = 0;
        yoloType m_yoloType;

        // preprocess config
        std::vector<float> m_mean;
        std::vector<float> m_std;
        bool m_bgr2rgb;
        int m_padValue;
        resizeType m_resizeType;

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

        stateType preProcess(const Mat* inputImages, int num);
        stateType inference();
        stateType postProcess(const Mat* inputImages, std::vector<detectBoxes>& outputBoxes, int num);
        
        int get_element_size(nvinfer1::DataType type) const;
        stateType resizeBox(const Mat*  inputImages, detectBoxes& outputBoxes);
        stateType yolov5Post(const Mat* inputImages, std::vector<detectBoxes>& outputBoxes, int num);

};


