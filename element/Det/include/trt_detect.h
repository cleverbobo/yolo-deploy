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
        trt_detect(const std::string& modelPath, const yoloType& type, const int devId = 0);
        ~trt_detect();

        std::vector<detectBoxes> process(void* inputImage, const int num) override;

    private:
        // algorithm config
        trt_logger m_logger;
        std::unique_ptr<IRuntime> m_trtRuntime;
        std::shared_ptr<ICudaEngine> m_trtEngine;
        cudaStream_t m_stream;

        // input and output config
        std::vector<char*> m_inputNames;
        std::vector<void*> m_inputMem;
        std::vector<int> m_inputSize;

        std::vector<char*> m_outputNames;
        std::vector<void*> m_outputMem;
        std::vector<int> m_outputSize;
        std::vector<std::unique_ptr<float[]>> m_outputCpuMem;

        int m_output_num, m_output_dim, m_nout;

        stateType preProcess(const Mat* inputImages, const int num);
        stateType inference();
        stateType postProcess(const Mat* inputImages, std::vector<detectBoxes>& outputBoxes, const int num);
        
        int get_element_size(const nvinfer1::DataType& type) const;
        stateType resizeBox(const Mat*  inputImages, detectBoxes& outputBoxes);
        stateType yolov5Post(const Mat* inputImages, std::vector<detectBoxes>& outputBoxes, const int num);

};


