#include "trt_pose.h"
#include "utils.h"

// debug
#ifdef _DEBUG
#include <cnpy.h>
#endif

void trt_logger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept{
    // Initialize logger if needed
    auto m_logger = yoloLogger::getInstance();
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            m_logger->error("TRT Internal Error: {}", msg);
            break;
        case Severity::kERROR:
            m_logger->error("TRT Error: {}", msg);
            break;
        case Severity::kWARNING:
            m_logger->warn("TRT Warning: {}", msg);
            break;
        case Severity::kINFO:
            m_logger->info("TRT Info: {}", msg);
            break;
        case Severity::kVERBOSE:
            m_logger->debug("TRT Verbose: {}", msg);
            break;
        default:
            m_logger->warn("TRT Unknown Severity: {}", msg);
            break; 
    }
}

trt_pose::trt_pose(const std::string& modelPath, const yoloType& type, const int devId)
    : pose(modelPath,type,devId) {
    m_logger = trt_logger();
    // runtime initialization
    m_trtRuntime.reset(nvinfer1::createInferRuntime(m_logger));

    // load the engine
    auto engineFileData = loadFile(modelPath);
    YOLO_CHECK(!engineFileData.empty(), "Failed to load TRT engine file: " + modelPath);
    m_trtEngine.reset(
        m_trtRuntime->deserializeCudaEngine(engineFileData.data(), engineFileData.size()));
    
    cudaStreamCreate(&m_stream);

    // 申请输入输出内存
    auto number_io = m_trtEngine->getNbIOTensors();
    std::vector<char*> tensor_names;
    for (int i = 0; i < number_io; i++) {
        tensor_names.push_back(const_cast<char*>(m_trtEngine->getIOTensorName(i)));
    }
    for (const auto& name:tensor_names) {
        auto tensor_shape = m_trtEngine->getTensorShape(name);
        auto tensor_type = m_trtEngine->getTensorDataType(name);
        int element_size = get_element_size(tensor_type);
        int tensor_size = element_size * std::accumulate(tensor_shape.d, tensor_shape.d + tensor_shape.nbDims, 1, std::multiplies<int64_t>());
        auto IOMode = m_trtEngine->getTensorIOMode(name);
        if ( IOMode == nvinfer1::TensorIOMode::kINPUT ) {
            m_inputNames.push_back(name);
            void* input_mem;
            cudaMallocAsync(&input_mem, tensor_size, m_stream);
            m_inputMem.push_back(input_mem);
            m_inputSize.push_back(tensor_size);
            // 记录输入的网络信息
            m_algorithmInfo.input_shape.push_back(std::vector<int>(tensor_shape.d, tensor_shape.d + tensor_shape.nbDims));
            
        }else{
            m_outputNames.push_back(name);
            void* output_mem;
            cudaMallocAsync(&output_mem, tensor_size, m_stream);
            m_outputMem.push_back(output_mem);
            m_outputSize.push_back(tensor_size);
            // 记录输出的网络信息
            m_algorithmInfo.output_shape.push_back(std::vector<int>(tensor_shape.d, tensor_shape.d + tensor_shape.nbDims));

            // 分配CPU内存用于输出
            std::unique_ptr<float[]> ouputCpuMem = std::make_unique<float[]>(tensor_size / element_size);
            m_outputCpuMem.push_back(std::move(ouputCpuMem));
        }

    }

    // 配置network config
    m_max_batch = m_trtEngine->getTensorShape(m_inputNames[0]).d[0];
    m_net_h = m_trtEngine->getTensorShape(m_inputNames[0]).d[2];
    m_net_w = m_trtEngine->getTensorShape(m_inputNames[0]).d[3];
    
    // postprocess config
    m_output_num = m_outputNames.size();
    m_output_dim = m_trtEngine->getTensorShape(m_outputNames[0]).nbDims;
    m_nout = m_trtEngine->getTensorShape(m_outputNames[0]).d[m_output_dim - 1];

    
    // 配置 algorithm info
    m_algorithmInfo.yolo_type = m_yoloType;
    m_algorithmInfo.algorithm_type = algorithmType::POSE;
    m_algorithmInfo.device_type = deviceType::TENSORRT;
    m_algorithmInfo.batch_size = m_trtEngine->getTensorShape(m_inputNames[0]).d[0];

    printAlgorithmInfo();
        
}

trt_pose::~trt_pose() {
    for(auto& mem: m_inputMem) {
        cudaFreeAsync(mem, m_stream);
    }

    for(auto& mem: m_outputMem) {
        cudaFreeAsync(mem, m_stream);
    }

    cudaStreamDestroy(m_stream);
}

std::vector<poseBoxes> trt_pose::process(void* inputImage, const int num) {
    const cv::Mat* imgPtr = static_cast<const cv::Mat*>(inputImage);

    stateType ret = stateType::SUCCESS;
    int calculateTime = (num-1) / m_max_batch + 1;
    std::vector<poseBoxes> outputBoxes;

    for (int i = 0; i < calculateTime; ++i) {
        int inputNum = std::min(num - i * m_max_batch, m_max_batch);
        // preProcess the input images
        auto ret = preProcess(imgPtr + m_max_batch*i, num);
        YOLO_CHECK(ret == stateType::SUCCESS, "TRT Preprocess failed");

        // inference the model
        ret = inference();
        YOLO_CHECK(ret == stateType::SUCCESS, "TRT Inference failed");

        // postProcess the output
        
        ret = postProcess(imgPtr + m_max_batch*i, outputBoxes, num);
        YOLO_CHECK(ret == stateType::SUCCESS, "TRT Postprocess failed");

        m_fpsCounter.add(inputNum);
    }
    return outputBoxes;
}


stateType trt_pose::preProcess(const Mat* inputImages, const int num){
    for (int i = 0; i < num; ++i) {
        cv::Mat img_letterbox = letterbox(inputImages[i], cv::Size(m_net_w, m_net_h), cv::Scalar(m_padValue, m_padValue, m_padValue));
        cv::Mat blob = cv::dnn::blobFromImage(img_letterbox, m_std[0], cv::Size(m_net_w, m_net_h),
                                              cv::Scalar(m_mean[0],m_mean[1],m_mean[2]), true, false);  
        cudaMemcpyAsync(m_inputMem[i], blob.data, m_inputSize[i], cudaMemcpyHostToDevice, m_stream);
    }
    return stateType::SUCCESS;
}

stateType trt_pose::inference() {
    // create execution context
    std::unique_ptr<nvinfer1::IExecutionContext> context(m_trtEngine->createExecutionContext());
    YOLO_CHECK(context != nullptr, "Failed to create TRT execution context");

    // execute inference
    for (size_t i = 0; i < m_inputNames.size(); ++i) {
        // set input tensor address
        context->setTensorAddress(m_inputNames[i], m_inputMem[i]);
    }
    for (size_t i = 0; i < m_outputNames.size(); ++i) {
        // set output tensor address
        context->setTensorAddress(m_outputNames[i], m_outputMem[i]);
    }
    // enqueue the inference
    bool status = context->enqueueV3(m_stream);

    // d2s

    YOLO_CHECK(status, "TRT Inference failed");
    
    return stateType::SUCCESS;
}

stateType trt_pose::postProcess(const Mat* inputImages, std::vector<poseBoxes>& outputBoxes, const int num) {
    auto ret = stateType::SUCCESS;
    switch (m_yoloType) {

        case yoloType::YOLOV8:
            ret = yolov8Post(inputImages, outputBoxes, num);
            break;
        case yoloType::YOLOV11:
            ret = yolov11Post(inputImages, outputBoxes, num);
            break;
        default:
            ret = stateType::UNMATCH_YOLO_TYPE_ERROR;
            break;
    }
    return ret;
}

int trt_pose::get_element_size(const nvinfer1::DataType& type) const {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return sizeof(float);
        case nvinfer1::DataType::kHALF:
            return sizeof(uint16_t);
        case nvinfer1::DataType::kINT8:
            return sizeof(int8_t);
        case nvinfer1::DataType::kINT32:
            return sizeof(int32_t);
        case nvinfer1::DataType::kINT64:
            return sizeof(int64_t);
        case nvinfer1::DataType::kUINT8:
            return sizeof(uint8_t);
        case nvinfer1::DataType::kBOOL:
            return sizeof(bool);
        case nvinfer1::DataType::kBF16:
            return sizeof(uint16_t);
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

stateType trt_pose::resizeBox(const cv::Mat* inputImage, poseBoxes& outputBoxes){

    auto img = *inputImage;
    int img_w = img.cols;
    int img_h = img.rows;

    int dx, dy;
    float scale_x, scale_y;
    getAspectParam(img_w, img_h, m_net_w, m_net_h, dx, dy, scale_x, scale_y, m_resizeType);
    for (auto& box : outputBoxes) {
        box.left = (box.left - dx) / scale_x;
        box.top = (box.top - dy) / scale_y;
        box.right = (box.right - dx) / scale_x;
        box.bottom = (box.bottom - dy) / scale_y;

        // clip
        box.left = std::min(std::max(box.left, 0), img_w);
        box.top = std::min(std::max(box.top, 0), img_h);
        box.right = std::min(std::max(box.right, 0), img_w);
        box.bottom = std::min(std::max(box.bottom, 0), img_h);

        // update w,h
        box.width = box.right - box.left;
        box.height = box.bottom - box.top;

        for(auto& kp : box.keypoints) {
            kp.x = (kp.x - dx) / scale_x;
            kp.y = (kp.y - dy) / scale_y;

            // clip keypoints
            kp.x = std::min(std::max(kp.x, 0), img_w);
            kp.y = std::min(std::max(kp.y, 0), img_h);
        }
    }
    return stateType::SUCCESS;
}

stateType trt_pose::yolov8Post(const Mat* inputImages, std::vector<poseBoxes>& outputBoxes, const int num) {
    for (int i = 0; i < m_outputMem.size(); ++i) {
        cudaMemcpyAsync(m_outputCpuMem[i].get(), m_outputMem[i], m_outputSize[i], cudaMemcpyDeviceToHost, m_stream);
    }
    // 同步数据
    cudaStreamSynchronize(m_stream);
    for(int batch_idx = 0; batch_idx < num; ++batch_idx) {
        poseBoxes yolobox_vec;

        auto output_shape = m_trtEngine->getTensorShape(m_outputNames[0]);
        YOLO_CHECK(output_shape.nbDims == 3, "The {} output's dim must be three. which means to [batch, box_num, feature]",enumName(m_yoloType));
        int box_num = output_shape.d[1];
        
        float *batch_output_data = m_outputCpuMem[0].get() + batch_idx * box_num * m_nout;
        int max_wh = 7680;
        for(int i = 0; i < box_num; ++i) {


            float confidence = batch_output_data[4];
            if (confidence > m_confThreshold) {
                float center_x = batch_output_data[0];
                float center_y = batch_output_data[1];
                float width = batch_output_data[2];
                float height = batch_output_data[3];

                poseBox box;
                box.left = center_x - width / 2;
                box.top = center_y - height / 2;
                box.right = box.left + width;
                box.bottom = box.top + height;
                
                box.score = confidence;

                // keypoints
                auto point_num = (m_nout - 5) / 3; // 5 is for [x, y, w, h, score]
                for (int j = 0; j < point_num; ++j) {
                    keypoint kp;
                    kp.x = static_cast<int>(batch_output_data[5 + j * 3]);
                    kp.y = static_cast<int>(batch_output_data[6 + j * 3]);
                    kp.visibility = batch_output_data[7 + j * 3];
                    box.keypoints.push_back(kp);
                }

                yolobox_vec.push_back(box);
            }
            batch_output_data += m_nout;
        }

        poseBoxes resVec;
        NMS(yolobox_vec, resVec, m_nmsThreshold);
        resizeBox(inputImages+batch_idx, resVec);
        YOLO_DEBUG("batch_id: {}, outputbox number is {}", batch_idx, resVec.size());
        outputBoxes.push_back(resVec);
    }
    return stateType::SUCCESS;
}

stateType trt_pose::yolov11Post(const Mat* inputImages, std::vector<poseBoxes>& outputBoxes, const int num) {
    return yolov8Post(inputImages, outputBoxes, num);
}