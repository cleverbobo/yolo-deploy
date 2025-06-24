#include "trt_detect.h"
#include "utils.h"

// debug
#ifdef YOLO_DEBUG
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

trt_detect::trt_detect(const std::string& modelPath, const yoloType& type, const int devId)
    : detect(modelPath,type,devId) {
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

    // 申请输出的物理内存


    // 配置 preprocess config
    // auto yoloConfig = getYOLOConfig(m_yoloType);
    // m_mean = yoloConfig.mean;
    // m_std = yoloConfig.std;
    // m_bgr2rgb = yoloConfig.bgr2rgb;
    // m_padValue = yoloConfig.padValue;
    // m_anchors = yoloConfig.anchors;
    // m_resizeType = yoloConfig.resize_type;

    // 配置network config
    m_max_batch = m_trtEngine->getTensorShape(m_inputNames[0]).d[0];
    m_net_h = m_trtEngine->getTensorShape(m_inputNames[0]).d[2];
    m_net_w = m_trtEngine->getTensorShape(m_inputNames[0]).d[3];
    
    // postprocess config
    m_output_num = m_outputNames.size();
    m_output_dim = m_trtEngine->getTensorShape(m_outputNames[0]).nbDims;
    m_nout = m_trtEngine->getTensorShape(m_outputNames[0]).d[m_output_dim - 1];
    m_class_num = m_nout - 5;

    
    // 配置 algorithm info
    m_algorithmInfo.yolo_type = m_yoloType;
    m_algorithmInfo.algorithm_type = algorithmType::DETECT;
    m_algorithmInfo.device_type = deviceType::TENSORRT;
    m_algorithmInfo.batch_size = m_trtEngine->getTensorShape(m_inputNames[0]).d[0];

    printAlgorithmInfo();
        
}

trt_detect::~trt_detect() {
    for(auto& mem: m_inputMem) {
        cudaFreeAsync(mem, m_stream);
    }

    for(auto& mem: m_outputMem) {
        cudaFreeAsync(mem, m_stream);
    }

    cudaStreamDestroy(m_stream);
}

std::vector<detectBoxes> trt_detect::process(void* inputImage, const int num) {
    const cv::Mat* imgPtr = static_cast<const cv::Mat*>(inputImage);

    stateType ret = stateType::SUCCESS;
    int calculateTime = (num-1) / m_max_batch + 1;
    std::vector<detectBoxes> outputBoxes;

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
    }
    return outputBoxes;
}

// algorithmInfo trt_detect::getAlgorithmInfo(){
//     return m_algorithmInfo;
// }

// void trt_detect::printAlgorithmInfo() {
//     std::cout << "----------------AlgorithmInfo-----------------" << std::endl;
//     std::cout << "YOLO Type: " << enumName(m_yoloType) << std::endl;
//     std::cout << "Device Type: " << enumName(m_deviceType) << std::endl;
//     std::cout << "Algorithm Type: " << enumName(m_algorithmInfo.algorithm_type) << std::endl;
//     std::cout << "Input Shape: ";
//     for (const auto& shape : m_algorithmInfo.input_shape) {
//         std::cout << "[";
//         for (const auto& dim : shape) {
//             std::cout << dim << " ";
//         }
//         std::cout << "] ";
//     }
//     std::cout << std::endl;

//     std::cout << "Output Shape: ";
//     for (const auto& shape : m_algorithmInfo.output_shape) {
//         std::cout << "[";        
//         for (const auto& dim : shape) {
//             std::cout << dim << " ";
//         }
//         std::cout << "] ";
//     }
//     std::cout << std::endl;
//     std::cout << "Batch Size: " << m_algorithmInfo.batch_size << std::endl;
//     std::cout << "----------------AlgorithmInfo-----------------" << std::endl;
// }

// stateType trt_detect::resetAnchor(std::vector<std::vector<std::vector<int>>> anchors) {
//     m_anchors = anchors;
//     return stateType::SUCCESS;
// }



stateType trt_detect::preProcess(const Mat* inputImages, const int num){
    for (int i = 0; i < num; ++i) {
        cv::Mat img_letterbox = letterbox(inputImages[i], cv::Size(m_net_w, m_net_h), cv::Scalar(m_padValue, m_padValue, m_padValue));
        cv::Mat blob = cv::dnn::blobFromImage(img_letterbox, m_std[0], cv::Size(m_net_w, m_net_h),
                                              cv::Scalar(m_mean[0],m_mean[1],m_mean[2]), true, false);  
        cudaMemcpyAsync(m_inputMem[i], blob.data, m_inputSize[i], cudaMemcpyHostToDevice, m_stream);
    }
    return stateType::SUCCESS;
}

stateType trt_detect::inference() {
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

stateType trt_detect::postProcess(const Mat* inputImages, std::vector<detectBoxes>& outputBoxes, const int num) {
    auto ret = stateType::SUCCESS;
    switch (m_yoloType) {
        case yoloType::YOLOV5:
            ret = yolov5Post(inputImages, outputBoxes, num);
            break;
        default:
            ret = stateType::UNMATCH_YOLO_TYPE_ERROR;
            break;
    }
    return ret;
}

int trt_detect::get_element_size(const nvinfer1::DataType& type) const {
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

stateType trt_detect::resizeBox(const cv::Mat* inputImage, detectBoxes& outputBoxes){

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
    }
    return stateType::SUCCESS;
}

stateType trt_detect::yolov5Post(const Mat* inputImages, std::vector<detectBoxes>& outputBoxes, const int num){
    for (int i = 0; i < m_outputMem.size(); ++i) {
        cudaMemcpyAsync(m_outputCpuMem[i].get(), m_outputMem[i], m_outputSize[i], cudaMemcpyDeviceToHost, m_stream);
    }
    // 同步数据
    cudaStreamSynchronize(m_stream);

    for(int batch_idx = 0; batch_idx < num; ++batch_idx)
    {
      detectBoxes yolobox_vec;
  
      int box_num = 0;
      for(auto outName : m_outputNames){
        auto output_shape = m_trtEngine->getTensorShape(outName);
        auto output_dims = output_shape.nbDims;
        YOLO_CHECK(output_dims == 5, "The Yolov5 output's dim must be five. which means to [batch, anchor_num, feature_height,feature_width,feature]")
        box_num += output_shape.d[1] * output_shape.d[2] * output_shape.d[3];
      }
  
  #if USE_MULTICLASS_NMS
      int out_nout = m_nout;
  #else
      int out_nout = 7;
  #endif
  
      // get transformed confidence threshold   
      float transformed_m_confThreshold = - std::log(1 / m_confThreshold - 1);
  
      // init detect head    
      std::vector<float> decoded_data(box_num*out_nout, -1);
      float *dst = decoded_data.data();

      for(int head_idx = 0; head_idx < m_output_num; head_idx++) {
          auto output_shape = m_trtEngine->getTensorShape(m_outputNames[head_idx]);
          int feat_c = output_shape.d[1];
          int feat_h = output_shape.d[2];
          int feat_w = output_shape.d[3];
          int area = feat_h * feat_w;
          int feature_size = feat_h * feat_w * m_nout;
          float *tensor_data = m_outputCpuMem[head_idx].get() + batch_idx*feat_c*area*m_nout;
          for (int anchor_idx = 0; anchor_idx < m_anchors[0].size(); anchor_idx++)
          {
              float *output_data_ptr = tensor_data + anchor_idx*feature_size;
              for (int i = 0; i < area; i++) {
                // confidence too low
                if(output_data_ptr[4] <= transformed_m_confThreshold){
                    output_data_ptr += m_nout;
                    continue;
                }

                // decode box
                dst[0] = (sigmoid(output_data_ptr[0]) * 2 - 0.5 + i % feat_w) / feat_w * m_net_w;
                dst[1] = (sigmoid(output_data_ptr[1]) * 2 - 0.5 + i / feat_w) / feat_h * m_net_h;
                dst[2] = pow((sigmoid(output_data_ptr[2]) * 2), 2) * m_anchors[head_idx][anchor_idx][0];
                dst[3] = pow((sigmoid(output_data_ptr[3]) * 2), 2) * m_anchors[head_idx][anchor_idx][1];
                dst[4] = sigmoid(output_data_ptr[4]);
  #if USE_MULTICLASS_NMS
              for(int d = 5; d < nout; d++)
                  dst[d] = output_data_ptr[d];
  #else
              dst[5] = output_data_ptr[5];
              dst[6] = 5;
              for(int d = 6; d < m_nout; d++){
                  if(output_data_ptr[d] > dst[5]){
                  dst[5] = output_data_ptr[d];
                  dst[6] = d;
                  }
              }
              dst[6] -= 5;
  #endif
              dst += out_nout;
              output_data_ptr += m_nout;
              }
          }
      }
      
      
      float* output_data = decoded_data.data();
      box_num = (dst - output_data) / out_nout;
  
  
      int max_wh = 7680;
      bool agnostic = false;
  
      for (int i = 0; i < box_num; i++) {
        float* output_data_ptr = output_data+i*out_nout;
        float score = output_data_ptr[4];
        float box_transformed_m_confThreshold = - std::log(score / m_confThreshold - 1);
  #if USE_MULTICLASS_NMS
        float centerX = output_data_ptr[0];
        float centerY = output_data_ptr[1];
        float width = output_data_ptr[2];
        float height = output_data_ptr[3];
        for (int j = 0; j < m_class_num; j++) {
          float confidence = output_data_ptr[5 + j];
          int class_id = j;
          if (confidence > box_transformed_m_confThreshold)
          {
              YoloV5Box box;
  
              box.x = std::max(centerX - width / 2 + class_id * max_wh,0.0f);
              box.y = std::max(centerY - height / 2 + class_id * max_wh,0.0f);
              box.width = width;
              box.height = height;
              box.class_id = class_id;
              box.score = sigmoid(confidence) * score;
  
              yolobox_vec.push_back(box);
          }
        }
  #else
        int class_id = output_data_ptr[6];
        float confidence = output_data_ptr[5];

        if (confidence > box_transformed_m_confThreshold)
        {
            float centerX = output_data_ptr[0];
            float centerY = output_data_ptr[1];
            float width = output_data_ptr[2];
            float height = output_data_ptr[3];
  
            detectBox box;
            box.left = centerX - width / 2;
            box.top = centerY - height / 2;
            box.right = box.left + width;
            box.bottom = box.top + height;

            // clip
            box.left = std::min(std::max(box.left,0), m_net_w);
            box.top = std::min(std::max(box.top,0), m_net_h);
            box.right = std::min(std::max(box.right,0), m_net_w);
            box.bottom = std::min(std::max(box.bottom,0), m_net_h);

            // update w,h
            box.width = box.right - box.left;
            box.height = box.bottom - box.top;

            // update class id and score
            box.classId = class_id;
            box.score = sigmoid(confidence) * score;

            if (!agnostic){
                box.left +=  class_id * max_wh;
                box.top += class_id * max_wh;
                box.right += class_id * max_wh;
                box.bottom += class_id * max_wh;
            } 

            yolobox_vec.push_back(box);
        }
  #endif
      }
  
      detectBoxes resVec;
      NMS(yolobox_vec, resVec, m_nmsThreshold);
      for (auto& box : resVec){
          if (!agnostic){
              box.left -= box.classId * max_wh;
              box.top -= box.classId * max_wh;
              box.right -= box.classId * max_wh;
              box.bottom -= box.classId * max_wh;
          }
      }
      resizeBox(inputImages+batch_idx, resVec);
      YOLO_DEBUG("batch_id: {}, outputbox number is {}", batch_idx, resVec.size());
      outputBoxes.push_back(resVec);
    }
    return stateType::SUCCESS;
}