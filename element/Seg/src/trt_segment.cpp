#include "trt_segment.h"
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

trt_segment::trt_segment(const std::string& modelPath, const yoloType& type, const int devId)
    : segment(modelPath,type,devId) {
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
    m_output_det_dim = m_trtEngine->getTensorShape(m_outputNames[0]).nbDims;
    m_output_seg_dim = m_trtEngine->getTensorShape(m_outputNames[m_output_num-1]).nbDims;
    m_seg_feature_size = m_trtEngine->getTensorShape(m_outputNames[m_output_num-1]).d[1];

    m_nout = m_trtEngine->getTensorShape(m_outputNames[0]).d[m_output_det_dim - 1];
    if (m_yoloType == yoloType::YOLOV6) {
        m_class_num = m_nout - 5;
    } else {
        m_class_num = m_nout - 5 - m_seg_feature_size;
    }
    
    // 配置 algorithm info
    m_algorithmInfo.yolo_type = m_yoloType;
    m_algorithmInfo.algorithm_type = algorithmType::SEGMENT;
    m_algorithmInfo.device_type = deviceType::TENSORRT;
    m_algorithmInfo.batch_size = m_trtEngine->getTensorShape(m_inputNames[0]).d[0];

    printAlgorithmInfo();
        
}

trt_segment::~trt_segment() {
    for(auto& mem: m_inputMem) {
        cudaFreeAsync(mem, m_stream);
    }

    for(auto& mem: m_outputMem) {
        cudaFreeAsync(mem, m_stream);
    }

    cudaStreamDestroy(m_stream);
}

std::vector<segmentBoxes> trt_segment::process(void* inputImage, const int num) {
    const cv::Mat* imgPtr = static_cast<const cv::Mat*>(inputImage);

    stateType ret = stateType::SUCCESS;
    int calculateTime = (num-1) / m_max_batch + 1;
    std::vector<segmentBoxes> outputBoxes;

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


stateType trt_segment::preProcess(const Mat* inputImages, const int num){
    for (int i = 0; i < num; ++i) {
        cv::Mat img_letterbox = letterbox(inputImages[i], cv::Size(m_net_w, m_net_h), cv::Scalar(m_padValue, m_padValue, m_padValue));
        cv::Mat blob = cv::dnn::blobFromImage(img_letterbox, m_std[0], cv::Size(m_net_w, m_net_h),
                                              cv::Scalar(m_mean[0],m_mean[1],m_mean[2]), true, false);  
        cudaMemcpyAsync(m_inputMem[i], blob.data, m_inputSize[i], cudaMemcpyHostToDevice, m_stream);
    }
    return stateType::SUCCESS;
}

stateType trt_segment::inference() {
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

stateType trt_segment::postProcess(const Mat* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    auto ret = stateType::UNMATCH_YOLO_TYPE_ERROR;
    switch (m_yoloType) {
        case yoloType::YOLOV5:
            ret = yolov5Post(inputImages, outputBoxes, num);
            break;
        case yoloType::YOLOV6:
            ret = yolov6Post(inputImages, outputBoxes, num);
            break;
        case yoloType::YOLOV7:
            ret = yolov7Post(inputImages, outputBoxes, num);
            break;
        case yoloType::YOLOV8:
            ret = yolov8Post(inputImages, outputBoxes, num);
            break;
        case yoloType::YOLOV9:
            ret = yolov9Post(inputImages, outputBoxes, num);
            break;
        case yoloType::YOLOV10:
            YOLO_ERROR("YOLOV10 is not supported in segment yet!");
            break;
        case yoloType::YOLOV11:
            ret = yolov11Post(inputImages, outputBoxes, num);
            break;
        case yoloType::YOLOV12:
            ret = yolov12Post(inputImages, outputBoxes, num);
            break;
        default:
            ret = stateType::UNMATCH_YOLO_TYPE_ERROR;
            break;
    }
    return ret;
}

int trt_segment::get_element_size(const nvinfer1::DataType& type) const {
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

stateType trt_segment::getSegmentBox(const cv::Mat& inputImages, segmentBoxes& outputBoxes, float* proto_data, const nvinfer1::Dims& proto_shape) {
    if (outputBoxes.empty()) {
        YOLO_WARN("outputBoxes is empty in trt_segment::getSegmentBox");
        return stateType::SUCCESS;
    }

    if (outputBoxes[0].mask.empty()) {
        YOLO_WARN("outputBoxes mask is empty in trt_segment::getSegmentBox");
        return stateType::ERROR;
    }

    std::vector<cv::Mat> maskVec(outputBoxes.size());
    for (auto i = 0; i < outputBoxes.size(); ++i) {
        maskVec[i] = cv::Mat(1,m_seg_feature_size, CV_32FC1, outputBoxes[i].mask.data());
    }

    int proto_width = proto_shape.d[3];
    int proto_height = proto_shape.d[2];
    

    std::vector<int> proto_shape_vec(proto_shape.d, proto_shape.d + proto_shape.nbDims);
    cv::Mat proto(proto_shape_vec.size(),proto_shape_vec.data(),  CV_32F, proto_data);

    int img_w = inputImages.cols;
    int img_h = inputImages.rows;

    // preprocess parameters
    int dx, dy;
    float scale_x, scale_y;
    getAspectParam(img_w, img_h, m_net_w, m_net_h, dx, dy, scale_x, scale_y, m_resizeType);

    // resize bbox
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
    
    

    // mask scale
    float mask_scale_x = scale_x * proto_width / m_net_w;
    float mask_scale_y = scale_y * proto_height / m_net_h;

    int mask_dx = static_cast<int>(dx * proto_width / m_net_w);
    int mask_dy = static_cast<int>(dy * proto_height / m_net_h);


    for (auto i = 0; i < outputBoxes.size(); ++i) {
        auto& outputBox = outputBoxes[i];
        auto mask_w = static_cast<int>(img_w * mask_scale_x);
        auto mask_h = static_cast<int>(img_h * mask_scale_y);
        std::vector<cv::Range> mask_range = {
            cv::Range(0, 1),
            cv::Range::all(),
            cv::Range(mask_dy, mask_dy + mask_h),
            cv::Range(mask_dx, mask_dx + mask_w)
        };

        cv::Mat mask_proto = (proto(mask_range).clone()).reshape(0, {m_seg_feature_size, mask_h * mask_w});
        cv::Mat res = maskVec[i] * mask_proto;
        res = res.reshape(1, {mask_h, mask_w});

        // resize
        cv::Mat maskRes;
        cv::resize(res, maskRes, cv::Size(img_w, img_h));

        cv::Rect rect(outputBox.left, outputBox.top, outputBox.width, outputBox.height);
        outputBox.maskImg = std::make_shared<cv::Mat>(maskRes(rect) > 0.5f);

    }
    return stateType::SUCCESS;
}

stateType trt_segment::yolov5Post(const Mat* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num){
    for (int i = 0; i < m_outputMem.size(); ++i) {
        cudaMemcpyAsync(m_outputCpuMem[i].get(), m_outputMem[i], m_outputSize[i], cudaMemcpyDeviceToHost, m_stream);
    }
    // 同步数据
    cudaStreamSynchronize(m_stream);

    auto proto_data = m_outputCpuMem[m_output_num - 1].get();
    auto proto_shape = m_trtEngine->getTensorShape(m_outputNames[m_output_num - 1]);
    int proto_size = 1;
    for (int i = 1; i < proto_shape.nbDims; ++i) {
        proto_size *= proto_shape.d[i];
    }

    for(int batch_idx = 0; batch_idx < num; ++batch_idx)
    {
      segmentBoxes yolobox_vec;
  
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
  
      // init segment head    
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
              for(int d = 6; d < m_nout - m_seg_feature_size; d++){
                  if(output_data_ptr[d] > dst[5]){
                  dst[5] = output_data_ptr[d];
                  dst[6] = d;
                  }
              }
              dst[6] -= 5;
  #endif
              // add seg feature
              for(int d = m_nout - m_seg_feature_size; d < m_nout; ++d) {
                dst[d] = output_data_ptr[d];
              }

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
  
            segmentBox box;
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

            // segmentation mask
            box.mask.resize(m_seg_feature_size);
            std::memcpy(box.mask.data(), output_data_ptr + 7, m_seg_feature_size * sizeof(float));

            yolobox_vec.push_back(box);
        }
  #endif
      }
  
      segmentBoxes resVec;
      NMS(yolobox_vec, resVec, m_nmsThreshold);
      for (auto& box : resVec){
          if (!agnostic){
              box.left -= box.classId * max_wh;
              box.top -= box.classId * max_wh;
              box.right -= box.classId * max_wh;
              box.bottom -= box.classId * max_wh;
          }
      }
      
      getSegmentBox(*(inputImages+batch_idx), resVec, proto_data, proto_shape);
      proto_data += proto_size;

      YOLO_DEBUG("batch_id: {}, outputbox number is {}", batch_idx, resVec.size());
      outputBoxes.push_back(resVec);
    }
    return stateType::SUCCESS;
}

stateType trt_segment::yolov6Post(const Mat* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    for (int i = 0; i < m_outputMem.size(); ++i) {
        cudaMemcpyAsync(m_outputCpuMem[i].get(), m_outputMem[i], m_outputSize[i], cudaMemcpyDeviceToHost, m_stream);
    }
    // 同步数据
    cudaStreamSynchronize(m_stream);

    auto proto_data = m_outputCpuMem[1].get();
    auto proto_shape = m_trtEngine->getTensorShape(m_outputNames[1]);
    int proto_size = 1;
    for (int i = 1; i < proto_shape.nbDims; ++i) {
        proto_size *= proto_shape.d[i];
    }

    auto seg_mask_data = m_outputCpuMem[2].get();

    for(int batch_idx = 0; batch_idx < num; ++batch_idx) {
        segmentBoxes yolobox_vec;

        auto output_shape = m_trtEngine->getTensorShape(m_outputNames[0]);
        YOLO_CHECK(output_shape.nbDims == 3, "The {} output's dim must be three. which means to [batch, box_num, feature]",enumName(m_yoloType));
        int box_num = output_shape.d[1];

        #if USE_MULTICLASS_NMS
            int out_nout = m_nout;
        #else
            int out_nout = 7;
        #endif
        
        float *batch_mask_data = seg_mask_data + batch_idx * box_num * (m_seg_feature_size + 1);
        float *batch_output_data = m_outputCpuMem[0].get() + batch_idx * box_num * m_nout;
        int max_wh = 7680;
        bool agnostic = false;
        for(int i = 0; i < box_num; ++i) {
            float score = batch_output_data[4];
            if (score < m_confThreshold) {
                batch_output_data += m_nout;
                continue;
            }
            #if USE_MULTICLASS_NMS
                for (int j = 0; j < m_class_num; ++j) {
                    float confidence = batch_output_data[j + 5];
                    int class_id = j;
                    if (confidence * score > m_confThreshold) {
                        center_x = batch_output_data[0];
                        center_y = batch_output_data[1];
                        width = batch_output_data[2];
                        height = batch_output_data[3];
                    }
                    segmentBox box;

                    box.left = center_x - width / 2;
                    box.top = center_y - height / 2;
                    box.right = box.left + width;
                    box.bottom = box.top + height;  
                    box.classId = class_id;
                    box.score = confidence * score;

                    if (!agnostic) {
                        box.left += class_id * max_wh;
                        box.top += class_id * max_wh;
                        box.right += class_id * max_wh;
                        box.bottom += class_id * max_wh;
                    }
                    yolobox_vec.push_back(box);
                    
                }
            #else
                int class_id = argmax(batch_output_data + 5, m_class_num);
                float confidence = batch_output_data[5 + class_id];
                if (confidence * score > m_confThreshold) {
                    float center_x = batch_output_data[0];
                    float center_y = batch_output_data[1];
                    float width = batch_output_data[2];
                    float height = batch_output_data[3];

                    segmentBox box;
                    box.left = center_x - width / 2;
                    box.top = center_y - height / 2;
                    box.right = box.left + width;
                    box.bottom = box.top + height;
                    
                    if (!agnostic) {
                        box.left += class_id * max_wh;
                        box.top += class_id * max_wh;
                        box.right += class_id * max_wh;
                        box.bottom += class_id * max_wh;
                    }
                    box.classId = class_id;
                    box.score = confidence * score;

                    // segment mask
                    box.mask.resize(m_seg_feature_size);
                    std::memcpy(box.mask.data(), batch_mask_data + 1, m_seg_feature_size * sizeof(float));

                    yolobox_vec.push_back(box);
                }
                batch_output_data += m_nout;
                batch_mask_data += m_seg_feature_size + 1;
        
            #endif
        }
        segmentBoxes resVec;
        NMS(yolobox_vec, resVec, m_nmsThreshold);
      for (auto& box : resVec){
          if (!agnostic){
              box.left -= box.classId * max_wh;
              box.top -= box.classId * max_wh;
              box.right -= box.classId * max_wh;
              box.bottom -= box.classId * max_wh;
          }
      }
      
      getSegmentBox(*(inputImages+batch_idx), resVec, proto_data, proto_shape);
      proto_data += proto_size;

      YOLO_DEBUG("batch_id: {}, outputbox number is {}", batch_idx, resVec.size());
      outputBoxes.push_back(resVec);
    }
    return stateType::SUCCESS;
}

stateType trt_segment::yolov7Post(const Mat* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    return yolov5Post(inputImages, outputBoxes, num);
}

stateType trt_segment::yolov8Post(const Mat* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    for (int i = 0; i < m_outputMem.size(); ++i) {
        cudaMemcpyAsync(m_outputCpuMem[i].get(), m_outputMem[i], m_outputSize[i], cudaMemcpyDeviceToHost, m_stream);
    }
    // 同步数据
    cudaStreamSynchronize(m_stream);

    auto proto_data = m_outputCpuMem[m_output_num - 1].get();
    auto proto_shape = m_trtEngine->getTensorShape(m_outputNames[m_output_num - 1]);
    int proto_size = 1;
    for (int i = 1; i < proto_shape.nbDims; ++i) {
        proto_size *= proto_shape.d[i];
    }

    for(int batch_idx = 0; batch_idx < num; ++batch_idx) {
        segmentBoxes yolobox_vec;

        auto output_shape = m_trtEngine->getTensorShape(m_outputNames[0]);
        YOLO_CHECK(output_shape.nbDims == 3, "The {} output's dim must be three. which means to [batch, box_num, feature]",enumName(m_yoloType));
        int box_num = output_shape.d[1];

        #if USE_MULTICLASS_NMS
            int out_nout = m_nout;
        #else
            int out_nout = 7;
        #endif
        
        float *batch_output_data = m_outputCpuMem[0].get() + batch_idx * box_num * m_nout;
        int max_wh = 7680;
        bool agnostic = false;
        for(int i = 0; i < box_num; ++i) {
            #if USE_MULTICLASS_NMS
                for (int j = 0; j < m_class_num; ++j) {
                    float confidence = batch_output_data[j + 4];
                    int class_id = j;
                    if (confidence > m_confThreshold) {
                        center_x = batch_output_data[0];
                        center_y = batch_output_data[1];
                        width = batch_output_data[2];
                        height = batch_output_data[3];
                    }
                    segmentBox box;

                    box.left = center_x - width / 2;
                    box.top = center_y - height / 2;
                    box.right = box.left + width;
                    box.bottom = box.top + height;  
                    box.classId = class_id;
                    box.score = confidence;

                    if (!agnostic) {
                        box.left += class_id * max_wh;
                        box.top += class_id * max_wh;
                        box.right += class_id * max_wh;
                        box.bottom += class_id * max_wh;
                    }
                    yolobox_vec.push_back(box);
                    
                }
            #else
                int class_id = argmax(batch_output_data + 4, m_class_num);
                float confidence = batch_output_data[4 + class_id];
                if (confidence > m_confThreshold) {
                    float center_x = batch_output_data[0];
                    float center_y = batch_output_data[1];
                    float width = batch_output_data[2];
                    float height = batch_output_data[3];

                    segmentBox box;
                    box.left = center_x - width / 2;
                    box.top = center_y - height / 2;
                    box.right = box.left + width;
                    box.bottom = box.top + height;
                    
                    if (!agnostic) {
                        box.left += class_id * max_wh;
                        box.top += class_id * max_wh;
                        box.right += class_id * max_wh;
                        box.bottom += class_id * max_wh;
                    }
                    box.classId = class_id;
                    box.score = confidence;

                    // segmentation mask
                    box.mask.resize(m_seg_feature_size);
                    std::memcpy(box.mask.data(), batch_output_data + m_nout - m_seg_feature_size, m_seg_feature_size * sizeof(float));

                    yolobox_vec.push_back(box);
                }
                batch_output_data += m_nout;
        
            #endif
        }

        segmentBoxes resVec;
        NMS(yolobox_vec, resVec, m_nmsThreshold);
        for (auto& box : resVec){
            if (!agnostic){
                box.left -= box.classId * max_wh;
                box.top -= box.classId * max_wh;
                box.right -= box.classId * max_wh;
                box.bottom -= box.classId * max_wh;
            }
        }
        getSegmentBox(*(inputImages+batch_idx), resVec, proto_data, proto_shape);
        proto_data += proto_size;

        YOLO_DEBUG("batch_id: {}, outputbox number is {}", batch_idx, resVec.size());
        outputBoxes.push_back(resVec);
    }
    return stateType::SUCCESS;
}

stateType trt_segment::yolov9Post(const Mat* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    return yolov8Post(inputImages, outputBoxes, num);
}

stateType trt_segment::yolov11Post(const Mat* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    return yolov8Post(inputImages, outputBoxes, num);
}

stateType trt_segment::yolov12Post(const Mat* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    return yolov8Post(inputImages, outputBoxes, num);
}