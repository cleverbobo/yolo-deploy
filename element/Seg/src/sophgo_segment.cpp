#include "sophgo_segment.h"

#include <cmath>
#include <opencv2/opencv.hpp>
#include "utils.h"

#include "bmcv_api_ext.h"
#include "bmnn_utils.h"


sophgo_segment::sophgo_segment(const std::string& modelPath, const yoloType& type, const int devId)
    : segment(modelPath, type, devId) {
    
    // init handle
    m_handle = std::make_shared<BMNNHandle>(m_devId);
    auto h = m_handle->handle();

    // init context
    m_bmContext = std::make_shared<BMNNContext>(m_handle, modelPath.c_str());

    // init network
    m_bmNetwork = m_bmContext->network(0);

    // init preprocess
    auto tensor = m_bmNetwork->inputTensor(0);
    // fuse preprocess(normalize) with input tensor
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];

    m_max_batch = m_bmNetwork->maxBatch();

    m_output_num = m_bmNetwork->outputTensorNum();
    auto det_output_tensor = m_bmNetwork->outputTensor(0);
    m_output_det_dim = det_output_tensor->get_shape()->num_dims;
    m_nout = det_output_tensor->get_shape()->dims[m_output_det_dim-1];

    std::shared_ptr<BMNNTensor> seg_output_tensor;
    if (m_yoloType != yoloType::YOLOV6) {
        seg_output_tensor = m_bmNetwork->outputTensor(m_output_num - 1);
    } else {
        seg_output_tensor = m_bmNetwork->outputTensor(m_output_num - 2);
    }
    m_seg_feature_size = seg_output_tensor->get_shape()->dims[1];
    m_class_num = m_nout - 5 - m_seg_feature_size; 

    // init preprocess data
    m_preprocess_images.resize(m_max_batch);
    bm_image_format_ext image_format = m_bgr2rgb ? FORMAT_RGB_PLANAR : FORMAT_BGR_PLANAR;
    bm_image_data_format_ext data_formate = tensor->get_dtype() == BM_UINT8 ? DATA_TYPE_EXT_1N_BYTE : DATA_TYPE_EXT_FLOAT32;
    for (auto& image : m_preprocess_images) {
        auto ret = bm_image_create(h, m_net_h, m_net_w, image_format, data_formate, &image);
    }
    auto ret = bm_image_alloc_contiguous_mem(m_max_batch, m_preprocess_images.data());
    YOLO_CHECK(ret == BM_SUCCESS, "bm_image_alloc_contiguous_mem failed in sophgo_segment::sophgo_segment!");

    // init m_algorithmInfo
    std::vector<std::vector<int>> input_shape;
    for (int i = 0; i < m_bmNetwork -> inputTensorNum(); ++i) {
        auto input_tensor = m_bmNetwork->inputTensor(i);
        input_shape.push_back(input_tensor->get_shape_vector());
    }
    std::vector<std::vector<int>> output_shape;
    for (int i = 0; i < m_bmNetwork -> outputTensorNum(); ++i) {
        auto output_tensor = m_bmNetwork->outputTensor(i);
        output_shape.push_back(output_tensor->get_shape_vector());
    }
    m_algorithmInfo = algorithmInfo{ m_yoloType,
                                     algorithmType::SEGMENT,
                                     deviceType::SOPHGO,
                                     input_shape,
                                     output_shape,
                                     m_max_batch };
    printAlgorithmInfo();
}

sophgo_segment::~sophgo_segment() {
    bm_image_free_contiguous_mem(m_max_batch, m_preprocess_images.data());
    for (auto& image : m_preprocess_images) {
        auto ret = bm_image_destroy(image);
    }
}

std::vector<segmentBoxes> sophgo_segment::process(void* inputImage, const int num) {
    bm_image* imageData = reinterpret_cast<bm_image*>(inputImage);

    stateType ret = stateType::SUCCESS;
    int calculateTime = (num-1) / m_max_batch + 1;
    std::vector<segmentBoxes> outputBoxes;

    for (int i = 0; i < calculateTime; ++i) {
        int inputNum = std::min(num - i * m_max_batch, m_max_batch);

        // preprocess
        ret = preProcess(imageData , inputNum);
        YOLO_CHECK(ret == stateType::SUCCESS, "preProcess failed in sophgo_segment::process")

        // inference
        ret = inference();
        YOLO_CHECK(ret == stateType::SUCCESS, "inference failed in sophgo_segment::process")

        // postprocess
        ret = postProcess(imageData, outputBoxes, inputNum);
        YOLO_CHECK(ret == stateType::SUCCESS, "postProcess failed inf sophgo_segment::process")

        m_fpsCounter.add(inputNum);

        imageData += m_max_batch * i;

    }

    return outputBoxes;
}

stateType sophgo_segment::preProcess(bm_image* inputImages, const int num){
    auto handle = m_handle->handle();

    std::vector<bmcv_padding_atrr_t> padding_attrs(num);
    std::vector<bmcv_rect_t> crop_rects(num);

    for(int i = 0; i < num; ++i){
        bm_image img = *(inputImages+i);
        auto img_w = img.width;
        auto img_h = img.height;
        int dx,dy;
        float scale_x, scale_y;
        getAspectParam(img_w, img_h, m_net_w, m_net_h, dx, dy, scale_x, scale_y, m_resizeType);
        padding_attrs[i].dst_crop_stx = dx;
        padding_attrs[i].dst_crop_sty = dy;
        padding_attrs[i].dst_crop_w = img.width * scale_x;
        padding_attrs[i].dst_crop_h = img.height * scale_y;
        padding_attrs[i].padding_b = m_padValue;
        padding_attrs[i].padding_g = m_padValue;
        padding_attrs[i].padding_r = m_padValue;
        padding_attrs[i].if_memset = true;

        crop_rects[i] = {0,0,img_w,img_h};

    }
    auto ret = bmcv_image_vpp_basic(handle, num, inputImages, m_preprocess_images.data(),
                                    NULL, NULL, padding_attrs.data(), BMCV_INTER_LINEAR);
    YOLO_CHECK(ret == BM_SUCCESS, "bmcv_image_vpp_basic failed in sophgo_segment::preProcess");

    // attach to tensor
    bm_device_mem_t input_dev_mem;
    auto num_ = num != m_max_batch ? m_max_batch : num;
    bm_image_get_contiguous_device_mem(num_, m_preprocess_images.data(), &input_dev_mem);
    std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
    input_tensor->set_device_mem(&input_dev_mem);
    return stateType::SUCCESS;
}

stateType sophgo_segment::inference(){
    auto ret = m_bmNetwork->forward();
    return ret == BM_SUCCESS ? stateType::SUCCESS : stateType::INFERENCE_ERROR;

}

stateType sophgo_segment::postProcess(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num){
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

stateType sophgo_segment::getSegmentBox(const bm_image* inputImages, segmentBoxes& outputBoxes, float* proto_data, const bm_shape_t* proto_shape) {
    if (outputBoxes.empty()) {
        YOLO_WARN("outputBoxes is empty in sophgo_segment::getSegmentBox");
        return stateType::SUCCESS;
    }

    if (outputBoxes[0].mask.empty()) {
        YOLO_WARN("outputBoxes mask is empty in sophgo_segment::getSegmentBox");
        return stateType::ERROR;
    }

    std::vector<cv::Mat> maskVec(outputBoxes.size());
    for (auto i = 0; i < outputBoxes.size(); ++i) {
        maskVec[i] = cv::Mat(1,m_seg_feature_size, CV_32FC1, outputBoxes[i].mask.data());
    }
    

    int proto_height = proto_shape->dims[2];
    int proto_width = proto_shape->dims[3];

    cv::Mat proto(proto_shape->num_dims, proto_shape->dims,  CV_32F, proto_data);

    int img_w = inputImages->width;
    int img_h = inputImages->height;

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


stateType sophgo_segment::yolov5Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    std::vector<std::shared_ptr<BMNNTensor>> outputTensors(m_output_num);
    for(int i=0; i<m_output_num; i++){
        outputTensors[i] = m_bmNetwork->outputTensor(i);
    }

    auto proto_shape = outputTensors[m_output_num - 1]->get_shape();
    auto proto_data = reinterpret_cast<float*>(outputTensors[m_output_num - 1]->get_cpu_data());
    int proto_size = bmrt_shape_count(proto_shape) / proto_shape->dims[0];
    
    for(int batch_idx = 0; batch_idx < num; ++batch_idx)
    {
      segmentBoxes yolobox_vec;
  
      int box_num = 0;
      for(int i=0; i<m_output_num - 1; i++){
        auto output_shape = m_bmNetwork->outputTensor(i)->get_shape();
        auto output_dims = output_shape->num_dims;
        YOLO_CHECK(output_dims == 5, "The {} output's dim must be five. which means to [batch, anchor_num, feature_height,feature_width,feature]",enumName(m_yoloType));
        box_num += output_shape->dims[1] * output_shape->dims[2] * output_shape->dims[3];
      }
  
  #if USE_MULTICLASS_NMS
      int out_nout = m_nout;
  #else
      int out_nout = 7 + m_seg_feature_size;
  #endif
  
      // get transformed confidence threshold   
      float transformed_m_confThreshold = - std::log(1 / m_confThreshold - 1);
  
      // init segment head    
      std::vector<float> decoded_data(box_num*out_nout);
      float *dst = decoded_data.data();

      for(int head_idx = 0; head_idx < m_output_num - 1; head_idx++) {
          auto output_tensor = m_bmNetwork->outputTensor(head_idx);
          int feat_c = output_tensor->get_shape()->dims[1];
          int feat_h = output_tensor->get_shape()->dims[2];
          int feat_w = output_tensor->get_shape()->dims[3];
          int area = feat_h * feat_w;
          int feature_size = feat_h * feat_w * m_nout;
          float *tensor_data = reinterpret_cast<float*>(output_tensor->get_cpu_data()) + batch_idx*feat_c*area*m_nout;
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
              for(int d = 5; d < m_nout - m_seg_feature_size; d++)
                  dst[d] = output_data_ptr[d];
              int idx = m_nout - m_seg_feature_size;
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
              int idx = 7;
  #endif
              // add seg feature
              
              for(int d = m_nout - m_seg_feature_size; d < m_nout; d++, idx++){
                dst[idx] = output_data_ptr[d];
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
        float* output_data_ptr = output_data + i * out_nout;
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
                    // update box position
                    box.left +=  class_id * max_wh;
                    box.top += class_id * max_wh;
                    box.right += class_id * max_wh;
                    box.bottom += class_id * max_wh;
                }

                // segment mask
                box.mask.resize(m_seg_feature_size);
                std::memcpy(box.mask.data(), output_data_ptr + m_nout - m_seg_feature_size, m_seg_feature_size * sizeof(float));
  
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

            // segment mask
            box.mask.resize(m_seg_feature_size);
            std::memcpy(box.mask.data(), output_data_ptr + 7, m_seg_feature_size * sizeof(float));

            yolobox_vec.push_back(box);
        }
  #endif
      }
  
      segmentBoxes resVec;
      NMS<segmentBoxes>(yolobox_vec, resVec, m_nmsThreshold);
      for (auto& box : resVec){
          if (!agnostic){
              box.left -= box.classId * max_wh;
              box.top -= box.classId * max_wh;
              box.right -= box.classId * max_wh;
              box.bottom -= box.classId * max_wh;
          }
      }

      
      getSegmentBox(inputImages+batch_idx, resVec, proto_data, proto_shape);
      proto_data += proto_size;

      YOLO_DEBUG("batch_id: {}, outputbox number is {}", batch_idx, resVec.size());
      outputBoxes.push_back(resVec);
    }
    return stateType::SUCCESS;

}

stateType sophgo_segment::yolov6Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    std::shared_ptr<BMNNTensor> output_tensor = m_bmNetwork->outputTensor(0);
    float* output_data = reinterpret_cast<float*>(output_tensor->get_cpu_data());

    auto seg_proto_tensor = m_bmNetwork->outputTensor(1);
    auto proto_shape = seg_proto_tensor->get_shape();
    auto proto_data = reinterpret_cast<float*>(seg_proto_tensor->get_cpu_data());
    int proto_size = bmrt_shape_count(proto_shape) / proto_shape->dims[0];

    auto seg_mask_tensor = m_bmNetwork->outputTensor(2);
    float* seg_mask_data = reinterpret_cast<float*>(seg_mask_tensor->get_cpu_data());

    for(int batch_idx = 0; batch_idx < num; ++batch_idx) {
        segmentBoxes yolobox_vec;

        auto output_shape = output_tensor->get_shape();
        YOLO_CHECK(output_shape->num_dims == 3, "The {} output's dim must be three. which means to [batch, box_num, feature]",enumName(m_yoloType));
        int box_num = output_tensor->get_shape()->dims[1];
        // m_class_num = output_tensor->get_shape()->dims[2] - 5;

        
        float *batch_output_data = output_data + batch_idx * box_num * m_nout;
        float *batch_mask_data = seg_mask_data + batch_idx * box_num * (m_seg_feature_size + 1);
        int max_wh = 7680;
        bool agnostic = false;
        for(int i = 0; i < box_num; ++i) {
            float score = batch_output_data[4];
            if (score < m_confThreshold) {
                batch_output_data += m_nout;
                batch_mask_data += m_seg_feature_size + 1;  
                continue;
            }
            #if USE_MULTICLASS_NMS
                for (int j = 0; j < m_class_num; ++j) {
                    float confidence = batch_output_data[j + 5];
                    int class_id = j;
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
            #endif
            batch_output_data += m_nout;
            batch_mask_data += m_seg_feature_size + 1;  
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

        getSegmentBox(inputImages+batch_idx, resVec, proto_data, proto_shape);
        proto_data += proto_size;
        YOLO_DEBUG("batch_id: {}, outputbox number is {}", batch_idx, resVec.size());
        outputBoxes.push_back(resVec);
    }
    return stateType::SUCCESS;
}

stateType sophgo_segment::yolov7Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    return yolov5Post(inputImages, outputBoxes, num);
}

stateType sophgo_segment::yolov8Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    auto det_output_tensor = m_bmNetwork->outputTensor(0);
    auto seg_proto_tensor = m_bmNetwork->outputTensor(1);

    auto proto_shape = seg_proto_tensor->get_shape();
    auto proto_data = reinterpret_cast<float*>(seg_proto_tensor->get_cpu_data());
    int proto_size = bmrt_shape_count(proto_shape) / proto_shape->dims[0];

    float* det_output_data = reinterpret_cast<float*>(det_output_tensor->get_cpu_data());

    for(int batch_idx = 0; batch_idx < num; ++batch_idx) {
        segmentBoxes yolobox_vec;

        auto output_shape = det_output_tensor->get_shape();
        YOLO_CHECK(output_shape->num_dims == 3, "The {} output's dim must be three. which means to [batch, box_num, feature]",enumName(m_yoloType));
        int box_num = det_output_tensor->get_shape()->dims[1];
        
        float *batch_output_data = det_output_data + batch_idx * box_num * m_nout;
        int max_wh = 7680;
        bool agnostic = false;
        for(int i = 0; i < box_num; ++i) {
            #if USE_MULTICLASS_NMS
                for (int j = 0; j < m_class_num; ++j) {
                    float confidence = batch_output_data[j + 4];
                    int class_id = j;
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

                        // segment mask
                        box.mask.resize(m_seg_feature_size);
                        std::memcpy(box.mask.data(), batch_output_data + m_nout - m_seg_feature_size, m_seg_feature_size * sizeof(float));

                        yolobox_vec.push_back(box);
                    }
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

                    // segment mask
                    box.mask.resize(m_seg_feature_size);
                    std::memcpy(box.mask.data(), batch_output_data + m_nout - m_seg_feature_size, m_seg_feature_size * sizeof(float));

                    yolobox_vec.push_back(box);
                }
                
        
            #endif
            batch_output_data += m_nout;
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

        getSegmentBox(inputImages+batch_idx, resVec, proto_data, proto_shape);
        proto_data += proto_size;

        YOLO_DEBUG("batch_id: {}, outputbox number is {}", batch_idx, resVec.size());
        outputBoxes.push_back(resVec);
    }
    return stateType::SUCCESS;
}

stateType sophgo_segment::yolov9Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    return yolov8Post(inputImages, outputBoxes, num);
}

stateType sophgo_segment::yolov11Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    return yolov8Post(inputImages, outputBoxes, num);
}

stateType sophgo_segment::yolov12Post(const bm_image* inputImages, std::vector<segmentBoxes>& outputBoxes, const int num) {
    return yolov8Post(inputImages, outputBoxes, num);
}
