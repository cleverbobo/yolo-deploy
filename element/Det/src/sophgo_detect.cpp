#include <cmath>

#include "sophgo_detect.h"

#include "utils.h"

#include "bmcv_api_ext.h"
#include "bmnn_utils.h"




sophgo_detect::sophgo_detect(std::string modelPath, yoloType type, int devId) {
    // init device id;
    m_devId = devId;

    // init handle
    m_handle = std::make_shared<BMNNHandle>(devId);
    auto h = m_handle->handle();

    // init context
    m_bmContext = std::make_shared<BMNNContext>(m_handle, modelPath.c_str());

    // init network
    m_bmNetwork = m_bmContext->network(0);
    
    // init preprocess
    auto tensor = m_bmNetwork->inputTensor(0);
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];

    m_yoloType = type;
    auto yoloConfig = getYOLOConfig(m_yoloType);
    m_mean = yoloConfig.mean;
    m_std = yoloConfig.std;
    m_bgr2rgb = yoloConfig.bgr2rgb;
    m_padValue = yoloConfig.padValue;
    m_anchors = yoloConfig.anchors;
    m_resizeType = yoloConfig.resize_type;


    // init postprocess
    m_max_batch = m_bmNetwork->maxBatch();
    auto output_tensor = m_bmNetwork->outputTensor(0);
    
    m_output_num = m_bmNetwork->outputTensorNum();
    m_output_dim = output_tensor->get_shape()->num_dims;
    m_nout = output_tensor->get_shape()->dims[m_output_dim-1];
    m_class_num = m_nout - 5;

    // init preprocess data
    m_preprocess_images.resize(m_max_batch);
    bm_image_format_ext image_format = m_bgr2rgb ? FORMAT_RGB_PLANAR : FORMAT_BGR_PLANAR;
    bm_image_data_format_ext data_formate = tensor->get_dtype() == BM_UINT8 ? DATA_TYPE_EXT_1N_BYTE : DATA_TYPE_EXT_FLOAT32;
    for (auto& image : m_preprocess_images) {
        auto ret = bm_image_create(h, m_net_h, m_net_w, image_format, data_formate, &image);
    }



}

sophgo_detect::~sophgo_detect() {
    for (auto& image : m_preprocess_images) {
        auto ret = bm_image_destroy(image);
    }
}

std::vector<detectBoxes> sophgo_detect::process(void* inputImage, int num) {
    bm_image* imageData = reinterpret_cast<bm_image*>(inputImage);

    stateType ret = stateType::SUCCESS;

    // preprocess
    ret = preProcess(imageData, num);

    // inference
    ret = inference();

    // postprocess
    std::vector<detectBoxes> outputBoxes;
    ret = postProcess(imageData, outputBoxes, num);

    return outputBoxes;
}

void sophgo_detect::printAlgorithmInfo() {
    std::cout << "Algorithm: " << int(m_yoloType) << std::endl;
    std::cout << "Device ID: " << m_devId << std::endl;
}

stateType sophgo_detect::resetAnchor(std::vector<std::vector<std::vector<int>>> anchors) {
    m_anchors = anchors;
    return stateType::SUCCESS;
}


stateType sophgo_detect::preProcess(bm_image* inputImages, int num){
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

        crop_rects[i] = {0,0,img_w,img_h};
    }

    auto ret = bmcv_image_vpp_convert_padding(handle, num, *inputImages, m_preprocess_images.data(),
                                              padding_attrs.data(), crop_rects.data(),BMCV_INTER_LINEAR);

    // attach to tensor
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(num, m_preprocess_images.data(), &input_dev_mem);
    std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
    input_tensor->set_device_mem(&input_dev_mem);
    return stateType::SUCCESS;
}

stateType sophgo_detect::inference(){
    auto ret = m_bmNetwork->forward();
    return stateType::SUCCESS;

}

stateType sophgo_detect::postProcess(bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, int num){
    auto ret = stateType::SUCCESS;
    switch (m_yoloType) {
        case yoloType::YOLOV5:
            ret = yolov5Post(inputImages, outputBoxes, num);
            break;
        default:
            ret = stateType::ERROR;
            break;
    }
    return ret;
    
}

stateType sophgo_detect::resizeBox(bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, int num){
    for (int i = 0; i < num; i++) {
        bm_image img = *(inputImages + i);
        int img_w = img.width;
        int img_h = img.height;

        int dx, dy;
        float scale_x, scale_y;
        getAspectParam(img_w, img_h, m_net_w, m_net_h, dx, dy, scale_x, scale_y, m_resizeType);
        for (auto& box : outputBoxes[i]) {
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
    }
}

stateType sophgo_detect::yolov5Post(bm_image* inputImages, std::vector<detectBoxes>& outputBoxes, int num){

    std::vector<std::shared_ptr<BMNNTensor>> outputTensors(m_output_num);
    for(int i=0; i<m_output_num; i++){
        outputTensors[i] = m_bmNetwork->outputTensor(i);
    }

    for(int batch_idx = 0; batch_idx < num; ++batch_idx)
    {
      detectBoxes yolobox_vec;
  
      int box_num = 0;
      for(int i=0; i<m_output_num; i++){
        auto output_shape = m_bmNetwork->outputTensor(i)->get_shape();
        auto output_dims = output_shape->num_dims;
        box_num += output_shape->dims[1] * output_shape->dims[2] * output_shape->dims[3];
      }
  
  #if USE_MULTICLASS_NMS
      int out_nout = m_nout;
  #else
      int out_nout = 7;
  #endif
  
      // get transformed confidence threshold   
      float transformed_m_confThreshold = - std::log(1 / m_confThreshold - 1);
  
      // init detect head    
      std::vector<float> decoded_data(box_num*out_nout);
      float *dst = decoded_data.data();

      for(int head_idx = 0; head_idx < m_output_num; head_idx++) {
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
  
      outputBoxes.push_back(resVec);
    }
    return stateType::SUCCESS;
}