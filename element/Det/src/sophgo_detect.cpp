#include <cmath>

#include "sophgo_detect.h"

#include "utils.h"

#include "bmcv_api_ext.h"
#include "bmnn_utils.h"




sophgo_detect::sophgo_detect(const std::string& modelPath, const yoloType& type, const int devId):detect(modelPath, type, devId):
    m_thead_preprocess(ThreadPool(2,10,10)),
    m_thead_inference(ThreadPool<BMNNTensor>(1,10,10)),
    m_thead_postprocess(ThreadPool<detectBoxes>(2,10,10)) {

    // init handle
    m_handle = std::make_shared<BMNNHandle>(m_devId);
    auto h = m_handle->handle();

    // init context
    m_bmContext = std::make_shared<BMNNContext>(m_handle, modelPath.c_str());

    // init network
    m_bmNetwork = m_bmContext->network(0);
    
    // init network config
    auto input_shapes = m_bmNetwork->input_shapes();
    m_net_h = input_shapes[0][2];
    m_net_w = input_shapes[0][3];
    m_max_batch = m_bmNetwork->maxBatch();
    auto output_shapes = m_bmNetwork->output_shapes();
    
    m_output_num = m_bmNetwork->outputTensorNum();
    m_output_dim = output_shapes[0].size();
    m_nout = output_shapes[0][m_output_dim-1];
    m_class_num = m_nout - 5;

    // init m_algorithmInfo
    m_algorithmInfo = algorithmInfo{ m_yoloType,
                                     algorithmType::DETECT,
                                     deviceType::SOPHGO,
                                     input_shapes,
                                     output_shapes,
                                     m_max_batch };

    // init threadpool
    
}

sophgo_detect::~sophgo_detect() {
    m_thead_preprocess.shutdown();
    m_thead_inference.shutdown();
    m_thead_postprocess.shutdown();
}

std::vector<detectBoxes> sophgo_detect::process(void* inputImage, const int num) {
    bm_image* imageData = reinterpret_cast<bm_image*>(inputImage);

    stateType ret = stateType::SUCCESS;
    int calculateTime = (num-1) / m_max_batch + 1;
    std::vector<detectBoxes> outputBoxes;

    for (int i = 0; i < calculateTime; ++i) {
        int inputNum = std::min(num - i * m_max_batch, m_max_batch);

        

        // preprocess
        ret = preProcess(imageData + m_max_batch*i, inputNum);
        YOLO_CHECK(ret == stateType::SUCCESS, "preProcess failed in sophgo_detect::process")

        // inference
        ret = inference();
        YOLO_CHECK(ret == stateType::SUCCESS, "inference failed in sophgo_detect::process")

        // postprocess
        ret = postProcess(imageData + m_max_batch*i, outputBoxes, inputNum);
        YOLO_CHECK(ret == stateType::SUCCESS, "postProcess failed inf sophgo_detect::process")

        m_fpsCounter.add(inputNum);

    }

    return outputBoxes;
}




stateType sophgo_detect::preProcess(bm_image* inputImages, bm_image* outputImages, const int num) {
    auto handle = m_handle->handle();

    std::vector<bmcv_padding_atrr_t> padding_attrs(num);
    std::vector<bmcv_rect_t> crop_rects(num);

    std::vector<bm_image> outputImagesVec(num);
    bm_image_format_ext image_format = m_bgr2rgb ? FORMAT_RGB_PLANAR : FORMAT_BGR_PLANAR;
    bm_image_data_format_ext data_formate = m_bmNetwork->input_dtypes()[0] == BM_UINT8 ? DATA_TYPE_EXT_1N_BYTE : DATA_TYPE_EXT_FLOAT32;
    for (auto& image : outputImagesVec) {
        auto ret = bm_image_create(handle, m_net_h, m_net_w, image_format, data_formate, &image);
    }

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
    auto ret = bmcv_image_vpp_basic(handle, num, inputImages, outputImages,
                                    NULL, NULL, padding_attrs.data(), BMCV_INTER_LINEAR);
    YOLO_CHECK(ret == BM_SUCCESS, "bmcv_image_vpp_basic failed in sophgo_detect::preProcess");

    // attach to tensor
    // bm_device_mem_t input_dev_mem;
    // auto num_ = num != m_max_batch ? m_max_batch : num;
    // bm_image_get_contiguous_device_mem(num_, m_preprocess_images.data(), &input_dev_mem);
    // outputTensors.set_device_mem(&input_dev_mem);
    return stateType::SUCCESS;
}

stateType sophgo_detect::inference(bm_image* inputImage, std::vector<BMNNTensor>& outputTensors, int num) {
    BMNNTensor inputTensor;
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(num, inputImage, &input_dev_mem);
    inputTensor.set_device_mem(&input_dev_mem);

    outputTensors = m_bmNetwork->forward(&inputTensor, num);

    for (int i = 0; i < m_output_num; ++i) {
        bm_image_destroy(*inputImage);
    }
    return stateType::SUCCESS;

}

stateType sophgo_detect::postProcess(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num){
    auto ret = stateType::SUCCESS;
    switch (m_yoloType) {
        case yoloType::YOLOV5:
            ret = yolov5Post(inputImages, outputTensors, outputBoxes, num);
            break;
        case yoloType::YOLOV6:
            ret = yolov6Post(inputImages, outputTensors, outputBoxes, num);
            break;
        case yoloType::YOLOV7:
            ret = yolov7Post(inputImages, outputTensors, outputBoxes, num);
            break; 
        case yoloType::YOLOV8:
            ret = yolov8Post(inputImages, outputTensors, outputBoxes, num);
            break;
        case yoloType::YOLOV9:
            ret = yolov9Post(inputImages, outputTensors, outputBoxes, num);
            break;
        case yoloType::YOLOV10:
            ret = yolov10Post(inputImages, outputTensors, outputBoxes, num);
            break;
        case yoloType::YOLOV11:
            ret = yolov11Post(inputImages, outputTensors, outputBoxes, num);
            break;
        case yoloType::YOLOV12:
            ret = yolov12Post(inputImages, outputTensors, outputBoxes, num);
            break;
        default:
            ret = stateType::UNMATCH_YOLO_TYPE_ERROR;
            break;
    }
    return ret;
    
}

stateType sophgo_detect::resizeBox(const bm_image* inputImage, detectBoxes& outputBoxes){

    bm_image img = *inputImage;
    int img_w = img.width;
    int img_h = img.height;

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

stateType sophgo_detect::yolov5Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num){
    for(int batch_idx = 0; batch_idx < num; ++batch_idx)
    {
      detectBoxes yolobox_vec;
  
      int box_num = 0;
      for(int i=0; i<m_output_num; i++){
        auto output_shape = outputTensors[i].get_shape();
        auto output_dims = output_shape->num_dims;
        YOLO_CHECK(output_dims == 5, "The output's dim must be five. which means to [batch, anchor_num, feature_height,feature_width,feature]",enumName(m_yoloType));
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
          auto output_tensor = &outputTensors[head_idx];
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
      resizeBox(inputImages+batch_idx, resVec);
      YOLO_DEBUG("batch_id: {}, outputbox number is {}", batch_idx, resVec.size());
      outputBoxes.push_back(resVec);
    }
    return stateType::SUCCESS;
}

stateType sophgo_detect::yolov6Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num){
    BMNNTensor* output_tensor = outputTensors.data();
    float* output_data = reinterpret_cast<float*>(output_tensor->get_cpu_data());

    for(int batch_idx = 0; batch_idx < num; ++batch_idx) {
        detectBoxes yolobox_vec;

        auto output_shape = output_tensor->get_shape();
        YOLO_CHECK(output_shape->num_dims == 3, "The {} output's dim must be three. which means to [batch, box_num, feature]",enumName(m_yoloType));
        int box_num = output_tensor->get_shape()->dims[1];

        #if USE_MULTICLASS_NMS
            int out_nout = m_nout;
        #else
            int out_nout = 7;
        #endif
        
        float *batch_output_data = output_data + batch_idx * box_num * m_nout;
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
                    detectBox box;

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

                    detectBox box;
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

                    yolobox_vec.push_back(box);
                }
                batch_output_data += m_nout;
        
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
      for (auto& box:resVec) {
        restrictBox(box, inputImages[batch_idx].width, inputImages[batch_idx].height);
      }
      YOLO_DEBUG("batch_id: {}, outputbox number is {}", batch_idx, resVec.size());
      outputBoxes.push_back(resVec);
    }
    return stateType::SUCCESS;
}

stateType sophgo_detect::yolov7Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num){
    return yolov5Post(inputImages, outputTensors, outputBoxes, num);
}

stateType sophgo_detect::yolov8Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num){
    BMNNTensor* output_tensor = outputTensors.data();
    float* output_data = reinterpret_cast<float*>(output_tensor->get_cpu_data());

    for(int batch_idx = 0; batch_idx < num; ++batch_idx) {
        detectBoxes yolobox_vec;

        auto output_shape = output_tensor->get_shape();
        YOLO_CHECK(output_shape->num_dims == 3, "The {} output's dim must be three. which means to [batch, box_num, feature]",enumName(m_yoloType));
        int box_num = output_tensor->get_shape()->dims[1];

        #if USE_MULTICLASS_NMS
            int out_nout = m_nout;
        #else
            int out_nout = 7;
        #endif
        
        float *batch_output_data = output_data + batch_idx * box_num * m_nout;
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
                    detectBox box;

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

                    detectBox box;
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

                    yolobox_vec.push_back(box);
                }
                batch_output_data += m_nout;
        
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
        for (auto& box:resVec) {
            restrictBox(box, inputImages[batch_idx].width, inputImages[batch_idx].height);
        }
        YOLO_DEBUG("batch_id: {}, outputbox number is {}", batch_idx, resVec.size());
        outputBoxes.push_back(resVec);
    }
    return stateType::SUCCESS;
}

stateType sophgo_detect::yolov9Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num){
    return yolov8Post(inputImages, outputTensors, outputBoxes, num);
}


stateType sophgo_detect::yolov10Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num){
    BMNNTensor* output_tensor = outputTensors.data();
    float* output_data = reinterpret_cast<float*>(output_tensor->get_cpu_data());

    for(int batch_idx = 0; batch_idx < num; ++batch_idx) {
        detectBoxes yolobox_vec;

        auto output_shape = output_tensor->get_shape();
        YOLO_CHECK(output_shape->num_dims == 3, "The {} output's dim must be three. which means to [batch, box_num, feature]", enumName(m_yoloType));
        int box_num = output_tensor->get_shape()->dims[1];

        // YOLOv10 only has one label
        int out_nout = 6;

        
        float *batch_output_data = output_data + batch_idx * box_num * m_nout;
        for(int i = 0; i < box_num; ++i) {
            if(batch_output_data[4] < m_confThreshold) {
                break;
            }

            detectBox box;
            box.left = batch_output_data[0];
            box.top = batch_output_data[1];
            box.right = batch_output_data[2];
            box.bottom = batch_output_data[3];
            box.classId =  batch_output_data[5];
            box.score = batch_output_data[4];

            yolobox_vec.push_back(box);
            batch_output_data += out_nout;
        }

        resizeBox(inputImages+batch_idx, yolobox_vec);
        for (auto& box:yolobox_vec) {
            restrictBox(box, inputImages[batch_idx].width, inputImages[batch_idx].height);
        }
        YOLO_DEBUG("batch_id: {}, outputbox number is {}", batch_idx, yolobox_vec.size());
        outputBoxes.push_back(yolobox_vec);
    }
    return stateType::SUCCESS;
}

stateType sophgo_detect::yolov11Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num){
    return yolov8Post(inputImages, outputTensors, outputBoxes, num);
}

stateType sophgo_detect::yolov12Post(const bm_image* inputImages, std::vector<BMNNTensor>& outputTensors, std::vector<detectBoxes>& outputBoxes, const int num){
    return yolov8Post(inputImages, outputTensors, outputBoxes, num);
}
