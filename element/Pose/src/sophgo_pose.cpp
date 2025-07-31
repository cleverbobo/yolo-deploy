#include <cmath>

#include "sophgo_pose.h"

#include "utils.h"

#include "bmcv_api_ext.h"
#include "bmnn_utils.h"




sophgo_pose::sophgo_pose(const std::string& modelPath, const yoloType& type, const int devId):pose(modelPath, type, devId) {

    // init handle
    m_handle = std::make_shared<BMNNHandle>(m_devId);
    auto h = m_handle->handle();

    // init context
    m_bmContext = std::make_shared<BMNNContext>(m_handle, modelPath.c_str());

    // init network
    m_bmNetwork = m_bmContext->network(0);
    
    // init preprocess
    auto tensor = m_bmNetwork->inputTensor(0);
    m_net_h = tensor->get_shape()->dims[2];
    m_net_w = tensor->get_shape()->dims[3];


    // init postprocess
    m_max_batch = m_bmNetwork->maxBatch();
    auto output_tensor = m_bmNetwork->outputTensor(0);
    
    m_output_num = m_bmNetwork->outputTensorNum();
    m_output_dim = output_tensor->get_shape()->num_dims;
    m_nout = output_tensor->get_shape()->dims[m_output_dim-1];
    

    // init preprocess data
    m_preprocess_images.resize(m_max_batch);
    bm_image_format_ext image_format = m_bgr2rgb ? FORMAT_RGB_PLANAR : FORMAT_BGR_PLANAR;
    bm_image_data_format_ext data_formate = tensor->get_dtype() == BM_UINT8 ? DATA_TYPE_EXT_1N_BYTE : DATA_TYPE_EXT_FLOAT32;
    for (auto& image : m_preprocess_images) {
        auto ret = bm_image_create(h, m_net_h, m_net_w, image_format, data_formate, &image);
    }
    auto ret = bm_image_alloc_contiguous_mem(m_max_batch, m_preprocess_images.data());
    YOLO_CHECK(ret == BM_SUCCESS, "bm_image_alloc_contiguous_mem failed in sophgo_pose::sophgo_pose!");

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
                                     algorithmType::POSE,
                                     deviceType::SOPHGO,
                                     input_shape,
                                     output_shape,
                                     m_max_batch };
}

sophgo_pose::~sophgo_pose() {
    bm_image_free_contiguous_mem(m_max_batch, m_preprocess_images.data());
    for (auto& image : m_preprocess_images) {
        auto ret = bm_image_destroy(image);
    }
}

std::vector<poseBoxes> sophgo_pose::process(void* inputImage, const int num) {
    bm_image* imageData = reinterpret_cast<bm_image*>(inputImage);

    stateType ret = stateType::SUCCESS;
    int calculateTime = (num-1) / m_max_batch + 1;
    std::vector<poseBoxes> outputBoxes;

    for (int i = 0; i < calculateTime; ++i) {
        int inputNum = std::min(num - i * m_max_batch, m_max_batch);

        // preprocess
        ret = preProcess(imageData + m_max_batch*i, inputNum);
        YOLO_CHECK(ret == stateType::SUCCESS, "preProcess failed in sophgo_pose::process")

        // inference
        ret = inference();
        YOLO_CHECK(ret == stateType::SUCCESS, "inference failed in sophgo_pose::process")

        // postprocess
        ret = postProcess(imageData + m_max_batch*i, outputBoxes, inputNum);
        YOLO_CHECK(ret == stateType::SUCCESS, "postProcess failed inf sophgo_pose::process")

        m_fpsCounter.add(inputNum);
    }

    return outputBoxes;
}


stateType sophgo_pose::preProcess(bm_image* inputImages, const int num){
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
    YOLO_CHECK(ret == BM_SUCCESS, "bmcv_image_vpp_basic failed in sophgo_pose::preProcess");

    // attach to tensor
    bm_device_mem_t input_dev_mem;
    auto num_ = num != m_max_batch ? m_max_batch : num;
    bm_image_get_contiguous_device_mem(num_, m_preprocess_images.data(), &input_dev_mem);
    std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0);
    input_tensor->set_device_mem(&input_dev_mem);
    return stateType::SUCCESS;
}

stateType sophgo_pose::inference(){
    auto ret = m_bmNetwork->forward();
    return ret == BM_SUCCESS ? stateType::SUCCESS : stateType::INFERENCE_ERROR;

}

stateType sophgo_pose::postProcess(const bm_image* inputImages, std::vector<poseBoxes>& outputBoxes, const int num){
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

stateType sophgo_pose::resizeBox(const bm_image* inputImage, poseBoxes& outputBoxes){

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


stateType sophgo_pose::yolov8Post(const bm_image* inputImages, std::vector<poseBoxes>& outputBoxes, const int num){
    std::shared_ptr<BMNNTensor> output_tensor = m_bmNetwork->outputTensor(0);
    float* output_data = reinterpret_cast<float*>(output_tensor->get_cpu_data());

    for(int batch_idx = 0; batch_idx < num; ++batch_idx) {
        poseBoxes yolobox_vec;

        auto output_shape = output_tensor->get_shape();
        YOLO_CHECK(output_shape->num_dims == 3, "The {} output's dim must be three. which means to [batch, box_num, feature]",enumName(m_yoloType));
        int box_num = output_tensor->get_shape()->dims[1];
        
        float *batch_output_data = output_data + batch_idx * box_num * m_nout;
        int max_wh = 7680;
        bool agnostic = false;
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

stateType sophgo_pose::yolov11Post(const bm_image* inputImages, std::vector<poseBoxes>& outputBoxes, const int num){
    return yolov8Post(inputImages, outputBoxes, num);
}


