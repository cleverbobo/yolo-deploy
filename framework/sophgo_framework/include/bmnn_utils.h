//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef BMNN_H
#define BMNN_H

#include <iostream>
#include <string>
#include <memory>
#include <set>

#include "yolo_common.h"

#include "bmruntime_interface.h"
#include "bmruntime_cpp.h"



/*
 * Help user managing input tensor and output tensor.
 * Feat 1. Free system memory automatically. 
 *      2. Any member in m_tensor has device memory like bm_tensor_t\bm_image\bm_device_mem_t must be freed outside.
 */
class BMNNTensor{
  /**
   *  members from bm_tensor {
   *    bm_data_type_t dtype;
   *    bm_shape_t shape;
   *    bm_device_mem_t device_mem;
   *    bm_store_mode_t st_mode;
   *  }
   */
  private:
    bm_handle_t m_handle;
    float *m_cpu_data;
    bm_tensor_t *m_tensor;
    bool can_mmap;
    int m_dev_id;
    std::shared_ptr<bm_tensor_t> m_tensor_ptr;

  public:
  BMNNTensor(bm_tensor_t* tensor=nullptr, int dev_id = 0):m_cpu_data(nullptr), m_tensor(tensor), m_dev_id(dev_id) {
    int ret = bm_dev_request(&m_handle, dev_id);
    YOLO_CHECK(BM_SUCCESS == ret);

    struct bm_misc_info misc_info;
    bm_status_t ret = bm_get_misc_info(m_handle, &misc_info);
    YOLO_CHECK(BM_SUCCESS == ret);
    can_mmap = misc_info.pcie_soc_mode == 1;
    
    if (m_tensor == nullptr) {
      m_tensor_ptr = std::make_shared<bm_tensor_t>();
      m_tensor = m_tensor_ptr.get();
    }
  }


  virtual ~BMNNTensor() {
    if (m_cpu_data == NULL) return;
    if(can_mmap && BM_FLOAT32 == m_tensor->dtype) {
      int tensor_size = bm_mem_get_device_size(m_tensor->device_mem);
      bm_status_t ret = bm_mem_unmap_device_mem(m_handle, m_cpu_data, tensor_size);
      YOLO_CHECK(BM_SUCCESS == ret);
    } else {
      delete [] m_cpu_data;
    }
    bm_dev_free(m_handle);
  }

  // Set tensor device memory.
  stateType set_device_mem(bm_device_mem_t *mem){
    m_tensor->device_mem = *mem;
    return stateType::SUCCESS;
  }

  const bm_device_mem_t* get_device_mem() {
    return &(m_tensor->device_mem);
  }

  // Return an array pointer to system memory of tensor.
  float *get_cpu_data() {
    if(m_cpu_data) return m_cpu_data;
    bm_status_t ret;
    float *pFP32 = nullptr;
    int count = bmrt_shape_count(&m_tensor->shape);
    // in SOC mode, device mem can be mapped to host memory, faster then using d2s
    if(can_mmap){
      if (m_tensor->dtype == BM_FLOAT32) {
        unsigned long long  addr;
        ret = bm_mem_mmap_device_mem(m_handle, &m_tensor->device_mem, &addr);
        YOLO_CHECK(BM_SUCCESS == ret);
        ret = bm_mem_invalidate_device_mem(m_handle, &m_tensor->device_mem);
        YOLO_CHECK(BM_SUCCESS == ret);
        pFP32 = (float*)addr;
      } else if (BM_INT8 == m_tensor->dtype) {
        int8_t * pI8 = nullptr;
        unsigned long long  addr;
        ret = bm_mem_mmap_device_mem(m_handle, &m_tensor->device_mem, &addr);
        YOLO_CHECK(BM_SUCCESS == ret);
        ret = bm_mem_invalidate_device_mem(m_handle, &m_tensor->device_mem);
        YOLO_CHECK(BM_SUCCESS == ret);
        pI8 = (int8_t*)addr;

        // dtype convert
        pFP32 = new float[count];
        YOLO_CHECK(pFP32 != nullptr);
        for(int i = 0;i < count; ++ i) {
          pFP32[i] = pI8[i];
        }
        ret = bm_mem_unmap_device_mem(m_handle, pI8, bm_mem_get_device_size(m_tensor->device_mem));
        YOLO_CHECK(BM_SUCCESS == ret);
      }else if (m_tensor->dtype == BM_INT32) {
        int32_t * pI32 = nullptr;
        unsigned long long  addr;
        ret = bm_mem_mmap_device_mem(m_handle, &m_tensor->device_mem, &addr);
        YOLO_CHECK(BM_SUCCESS == ret);
        ret = bm_mem_invalidate_device_mem(m_handle, &m_tensor->device_mem);
        YOLO_CHECK(BM_SUCCESS == ret);
        pI32 = (int32_t*)addr;
        // dtype convert
        pFP32 = new float[count];
        YOLO_CHECK(pFP32 != nullptr);
        for(int i = 0;i < count; ++ i) {
          pFP32[i] = pI32[i];
        }
        ret = bm_mem_unmap_device_mem(m_handle, pI32, bm_mem_get_device_size(m_tensor->device_mem));
        YOLO_CHECK(BM_SUCCESS == ret);
      } else{
        std::cout << "NOT support dtype=" << m_tensor->dtype << std::endl;
      }
    } else {
      // the common method using d2s
      if (m_tensor->dtype == BM_FLOAT32) {
        pFP32 = new float[count];
        YOLO_CHECK(pFP32 != nullptr);
        ret = bm_memcpy_d2s_partial(m_handle, pFP32, m_tensor->device_mem, count * sizeof(float));
        YOLO_CHECK(BM_SUCCESS ==ret);
      } else if (BM_INT8 == m_tensor->dtype) {
        int8_t * pI8 = nullptr;
        int tensor_size = bmrt_tensor_bytesize(m_tensor);
        pI8 = new int8_t[tensor_size];
        YOLO_CHECK(pI8 != nullptr);

        // dtype convert
        pFP32 = new float[count];
        YOLO_CHECK(pFP32 != nullptr);
        ret = bm_memcpy_d2s_partial(m_handle, pI8, m_tensor->device_mem, tensor_size);
        YOLO_CHECK(BM_SUCCESS ==ret);
        for(int i = 0;i < count; ++ i) {
          pFP32[i] = pI8[i];
        }
        delete [] pI8;
      }else if(m_tensor->dtype == BM_INT32){
        int32_t *pI32=nullptr;
        int tensor_size = bmrt_tensor_bytesize(m_tensor);
        pI32 =new int32_t[tensor_size];
        YOLO_CHECK(pI32 != nullptr);

        // dtype convert
        pFP32 = new float[count];
        YOLO_CHECK(pFP32 != nullptr);
        ret = bm_memcpy_d2s_partial(m_handle, pI32, m_tensor->device_mem, tensor_size);
        YOLO_CHECK(BM_SUCCESS ==ret);
        for(int i = 0;i < count; ++ i) {
          pFP32[i] = pI32[i];
        }
        delete [] pI32;
        
      }
       else{
        std::cout << "NOT support dtype=" << m_tensor->dtype << std::endl;
      }
    }

    m_cpu_data = pFP32;
    return m_cpu_data;
  }

  const bm_shape_t* get_shape() {
    return &m_tensor->shape;
  }

  std::vector<int> get_shape_vector() {
    std::vector<int> shape;
    for(int i=0; i<m_tensor->shape.num_dims; i++){
      shape.push_back(m_tensor->shape.dims[i]);
    }
    return shape;
  }

  bm_data_type_t get_dtype() {
    return m_tensor->dtype;
  }


  bm_tensor_t* get_tensor() {
    return m_tensor;
  }

  stateType release_tensor_mem() {
    int tensor_size = bm_mem_get_device_size(m_tensor->device_mem);
    if(tensor_size > 0) {
      if(m_cpu_data && can_mmap && BM_FLOAT32 == m_tensor->dtype) {
        bm_status_t ret = bm_mem_unmap_device_mem(m_handle, m_cpu_data, tensor_size);
        YOLO_CHECK(BM_SUCCESS == ret);
      }
      bm_free_device(m_handle, m_tensor->device_mem);
    } else {
      YOLO_WARN("The device memory of tensor is already released.");
      return stateType::SUCCESS;
    }
  }

};

/*
 * Help user managing network to do inference.
 * Feat 1. Create and free device memory of output tensors automatically.
 *      2. Device memory of input tensors must be provided outside, and will not be freed here.
 *      3. Print Network information.
 */
class BMNNNetwork : public NoCopyable {

  bm_handle_t  m_handle;
  void *m_bmrt;
  bool is_soc;
  std::set<int> m_batches;
  int m_max_batch;


  public:
  // Initialize a network for inference, including handle\netinfo\io tensors.
  const bm_net_info_t *m_netinfo;
  BMNNNetwork(void *bmrt, const std::string& name):m_bmrt(bmrt) {
    m_handle = static_cast<bm_handle_t>(bmrt_get_bm_handle(bmrt));
    m_netinfo = bmrt_get_network_info(bmrt, name.c_str());
    m_max_batch = -1;
    std::vector<int> batches;
    for(int i=0; i<m_netinfo->stage_num; i++){
      batches.push_back(m_netinfo->stages[i].input_shapes[0].dims[0]);
      if(m_max_batch<batches.back()){
        m_max_batch = batches.back();
      }
    }
    m_batches.insert(batches.begin(), batches.end());
    struct bm_misc_info misc_info;
    bm_status_t ret = bm_get_misc_info(m_handle, &misc_info);
    YOLO_CHECK(BM_SUCCESS == ret);
    is_soc = misc_info.pcie_soc_mode == 1;

    printf("*** Run in %s mode ***\n", is_soc?"SOC": "PCIE");

    //YOLO_CHECK(m_netinfo->stage_num == 1);
    showInfo();
  }

  ~BMNNNetwork() {
  }

  int maxBatch() const {
    return m_max_batch;
  }
  int get_nearest_batch(int real_batch){
      for(auto batch: m_batches){
          if(batch>=real_batch){
             return batch;
					}
      }
      YOLO_CHECK(0);
      return m_max_batch;
  }

  int inputTensorNum() {
    return m_netinfo->input_num;
  }



  int outputTensorNum() {
    return m_netinfo->output_num;
  }



  std::vector<BMNNTensor> forward(BMNNTensor* inputTensors, const int num) {
    //attach input tensors 
    std::vector<bm_tensor_t> inputTensorsVec(num);
    for (int i = 0; i < num; ++i) {
      inputTensorsVec[i] = *inputTensors[i].get_tensor();
    }

    // malloc output tensors
    std::vector<BMNNTensor> outputTensors(m_netinfo->output_num);
    for(int i = 0; i < m_netinfo->output_num; ++i) {
      auto outputTensor = outputTensors[i].get_tensor();
      outputTensor->dtype = m_netinfo->output_dtypes[i];
      outputTensor->shape = m_netinfo->stages[0].output_shapes[i];
      outputTensor->st_mode = BM_STORE_1N;
      
      // alloc as max size to reuse device mem, avoid to alloc and free everytime
      size_t max_size = 0;
			for(int s=0; s<m_netinfo->stage_num; s++){
         size_t out_size = bmrt_shape_count(&m_netinfo->stages[s].output_shapes[i]);
         if(max_size < out_size){
            max_size = out_size;
         }
      }
      max_size *= bmruntime::ByteSize(m_netinfo->output_dtypes[i]);
      auto ret =  bm_malloc_device_byte(m_handle, &(outputTensor->device_mem), max_size);
			YOLO_CHECK(BM_SUCCESS == ret);
    }
    std::vector<bm_tensor_t> outputTensorsVec(m_netinfo->output_num);
    for(int i=0; i<m_netinfo->output_num; i++){
      outputTensorsVec[i] = *outputTensors[i].get_tensor();
    }
    

    bool ok=bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, inputTensorsVec.data(), m_netinfo->input_num,
                                  outputTensorsVec.data(), m_netinfo->output_num, true, false);
    YOLO_CHECK(ok == BM_SUCCESS);

    bool status = bm_thread_sync(m_handle);
    YOLO_CHECK(BM_SUCCESS == status);

    return outputTensors;
  }

  static std::string shape_to_str(const bm_shape_t& shape) {
    std::string str ="[ ";
    for(int i=0; i<shape.num_dims; i++){
      str += std::to_string(shape.dims[i]) + " ";
    }
    str += "]";
    return str;
  }

  
  std::vector<bm_data_type_t> input_dtypes() const {
    std::vector<bm_data_type_t> dtypes;
    for(int i=0; i<m_netinfo->input_num; i++){
      dtypes.push_back(m_netinfo->input_dtypes[i]);
    }
    return dtypes;
  }

  std::vector<bm_data_type_t> output_dtypes() const {
    std::vector<bm_data_type_t> dtypes;
    for(int i=0; i<m_netinfo->output_num; i++){
      dtypes.push_back(m_netinfo->output_dtypes[i]);
    }
    return dtypes;
  }

  std::vector<std::vector<int>> input_shapes() const {
    std::vector<std::vector<int>> shapes;
    for(int i=0; i<m_netinfo->input_num; i++){
      std::vector<int> shape;
      for(int j=0; j<m_netinfo->stages[0].input_shapes[i].num_dims; j++){
        shape.push_back(m_netinfo->stages[0].input_shapes[i].dims[j]);
      }
      shapes.push_back(shape);
    }
    return shapes;
  }
  
  std::vector<std::vector<int>> output_shapes() const {
    std::vector<std::vector<int>> shapes;
    for(int i=0; i<m_netinfo->output_num; i++){
      std::vector<int> shape;
      for(int j=0; j<m_netinfo->stages[0].output_shapes[i].num_dims; j++){
        shape.push_back(m_netinfo->stages[0].output_shapes[i].dims[j]);
      }
      shapes.push_back(shape);
    }
    return shapes;
  }

  void showInfo()
  {
    const char* dtypeMap[] = {
      "FLOAT32",
      "FLOAT16",
      "INT8",
      "UINT8",
      "INT16",
      "UINT16",
      "INT32",
      "UINT32",
    };
    printf("\n########################\n");
    printf("NetName: %s\n", m_netinfo->name);
    for(int s=0; s<m_netinfo->stage_num; s++){
      printf("---- stage %d ----\n", s);
      for(int i=0; i<m_netinfo->input_num; i++){
        auto shapeStr = shape_to_str(m_netinfo->stages[s].input_shapes[i]);
        printf("  Input %d) '%s' shape=%s dtype=%s scale=%g\n",
            i,
            m_netinfo->input_names[i],
            shapeStr.c_str(),
            dtypeMap[m_netinfo->input_dtypes[i]],
            m_netinfo->input_scales[i]);
      }
      for(int i=0; i<m_netinfo->output_num; i++){
        auto shapeStr = shape_to_str(m_netinfo->stages[s].output_shapes[i]);
        printf("  Output %d) '%s' shape=%s dtype=%s scale=%g\n",
            i,
            m_netinfo->output_names[i],
            shapeStr.c_str(),
            dtypeMap[m_netinfo->output_dtypes[i]],
            m_netinfo->output_scales[i]);
      }
    }
    printf("########################\n\n");

  }

};

// Device handle auto manager.
class BMNNHandle: public NoCopyable {
  bm_handle_t m_handle;
  int m_dev_id;
  public:
  BMNNHandle(int dev_id=0):m_dev_id(dev_id) {
    int ret = bm_dev_request(&m_handle, dev_id);
    YOLO_CHECK(BM_SUCCESS == ret);
  }

  ~BMNNHandle(){
    bm_dev_free(m_handle);
  }

  bm_handle_t handle() {
    return m_handle;
  }

  int dev_id() {
    return m_dev_id;
  }
};

using BMNNHandlePtr = std::shared_ptr<BMNNHandle>;

/*
 * Help user managing handles and networks of a bmodel, using class instances above.
 */
class BMNNContext : public NoCopyable {
  BMNNHandlePtr m_handlePtr;
  void *m_bmrt;
  std::vector<std::string> m_network_names;

  public:
  BMNNContext(BMNNHandlePtr handle, const char* bmodel_file):m_handlePtr(handle){
    bm_handle_t hdev = m_handlePtr->handle();
    m_bmrt = bmrt_create(hdev);
    if (NULL == m_bmrt) {
      std::cout << "bmrt_create() failed!" << std::endl;
      exit(-1);
    }

    if (!bmrt_load_bmodel(m_bmrt, bmodel_file)) {
      std::cout << "load bmodel(" << bmodel_file << ") failed" << std::endl;
    }

    load_network_names();


  }

  ~BMNNContext() {
    if (m_bmrt!=NULL) {
      bmrt_destroy(m_bmrt);
      m_bmrt = NULL;
    }
  }

  bm_handle_t handle() {
    return m_handlePtr->handle();
  }

  void* bmrt() {
    return m_bmrt;
  }

  void load_network_names() {
    const char **names;
    int num;
    num = bmrt_get_network_number(m_bmrt);
    bmrt_get_network_names(m_bmrt, &names);
    for(int i=0;i < num; ++i) {
      m_network_names.push_back(names[i]);
    }

    free(names);
  }

  std::string network_name(int index){
    if (index >= (int)m_network_names.size()) {
      return "Invalid index";
    }

    return m_network_names[index];
  }

  std::shared_ptr<BMNNNetwork> network(const std::string& net_name)
  {
    return std::make_shared<BMNNNetwork>(m_bmrt, net_name);
  }

  std::shared_ptr<BMNNNetwork> network(int net_index) {
    YOLO_CHECK(net_index < (int)m_network_names.size());
    return std::make_shared<BMNNNetwork>(m_bmrt, m_network_names[net_index]);
  }

};

#endif 
