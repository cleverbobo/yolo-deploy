cmake_minimum_required(VERSION 3.10)
project(detectAPI)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -fprofile-arcs -g")

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH sophgo)
endif()

message(STATUS "[INFO] BUILD DETECT FOR TARGET_ARCH: ${TARGET_ARCH}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -pthread -fpermissive")

# 公共的include目录
include_directories(../../framework/include)
include_directories(../../framework/utils/include)
include_directories(include)

# 公共的lib目录
link_directories(../../build/lib)


if (${TARGET_ARCH} STREQUAL "sophgo")
    set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})

    set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})

    set(LIBSOPHON_DIR  /opt/sophon/libsophon-current/data/libsophon-config.cmake)
    find_package(LIBSOPHON REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    set(BM_LIBS bmlib bmrt bmcv yuv)
    find_library(BMJPU bmjpuapi)
    if(BMJPU)
        set(JPU_LIBS bmjpuapi bmjpulite)
    endif()

    
    include_directories(../../framework/sophgo_framework/include)
    

    add_library(detect SHARED  
        src/detect.cpp
        src/detect_factory.cpp
        src/sophgo_detect.cpp)
    
    # target_link_options(detect PRIVATE "LINKER:-no-undefined")
    target_link_libraries(detect ${FFMPEG_LIBS} ${OpenCV_LIBS} ${BM_LIBS} ${JPU_LIBS} -lpthread -lutils -lsophgo_framework)
    

elseif (${TARGET_ARCH} STREQUAL "tensorrt")
    set(TENSORRT_ROOT "/usr/src/tensorrt")

    # trt runtime
    include_directories(${TENSORRT_ROOT}/include)
    include_directories(/usr/local/cuda-12.1/targets/x86_64-linux/include/)
    link_directories(${TENSORRT_ROOT}/lib)
    find_package(CUDA REQUIRED)

    # opencv
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})

    add_library(detect SHARED 
        src/detect.cpp
        src/detect_factory.cpp
        src/trt_detect.cpp)

    
    # target_link_libraries(detect nvinfer ${CUDA_LIBRARIES} ${OpenCV_LIBS} -lpthread -lutils)
    # debug
    link_directories(/usr/local/lib)
    target_link_libraries(detect nvinfer ${CUDA_LIBRARIES} ${OpenCV_LIBS} -lpthread -lutils -lcnpy)

endif()