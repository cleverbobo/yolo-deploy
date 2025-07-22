cmake_minimum_required(VERSION 3.10)
project(yolo-deploy-seg-sophgo)

set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH sophgo)
endif()

message(STATUS "[INFO] BUILD DET SAMPLE FOR TARGET_ARCH: ${TARGET_ARCH}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -fprofile-arcs")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -rdynamic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -rdynamic")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")

# 公共的头文件
include_directories(../../../element/Seg/include)
include_directories(../../../framework/include)
include_directories(../../../framework/utils/include)

# 公共的库目录
link_directories(../../../build/lib)


if (${TARGET_ARCH} STREQUAL "sophgo")
    set(FFMPEG_DIR  /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})

    set(OpenCV_DIR  /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio)

    set(LIBSOPHON_DIR  /opt/sophon/libsophon-current/data/libsophon-config.cmake)
    find_package(LIBSOPHON REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    include_directories(../../../framework/sophgo_framework/include)

    add_executable(detect_sophgo.run detect.cpp)
    target_link_libraries(detect_sophgo.run ${OpenCV_LIBS} -ldetect -lsophgo_framework -lutils)

endif()
