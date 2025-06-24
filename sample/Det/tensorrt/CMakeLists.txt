cmake_minimum_required(VERSION 3.14)
project(yolo-deploy-det-tensorrt)

set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH tensorrt)
endif()

message(STATUS "[INFO] BUILD DET SAMPLE FOR TARGET_ARCH: ${TARGET_ARCH}")
message(STATUS "[INFO] BUILD DET SAMPLE FOR TARGET_ARCH: ${TARGET_ARCH}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -fprofile-arcs")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -rdynamic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -rdynamic")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
# 设置为debug模式
set(CMAKE_BUILD_TYPE Debug)
# 编译优化设置为O0
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

# 公共的头文件
include_directories(../../../element/Det/include)
include_directories(../../../framework/include)
include_directories(../../../framework/utils/include)

# include_directories(../../3rdparty/argparse-3.2/include)

# 公共的库目录
link_directories(../../../build/lib)

if (${TARGET_ARCH} STREQUAL "tensorrt")

    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})

    add_executable(detect_tensorrt.run detect.cpp)
    target_link_libraries(detect_tensorrt.run ${OpenCV_LIBS} -ldetect -lutils)

endif()
