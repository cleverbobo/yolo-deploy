cmake_minimum_required(VERSION 3.14)
project(yolo-deploy-pose-tensorrt)

set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH tensorrt)
endif()

message(STATUS "[INFO] BUILD POSE SAMPLE FOR TARGET_ARCH: ${TARGET_ARCH}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -fprofile-arcs")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -rdynamic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -rdynamic")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")

# 获取当前 CMakeLists.txt 所在的路径
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# 设置可执行文件输出到 build 目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")

# 公共的头文件
include_directories(../../../element/Pose/include)
include_directories(../../../framework/include)
include_directories(../../../framework/utils/include)

# include_directories(../../3rdparty/argparse-3.2/include)

# 公共的库目录
link_directories(../../../build/lib)

if (${TARGET_ARCH} STREQUAL "tensorrt")

    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})

    add_executable(pose_tensorrt.run pose.cpp)
    target_link_libraries(pose_tensorrt.run ${OpenCV_LIBS} -lpose -lutils)

endif()
