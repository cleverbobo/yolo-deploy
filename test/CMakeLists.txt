cmake_minimum_required(VERSION 3.10)
project(yolo-deploy-test-case)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH sophgo)
endif()

message(STATUS "[INFO] BUILD TEST CASE FOR TARGET_ARCH: ${TARGET_ARCH}")

if (${TARGET_ARCH} STREQUAL "sophgo")

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -fprofile-arcs")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -rdynamic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -rdynamic -fpermissive")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")

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

    # 3rdparty include
    include_directories(../3rdparty/argarse-3.2/include)
    include_directories(../3rdparty/spdlog-1.15.3/include)

    # yolo-deploy include
    include_directories(../framework/include)
    include_directories(../framework/sophgo_framework/include)
    include_directories(../framework/utils/include)


    # test case binary
    add_executable(test_case_log test_log.cpp)
    add_executable(test_case_check test_check.cpp)

elseif(${TARGET_ARCH} STREQUAL "soc")


endif()
