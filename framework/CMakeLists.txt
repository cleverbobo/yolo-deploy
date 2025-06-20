cmake_minimum_required(VERSION 3.10)
project(yolo-deploy-framework)

set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH sophgo)
endif()

message(STATUS "[INFO] BUILD FRAMEWORK FOR TARGET_ARCH: ${TARGET_ARCH}")

# 设置公共的头文件和库目录
include_directories(./include)
include_directories(./utils/include)
include_directories(../3rdparty/spdlog-1.15.3/include)
include_directories(../3rdparty/magic_enum/include)


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

    
    include_directories(./sophgo_framework/include)
    

    add_library(sophgo_framework SHARED
        ./sophgo_framework/src/ff_decode.cpp
    )
    # target_link_options(sophgo_framework PRIVATE "LINKER:-no-undefined")
    target_link_libraries(sophgo_framework ${FFMPEG_LIBS} ${OpenCV_LIBS} ${BM_LIBS} ${JPU_LIBS} -lpthread)
    

    add_library(utils SHARED
        src/log.cpp
        utils/src/utils.cpp)
    # target_link_options(utils PRIVATE "LINKER:-no-undefined")
    target_link_libraries(utils PUBLIC ${OpenCV_LIBS})
    
    

elseif(${TARGET_ARCH} STREQUAL "nvidia")
    # 使用公版的opencv
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})

    dd_library(utils SHARED
        src/log.cpp
        utils/src/utils.cpp)
    target_link_options(utils PRIVATE "LINKER:-no-undefined")
    target_link_libraries(utils PUBLIC ${OpenCV_LIBS})


endif()
