cmake_minimum_required(VERSION 3.10)
project(detectAPI)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -fprofile-arcs -g")

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH sophgo_pcie)
endif()

message(STATUS "[INFO] BUILD DETECT FOR TARGET_ARCH: ${TARGET_ARCH}")

if (${TARGET_ARCH} STREQUAL "sophgo_pcie")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -pthread -fpermissive")

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

    link_directories(../../build/lib)


    include_directories(../../framework/include)
    include_directories(../../framework/sophgo_framework/include)
    include_directories(../../framework/utils/include)

    include_directories(include)

    add_library(detect SHARED  
        src/detect_factory.cpp
        src/sophgo_detect.cpp)
    
    # target_link_options(detect PRIVATE "LINKER:-no-undefined")
    target_link_libraries(detect ${FFMPEG_LIBS} ${OpenCV_LIBS} ${BM_LIBS} ${JPU_LIBS} -lpthread -lutils -lsophgo_framework)
    

elseif (${TARGET_ARCH} STREQUAL "sophgo_soc")
    add_compile_options(-fPIC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -fprofile-arcs -ftest-coverage -g -rdynamic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -rdynamic -fpermissive")

    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

    include_directories("${SOPHON_SDK_SOC}/include/")
    include_directories("${SOPHON_SDK_SOC}/include/opencv4")
    link_directories("${SOPHON_SDK_SOC}/lib/")
    set(BM_LIBS bmlib bmrt bmcv yuv)
    find_library(BMJPU bmjpuapi)
    if(BMJPU)
        set(JPU_LIBS bmjpuapi bmjpulite)
    endif()

    include_directories(../../../framework/include)
    include_directories(../../../framework/sophgo_framework/include)
    include_directories(../../../framework/util)
    
    include_directories(include)

    add_library(detect SHARED  
        src/detect_factory.cpp
        src/sophgo_detect.cpp)

    target_link_libraries(detect ${BM_LIBS} ${FFMPEG_LIBS} ${OpenCV_LIBS}  ${JPU_LIBS} -lopencv_video -fprofile-arcs -lgcov -lpthread)
endif()