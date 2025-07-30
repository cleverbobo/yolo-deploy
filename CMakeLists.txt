cmake_minimum_required(VERSION 3.10)
project(sophon-stream)
set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH sophgo)
endif()
add_compile_definitions("${TARGET_ARCH}")

option(USE_MULTICLASS_NMS "Enable multiclass NMS" OFF)
if(USE_MULTICLASS_NMS)
    add_compile_definitions(USE_MULTICLASS_NMS=1)
endif()

message(STATUS "[INFO] START BUILD FOR TARGET_ARCH: ${TARGET_ARCH}")

if (NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "[INFO] CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# 公共的第三方库的头文件查找路径
include_directories(3rdparty/argparse-3.2/include)
include_directories(3rdparty/spdlog-1.15.3/include)
include_directories(3rdparty/magic_enum/include)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/lib)
add_subdirectory(framework)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
message("PROJECT_ROOT = ${PROJECT_ROOT}")

function (checkAndAddElement dir)
    if(EXISTS "${PROJECT_ROOT}/${dir}" AND IS_DIRECTORY "${PROJECT_ROOT}/${dir}")
        add_subdirectory(${dir})
    endif()
endfunction()

function (checkAndAddSample dir)
    if(EXISTS "${PROJECT_ROOT}/${dir}" AND IS_DIRECTORY "${PROJECT_ROOT}/${dir}")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_ROOT}/${dir}/build)
        add_subdirectory(${dir})
    endif()
endfunction()

checkAndAddElement(element/Det)
add_dependencies(detect utils)

checkAndAddElement(element/Seg)
add_dependencies(segment utils)


# checkAndAddElement(element/Pose)



if (${TARGET_ARCH} STREQUAL "sophgo")
    checkAndAddSample(sample/Det/sophgo)
    add_dependencies(detect_sophgo.run detect)

    checkAndAddElement(sample/Seg/sophgo)
    add_dependencies(segment_sophgo.run segment)

    checkAndAddElement(sample/sophgo)
    add_dependencies(yolo_sophgo.run segment detect)

elseif(${TARGET_ARCH} STREQUAL "tensorrt")
    checkAndAddSample(sample/Det/tensorrt)
    add_dependencies(detect_tensorrt.run detect)

    checkAndAddElement(sample/Seg/tensorrt)
    add_dependencies(segment_tensorrt.run segment)

endif()

