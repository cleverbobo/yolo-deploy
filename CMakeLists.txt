cmake_minimum_required(VERSION 3.10)
project(sophon-stream)
set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH sophgo)
endif()

message(STATUS "[INFO] START BUILD FOR TARGET_ARCH: ${TARGET_ARCH}")

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
# checkAndAddElement(element/Pose)
# checkAndAddElement(element/Seg)

checkAndAddSample(sample/Det)
add_dependencies(detect.sophgo detect)

