cmake_minimum_required(VERSION 3.26)
project(mandel CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_executable(mandel mandel.cu
        save_image.cpp
        save_image.h
        utils.cpp
        utils.h
        window.h)

target_include_directories(mandel PUBLIC "${PROJECT_SOURCE_DIR}")

set_target_properties(mandel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(mandel -lfreeimage)
