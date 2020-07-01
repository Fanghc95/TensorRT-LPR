cmake_minimum_required(VERSION 2.8)
project(demo)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

#配置添加tensorRT路径
find_package(CUDA REQUIRED)
set(TENSORRT_ROOT /home/fhc/trt/TensorRT-6.0.1.5/)


set(CUDA_VERBOSE_BUILD ON)
# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
    HINTS  ${TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PARSER nvparsers
    HINTS  ${TENSORRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()

list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")
CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
message(STATUS "Find CUDA at ${CUDA_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR}")


#添加caffe路径
set(Caffe_Root /home/fhc/caffe-master)
set(Caffe_INCLUDE_DIRS 
	${Caffe_Root}/include
)


#添加opencv路径
FIND_PACKAGE(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARIES_DIRS})

include_directories(
  TrtNet/include 
  include
  ${OpenCV_INCLUDE_DIRS} 
  ${CUDA_INCLUDE_DIRS} 
  ${TENSORRT_INCLUDE_DIR} 
  ${CUDNN_INCLUDE_DIR}
  ${Caffe_INCLUDE_DIRS}
)


file(GLOB CPP_SRC TrtNet/src/*.cpp)
file(GLOB CU_SRC TrtNet/src/*.cu)
file(GLOB PLATE_SRC src/*.cpp)

#plate 
cuda_add_executable(testPlate plate.cpp ${CPP_SRC} ${CU_SRC} ${PLATE_SRC})
target_link_libraries(testPlate ${OpenCV_LIBS} ${TENSORRT_LIBRARY} /home/fhc/caffe-master/build/lib/libcaffe.so /usr/lib/x86_64-linux-gnu/libglog.so  -lboost_system)
