cmake_minimum_required(VERSION 3.2)
project(tvm_inference)

set(CMAKE_CXX_STANDARD 14)

#Needs to be enabled for FindROCM.cmake
option(USE_ROCM "Build with ROCM" ON)

if(DEFINED ENV{TVM_SRC})
  set(TVM_SRC_DIR ${TVM_SRC})
else()
  set(TVM_SRC_DIR "/src/tvm")
endif()
message("------------TVM_SRC_DIR: ${TVM_SRC_DIR}")

include(${TVM_SRC_DIR}/cmake/utils/Utils.cmake)
include(${TVM_SRC_DIR}/cmake/utils/FindROCM.cmake)
include(${TVM_SRC_DIR}/cmake/modules/ROCM.cmake)

include_directories(${TVM_SRC_DIR}/3rdparty/dlpack/include)
include_directories(${TVM_SRC_DIR}/3rdparty/dmlc-core/include)
include_directories(${TVM_SRC_DIR}/src/runtime)
include_directories(${TVM_SRC_DIR}/include)

add_executable(
  inference
  tvm_runtime_pack.cc
  inference.cc
  )

target_compile_definitions(inference PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
target_link_libraries(inference
  ${ROCM_HIPHCC_LIBRARY}
  ${ROCM_HSA_LIBRARY}
  ${ROCM_MIOPEN_LIBRARY}
  ${ROCM_ROCBLAS_LIBRARY}
  pthread
  dl
  )
