set(TARGET_NAME "renderer")

file(
    GLOB_RECURSE SOURCES
    *.h *.hpp *.c *.cpp)

set(
    CUDA_KERNELS
    "gpu_kernels/dummy_kernel.cu"
    "gpu_kernels/post_process_kernels.cu"
    "gpu_kernels/compute_light_probs.cu"
)

set(
    OPTIX_KERNELS
    "gpu_kernels/optix_pathtracing_kernels.cu"
    "gpu_kernels/optix_lighttracing_kernels.cu"
)

set(
    GPU_KERNEL_DEPENDENCIES
    "../common/utils/cuda_util.h"
    "../common/utils/optix_util.h"
    "../common/utils/optixu_on_cudau.h"
    "../common/common_shared.h"
    "../common/common_renderer_types.h"
    "../common/basic_types.h"
    "${TARGET_NAME}_shared.h"
    "gpu_kernels/renderer_kernel_common.h"
)

nvcuda_compile_ptx(
    SOURCES ${CUDA_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES CUDA_PTXES
    NVCC_OPTIONS
    "$<$<CONFIG:Debug>:-G>"
    "$<$<CONFIG:Debug>:-D_DEBUG=1>"
    "--gpu-architecture=compute_52"
    "-std=c++20"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
    "-I${CMAKE_SOURCE_DIR}/ext/nanovdb"
)

nvcuda_compile_ptx(
    SOURCES ${OPTIX_KERNELS}
    DEPENDENCIES ${GPU_KERNEL_DEPENDENCIES}
    TARGET_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/${TARGET_NAME}/ptxes"
    GENERATED_FILES OPTIX_PTXES
    NVCC_OPTIONS
    "$<$<CONFIG:Debug>:-lineinfo>"
    "$<$<CONFIG:Debug>:-D_DEBUG=1>"
    "--gpu-architecture=compute_${TARGET_CUDA_ARCH}"
    "-std=c++20"
    "-cudart" "shared"
    "--use_fast_math"
    "--relocatable-device-code=true"
    "-I${OPTIX_INCLUDE_DIR}"
    "-I${CMAKE_SOURCE_DIR}/ext/nanovdb"
)

set_common_sources()
set_source_files_properties(
    ${GL3W_SOURCES} PROPERTIES
    GENERATED TRUE
)

# essentials
source_group(
    "essentials" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/[^/]*\.(h|hpp|c|cpp)$")
source_group(
    "essentials/GPU kernels" REGULAR_EXPRESSION
    "${CMAKE_CURRENT_SOURCE_DIR}/gpu_kernels/[^/]*\.(h|cu)$")

list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)
list(APPEND DEFINITIONS -DTCNN_MIN_GPU_ARCH=${MIN_GPU_ARCH})

list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "$<$<CONFIG:Debug>:-G>")

add_executable(
    "${TARGET_NAME}"
    ${COMMON_SOURCES}
    ${COMMON_RENDERER_SOURCES}
    ${SOURCES}
    ${GL3W_SOURCES}
    ${CUDA_KERNELS}
    ${OPTIX_KERNELS}
)
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF CUDA_EXTENSIOINS OFF)
target_compile_definitions("${TARGET_NAME}" PRIVATE ${DEFINITIONS})
target_compile_options(
    "${TARGET_NAME}" PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>"
)
target_include_directories(
    "${TARGET_NAME}" PRIVATE
    "../ext/tiny-cuda-nn/dependencies"
    "../ext/tiny-cuda-nn/include"
)
target_link_libraries(
    "${TARGET_NAME}"
    fakelib
)
    