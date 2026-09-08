#ifndef NUMERIC_HIP_RUNTIME_HPP_
#define NUMERIC_HIP_RUNTIME_HPP_

#include <numeric/config.hpp>

#if NUMERIC_GCC_COMPILER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#endif
#if NUMERIC_CLANG_COMPILER
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wreturn-stack-address"
#endif

#ifndef __HIP_DEVICE_COMPILE__
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#ifdef __HIP_PLATFORM_NVIDIA__
// These things were forgotten by the AMD people to wrap in HIP

hipError_t hipOccupancyMaxPotentialBlockSize(int *gridSize, int *blockSize,
                                             hipFunction_t f,
                                             size_t dynSharedMemPerBlk,
                                             int blockSizeLimit);

template <typename UnaryFunction, class T>
hipError_t hipOccupancyMaxPotentialBlockSizeVariableSMem(
    int *min_grid_size, int *block_size, T func,
    UnaryFunction block_size_to_dynamic_smem_size, int block_size_limit = 0) {
  return hipCUDAErrorTohipError(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
      min_grid_size, block_size, func, block_size_to_dynamic_smem_size,
      block_size_limit));
}
#endif
#endif
// HACK TO FIX HIP BUG
#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_API_H
#define HIP_INCLUDE_HIP_HIP_RUNTIME_API_H
#define NUMERIC_UNSET_HACK_AGAIN
#endif
#include <hip/hip_runtime.h>
#include <hip/math_functions.h>
#ifdef NUMERIC_UNSET_HACK_AGAIN
#undef HIP_INCLUDE_HIP_HIP_RUNTIME_API_H
#endif
#undef NUMERIC_UNSET_HACK_AGAIN

#if NUMERIC_GCC_COMPILER
#pragma GCC diagnostic pop
#endif
#if NUMERIC_CLANG_COMPILER
#pragma clang diagnostic pop
#endif

#endif
