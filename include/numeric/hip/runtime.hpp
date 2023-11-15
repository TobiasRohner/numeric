#ifndef NUMERIC_HIP_RUNTIME_HPP_
#define NUMERIC_HIP_RUNTIME_HPP_

#include <numeric/config.hpp>

#if NUMERIC_GCC_COMPILER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#if NUMERIC_CLANG_COMPILER
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-result"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#ifndef __HIP_DEVICE_COMPILE__
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
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
