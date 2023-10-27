#ifndef NUMERIC_MEMORY_MEMCPY_HPP_
#define NUMERIC_MEMORY_MEMCPY_HPP_

#include <cstring>
#include <numeric/config.hpp>
#include <numeric/memory/array_const_view_decl.hpp>
#include <numeric/memory/array_view_decl.hpp>
#if NUMERIC_ENABLE_HIP
#include <hip/hip_runtime_api.h>
#endif

namespace numeric::memory {

template <typename ScalarL, dim_t NL, typename ScalarR, dim_t NR>
void memcpy(ArrayView<ScalarL, NL> dst, const ArrayConstView<ScalarR, NR> &src,
            dim_t N = -1) {
  if (N < 0) {
    N = std::min(dst.size() * sizeof(ScalarL), src.size() * sizeof(ScalarR));
  }
  if (is_host_accessible(dst.memory_type()) &&
      is_host_accessible(src.memory_type())) {
    std::memcpy(dst.raw(), src.raw(), N);
  }
#if NUMERIC_ENABLE_HIP
  else if (is_host_accessible(dst.memory_type()) &&
           is_device_accessible(src.memory_type())) {
    hipMemcpy(dst.raw(), src.raw(), N, hipMemcpyDeviceToHost);
  } else if (is_device_accessible(dst.memory_type()) &&
             is_host_accessible(src.memory_type())) {
    hipMemcpy(dst.raw(), src.raw(), N, hipMemcpyHostToDevice);
  } else if (is_device_accessible(dst.memory_type()) &&
             is_device_accessible(src.memory_type())) {
    hipMemcpy(dst.raw(), src.raw(), N, hipMemcpyDeviceToDevice);
  }
#endif
  else {
    NUMERIC_ERROR("Unsupported memory types for memcpy: {} -> {}",
                  to_string(src.memory_type()), to_string(dst.memory_type()));
  }
}

} // namespace numeric::memory

#endif
