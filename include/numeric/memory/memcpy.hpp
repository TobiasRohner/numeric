#ifndef NUMERIC_MEMORY_MEMCPY_HPP_
#define NUMERIC_MEMORY_MEMCPY_HPP_

#include <cstring>
#include <numeric/config.hpp>
#include <numeric/memory/array_const_view_decl.hpp>
#include <numeric/memory/array_view_decl.hpp>
#include <numeric/utils/error.hpp>
#if NUMERIC_ENABLE_HIP
#include <numeric/hip/runtime.hpp>
#endif

namespace numeric::memory {

/**
 * @brief Copies data between arrays.
 *
 * This function copies data from the source array `src` to the destination
 * array `dst`. The number of elements to copy can be specified by `N`. If `N`
 * is not provided, it defaults to the minimum of the sizes of the source and
 * destination arrays.
 *
 * @tparam ScalarL The scalar type of the destination array.
 * @tparam NL The dimensionality of the destination array.
 * @tparam ScalarR The scalar type of the source array.
 * @tparam NR The dimensionality of the source array.
 * @param dst The destination array view.
 * @param src The source array const view.
 * @param N The number of elements to copy.
 */
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
    NUMERIC_CHECK_HIP(
        hipMemcpy(dst.raw(), src.raw(), N, hipMemcpyDeviceToHost));
  } else if (is_device_accessible(dst.memory_type()) &&
             is_host_accessible(src.memory_type())) {
    NUMERIC_CHECK_HIP(
        hipMemcpy(dst.raw(), src.raw(), N, hipMemcpyHostToDevice));
  } else if (is_device_accessible(dst.memory_type()) &&
             is_device_accessible(src.memory_type())) {
    NUMERIC_CHECK_HIP(
        hipMemcpy(dst.raw(), src.raw(), N, hipMemcpyDeviceToDevice));
  }
#endif
  else {
    NUMERIC_ERROR("Unsupported memory types for memcpy: {} -> {}",
                  to_string(src.memory_type()), to_string(dst.memory_type()));
  }
}

} // namespace numeric::memory

#endif
