#ifndef NUMERIC_MEMORY_SLICE_HPP_
#define NUMERIC_MEMORY_SLICE_HPP_

#include <numeric/config.hpp>
#include <numeric/math/functions.hpp>
#ifndef __HIP_DEVICE_COMPILE__
#include <iostream>
#endif

namespace numeric::memory {

/**
 * @brief Represents a slice of indices.
 *
 * A slice defines a range of indices with a start, stop, and step.
 */
struct Slice {
  dim_t start;
  dim_t stop;
  dim_t step;

  /**
   * @brief Default constructor.
   *
   * Initializes the slice with default values.
   */
  NUMERIC_HOST_DEVICE Slice() : start(0), stop(-1), step(1) {}

  /**
   * @brief Constructor with specified parameters.
   *
   * Constructs a slice with the provided start, stop, and step values.
   *
   * @param start_ Start index of the slice.
   * @param stop_ Stop index of the slice.
   * @param step_ Step size of the slice.
   */
  NUMERIC_HOST_DEVICE explicit Slice(dim_t start_, dim_t stop_, dim_t step_ = 1)
      : start(start_), stop(stop_), step(step_) {}
  Slice(const Slice &) = default;
  Slice &operator=(const Slice &) = default;

  /**
   * @brief Calculates the size of the slice.
   *
   * Calculates the number of elements in the slice based on the size of the
   * dimension.
   *
   * @param N Size of the dimension.
   * @return Size of the slice.
   */
  NUMERIC_HOST_DEVICE dim_t size(dim_t N) const noexcept {
    dim_t real_stop = stop;
    if (real_stop < 0) {
      while (real_stop < 0) {
        real_stop += N;
      }
      real_stop += 1;
    }
    return math::div_up((real_stop - start), step);
  }
};

#ifndef __HIP_DEVICE_COMPILE__
std::ostream &operator<<(std::ostream &os, const Slice &slice);
#endif

} // namespace numeric::memory

#endif
