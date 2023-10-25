#ifndef NUMERIC_MEMORY_SLICE_HPP_
#define NUMERIC_MEMORY_SLICE_HPP_

#include <numeric/config.hpp>
#ifndef __HIP_DEVICE_COMPILE__
#include <iostream>
#endif

namespace numeric::memory {

struct Slice {
  dim_t start;
  dim_t stop;
  dim_t step;

  NUMERIC_HOST_DEVICE Slice() : start(0), stop(-1), step(1) {}
  NUMERIC_HOST_DEVICE explicit Slice(dim_t start_, dim_t stop_, dim_t step_ = 1)
      : start(start_), stop(stop_), step(step_) {}
  Slice(const Slice &) = default;
  Slice &operator=(const Slice &) = default;
};

#ifndef __HIP_DEVICE_COMPILE__
std::ostream &operator<<(std::ostream &os, const Slice &slice);
#endif

} // namespace numeric::memory

#endif
