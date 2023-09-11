#ifndef NUMERIC_MEMORY_SLICE_HPP_
#define NUMERIC_MEMORY_SLICE_HPP_

#include <iostream>
#include <numeric/config.hpp>

namespace numeric::memory {

struct Slice {
  dim_t start;
  dim_t stop;
  dim_t step;

  Slice() : start(0), stop(-1), step(1) {}
  explicit Slice(dim_t start_, dim_t stop_, dim_t step_ = 1)
      : start(start_), stop(stop_), step(step_) {}
  Slice(const Slice &) = default;
  Slice &operator=(const Slice &) = default;
};

std::ostream &operator<<(std::ostream &os, const Slice &slice);

} // namespace numeric::memory

#endif
