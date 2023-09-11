#include <numeric/memory/slice.hpp>

namespace numeric::memory {

std::ostream &operator<<(std::ostream &os, const Slice &slice) {
  return os << slice.start << ':' << slice.stop << ':' << slice.step;
}

} // namespace numeric::memory
