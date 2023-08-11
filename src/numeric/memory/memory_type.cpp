#include <numeric/memory/memory_type.hpp>

namespace numeric::memory {

std::string_view to_string(MemoryType t) noexcept {
  switch (t) {
  case MemoryType::HOST:
    return "HOST";
  case MemoryType::DEVICE:
    return "DEVICE";
  case MemoryType::UNKNOWN:
    return "UNKNOWN";
  case MemoryType::PINNED:
    return "PINNED";
  default:
    return "";
  }
}

} // namespace numeric::memory
