#ifndef NUMERIC_MEMORY_MEMORY_TYPE_HPP_
#define NUMERIC_MEMORY_MEMORY_TYPE_HPP_

#ifndef __HIP_DEVICE_COMPILE__
#include <string_view>
#include <type_traits>
#endif

namespace numeric::memory {

enum struct MemoryType {
  HOST = 1,
  DEVICE = 2,
  UNKNOWN = (1 << 2),
  PINNED = (2 << 2) | HOST,
};

#ifndef __HIP_DEVICE_COMPILE__
[[nodiscard]] inline constexpr bool is_host_accessible(MemoryType t) noexcept {
  using int_t = std::underlying_type_t<MemoryType>;
  return static_cast<int_t>(t) & static_cast<int_t>(MemoryType::HOST);
}

[[nodiscard]] inline constexpr bool
is_device_accessible(MemoryType t) noexcept {
  using int_t = std::underlying_type_t<MemoryType>;
  return static_cast<int_t>(t) & static_cast<int_t>(MemoryType::DEVICE);
}

[[nodiscard]] std::string_view to_string(MemoryType t) noexcept;
#endif

} // namespace numeric::memory

#endif
