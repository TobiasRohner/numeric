#ifndef NUMERIC_MEMORY_MEMORY_TYPE_HPP_
#define NUMERIC_MEMORY_MEMORY_TYPE_HPP_

#ifndef __HIP_DEVICE_COMPILE__
#include <string_view>
#include <type_traits>
#endif

namespace numeric::memory {

/**
 * @brief Enumeration representing different memory types.
 */
enum struct MemoryType {
  HOST = 1,                 /**< Host memory */
  DEVICE = 2,               /**< Device memory */
  UNKNOWN = (1 << 2),       /**< Unknown memory type */
  PINNED = (2 << 2) | HOST, /**< Pinned memory */
};

#ifndef __HIP_DEVICE_COMPILE__
/**
 * @brief Checks if the memory type allows access from the host.
 *
 * @param t The memory type.
 * @return true If the memory type allows access from the host.
 * @return false Otherwise.
 */
inline constexpr bool is_host_accessible(MemoryType t) noexcept {
  using int_t = std::underlying_type_t<MemoryType>;
  return static_cast<int_t>(t) & static_cast<int_t>(MemoryType::HOST);
}

/**
 * @brief Checks if the memory type allows access from the device.
 *
 * @param t The memory type.
 * @return true If the memory type allows access from the device.
 * @return false Otherwise.
 */
inline constexpr bool is_device_accessible(MemoryType t) noexcept {
  using int_t = std::underlying_type_t<MemoryType>;
  return static_cast<int_t>(t) & static_cast<int_t>(MemoryType::DEVICE);
}

/**
 * @brief Converts a memory type to a string representation.
 *
 * @param t The memory type.
 * @return std::string_view The string representation of the memory type.
 */
std::string_view to_string(MemoryType t) noexcept;
#endif

} // namespace numeric::memory

#endif
