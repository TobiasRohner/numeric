#ifndef NUMERIC_MEMORY_MEMORY_RESOURCE_FACTORY_HPP_
#define NUMERIC_MEMORY_MEMORY_RESOURCE_FACTORY_HPP_

#include <memory>
#include <numeric/memory/host_memory_resource.hpp>
#include <numeric/memory/memory_resource.hpp>
#include <numeric/memory/memory_type.hpp>
#if NUMERIC_ENABLE_HIP
#include <numeric/memory/device_memory_resource.hpp>
#include <numeric/memory/pinned_memory_resource.hpp>
#endif

namespace numeric::memory {

/**
 * @brief Factory function to create a memory resource based on the specified
 * memory type.
 *
 * This function creates and returns a shared pointer to a memory resource of
 * type `MemoryResource<T>`. The memory type is specified by the enum
 * `MemoryType`.
 *
 * @tparam T The type of data handled by the memory resource.
 * @param mem_type The memory type.
 * @return std::shared_ptr<MemoryResource<T>> A shared pointer to the created
 * memory resource.
 */
template <typename T>
[[nodiscard]] std::shared_ptr<MemoryResource<T>>
make_memory_resource(MemoryType mem_type) noexcept {
  switch (mem_type) {
  case MemoryType::HOST:
    return std::make_shared<HostMemoryResource<T>>();
#if NUMERIC_ENABLE_HIP
  case MemoryType::DEVICE:
    return std::make_shared<DeviceMemoryResource<T>>();
  case MemoryType::PINNED:
    return std::make_shared<PinnedMemoryResource<T>>();
#endif
  default:
    return nullptr;
  }
}

} // namespace numeric::memory

#endif
