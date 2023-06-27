#ifndef NUMERIC_MEMORY_MEMORY_RESOURCE_FACTORY_HPP_
#define NUMERIC_MEMORY_MEMORY_RESOURCE_FACTORY_HPP_

#include <memory>
#include <numeric/memory/memory_type.hpp>
#include <numeric/memory/memory_resource.hpp>
#include <numeric/memory/host_memory_resource.hpp>
#if NUMERIC_ENABLE_HIP
#include <numeric/memory/device_memory_resource.hpp>
#include <numeric/memory/pinned_memory_resource.hpp>
#endif


namespace numeric::memory {

template <typename T>
[[nodiscard]] std::shared_ptr<MemoryResource<T>> make_memory_resource(MemoryType mem_type) noexcept {
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

}


#endif
