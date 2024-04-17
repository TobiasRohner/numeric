#ifndef NUMERIC_MEMORY_DEVICE_MEMORY_RESOURCE_HPP_
#define NUMERIC_MEMORY_DEVICE_MEMORY_RESOURCE_HPP_

#include <cstdlib>
#include <numeric/hip/device.hpp>
#include <numeric/hip/runtime.hpp>
#include <numeric/hip/safe_call.hpp>
#include <numeric/memory/memory_resource.hpp>

namespace numeric::memory {

/**
 * @brief Memory resource for allocating memory on a device.
 *
 * This class allocates memory on a device using hipMalloc for allocation
 * and hipFree for deallocation.
 *
 * @tparam T The type of elements to allocate.
 */
template <typename T> class DeviceMemoryResource : public MemoryResource<T> {
  using super = MemoryResource<T>;

public:
  using value_type = typename super::value_type;
  using size_type = typename super::size_type;
  using pointer = typename super::pointer;

  /**
   * @brief Constructs a DeviceMemoryResource for the default device.
   */
  DeviceMemoryResource() : DeviceMemoryResource(hip::Device()) {}

  /**
   * @brief Constructs a DeviceMemoryResource for the specified device.
   *
   * @param device The HIP device to allocate memory on.
   */
  DeviceMemoryResource(const hip::Device &device)
      : super(MemoryType::DEVICE), device_(device) {}

  DeviceMemoryResource(const DeviceMemoryResource &) = default;
  DeviceMemoryResource(DeviceMemoryResource &&) = default;
  DeviceMemoryResource &operator=(const DeviceMemoryResource &) = default;
  DeviceMemoryResource &operator=(DeviceMemoryResource &&) = default;
  virtual ~DeviceMemoryResource() override = default;

  using super::memory_type;

protected:
  /**
   * @brief Allocates memory on the device.
   *
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated memory.
   */
  virtual pointer do_allocate(size_type n) override {
    pointer ptr;
    device_.do_while_active(
        [&]() { NUMERIC_CHECK_HIP(hipMalloc(&ptr, n * sizeof(T))); });
    return ptr;
  }

  /**
   * @brief Deallocates memory on the device.
   *
   * @param p Pointer to the memory to deallocate.
   * @param n The number of elements previously allocated (unused).
   */
  virtual void do_deallocate(pointer p, size_type /*n*/) override {
    device_.do_while_active([&]() { NUMERIC_CHECK_HIP(hipFree(p)); });
  }

private:
  hip::Device device_;
};

} // namespace numeric::memory

#endif
