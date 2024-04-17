#ifndef NUMERIC_MEMORY_PINNED_MEMORY_RESOURCE_HPP_
#define NUMERIC_MEMORY_PINNED_MEMORY_RESOURCE_HPP_

#include <cstdlib>
#include <numeric/hip/device.hpp>
#include <numeric/hip/runtime.hpp>
#include <numeric/hip/safe_call.hpp>
#include <numeric/memory/memory_resource.hpp>

namespace numeric::memory {

/**
 * @brief Memory resource for allocating pinned memory.
 *
 * This class allocates pinned memory using hipHostMalloc for allocation
 * and hipHostFree for deallocation.
 *
 * @tparam T The type of elements to allocate.
 */
template <typename T> class PinnedMemoryResource : public MemoryResource<T> {
  using super = MemoryResource<T>;

public:
  using value_type = typename super::value_type;
  using size_type = typename super::size_type;
  using pointer = typename super::pointer;

  /**
   * @brief Constructs a PinnedMemoryResource for the default device.
   */
  PinnedMemoryResource() : PinnedMemoryResource(hip::Device()) {}

  /**
   * @brief Constructs a PinnedMemoryResource for the specified device.
   *
   * @param device The HIP device to allocate pinned memory on.
   */
  PinnedMemoryResource(const hip::Device &device)
      : super(MemoryType::PINNED), device_(device) {}

  PinnedMemoryResource(const PinnedMemoryResource &) = default;
  PinnedMemoryResource(PinnedMemoryResource &&) = default;
  PinnedMemoryResource &operator=(const PinnedMemoryResource &) = default;
  PinnedMemoryResource &operator=(PinnedMemoryResource &&) = default;
  virtual ~PinnedMemoryResource() override = default;

  using super::memory_type;

protected:
  /**
   * @brief Allocates pinned memory.
   *
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated memory.
   */
  virtual pointer do_allocate(size_type n) override {
    pointer ptr;
    device_.do_while_active(
        [&]() { NUMERIC_CHECK_HIP(hipHostMalloc(&ptr, n * sizeof(T), 0)); });
    return ptr;
  }

  /**
   * @brief Deallocates pinned memory.
   *
   * @param p Pointer to the memory to deallocate.
   * @param n The number of elements previously allocated (unused).
   */
  virtual void do_deallocate(pointer p, size_type /*n*/) override {
    device_.do_while_active([&]() { NUMERIC_CHECK_HIP(hipHostFree(p)); });
  }

private:
  hip::Device device_;
};

} // namespace numeric::memory

#endif
