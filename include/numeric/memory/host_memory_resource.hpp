#ifndef NUMERIC_MEMORY_HOST_MEMORY_RESOURCE_HPP_
#define NUMERIC_MEMORY_HOST_MEMORY_RESOURCE_HPP_

#include <cstdlib>
#include <numeric/memory/memory_resource.hpp>

namespace numeric::memory {

/**
 * @brief Memory resource for allocating memory on the host.
 *
 * This class allocates memory on the host using std::malloc for allocation
 * and std::free for deallocation.
 *
 * @tparam T The type of elements to allocate.
 */
template <typename T> class HostMemoryResource : public MemoryResource<T> {
  using super = MemoryResource<T>;

public:
  using value_type = typename super::value_type;
  using size_type = typename super::size_type;
  using pointer = typename super::pointer;

  HostMemoryResource() : super(MemoryType::HOST) {}
  HostMemoryResource(const HostMemoryResource &) = default;
  HostMemoryResource(HostMemoryResource &&) = default;
  HostMemoryResource &operator=(const HostMemoryResource &) = default;
  HostMemoryResource &operator=(HostMemoryResource &&) = default;
  virtual ~HostMemoryResource() override = default;

  using super::memory_type;

protected:
  /**
   * @brief Allocates memory on the host.
   *
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated memory.
   */
  virtual pointer do_allocate(size_type n) override {
    return static_cast<T *>(malloc(n * sizeof(T)));
  }

  /**
   * @brief Deallocates memory on the host.
   *
   * @param p Pointer to the memory to deallocate.
   * @param n The number of elements previously allocated (unused).
   */
  virtual void do_deallocate(pointer p, size_type /*n*/) override { free(p); }
};

} // namespace numeric::memory

#endif
