#ifndef NUMERIC_MEMORY_MEMORY_RESOURCE_HPP_
#define NUMERIC_MEMORY_MEMORY_RESOURCE_HPP_

#include <numeric/memory/memory_type.hpp>

namespace numeric::memory {

/**
 * @brief Base class for memory resources used by Allocator.
 *
 * This class provides an interface for allocating and deallocating memory.
 *
 * @tparam T The type of elements to allocate.
 */
template <typename T> class MemoryResource {
public:
  using value_type = T;
  using size_type = size_t;
  using pointer = T *;

  /**
   * @brief Constructs a MemoryResource with the specified memory type.
   *
   * @param mem_type The memory type associated with this resource.
   */
  MemoryResource(MemoryType mem_type) : mem_type_(mem_type) {}
  MemoryResource(const MemoryResource &) = default;
  MemoryResource(MemoryResource &&) = default;
  MemoryResource &operator=(const MemoryResource &) = default;
  MemoryResource &operator=(MemoryResource &&) = default;
  virtual ~MemoryResource() = default;

  /**
   * @brief Gets the memory type associated with this resource.
   *
   * @return The memory type.
   */
  MemoryType memory_type() const noexcept { return mem_type_; }

  /**
   * @brief Allocates memory for an array of n elements.
   *
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated memory.
   */
  [[nodiscard]] pointer allocate(size_type n) { return do_allocate(n); }

  /**
   * @brief Deallocates memory previously allocated by allocate.
   *
   * @param p Pointer to the memory to deallocate.
   * @param n The number of elements previously allocated.
   */
  void deallocate(pointer p, size_type n) { do_deallocate(p, n); }

protected:
  /**
   * @brief Performs the allocation of memory.
   *
   * This method should be overridden by derived classes to provide custom
   * allocation behavior.
   *
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated memory.
   */
  virtual pointer do_allocate(size_type n) = 0;

  /**
   * @brief Performs the deallocation of memory.
   *
   * This method should be overridden by derived classes to provide custom
   * deallocation behavior.
   *
   * @param p Pointer to the memory to deallocate.
   * @param n The number of elements previously allocated.
   */
  virtual void do_deallocate(pointer p, size_type n) = 0;

private:
  MemoryType mem_type_;
};

} // namespace numeric::memory

#endif
