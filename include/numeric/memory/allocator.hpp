#ifndef NUMERIC_MEMORY_ALLOCATOR_HPP_
#define NUMERIC_MEMORY_ALLOCATOR_HPP_

#include <memory>
#include <numeric/memory/memory_resource.hpp>
#include <numeric/memory/memory_resource_factory.hpp>
#include <numeric/memory/memory_type.hpp>

namespace numeric::memory {

/**
 * @brief Allocator class for managing memory allocation and deallocation.
 *
 * This allocator utilizes a MemoryResource for allocation and deallocation.
 *
 * @tparam T The type of elements to allocate.
 */
template <typename T> class Allocator {
  using resource_t = MemoryResource<T>;

public:
  using pointer = typename resource_t::pointer;
  using value_type = typename resource_t::value_type;
  using size_type = typename resource_t::size_type;
  template <typename U> using rebind = Allocator<U>;

  /**
   * @brief Constructs an Allocator with the specified memory type.
   *
   * @param mem_type The memory type to use for allocation.
   */
  explicit Allocator(MemoryType mem_type)
      : Allocator(make_memory_resource<T>(mem_type)) {}

  /**
   * @brief Constructs an Allocator with the specified memory resource.
   *
   * @param resource The shared pointer to the memory resource.
   */
  explicit Allocator(const std::shared_ptr<resource_t> &resource)
      : resource_(resource) {}

  Allocator(const Allocator &) = default;
  Allocator(Allocator &&) = default;
  Allocator &operator=(const Allocator &) = default;
  Allocator &operator=(Allocator &&) = default;
  ~Allocator() = default;

  bool operator==(const Allocator<T> &other) const noexcept {
    return memory_type() == other.memory_type();
  }
  bool operator!=(const Allocator<T> &other) const noexcept {
    return memory_type() != other.memory_type();
  }

  /**
   * @brief Allocates memory for an array of n elements.
   *
   * @param n The number of elements to allocate.
   * @return A pointer to the allocated memory.
   */
  [[nodiscard]] pointer allocate(size_type n) { return resource_->allocate(n); }

  /**
   * @brief Deallocates memory previously allocated by allocate.
   *
   * @param p Pointer to the memory to deallocate.
   * @param n The number of elements previously allocated.
   */
  void deallocate(pointer p, size_type n) { resource_->deallocate(p, n); }

  /**
   * @brief Gets the memory type associated with this allocator.
   *
   * @return The memory type.
   */
  MemoryType memory_type() const noexcept { return resource_->memory_type(); }

private:
  std::shared_ptr<resource_t> resource_;
};

} // namespace numeric::memory

#endif
