#ifndef NUMERIC_MEMORY_ALLOCATOR_HPP_
#define NUMERIC_MEMORY_ALLOCATOR_HPP_

#include <memory>
#include <numeric/memory/memory_type.hpp>
#include <numeric/memory/memory_resource.hpp>
#include <numeric/memory/memory_resource_factory.hpp>

namespace numeric::memory {

template <typename T> class Allocator {
  using resource_t = MemoryResource<T>;

public:
  using pointer = typename resource_t::pointer;
  using value_type = typename resource_t::value_type;
  using size_type = typename resource_t::size_type;
  template <typename U> using rebind = Allocator<U>;

  explicit Allocator(MemoryType mem_type) : Allocator(make_memory_resource<T>(mem_type)) { }
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

  [[nodiscard]] pointer allocate(size_type n) { return resource_->allocate(n); }
  void deallocate(pointer p, size_type n) { resource_->deallocate(p, n); }

  [[nodiscard]] MemoryType memory_type() const noexcept { return resource_->memory_type(); }

private:
  std::shared_ptr<resource_t> resource_;
};

}

#endif
