#ifndef NUMERIC_MEMORY_MEMORY_RESOURCE_HPP_
#define NUMERIC_MEMORY_MEMORY_RESOURCE_HPP_

#include <numeric/memory/memory_type.hpp>

namespace numeric::memory {

template <typename T> class MemoryResource {
public:
  using value_type = T;
  using size_type = size_t;
  using pointer = T *;

  MemoryResource(MemoryType mem_type) : mem_type_(mem_type) {}
  MemoryResource(const MemoryResource &) = default;
  MemoryResource(MemoryResource &&) = default;
  MemoryResource &operator=(const MemoryResource &) = default;
  MemoryResource &operator=(MemoryResource &&) = default;
  virtual ~MemoryResource() = default;

  [[nodiscard]] MemoryType memory_type() const noexcept { return mem_type_; }

  [[nodiscard]] pointer allocate(size_type n) { return do_allocate(n); }
  void deallocate(pointer p, size_type n) { do_deallocate(p, n); }

protected:
  virtual pointer do_allocate(size_type n) = 0;
  virtual void do_deallocate(pointer p, size_type n) = 0;

private:
  MemoryType mem_type_;
};

} // namespace numeric::memory

#endif
