#ifndef NUMERIC_MEMORY_PINNED_MEMORY_RESOURCE_HPP_
#define NUMERIC_MEMORY_PINNED_MEMORY_RESOURCE_HPP_

#include <cstdlib>
#include <numeric/memory/memory_resource.hpp>
#include <numeric/hip/safe_call.hpp>
#include <numeric/hip/device.hpp>
#include <hip/hip_runtime_api.h>

namespace numeric::memory {

template <typename T> class PinnedMemoryResource : public MemoryResource<T> {
  using super = MemoryResource<T>;

public:
  using value_type = typename super::value_type;
  using size_type = typename super::size_type;
  using pointer = typename super::pointer;

  PinnedMemoryResource() : PinnedMemoryResource(hip::Device()) {}
  PinnedMemoryResource(const hip::Device &device) : super(MemoryType::PINNED), device_(device) {}
  PinnedMemoryResource(const PinnedMemoryResource &) = default;
  PinnedMemoryResource(PinnedMemoryResource &&) = default;
  PinnedMemoryResource &operator=(const PinnedMemoryResource &) = default;
  PinnedMemoryResource &operator=(PinnedMemoryResource &&) = default;
  virtual ~PinnedMemoryResource() override = default;

  using super::memory_type;

protected:
  virtual pointer do_allocate(size_type n) override {
    pointer ptr;
    device_.do_while_active([&]() {
      NUMERIC_CHECK_HIP(hipHostMalloc(&ptr, n * sizeof(T), 0));
    });
    return ptr;
  }
  virtual void do_deallocate(pointer p, size_type /*n*/) override {
    device_.do_while_active([&]() {
      NUMERIC_CHECK_HIP(hipHostFree(p));
    });
  }

private:
  hip::Device device_;
};

}

#endif
