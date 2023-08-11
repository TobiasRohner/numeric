#ifndef NUMERIC_MEMORY_ARRAY_HPP_
#define NUMERIC_MEMORY_ARRAY_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_view.hpp>
#include <numeric/memory/allocator.hpp>


namespace numeric::memory {

template<typename Scalar, dim_t N>
class Array : public ArrayView<Scalar, N> {
  using super = ArrayView<Scalar, N>;

public:
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;

  Array(): super(nullptr, {}, MemoryType::UNKNOWN), alloc_(nullptr) {}

  Array(const Layout<dim> &layout, Allocator<scalar_t> alloc)
    : super(alloc.allocate(layout.size()), layout, alloc.memory_type()),
    alloc_(alloc) {
  }
  explicit Array(const Layout<dim> &layout, MemoryType mem_type = MemoryType::HOST)
    : Array(layout, Allocator<scalar_t>(mem_type)) {
  }
  Array(const Array &) = delete;
  Array &operator=(const Array &) = delete;
  Array(Array &&other) : super(other.data_, other.layout_, other.memory_type_), alloc_(std::move(other.alloc_)) {
    other.data_ = nullptr;
  }
  Array &operator=(Array &&other) {
    super::operator=(std::move(other));
    alloc_ = other.alloc_;
    other.data_ = nullptr;
    return *this;
  }

  ~Array() {
    if (raw()) {
      alloc_.deallocate(raw(), size());
    }
  }

  [[nodiscard]] ArrayView<Scalar, N> view() noexcept { return *this; }

  using super::memory_type;
  using super::const_view;
  using super::operator();
  using super::raw;
  using super::layout;
  using super::shape;
  using super::stride;
  using super::size;

protected:
  Allocator<scalar_t> alloc_;
  using super::data_;
  using super::layout_;
  using super::memory_type_;

  using super::memory_index;
};

}


#endif
