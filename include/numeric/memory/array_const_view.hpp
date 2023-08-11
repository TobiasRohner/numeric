#ifndef NUMERIC_MEMORY_ARRAY_CONST_VIEW_HPP_
#define NUMERIC_MEMORY_ARRAY_CONST_VIEW_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/layout.hpp>
#include <numeric/memory/memory_type.hpp>

namespace numeric::memory {

template <typename Scalar, dim_t N> class ArrayConstView {
public:
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;

  NUMERIC_HOST_DEVICE ArrayConstView()
      : data_(nullptr), layout_(), memory_type_(MemoryType::UNKNOWN) {}
  NUMERIC_HOST_DEVICE
  ArrayConstView(const scalar_t *data, const Layout<dim> &layout,
                 MemoryType memory_type = MemoryType::UNKNOWN)
      : data_(data), layout_(layout), memory_type_(memory_type) {}
  NUMERIC_HOST_DEVICE ArrayConstView(const ArrayConstView &) = default;
  NUMERIC_HOST_DEVICE ArrayConstView(ArrayConstView &&) = default;
  NUMERIC_HOST_DEVICE ArrayConstView &
  operator=(const ArrayConstView &) = default;
  NUMERIC_HOST_DEVICE ArrayConstView &operator=(ArrayConstView &&) = default;

  NUMERIC_HOST_DEVICE [[nodiscard]] MemoryType memory_type() const noexcept {
    return memory_type_;
  }

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] const scalar_t &
  operator()(Idxs... idxs) const noexcept {
    return data_[memory_index(idxs...)];
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] const scalar_t *raw() const noexcept {
    return data_;
  }
  NUMERIC_HOST_DEVICE [[nodiscard]] const Layout<dim> &layout() const noexcept {
    return layout_;
  }
  NUMERIC_HOST_DEVICE [[nodiscard]] const dim_t *shape() const noexcept {
    return layout_.shape();
  }
  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t shape(size_t idx) const noexcept {
    return layout_.shape(idx);
  }
  NUMERIC_HOST_DEVICE [[nodiscard]] const dim_t *stride() const noexcept {
    return layout_.stride();
  }
  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t stride(size_t idx) const noexcept {
    return layout_.stride(idx);
  }
  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t size() const noexcept {
    return layout_.size();
  }

  template <dim_t M>
  NUMERIC_HOST_DEVICE [[nodiscard]] ArrayConstView<scalar_t, M>
  broadcast(const Layout<M> &layout) const noexcept {
    const Layout<M> new_layout = broadcasted_layout(layout);
    return ArrayConstView<scalar_t, M>(raw(), new_layout, memory_type());
  }

protected:
  const scalar_t *data_;
  Layout<dim> layout_;
  MemoryType memory_type_;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] size_t
  memory_index(Idxs... idxs) const noexcept {
    size_t stride_idx = 0;
    return (... + (idxs * stride(stride_idx++)));
  }

  template <dim_t M>
  NUMERIC_HOST_DEVICE [[nodiscard]] Layout<M>
  broadcasted_shape(Layout<M> layout) const noexcept {
    static_assert(M >= N, "Broadcasted shape needs to have at least as many "
                          "dimensions as original array");
    for (dim_t d = 0; d < M - N; ++d) {
      layout.stride(d) = 0;
    }
    for (dim_t d = 0; d < N; ++d) {
      if (layout_.shape(d) == 1) {
        layout.stride(M - N + d) = 0;
      } else {
        layout.stride(M - N + d) = layout_.stride(d);
      }
    }
    return layout;
  }
};

} // namespace numeric::memory

#endif
