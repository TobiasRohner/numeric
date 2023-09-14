#ifndef NUMERIC_MEMORY_ARRAY_CONST_VIEW_HPP_
#define NUMERIC_MEMORY_ARRAY_CONST_VIEW_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_base.hpp>
#include <numeric/memory/layout.hpp>
#include <numeric/memory/memory_type.hpp>
#include <numeric/memory/slice.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::memory {

template <typename Scalar, dim_t N>
class ArrayConstView : public ArrayBase<ArrayConstView<Scalar, N>> {
  using super = ArrayBase<ArrayConstView<Scalar, N>>;

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
  operator=(const ArrayConstView &) = delete;
  NUMERIC_HOST_DEVICE ArrayConstView &operator=(ArrayConstView &&) = delete;

  NUMERIC_HOST_DEVICE [[nodiscard]] MemoryType memory_type() const noexcept {
    return memory_type_;
  }

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] decltype(auto)
  operator()(Idxs... idxs) const noexcept {
    if constexpr ((!meta::is_same_v<Idxs, Slice> && ...)) {
      return data_[memory_index(idxs...)];
    } else {
      return sub_view(*this, 0, idxs...);
    }
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
    const Layout<M> new_layout = broadcasted_layout(layout_, layout);
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
    dim_t idx = 0;
    (..., (idx += idxs * stride(stride_idx++)));
    return idx;
  }

  template <dim_t M, typename Idx, typename... Idxs>
  static NUMERIC_HOST_DEVICE [[nodiscard]] decltype(auto)
  sub_view(const ArrayConstView<scalar_t, M> &view, dim_t d, Idx idx,
           Idxs... idxs) noexcept {
    if constexpr (meta::is_same_v<Idx, Slice>) {
      if (idx.stop < 0) {
        idx.stop += view.shape(d) + 1;
      }
      const scalar_t *new_data = view.raw() + idx.start * view.stride(d);
      Layout<M> new_layout;
      for (dim_t i = 0; i < M; ++i) {
        new_layout.shape(i) = view.shape(i);
        new_layout.stride(i) = view.stride(i);
      }
      new_layout.shape(d) = (idx.stop - idx.start) / idx.step;
      new_layout.stride(d) = view.stride(d) * idx.step;
      return sub_view(
          ArrayConstView<scalar_t, M>(new_data, new_layout, view.memory_type()),
          d + 1, idxs...);
    } else {
      const scalar_t *new_data = view.raw() + idx * view.stride(d);
      Layout<M - 1> new_layout;
      for (dim_t i = 0; i < M - 1; ++i) {
        new_layout.shape(i) = view.shape(i + (i < d ? 0 : 1));
        new_layout.stride(i) = view.stride(i + (i < d ? 0 : 1));
      }
      return sub_view(ArrayConstView<scalar_t, M - 1>(new_data, new_layout,
                                                      view.memory_type()),
                      d, idxs...);
    }
  }

  template <dim_t M>
  static NUMERIC_HOST_DEVICE [[nodiscard]] decltype(auto)
  sub_view(const ArrayConstView<scalar_t, M> &view, dim_t) noexcept {
    return view;
  }

  using super::broadcasted_layout;
};

} // namespace numeric::memory

#endif
