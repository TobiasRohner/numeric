#ifndef NUMERIC_MEMORY_ARRAY_VIEW_HPP_
#define NUMERIC_MEMORY_ARRAY_VIEW_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/copy.hpp>

namespace numeric::memory {

template <typename Scalar, dim_t N>
class ArrayView : public ArrayConstView<Scalar, N> {
  using super = ArrayConstView<Scalar, N>;

public:
  using scalar_t = Scalar;
  static constexpr dim_t dim = N;

  NUMERIC_HOST_DEVICE ArrayView() : super(nullptr, {}, MemoryType::UNKNOWN) {}
  NUMERIC_HOST_DEVICE ArrayView(scalar_t *data, const Layout<dim> &layout,
                                MemoryType memory_type = MemoryType::UNKNOWN)
      : super(data, layout, memory_type) {}
  NUMERIC_HOST_DEVICE ArrayView(const ArrayView &) = default;
  NUMERIC_HOST_DEVICE ArrayView(ArrayView &&) = default;

  template <typename Arg>
  NUMERIC_HOST_DEVICE ArrayView &operator=(const ArrayBase<Arg> &other) {
    copy(*this, other);
    return *this;
  }

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE [[nodiscard]] decltype(auto)
  operator()(Idxs... idxs) noexcept {
    if constexpr ((!meta::is_same_v<Idxs, Slice> && ...)) {
      return raw()[memory_index(idxs...)];
    } else {
      return sub_view(*this, 0, idxs...);
    }
  }

  template <dim_t M>
  NUMERIC_HOST_DEVICE [[nodiscard]] ArrayView<scalar_t, M>
  broadcast(const Layout<M> &layout) noexcept {
    const Layout<M> new_layout = broadcasted_layout(layout_, layout);
    return ArrayView<scalar_t, M>(raw(), new_layout, memory_type());
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] ArrayConstView<Scalar, N>
  const_view() const noexcept {
    return *this;
  }

  using super::memory_type;
  using super::operator();
  using super::raw;
  NUMERIC_HOST_DEVICE [[nodiscard]] scalar_t *raw() noexcept {
    return const_cast<scalar_t *>(data_);
  }
  using super::broadcast;
  using super::layout;
  using super::shape;
  using super::size;
  using super::stride;

protected:
  using super::data_;
  using super::layout_;
  using super::memory_type_;

  using super::broadcasted_layout;
  using super::memory_index;

  template <dim_t M, typename Idx, typename... Idxs>
  static NUMERIC_HOST_DEVICE [[nodiscard]] decltype(auto)
  sub_view(ArrayView<scalar_t, M> view, dim_t d, Idx idx,
           Idxs... idxs) noexcept {
    if constexpr (meta::is_same_v<Idx, Slice>) {
      if (idx.stop < 0) {
        idx.stop += view.shape(d) + 1;
      }
      scalar_t *new_data = view.raw() + idx.start * view.stride(d);
      Layout<M> new_layout;
      for (dim_t i = 0; i < M; ++i) {
        new_layout.shape(i) = view.shape(i);
        new_layout.stride(i) = view.stride(i);
      }
      new_layout.shape(d) = (idx.stop - idx.start) / idx.step;
      new_layout.stride(d) = view.stride(d) * idx.step;
      return sub_view(
          ArrayView<scalar_t, M>(new_data, new_layout, view.memory_type()),
          d + 1, idxs...);
    } else {
      scalar_t *new_data = view.raw() + idx * view.stride(d);
      Layout<M - 1> new_layout;
      for (dim_t i = 0; i < M - 1; ++i) {
        new_layout.shape(i) = view.shape(i + (i < d ? 0 : 1));
        new_layout.stride(i) = view.stride(i + (i < d ? 0 : 1));
      }
      return sub_view(
          ArrayView<scalar_t, M - 1>(new_data, new_layout, view.memory_type()),
          d, idxs...);
    }
  }

  template <dim_t M>
  static NUMERIC_HOST_DEVICE [[nodiscard]] ArrayView<scalar_t, M>
  sub_view(ArrayView<scalar_t, M> view, dim_t) noexcept {
    return view;
  }
};

} // namespace numeric::memory

#endif
