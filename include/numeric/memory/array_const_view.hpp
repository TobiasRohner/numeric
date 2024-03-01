#ifndef NUMERIC_MEMORY_ARRAY_CONST_VIEW_HPP_
#define NUMERIC_MEMORY_ARRAY_CONST_VIEW_HPP_

#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_const_view_decl.hpp>
#include <numeric/memory/slice.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::memory {

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE ArrayConstView<Scalar, N>::ArrayConstView()
    : data_(nullptr), layout_(), memory_type_(MemoryType::UNKNOWN) {}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE ArrayConstView<Scalar, N>::ArrayConstView(
    const Scalar *data, const Layout<dim> &layout, MemoryType memory_type)
    : data_(data), layout_(layout), memory_type_(memory_type) {}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE void
ArrayConstView<Scalar, N>::set(const scalar_t *data, const Layout<dim> &layout,
                               MemoryType memory_type) noexcept {
  data_ = data;
  layout_ = layout;
  memory_type_ = memory_type;
}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE [[nodiscard]] MemoryType
ArrayConstView<Scalar, N>::memory_type() const noexcept {
  return memory_type_;
}

template <typename Scalar, dim_t N>
template <typename... Idxs>
NUMERIC_HOST_DEVICE [[nodiscard]] decltype(auto)
ArrayConstView<Scalar, N>::operator()(Idxs... idxs) const noexcept {
  if constexpr (sizeof...(idxs) == dim &&
                (!meta::is_same_v<Idxs, Slice> && ...)) {
    return data_[memory_index(idxs...)];
  } else {
    return sub_view(*this, 0, idxs...);
  }
}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE [[nodiscard]] const Scalar *
ArrayConstView<Scalar, N>::raw() const noexcept {
  return data_;
}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE [[nodiscard]] const Layout<ArrayConstView<Scalar, N>::dim> &
ArrayConstView<Scalar, N>::layout() const noexcept {
  return layout_;
}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE [[nodiscard]] const Shape<N> &
ArrayConstView<Scalar, N>::shape() const noexcept {
  return layout_.shape();
}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE [[nodiscard]] dim_t
ArrayConstView<Scalar, N>::shape(size_t idx) const noexcept {
  return layout_.shape(idx);
}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE [[nodiscard]] const Stride<N> &
ArrayConstView<Scalar, N>::stride() const noexcept {
  return layout_.stride();
}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE [[nodiscard]] dim_t
ArrayConstView<Scalar, N>::stride(size_t idx) const noexcept {
  return layout_.stride(idx);
}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE [[nodiscard]] dim_t
ArrayConstView<Scalar, N>::size() const noexcept {
  return layout_.size();
}

template <typename Scalar, dim_t N>
template <dim_t M>
NUMERIC_HOST_DEVICE [[nodiscard]] ArrayConstView<Scalar, M>
ArrayConstView<Scalar, N>::broadcast(const Shape<M> &shape) const noexcept {
  const Layout<M> new_layout = broadcasted_layout(layout_, shape);
  return ArrayConstView<Scalar, M>(raw(), new_layout, memory_type());
}

#if NUMERIC_ENABLE_EIGEN
template <typename Scalar, dim_t N>
[[nodiscard]] Eigen::Map<
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, 0,
    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
ArrayConstView<Scalar, N>::matrix_view() const noexcept {
  static_assert(dim == 2);
  return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                    0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(
      raw(), shape(0), shape(1), Eigen::Stride(stride(0), stride(1)));
}
#endif

template <typename Scalar, dim_t N>
template <typename... Idxs>
NUMERIC_HOST_DEVICE size_t
ArrayConstView<Scalar, N>::memory_index(Idxs... idxs) const noexcept {
  size_t stride_idx = 0;
  dim_t idx = 0;
  (..., (idx += idxs * stride(stride_idx++)));
  return idx;
}

template <typename Scalar, dim_t N>
template <dim_t M, typename Idx, typename... Idxs>
NUMERIC_HOST_DEVICE decltype(auto)
ArrayConstView<Scalar, N>::sub_view(const ArrayConstView<Scalar, M> &view,
                                    dim_t d, Idx idx, Idxs... idxs) noexcept {
  if constexpr (meta::is_same_v<Idx, Slice>) {
    if (idx.stop < 0) {
      idx.stop += view.shape(d) + 1;
    }
    const Scalar *new_data = view.raw() + idx.start * view.stride(d);
    Layout<M> new_layout;
    for (dim_t i = 0; i < M; ++i) {
      new_layout.shape(i) = view.shape(i);
      new_layout.stride(i) = view.stride(i);
    }
    new_layout.shape(d) = (idx.stop - idx.start) / idx.step;
    new_layout.stride(d) = view.stride(d) * idx.step;
    return sub_view(
        ArrayConstView<Scalar, M>(new_data, new_layout, view.memory_type()),
        d + 1, idxs...);
  } else {
    const Scalar *new_data = view.raw() + idx * view.stride(d);
    Layout<M - 1> new_layout;
    for (dim_t i = 0; i < M - 1; ++i) {
      new_layout.shape(i) = view.shape(i + (i < d ? 0 : 1));
      new_layout.stride(i) = view.stride(i + (i < d ? 0 : 1));
    }
    return sub_view(
        ArrayConstView<Scalar, M - 1>(new_data, new_layout, view.memory_type()),
        d, idxs...);
  }
}

template <typename Scalar, dim_t N>
template <dim_t M>
NUMERIC_HOST_DEVICE ArrayConstView<Scalar, M>
ArrayConstView<Scalar, N>::sub_view(const ArrayConstView<Scalar, M> &view,
                                    dim_t) noexcept {
  return view;
}

} // namespace numeric::memory

#endif
