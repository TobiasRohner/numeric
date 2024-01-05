#ifndef NUMERIC_MEMORY_ARRAY_VIEW_HPP_
#define NUMERIC_MEMORY_ARRAY_VIEW_HPP_

#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view_decl.hpp>
#include <numeric/memory/slice.hpp>
#include <numeric/meta/meta.hpp>
#ifdef __HIP_DEVICE_COMPILE__
#include <numeric/memory/copy_kernels.hpp>
#else
#include <numeric/memory/copy.hpp>
#endif

namespace numeric::memory {

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE ArrayView<Scalar, N>::ArrayView()
    : super(nullptr, {}, MemoryType::UNKNOWN) {}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE ArrayView<Scalar, N>::ArrayView(Scalar *data,
                                                    const Layout<dim> &layout,
                                                    MemoryType memory_type)
    : super(data, layout, memory_type) {}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE ArrayView<Scalar, N> &
ArrayView<Scalar, N>::operator=(const ArrayView &other) {
  copy(*this, other);
  return *this;
}

template <typename Scalar, dim_t N>
template <typename Src>
NUMERIC_HOST_DEVICE ArrayView<Scalar, N> &
ArrayView<Scalar, N>::operator=(const ArrayBase<Src> &src) {
#ifdef __HIP_DEVICE_COMPILE__
  copy_naive_elm(*this, src.derived());
#else
  copy(*this, src);
#endif
  return *this;
}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE ArrayView<Scalar, N> &
ArrayView<Scalar, N>::operator=(Scalar val) {
#ifdef __HIP_DEVICE_COMPILE__
  static constexpr MemoryType mt = MemoryType::DEVICE;
#else
  static constexpr MemoryType mt = MemoryType::HOST;
#endif
  const ArrayConstView<Scalar, 1> view(&val, Layout<1>(1), mt);
  *this = view;
  return *this;
}

#define NUMERIC_ARRAY_VIEW_DEFINE_ASSIGNMENT(op)                               \
  template <typename Scalar, dim_t N>                                          \
  template <typename Src>                                                      \
  NUMERIC_HOST_DEVICE ArrayView<Scalar, N> &ArrayView<Scalar, N>::operator op( \
      const ArrayBase<Src> &src) {                                             \
    return *this = *this op src;                                               \
  }                                                                            \
  template <typename Scalar, dim_t N>                                          \
  NUMERIC_HOST_DEVICE ArrayView<Scalar, N> &ArrayView<Scalar, N>::operator op( \
      Scalar val) {                                                            \
    return *this = *this op val;                                               \
  }
NUMERIC_ARRAY_VIEW_DEFINE_ASSIGNMENT(+=);
NUMERIC_ARRAY_VIEW_DEFINE_ASSIGNMENT(-=);
NUMERIC_ARRAY_VIEW_DEFINE_ASSIGNMENT(*=);
NUMERIC_ARRAY_VIEW_DEFINE_ASSIGNMENT(/=);
NUMERIC_ARRAY_VIEW_DEFINE_ASSIGNMENT(%=);
NUMERIC_ARRAY_VIEW_DEFINE_ASSIGNMENT(&=);
NUMERIC_ARRAY_VIEW_DEFINE_ASSIGNMENT(|=);
NUMERIC_ARRAY_VIEW_DEFINE_ASSIGNMENT(^=);
#undef NUMERIC_ARRAY_VIEW_DEFINE_ASSIGNMENT

template <typename Scalar, dim_t N>
template <typename... Idxs>
NUMERIC_HOST_DEVICE [[nodiscard]] decltype(auto)
ArrayView<Scalar, N>::operator()(Idxs... idxs) noexcept {
  if constexpr (sizeof...(idxs) == dim &&
                (!meta::is_same_v<Idxs, Slice> && ...)) {
    return raw()[memory_index(idxs...)];
  } else {
    return sub_view(*this, 0, idxs...);
  }
}

template <typename Scalar, dim_t N>
template <dim_t M>
NUMERIC_HOST_DEVICE [[nodiscard]] ArrayView<Scalar, M>
ArrayView<Scalar, N>::broadcast(const Layout<M> &layout) noexcept {
  const Layout<M> new_layout = broadcasted_layout(layout_, layout);
  return ArrayView<Scalar, M>(raw(), new_layout, memory_type());
}

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE [[nodiscard]] ArrayConstView<Scalar, N>
ArrayView<Scalar, N>::const_view() const noexcept {
  return *this;
}

#if NUMERIC_ENABLE_EIGEN
template <typename Scalar, dim_t N>
[[nodiscard]] Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                         0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
ArrayView<Scalar, N>::matrix_view() noexcept {
  static_assert(dim == 2);
  return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, 0,
                    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(
      raw(), shape(0), shape(1), Eigen::Stride(stride(0), stride(1)));
}
#endif

template <typename Scalar, dim_t N>
NUMERIC_HOST_DEVICE [[nodiscard]] Scalar *ArrayView<Scalar, N>::raw() noexcept {
  return const_cast<Scalar *>(data_);
}

template <typename Scalar, dim_t N>
template <dim_t M, typename Idx, typename... Idxs>
NUMERIC_HOST_DEVICE decltype(auto)
ArrayView<Scalar, N>::sub_view(ArrayView<Scalar, M> view, dim_t d, Idx idx,
                               Idxs... idxs) noexcept {
  if constexpr (meta::is_same_v<Idx, Slice>) {
    if (idx.stop < 0) {
      idx.stop += view.shape(d) + 1;
    }
    Scalar *new_data = view.raw() + idx.start * view.stride(d);
    Layout<M> new_layout;
    for (dim_t i = 0; i < M; ++i) {
      new_layout.shape(i) = view.shape(i);
      new_layout.stride(i) = view.stride(i);
    }
    new_layout.shape(d) = (idx.stop - idx.start) / idx.step;
    new_layout.stride(d) = view.stride(d) * idx.step;
    return sub_view(
        ArrayView<Scalar, M>(new_data, new_layout, view.memory_type()), d + 1,
        idxs...);
  } else {
    Scalar *new_data = view.raw() + idx * view.stride(d);
    Layout<M - 1> new_layout;
    for (dim_t i = 0; i < M - 1; ++i) {
      new_layout.shape(i) = view.shape(i + (i < d ? 0 : 1));
      new_layout.stride(i) = view.stride(i + (i < d ? 0 : 1));
    }
    return sub_view(
        ArrayView<Scalar, M - 1>(new_data, new_layout, view.memory_type()), d,
        idxs...);
  }
}

template <typename Scalar, dim_t N>
template <dim_t M>
NUMERIC_HOST_DEVICE ArrayView<Scalar, M>
ArrayView<Scalar, N>::sub_view(ArrayView<Scalar, M> view, dim_t) noexcept {
  return view;
}

} // namespace numeric::memory

#endif
