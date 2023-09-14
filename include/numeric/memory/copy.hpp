#ifndef NUMERIC_MEMORY_COPY_HPP_
#define NUMERIC_MEMORY_COPY_HPP_

#include <numeric/config.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::memory {

template <typename Derived> class ArrayBase;
template <typename Scalar, dim_t N> class ArrayView;

namespace internal {

template <typename Scalar, dim_t N, typename Src, typename... Idxs>
void copy_naive_host_loop(ArrayView<Scalar, N> &dst, const Src &src,
                          Idxs... idxs) {
  static constexpr dim_t d = sizeof...(Idxs);
  if constexpr (d == N) {
    dst(idxs...) = src(idxs...);
  } else {
    for (dim_t i = 0; i < dst.shape(d); ++i) {
      copy_naive_host_loop(dst, src, idxs..., i);
    }
  }
}

template <typename Scalar, dim_t N, typename Src>
void copy_naive_host(ArrayView<Scalar, N> &dst, const Src &src) {
  const auto src_brdc = src.broadcast(dst.layout());
  copy_naive_host_loop(dst, src_brdc);
}

} // namespace internal

template <typename Scalar, dim_t N, typename Src>
void copy_host(ArrayView<Scalar, N> dst, const ArrayBase<Src> &src) {
  internal::copy_naive_host(dst, src.derived());
}

template <typename Scalar, dim_t N, typename Src>
void copy(ArrayView<Scalar, N> dst, const ArrayBase<Src> &src) {
  copy_host(dst, src.derived());
}

} // namespace numeric::memory

#endif
