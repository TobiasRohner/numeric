#ifndef NUMERIC_MEMORY_COPY_KERNELS_HPP_
#define NUMERIC_MEMORY_COPY_KERNELS_HPP_

#include <numeric/memory/array_view_decl.hpp>

namespace numeric::memory {

template <typename, dim_t> class ArrayView;

namespace internal {

template <typename Scalar, dim_t N, typename Src, typename... Idxs>
NUMERIC_HOST_DEVICE void copy_naive_elm_loop(ArrayView<Scalar, N> &dst,
                                             const Src &src, Idxs... idxs) {
  static constexpr dim_t d = sizeof...(Idxs);
  if constexpr (d == N) {
    dst(idxs...) = src(idxs...);
  } else {
    for (dim_t i = 0; i < dst.shape(d); ++i) {
      copy_naive_elm_loop(dst, src, idxs..., i);
    }
  }
}

} // namespace internal

template <typename Scalar, dim_t N, typename Src>
NUMERIC_HOST_DEVICE void copy_naive_elm(ArrayView<Scalar, N> &dst,
                                        const Src &src) {
  const auto src_brdc = src.broadcast(dst.layout());
  internal::copy_naive_elm_loop(dst, src_brdc);
}

} // namespace numeric::memory

#endif
