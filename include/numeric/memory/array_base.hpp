#ifndef NUMERIC_MEMORY_ARRAY_BASE_HPP_
#define NUMERIC_MEMORY_ARRAY_BASE_HPP_

#include <numeric/memory/array_base_decl.hpp>

namespace numeric::memory {

template <typename Derived>
NUMERIC_HOST_DEVICE [[nodiscard]] dim_t
ArrayBase<Derived>::shape(size_t idx) const noexcept {
  return derived().shape(idx);
}

template <typename Derived>
NUMERIC_HOST_DEVICE [[nodiscard]] Derived &
ArrayBase<Derived>::derived() noexcept {
  return static_cast<Derived &>(*this);
}

template <typename Derived>
NUMERIC_HOST_DEVICE [[nodiscard]] const Derived &
ArrayBase<Derived>::derived() const noexcept {
  return static_cast<const Derived &>(*this);
}

template <typename Derived>
NUMERIC_HOST_DEVICE [[nodiscard]] MemoryType
ArrayBase<Derived>::memory_type() const noexcept {
  return derived().memory_type();
}

template <typename Derived>
template <dim_t N, dim_t M>
NUMERIC_HOST_DEVICE Layout<M>
ArrayBase<Derived>::broadcasted_layout(const Layout<N> &from,
                                       const Layout<M> &to) noexcept {
  static_assert(M >= N, "Broadcasted shape needs to have at least as many "
                        "dimensions as original array");
  Layout<M> brd;
  for (dim_t d = 0; d < M - N; ++d) {
    brd.shape(d) = to.shape(d);
    brd.stride(d) = 0;
  }
  for (dim_t d = 0; d < N; ++d) {
    if (from.shape(d) == 1) {
      brd.shape(M - N + d) = to.shape(M - N + d);
      brd.stride(M - N + d) = 0;
    } else {
      brd.shape(M - N + d) = from.shape(d);
      brd.stride(M - N + d) = from.stride(d);
    }
  }
  return brd;
}

} // namespace numeric::memory

#endif
