#ifndef NUMERIC_MEMORY_ARRAY_BASE_HPP_
#define NUMERIC_MEMORY_ARRAY_BASE_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/layout.hpp>

namespace numeric::memory {

template <typename Derived> class ArrayBase {
public:
  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t shape(size_t idx) const
      noexcept(noexcept(derived().shape(idx))) {
    return derived().shape(idx);
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] Derived &derived() noexcept {
    return static_cast<Derived &>(*this);
  }

  NUMERIC_HOST_DEVICE [[nodiscard]] const Derived &derived() const noexcept {
    return static_cast<const Derived &>(*this);
  }

protected:
  template <dim_t N, dim_t M>
  static NUMERIC_HOST_DEVICE [[nodiscard]] Layout<M>
  broadcasted_layout(const Layout<N> &from, const Layout<M> &to) noexcept {
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
};

} // namespace numeric::memory

#endif
