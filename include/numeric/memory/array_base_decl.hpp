#ifndef NUMERIC_MEMORY_ARRAY_BASE_DECL_HPP_
#define NUMERIC_MEMORY_ARRAY_BASE_DECL_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/layout.hpp>
#include <numeric/memory/memory_type.hpp>

namespace numeric::memory {

template <typename Derived> class ArrayBase {
public:
  NUMERIC_HOST_DEVICE [[nodiscard]] dim_t shape(size_t idx) const noexcept;
  NUMERIC_HOST_DEVICE [[nodiscard]] Derived &derived() noexcept;
  NUMERIC_HOST_DEVICE [[nodiscard]] const Derived &derived() const noexcept;
  NUMERIC_HOST_DEVICE [[nodiscard]] MemoryType memory_type() const noexcept;

protected:
  template <dim_t N, dim_t M>
  static NUMERIC_HOST_DEVICE Layout<M>
  broadcasted_layout(const Layout<N> &from, const Layout<M> &to) noexcept;
};

} // namespace numeric::memory

#endif
