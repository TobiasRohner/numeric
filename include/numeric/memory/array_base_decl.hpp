#ifndef NUMERIC_MEMORY_ARRAY_BASE_DECL_HPP_
#define NUMERIC_MEMORY_ARRAY_BASE_DECL_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/layout.hpp>
#include <numeric/memory/memory_type.hpp>

namespace numeric::memory {

template <typename Derived> class ArrayBase {
public:
  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept;
  NUMERIC_HOST_DEVICE Derived &derived() noexcept;
  NUMERIC_HOST_DEVICE const Derived &derived() const noexcept;
  NUMERIC_HOST_DEVICE MemoryType memory_type() const noexcept;
  NUMERIC_HOST_DEVICE decltype(auto) layout() const noexcept;
  NUMERIC_HOST_DEVICE dim_t size() const noexcept;

protected:
  template <dim_t N, dim_t M>
  static NUMERIC_HOST_DEVICE Layout<M>
  broadcasted_layout(const Layout<N> &from, const Layout<M> &to) noexcept;
};

} // namespace numeric::memory

#endif
