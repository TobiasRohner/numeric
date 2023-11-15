#ifndef NUMERIC_MEMORY_BROADCAST_DECL_HPP_
#define NUMERIC_MEMORY_BROADCAST_DECL_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_base_decl.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/memory/slice.hpp>

namespace numeric::memory {

template <typename Src, dim_t N> class Broadcast {
public:
  static constexpr dim_t dim = N;
  using scalar_t = typename ArrayTraits<Src>::scalar_t;

  NUMERIC_HOST_DEVICE Broadcast(const ArrayBase<Src> &src,
                                const Layout<N> &layout);
  Broadcast(const Broadcast &) = default;
  Broadcast(Broadcast &&) = default;
  Broadcast &operator=(const Broadcast &) = default;
  Broadcast &operator=(Broadcast &&) = default;

  NUMERIC_HOST_DEVICE MemoryType memory_type() const noexcept;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE decltype(auto) operator()(Idxs... idxs) const noexcept;

  NUMERIC_HOST_DEVICE const Layout<N> &layout() const noexcept;
  NUMERIC_HOST_DEVICE const dim_t *shape() const noexcept;
  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept;
  NUMERIC_HOST_DEVICE dim_t size() const noexcept;

private:
  Src src_;
  Layout<N> layout_;

  template <typename Idx, typename... Idxs>
  NUMERIC_HOST_DEVICE scalar_t element(Idx idx, Idxs... idxs) const noexcept;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE decltype(auto) sub_view(dim_t d,
                                              Idxs... idxs) const noexcept;
};

template <typename Src, dim_t N> struct ArrayTraits<Broadcast<Src, N>> {
  static constexpr bool is_array = true;
  static constexpr dim_t dim = N;
  using scalar_t = typename ArrayTraits<Src>::scalar_t;
};

template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE Broadcast<Src, N>
broadcast(const ArrayBase<Src> &src, const Layout<N> &layout) noexcept;

} // namespace numeric::memory

#endif
