#ifndef NUMERIC_MEMORY_BROADCAST_DECL_HPP_
#define NUMERIC_MEMORY_BROADCAST_DECL_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/array_base_decl.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/memory/slice.hpp>
#include <numeric/meta/integer_sequence.hpp>
#include <numeric/utils/tuple.hpp>

namespace numeric::memory {

template <typename Src, dim_t N> class Broadcast {
public:
  static constexpr dim_t dim = N;
  using scalar_t = typename ArrayTraits<Src>::scalar_t;

  NUMERIC_HOST_DEVICE Broadcast(const ArrayBase<Src> &src,
                                const Shape<N> &shape);
  Broadcast(const Broadcast &) = default;
  Broadcast(Broadcast &&) = default;
  Broadcast &operator=(const Broadcast &) = default;
  Broadcast &operator=(Broadcast &&) = default;

  NUMERIC_HOST_DEVICE MemoryType memory_type() const noexcept;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE decltype(auto) operator()(Idxs... idxs) const noexcept;

  NUMERIC_HOST_DEVICE const Shape<N> &shape() const noexcept;
  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept;
  NUMERIC_HOST_DEVICE dim_t size() const noexcept;

  template <dim_t M>
  NUMERIC_HOST_DEVICE [[nodiscard]] Broadcast<Src, M>
  broadcast(const Shape<M> &shape) const noexcept;

private:
  Src src_;
  Shape<N> shape_;

  template <typename Idx, typename... Idxs>
  NUMERIC_HOST_DEVICE scalar_t element(Idx idx, Idxs... idxs) const noexcept;

  template <typename... Idxs>
  NUMERIC_HOST_DEVICE decltype(auto) sub_view(Idxs... idxs) const noexcept;
  template <typename... Idxs, size_t... IdxsIdxs>
  NUMERIC_HOST_DEVICE decltype(auto)
  sub_view_impl(meta::index_sequence<IdxsIdxs...>, Idxs... idxs) const noexcept;
  template <typename... Idxs, size_t... IdxsIdxs, size_t... IdxsIdxsBroadcast,
            size_t... IdxsIdxsSrc>
  NUMERIC_HOST_DEVICE decltype(auto)
  sub_view_impl(const utils::Tuple<Idxs...> &idxs,
                meta::index_sequence<IdxsIdxs...>,
                meta::index_sequence<IdxsIdxsBroadcast...>,
                meta::index_sequence<IdxsIdxsSrc...>) const noexcept;
};

template <typename Src, dim_t N> struct ArrayTraits<Broadcast<Src, N>> {
  static constexpr bool is_array = true;
  static constexpr dim_t dim = N;
  using scalar_t = typename ArrayTraits<Src>::scalar_t;
};

template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE Broadcast<Src, N> broadcast(const ArrayBase<Src> &src,
                                                const Shape<N> &shape) noexcept;

} // namespace numeric::memory

#endif
