#ifndef NUMERIC_MEMORY_BROADCAST_HPP_
#define NUMERIC_MEMORY_BROADCAST_HPP_

#include <numeric/memory/array_base.hpp>
#include <numeric/memory/broadcast_decl.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::memory {

template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE Broadcast<Src, N>::Broadcast(const ArrayBase<Src> &src,
                                                 const Shape<N> &shape)
    : src_(src.derived()), shape_(shape) {}

template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE MemoryType Broadcast<Src, N>::memory_type() const noexcept {
  return src_.memory_type();
}

template <typename Src, dim_t N>
template <typename... Idxs>
NUMERIC_HOST_DEVICE decltype(auto)
Broadcast<Src, N>::operator()(Idxs... idxs) const noexcept {
  if constexpr ((!meta::is_same_v<Idxs, Slice> && ...)) {
    return element(idxs...);
  } else {
    return sub_view(*this, 0, idxs...);
  }
}

template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE const Shape<N> &Broadcast<Src, N>::shape() const noexcept {
  return shape_;
}
template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE dim_t Broadcast<Src, N>::shape(size_t idx) const noexcept {
  return shape_[idx];
}
template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE dim_t Broadcast<Src, N>::size() const noexcept {
  return shape_.size();
}

template <typename Src, dim_t N>
template <dim_t M>
NUMERIC_HOST_DEVICE [[nodiscard]] Broadcast<Src, M>
Broadcast<Src, N>::broadcast(const Shape<M> &shape) const noexcept {
  return Broadcast<Src, M>(src_, shape);
}

template <typename Src, dim_t N>
template <typename Idx, typename... Idxs>
NUMERIC_HOST_DEVICE typename ArrayTraits<Src>::scalar_t
Broadcast<Src, N>::element(Idx idx, Idxs... idxs) const noexcept {
  if constexpr (sizeof...(Idxs) >= ArrayTraits<Src>::dim) {
    return element(idxs...);
  } else {
    dim_t ds = 0;
    idx = src_.shape(ds++) == 1 ? Idx{0} : idx;
    ((idxs = src_.shape(ds++) == 1 ? Idxs{0} : idxs), ...);
    return src_(idx, idxs...);
  }
}

template <typename Src, dim_t N>
template <typename... Idxs>
NUMERIC_HOST_DEVICE decltype(auto)
Broadcast<Src, N>::sub_view(dim_t d, Idxs... idxs) const noexcept {
  static_assert(!meta::is_same_v<Src, Src>,
                "Slicing not yet supported for Broadcast views");
  NUMERIC_ERROR("Slicing not yet implemented for Broadcast views");
}

template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE Broadcast<Src, N>
broadcast(const ArrayBase<Src> &src, const Shape<N> &shape) noexcept {
  return Broadcast<Src, N>(src, shape);
}

} // namespace numeric::memory

#endif
