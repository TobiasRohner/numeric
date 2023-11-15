#ifndef NUMERIC_MEMORY_BROADCAST_HPP_
#define NUMERIC_MEMORY_BROADCAST_HPP_

#include <numeric/memory/array_base.hpp>
#include <numeric/memory/broadcast_decl.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::memory {

template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE Broadcast<Src, N>::Broadcast(const ArrayBase<Src> &src,
                                                 const Layout<N> &layout)
    : src_(src.derived()), layout_(layout) {}

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
NUMERIC_HOST_DEVICE const Layout<N> &
Broadcast<Src, N>::layout() const noexcept {
  return layout_;
}
template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE const dim_t *Broadcast<Src, N>::shape() const noexcept {
  return layout_.shape();
}
template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE dim_t Broadcast<Src, N>::shape(size_t idx) const noexcept {
  return layout_.shape(idx);
}
template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE dim_t Broadcast<Src, N>::size() const noexcept {
  return layout_.size();
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
  static_assert((idxs != ...), "Slicing not yet supported for Broadcast views");
}

template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE Broadcast<Src, N>
broadcast(const ArrayBase<Src> &src, const Layout<N> &layout) noexcept {
  return Broadcast<Src, N>(src, layout);
}

} // namespace numeric::memory

#endif
