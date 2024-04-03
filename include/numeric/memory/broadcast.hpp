#ifndef NUMERIC_MEMORY_BROADCAST_HPP_
#define NUMERIC_MEMORY_BROADCAST_HPP_

#include <numeric/memory/array_base.hpp>
#include <numeric/memory/broadcast_decl.hpp>
#include <numeric/meta/integer_sequence.hpp>
#include <numeric/meta/meta.hpp>

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
  if constexpr (sizeof...(Idxs) == N &&
                (!meta::is_same_v<Idxs, Slice> && ...)) {
    return element(idxs...);
  } else {
    return sub_view(idxs...);
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
Broadcast<Src, N>::sub_view(Idxs... idxs) const noexcept {
  static constexpr dim_t dim_src = ArrayTraits<Src>::dim;
  static constexpr dim_t num_expanded_dims = N - dim_src;
  if constexpr (sizeof...(Idxs) <= num_expanded_dims) {
    return sub_view_impl(meta::make_index_sequence<sizeof...(Idxs)>(), idxs...);
  } else {
    static constexpr auto idxs_all =
        meta::make_index_sequence<sizeof...(Idxs)>();
    static constexpr auto idxs_broadcast =
        meta::make_index_sequence<num_expanded_dims>();
    static constexpr auto idxs_src =
        meta::make_index_sequence<sizeof...(Idxs) - num_expanded_dims>();
    return sub_view_impl(utils::Tuple<Idxs...>(idxs...), idxs_all,
                         idxs_broadcast, idxs_src);
  }
}

template <typename Src, dim_t N>
template <typename... Idxs, size_t... IdxsIdxs>
NUMERIC_HOST_DEVICE decltype(auto)
Broadcast<Src, N>::sub_view_impl(meta::index_sequence<IdxsIdxs...>,
                                 Idxs... idxs) const noexcept {
  static constexpr dim_t num_slices_in_idxs =
      (0 + ... + meta::is_same_v<Idxs, Slice>);
  static constexpr dim_t new_dim = N - sizeof...(Idxs) + num_slices_in_idxs;
  const auto get_size = [&](auto sl, size_t size) {
    if constexpr (meta::is_same_v<decltype(sl), Slice>) {
      return sl.size(size);
    } else {
      return 0;
    }
  };
  Shape<new_dim> new_shape;
  dim_t i = 0;
  ((meta::is_same_v<Idxs, Slice> &&
    (new_shape[i++] = get_size(idxs, shape_[IdxsIdxs]))),
   ...);
  for (; i < new_dim; ++i) {
    new_shape[i] = shape_[sizeof...(Idxs) + i];
  }
  return Broadcast<Src, new_dim>(src_, new_shape);
}

template <typename Src, dim_t N>
template <typename... Idxs, size_t... IdxsIdxs, size_t... IdxsIdxsBroadcast,
          size_t... IdxsIdxsSrc>
NUMERIC_HOST_DEVICE decltype(auto) Broadcast<Src, N>::sub_view_impl(
    const utils::Tuple<Idxs...> &idxs, meta::index_sequence<IdxsIdxs...>,
    meta::index_sequence<IdxsIdxsBroadcast...>,
    meta::index_sequence<IdxsIdxsSrc...>) const noexcept {
  static constexpr dim_t num_broadcast_dims = N - ArrayTraits<Src>::dim;
  static constexpr dim_t num_slices_in_idxs_broadcast =
      (0 + ... +
       meta::is_same_v<typename std::tuple_element<IdxsIdxsBroadcast,
                                                   utils::Tuple<Idxs...>>::type,
                       Slice>);
  static constexpr dim_t num_slices_in_idxs =
      (0 + ... + meta::is_same_v<Idxs, Slice>);
  static constexpr dim_t new_dim = N - sizeof...(Idxs) + num_slices_in_idxs;
  const auto get_size = [&](auto sl, size_t size) {
    if constexpr (meta::is_same_v<decltype(sl), Slice>) {
      return sl.size(size);
    } else {
      return 0;
    }
  };
  Shape<new_dim> new_shape;
  dim_t i = 0;
  ((meta::is_same_v<Idxs, Slice> &&
    (new_shape[i++] = get_size(idxs, shape_[IdxsIdxs]))),
   ...);
  for (; i < new_dim; ++i) {
    new_shape[i] = shape_[sizeof...(Idxs) + i];
  }
  if constexpr (sizeof...(IdxsIdxsSrc) == 0) {
    return Broadcast<Src, new_dim>(src_, new_shape);
  } else {
    static constexpr dim_t num_slices_in_idxs_src =
        (0 + ... +
         meta::is_same_v<
             typename std::tuple_element<IdxsIdxsSrc + num_broadcast_dims,
                                         utils::Tuple<Idxs...>>::type,
             Slice>);
    if constexpr (sizeof...(IdxsIdxsSrc) == ArrayTraits<Src>::dim &&
                  num_slices_in_idxs_src == 0) {
      using scalar_t = typename ArrayTraits<Src>::scalar_t;
      const scalar_t value =
          src_(idxs.template get<IdxsIdxsSrc + num_broadcast_dims>()...);
      return Constant<scalar_t, new_dim>(new_shape, value, src_.memory_type());
    } else {
      auto src_slice =
          src_(idxs.template get<IdxsIdxsSrc + num_broadcast_dims>()...);
      return Broadcast<decltype(src_slice), new_dim>(src_slice, new_shape);
    }
  }
}

template <typename Src, dim_t N>
NUMERIC_HOST_DEVICE Broadcast<Src, N>
broadcast(const ArrayBase<Src> &src, const Shape<N> &shape) noexcept {
  return Broadcast<Src, N>(src, shape);
}

} // namespace numeric::memory

#endif
