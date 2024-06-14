#ifndef NUMERIC_META_INTEGER_SEQUENCE_HPP_
#define NUMERIC_META_INTEGER_SEQUENCE_HPP_

#include <numeric/config.hpp>
#ifndef __HIP_DEVICE_COMPILE__
#include <cstddef>
#endif

namespace numeric::meta {

template <typename T, T... Ints> struct integer_sequence {
  using value_type = T;
  NUMERIC_HOST_DEVICE static constexpr size_t size() { return sizeof...(Ints); }
};

namespace detail {

template <typename T, T N, T I, T... Idxs> struct make_integer_sequence_impl {
  using result =
      typename make_integer_sequence_impl<T, N, I + 1, Idxs..., I>::result;
};

template <typename T, T N, T... Idxs>
struct make_integer_sequence_impl<T, N, N, Idxs...> {
  using result = integer_sequence<T, Idxs...>;
};

} // namespace detail

template <typename T, T N>
using make_integer_sequence =
    typename detail::make_integer_sequence_impl<T, N, 0>::result;

template <size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

template <typename... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;

} // namespace numeric::meta

#endif
