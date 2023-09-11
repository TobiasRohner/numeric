#ifndef NUMERIC_META_META_HPP_
#define NUMERIC_META_META_HPP_

#include <numeric/config.hpp>

namespace numeric::meta {

template <typename T> NUMERIC_HOST_DEVICE T declval();

template <typename T1, typename T2> struct is_same {
  static constexpr bool value = false;
};

template <typename T> struct is_same<T, T> {
  static constexpr bool value = true;
};

template <typename T1, typename T2>
static constexpr bool is_same_v = is_same<T1, T2>::value;

} // namespace numeric::meta

#endif
