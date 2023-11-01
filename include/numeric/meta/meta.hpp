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

template <typename T> struct remove_reference { using type = T; };

template <typename T> struct remove_reference<T &> { using type = T; };

template <typename T> struct remove_reference<T &&> { using type = T; };

template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

template <typename T> struct remove_cv { using type = T; };

template <typename T> struct remove_cv<const T> { using type = T; };

template <typename T> struct remove_cv<volatile T> { using type = T; };

template <typename T> struct remove_cv<const volatile T> { using type = T; };

template <typename T> using remove_cv_t = typename remove_cv<T>::type;

template <typename T> struct remove_const { using type = T; };

template <typename T> struct remove_const<const T> { using type = T; };

template <typename T> using remove_const_t = typename remove_const<T>::type;

template <typename T> struct remove_volatile { using type = T; };

template <typename T> struct remove_volatile<volatile T> { using type = T; };

template <typename T>
using remove_volatile_t = typename remove_volatile<T>::type;

template <typename T> struct remove_cvref {
  using type = remove_cv_t<remove_reference_t<T>>;
};

template <typename T> using remove_cvref_t = typename remove_cvref<T>::type;

} // namespace numeric::meta

#endif
