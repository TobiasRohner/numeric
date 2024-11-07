#ifndef NUMERIC_META_META_HPP_
#define NUMERIC_META_META_HPP_

#include <numeric/config.hpp>

namespace numeric::meta {

template <typename T, T v> struct integral_constant {
  static constexpr T value = v;
  using type = T;
  NUMERIC_HOST_DEVICE constexpr operator T() { return v; }
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

struct nonesuch {
  ~nonesuch() = delete;
  nonesuch(const nonesuch &) = delete;
  void operator=(const nonesuch &) = delete;
};

template <typename...> using void_t = void;

template <typename T> NUMERIC_HOST_DEVICE T declval();

template <typename T1, typename T2> struct is_same {
  static constexpr bool value = false;
};

template <typename T> struct is_same<T, T> {
  static constexpr bool value = true;
};

template <typename T1, typename T2>
static constexpr bool is_same_v = is_same<T1, T2>::value;

template <typename T> struct is_lvalue_reference {
  static constexpr bool value = false;
};

template <typename T> struct is_lvalue_reference<T &> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_lvalue_reference_v = is_lvalue_reference<T>::value;

template <typename T> struct remove_reference {
  using type = T;
};

template <typename T> struct remove_reference<T &> {
  using type = T;
};

template <typename T> struct remove_reference<T &&> {
  using type = T;
};

template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

template <typename T> struct remove_cv {
  using type = T;
};

template <typename T> struct remove_cv<const T> {
  using type = T;
};

template <typename T> struct remove_cv<volatile T> {
  using type = T;
};

template <typename T> struct remove_cv<const volatile T> {
  using type = T;
};

template <typename T> using remove_cv_t = typename remove_cv<T>::type;

template <typename T> struct remove_const {
  using type = T;
};

template <typename T> struct remove_const<const T> {
  using type = T;
};

template <typename T> using remove_const_t = typename remove_const<T>::type;

template <typename T> struct remove_volatile {
  using type = T;
};

template <typename T> struct remove_volatile<volatile T> {
  using type = T;
};

template <typename T>
using remove_volatile_t = typename remove_volatile<T>::type;

template <typename T> struct remove_cvref {
  using type = remove_cv_t<remove_reference_t<T>>;
};

template <typename T> using remove_cvref_t = typename remove_cvref<T>::type;

template <bool B> struct enable_if {};
template <> struct enable_if<true> {
  using type = void;
};
template <bool B> using enable_if_t = typename enable_if<B>::type;

namespace detail {

template <typename Default, typename AlwaysVoid,
          template <typename...> typename Op, typename... Args>
struct Detector {
  using value_t = false_type;
  using type = Default;
};

template <typename Default, template <typename...> typename Op,
          typename... Args>
struct Detector<Default, void_t<Op<Args...>>, Op, Args...> {
  using value_t = true_type;
  using type = Op<Args...>;
};

} // namespace detail

template <template <typename...> typename Op, typename... Args>
using is_detected =
    typename detail::Detector<nonesuch, void, Op, Args...>::value_t;

template <template <typename...> typename Op, typename... Args>
static constexpr bool is_detected_v = is_detected<Op, Args...>::value;

template <template <typename...> typename Op, typename... Args>
using is_detected_t =
    typename detail::Detector<nonesuch, void, Op, Args...>::type;

} // namespace numeric::meta

#endif
