#ifndef NUMERIC_UTILS_TUPLE_HPP_
#define NUMERIC_UTILS_TUPLE_HPP_

#include <numeric/meta/integer_sequence.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>
#include <numeric/utils/forward.hpp>

namespace numeric::utils {

namespace detail {

template <size_t I, typename T> struct TupleLeaf {
  TupleLeaf() = default;
  NUMERIC_HOST_DEVICE TupleLeaf(const T &val) : value(val) {}
  NUMERIC_HOST_DEVICE TupleLeaf(T &&val) : value(utils::forward<T>(val)) {}
  TupleLeaf(const TupleLeaf &) = default;
  TupleLeaf(TupleLeaf &&) = default;
  TupleLeaf &operator=(const TupleLeaf &) = default;
  TupleLeaf &operator=(TupleLeaf &&) = default;

  T value;
  NUMERIC_HOST_DEVICE T element_type(meta::integral_constant<size_t, I>) const;
  NUMERIC_HOST_DEVICE T &get(meta::integral_constant<size_t, I>) {
    return value;
  }
  NUMERIC_HOST_DEVICE const T &get(meta::integral_constant<size_t, I>) const {
    return value;
  }
  NUMERIC_HOST_DEVICE T &get(meta::type_tag<T>) { return value; }
  NUMERIC_HOST_DEVICE const T &get(meta::type_tag<T>) const { return value; }
};

template <typename Idxs, typename... Ts> struct TupleBase;

template <size_t... Idxs, typename... Ts>
struct TupleBase<meta::index_sequence<Idxs...>, Ts...>
    : public TupleLeaf<Idxs, Ts>... {
  using TupleLeaf<Idxs, Ts>::get...;
  using TupleLeaf<Idxs, Ts>::element_type...;

  TupleBase() = default;
  NUMERIC_HOST_DEVICE TupleBase(Ts &&...args)
      : TupleLeaf<Idxs, Ts>(utils::forward<Ts>(args))... {}
  TupleBase(const TupleBase &) = default;
  TupleBase(TupleBase &&) = default;
  TupleBase &operator=(const TupleBase &) = default;
  TupleBase &operator=(TupleBase &&) = default;
};

} // namespace detail

template <typename... Ts>
struct Tuple
    : public detail::TupleBase<meta::make_index_sequence<sizeof...(Ts)>,
                               Ts...> {
private:
  using super =
      detail::TupleBase<meta::make_index_sequence<sizeof...(Ts)>, Ts...>;

public:
  using super::element_type;
  using super::get;

  Tuple() = default;
  explicit NUMERIC_HOST_DEVICE Tuple(Ts &&...args)
      : super(utils::forward<Ts>(args)...) {}
  Tuple(const Tuple &) = default;
  Tuple(Tuple &&) = default;
  Tuple &operator=(const Tuple &) = default;
  Tuple &operator=(Tuple &&) = default;

  template <size_t I> NUMERIC_HOST_DEVICE decltype(auto) get() {
    return get(meta::integral_constant<size_t, I>());
  }

  template <size_t I> NUMERIC_HOST_DEVICE decltype(auto) get() const {
    return get(meta::integral_constant<size_t, I>());
  }

  template <typename T> NUMERIC_HOST_DEVICE const T &get() const {
    static constexpr int num_Ts = (meta::is_same_v<T, Ts> + ...);
    static_assert(num_Ts <= 1, "Tuple contains given type multiple times");
    static_assert(num_Ts > 0, "Tuple does not contain given type");
    return get(meta::type_tag<T>());
  }

  template <typename T> NUMERIC_HOST_DEVICE T &get() {
    static constexpr int num_Ts = (meta::is_same_v<T, Ts> + ...);
    static_assert(num_Ts <= 1, "Tuple contains given type multiple times");
    static_assert(num_Ts > 0, "Tuple does not contain given type");
    return get(meta::type_tag<T>());
  }
};

} // namespace numeric::utils

namespace std {

#ifdef __HIP_DEVICE_COMPILE__
template <typename T> struct tuple_size {};
template <size_t I, typename T> struct tuple_element {};
#endif

template <typename... Ts> struct tuple_size<numeric::utils::Tuple<Ts...>> {
  static constexpr size_t value = sizeof...(Ts);
  using type = size_t;
  NUMERIC_HOST_DEVICE constexpr operator size_t() { return value; }
};

template <size_t I, typename... Ts>
struct tuple_element<I, numeric::utils::Tuple<Ts...>> {
  using type = decltype(numeric::meta::declval<numeric::utils::Tuple<Ts...>>()
                            .element_type(
                                numeric::meta::integral_constant<size_t, I>()));
};

} // namespace std

#endif
