#ifndef NUMERIC_UTILS_TUPLE_HPP_
#define NUMERIC_UTILS_TUPLE_HPP_

#include <numeric/meta/integer_sequence.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::utils {

namespace detail {

template <size_t I, typename T> struct TupleLeaf {
  T value;
  T element_type(meta::integral_constant<size_t, I>) const;
  T &get(meta::integral_constant<size_t, I>) { return value; }
  const T &get(meta::integral_constant<size_t, I>) const { return value; }
};

template <typename Idxs, typename... Ts> struct TupleBase;

template <size_t... Idxs, typename... Ts>
struct TupleBase<meta::index_sequence<Idxs...>, Ts...>
    : public TupleLeaf<Idxs, Ts>... {
  using TupleLeaf<Idxs, Ts>::get...;
  using TupleLeaf<Idxs, Ts>::element_type...;

  TupleBase() = default;
  TupleBase(const Ts &...args) : TupleLeaf<Idxs, Ts>(args)... {}
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
  Tuple(const Ts &...args) : super(args...) {}
  Tuple(const Tuple &) = default;
  Tuple(Tuple &&) = default;
  Tuple &operator=(const Tuple &) = default;
  Tuple &operator=(Tuple &&) = default;

  template <size_t I> decltype(auto) get() {
    return get(meta::integral_constant<size_t, I>());
  }

  template <size_t I> decltype(auto) get() const {
    return get(meta::integral_constant<size_t, I>());
  }
};

} // namespace numeric::utils

namespace std {

template <typename... Ts> struct tuple_size<numeric::utils::Tuple<Ts...>> {
  static constexpr size_t value = sizeof...(Ts);
  using type = size_t;
  constexpr operator size_t() { return value; }
};

template <size_t I, typename... Ts>
struct tuple_element<I, numeric::utils::Tuple<Ts...>> {
  using type = decltype(numeric::meta::declval<numeric::utils::Tuple<Ts...>>()
                            .element_type(
                                numeric::meta::integral_constant<size_t, I>()));
};

} // namespace std

#endif
