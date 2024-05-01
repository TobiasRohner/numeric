#ifndef NUMERIC_UTILS_TYPE_INDEXED_MAP_HPP_
#define NUMERIC_UTILS_TYPE_INDEXED_MAP_HPP_

#include <numeric/meta/meta.hpp>
#include <numeric/utils/forward.hpp>
#include <numeric/utils/tuple.hpp>

namespace numeric::utils {

template <typename Value, typename... Keys> class TypeIndexedMap {
  template <typename Key> struct Container {
    Value value;

    Container(const Value &val) : value(val) {}
    Container(Value &&val) : value(val) {}
    Container() = default;
    Container(const Container &) = default;
    Container(Container &&) = default;
    Container &operator=(const Container &) = default;
    Container &operator=(Container &&) = default;
  };

public:
  template <typename... Values,
            typename = meta::enable_if_t<
                (meta::is_same_v<meta::remove_cvref_t<Values>, Value> && ...)>>
  TypeIndexedMap(Values &&...values)
      : values_(utils::forward<Values>(values)...) {}
  TypeIndexedMap() = default;
  TypeIndexedMap(const TypeIndexedMap &) = default;
  TypeIndexedMap(TypeIndexedMap &&) = default;
  TypeIndexedMap &operator=(const TypeIndexedMap &) = default;
  TypeIndexedMap &operator=(TypeIndexedMap &&) = default;

  template <typename Key> Value &get() {
    return values_.template get<Container<Key>>().value;
  }

  template <typename Key> const Value &get() const {
    return values_.template get<Container<Key>>().value;
  }

private:
  Tuple<Container<Keys>...> values_;
};

} // namespace numeric::utils

#endif
