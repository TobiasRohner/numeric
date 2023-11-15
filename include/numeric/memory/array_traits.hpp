#ifndef NUMERIC_MEMORY_ARRAY_TRAITS_HPP_
#define NUMERIC_MEMORY_ARRAY_TRAITS_HPP_

#include <numeric/meta/meta.hpp>

namespace numeric::memory {

template <typename T> struct ArrayTraits {
  static constexpr bool is_array = false;
};

template <typename T> struct ArrayTraits<const T> : public ArrayTraits<T> {};
template <typename T> struct ArrayTraits<T &> : public ArrayTraits<T> {};
template <typename T> struct ArrayTraits<const T &> : public ArrayTraits<T> {};

} // namespace numeric::memory

#endif
