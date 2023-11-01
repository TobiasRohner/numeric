#ifndef NUMERIC_MEMORY_ARRAY_TRAITS_HPP_
#define NUMERIC_MEMORY_ARRAY_TRAITS_HPP_

#include <numeric/meta/meta.hpp>

namespace numeric::memory {

template <typename T> struct ArrayTraits {
  static_assert(!meta::is_same_v<T, T>, "Forgot overload ArrayTraits");
};

} // namespace numeric::memory

#endif
