#ifndef NUMERIC_MEMORY_ARRAY_TRAITS_HPP_
#define NUMERIC_MEMORY_ARRAY_TRAITS_HPP_

#include <numeric/meta/meta.hpp>

namespace numeric::memory {

/**
 * @brief Template struct for traits of array types.
 *
 * This struct provides traits information for array types.
 *
 * @tparam T The type of the array.
 */
template <typename T> struct ArrayTraits {
  static constexpr bool is_array = false;
  // using scalar_t = /* specicalize */
  // static constexpr dim_t dim = /* specialize */
};

/**
 * @brief Specialization of ArrayTraits for const-qualified types.
 *
 * This struct inherits traits information for const-qualified types.
 *
 * @tparam T The type of the array.
 */
template <typename T> struct ArrayTraits<const T> : public ArrayTraits<T> {};

/**
 * @brief Specialization of ArrayTraits for reference types.
 *
 * This struct inherits traits information for reference types.
 *
 * @tparam T The type of the array.
 */
template <typename T> struct ArrayTraits<T &> : public ArrayTraits<T> {};

/**
 * @brief Specialization of ArrayTraits for const reference types.
 *
 * This struct inherits traits information for const reference types.
 *
 * @tparam T The type of the array.
 */
template <typename T> struct ArrayTraits<const T &> : public ArrayTraits<T> {};

} // namespace numeric::memory

#endif
