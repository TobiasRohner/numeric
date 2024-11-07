#ifndef NUMERIC_UTILS_DATATYPE_HPP_
#define NUMERIC_UTILS_DATATYPE_HPP_

#include <cstdint>
#include <numeric/meta/meta.hpp>
#include <string_view>

namespace numeric::utils {

enum class Datatype {
  FLOAT,
  DOUBLE,
  LONG_DOUBLE,
  INT8_T,
  UINT8_T,
  INT16_T,
  UINT16_T,
  INT32_T,
  UINT32_T,
  INT64_T,
  UINT64_T
};

std::string_view to_string(Datatype type);

template <typename T> struct to_datatype {
  static_assert(!meta::is_same_v<T, T>, "Unsupported Datatype");
};

template <> struct to_datatype<float> {
  static constexpr Datatype value = Datatype::FLOAT;
};
template <> struct to_datatype<double> {
  static constexpr Datatype value = Datatype::DOUBLE;
};
template <> struct to_datatype<long double> {
  static constexpr Datatype value = Datatype::LONG_DOUBLE;
};
template <> struct to_datatype<int8_t> {
  static constexpr Datatype value = Datatype::INT8_T;
};
template <> struct to_datatype<uint8_t> {
  static constexpr Datatype value = Datatype::UINT8_T;
};
template <> struct to_datatype<int16_t> {
  static constexpr Datatype value = Datatype::INT16_T;
};
template <> struct to_datatype<uint16_t> {
  static constexpr Datatype value = Datatype::UINT16_T;
};
template <> struct to_datatype<int32_t> {
  static constexpr Datatype value = Datatype::INT32_T;
};
template <> struct to_datatype<uint32_t> {
  static constexpr Datatype value = Datatype::UINT32_T;
};
template <> struct to_datatype<int64_t> {
  static constexpr Datatype value = Datatype::INT64_T;
};
template <> struct to_datatype<uint64_t> {
  static constexpr Datatype value = Datatype::UINT64_T;
};

template <typename T>
static constexpr Datatype to_datatype_v = to_datatype<T>::value;

template <Datatype type> struct from_datatype {
  static_assert(type != type, "Unsupported Datatype");
};

template <> struct from_datatype<Datatype::FLOAT> {
  using type = float;
};
template <> struct from_datatype<Datatype::DOUBLE> {
  using type = double;
};
template <> struct from_datatype<Datatype::LONG_DOUBLE> {
  using type = long double;
};
template <> struct from_datatype<Datatype::INT8_T> {
  using type = int8_t;
};
template <> struct from_datatype<Datatype::UINT8_T> {
  using type = uint8_t;
};
template <> struct from_datatype<Datatype::INT16_T> {
  using type = int16_t;
};
template <> struct from_datatype<Datatype::UINT16_T> {
  using type = uint16_t;
};
template <> struct from_datatype<Datatype::INT32_T> {
  using type = int32_t;
};
template <> struct from_datatype<Datatype::UINT32_T> {
  using type = uint32_t;
};
template <> struct from_datatype<Datatype::INT64_T> {
  using type = int64_t;
};
template <> struct from_datatype<Datatype::UINT64_T> {
  using type = uint64_t;
};

template <Datatype type>
using from_datatype_t = typename from_datatype<type>::type;

} // namespace numeric::utils

#endif
