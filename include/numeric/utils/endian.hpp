#ifndef NUMERIC_UTILS_ENDIAN_HPP_
#define NUMERIC_UTILS_ENDIAN_HPP_

#include <bit>
#include <cstring>

namespace numeric::utils {

template <typename T, std::endian E> struct alignas(T) Endian {
  using type = T;
  static constexpr std::endian endianness = E;

  char data[sizeof(T)];

  Endian() = default;

  Endian(T value) { *this = value; }

  template <std::endian EO> Endian(Endian<T, EO> other) { *this = other; }

  Endian &operator=(T value) {
    if (endianness == std::endian::native) {
      std::memcpy(data, &value, sizeof(T));
    } else {
      char tmp[sizeof(T)];
      std::memcpy(tmp, &value, sizeof(T));
      for (size_t i = 0; i < sizeof(T); ++i) {
        data[i] = tmp[sizeof(T) - i - 1];
      }
    }
    return *this;
  }

  template <std::endian EO> Endian &operator=(Endian<T, EO> other) {
    if (EO == endianness) {
      std::memcpy(data, other.data, sizeof(T));
    } else {
      for (size_t i = 0; i < sizeof(T); ++i) {
        data[i] = other.data[sizeof(T) - i - 1];
      }
    }
    return *this;
  }

  operator T() const {
    T value;
    if (endianness == std::endian::native) {
      std::memcpy(&value, data, sizeof(T));
    } else {
      char tmp[sizeof(T)];
      for (size_t i = 0; i < sizeof(T); ++i) {
        tmp[i] = data[sizeof(T) - i - 1];
        std::memcpy(&value, tmp, sizeof(T));
      }
    }
    return value;
  }
};

} // namespace numeric::utils

#endif
