#ifndef NUMERIC_MATH_DIHEDRAL_GROUP_HPP_
#define NUMERIC_MATH_DIHEDRAL_GROUP_HPP_

#include <numeric/config.hpp>
#ifndef __HIP_DEVICE_COMPILE__
#include <ostream>
#endif

namespace numeric::math {

template <dim_t> struct DihedralGroupElement;
template <dim_t N>
constexpr NUMERIC_HOST_DEVICE DihedralGroupElement<N>
operator*(const DihedralGroupElement<N> &,
          const DihedralGroupElement<N> &) noexcept;
template <dim_t N>
constexpr NUMERIC_HOST_DEVICE DihedralGroupElement<N>
operator/(const DihedralGroupElement<N> &,
          const DihedralGroupElement<N> &) noexcept;
template <dim_t N>
constexpr NUMERIC_HOST_DEVICE bool
operator==(const DihedralGroupElement<N> &,
           const DihedralGroupElement<N> &) noexcept;
template <dim_t N>
constexpr NUMERIC_HOST_DEVICE bool
operator!=(const DihedralGroupElement<N> &,
           const DihedralGroupElement<N> &) noexcept;

template <dim_t N_> struct DihedralGroupElement {
  enum Type { REFLECTION, ROTATION };

  static constexpr dim_t N = N_;
  Type type;
  dim_t n;

  NUMERIC_HOST_DEVICE DihedralGroupElement() { *this = identity(); }
  NUMERIC_HOST_DEVICE DihedralGroupElement(Type type_, dim_t n_)
      : type(type_), n(n_) {}
  DihedralGroupElement(const DihedralGroupElement &) = default;
  DihedralGroupElement &operator=(const DihedralGroupElement &) = default;

  constexpr NUMERIC_HOST_DEVICE bool is_identity() const noexcept {
    return *this == identity();
  }

  constexpr NUMERIC_HOST_DEVICE DihedralGroupElement &
  operator*=(const DihedralGroupElement &other) noexcept {
    *this = *this * other;
    return *this;
  }

  constexpr NUMERIC_HOST_DEVICE DihedralGroupElement &
  operator/=(const DihedralGroupElement &other) noexcept {
    *this = *this / other;
    return *this;
  }

  constexpr NUMERIC_HOST_DEVICE DihedralGroupElement inverse() const noexcept {
    return identity() / *this;
  }

  static constexpr NUMERIC_HOST_DEVICE DihedralGroupElement
  identity() noexcept {
    return DihedralGroupElement<N>{ROTATION, 0};
  }

  static constexpr NUMERIC_HOST_DEVICE DihedralGroupElement
  reflection(dim_t n) noexcept {
    while (n < 0) {
      n += N;
    }
    return DihedralGroupElement<N>{REFLECTION, n % N};
  }

  static constexpr NUMERIC_HOST_DEVICE DihedralGroupElement
  rotation(dim_t n) noexcept {
    while (n < 0) {
      n += N;
    }
    return DihedralGroupElement<N>{ROTATION, n % N};
  }
};

template <dim_t N>
constexpr NUMERIC_HOST_DEVICE DihedralGroupElement<N>
operator*(const DihedralGroupElement<N> &a,
          const DihedralGroupElement<N> &b) noexcept {
  using DHN = DihedralGroupElement<N>;
  if (a.type == DHN::REFLECTION && b.type == DHN::REFLECTION) {
    return DHN::rotation((N + a.n - b.n) % N);
  } else if (a.type == DHN::REFLECTION && b.type == DHN::ROTATION) {
    return DHN::reflection((N + a.n - b.n) % N);
  } else if (a.type == DHN::ROTATION && b.type == DHN::REFLECTION) {
    return DHN::reflection((a.n + b.n) % N);
  } else if (a.type == DHN::ROTATION && b.type == DHN::ROTATION) {
    return DHN::rotation((a.n + b.n) % N);
  }
  return DHN(); // This should never be reached
}

template <dim_t N>
constexpr NUMERIC_HOST_DEVICE DihedralGroupElement<N>
operator/(const DihedralGroupElement<N> &a,
          const DihedralGroupElement<N> &b) noexcept {
  using DHN = DihedralGroupElement<N>;
  if (a.type == DHN::REFLECTION && b.type == DHN::REFLECTION) {
    return DHN::rotation((N + a.n - b.n) % N);
  } else if (a.type == DHN::REFLECTION && b.type == DHN::ROTATION) {
    return DHN::reflection((a.n + b.n) % N);
  } else if (a.type == DHN::ROTATION && b.type == DHN::REFLECTION) {
    return DHN::reflection((a.n + b.n) % N);
  } else if (a.type == DHN::ROTATION && b.type == DHN::ROTATION) {
    return DHN::rotation((N + a.n - b.n) % N);
  }
  return DHN(); // This should never be reached
}

template <dim_t N>
constexpr NUMERIC_HOST_DEVICE bool
operator==(const DihedralGroupElement<N> &a,
           const DihedralGroupElement<N> &b) noexcept {
  return a.type == b.type && a.n == b.n;
}

template <dim_t N>
constexpr NUMERIC_HOST_DEVICE bool
operator!=(const DihedralGroupElement<N> &a,
           const DihedralGroupElement<N> &b) noexcept {
  return !(a == b);
}

#ifndef __HIP_DEVICE_COMPILE__
template <dim_t N>
std::ostream &operator<<(std::ostream &os,
                         const DihedralGroupElement<N> &element) {
  return os << (element.type == DihedralGroupElement<N>::ROTATION ? "R" : "S")
            << element.n;
}
#endif

} // namespace numeric::math

#endif
