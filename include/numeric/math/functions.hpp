#ifndef NUMERIC_MATH_FUNCTIONS_HPP_
#define NUMERIC_MATH_FUNCTIONS_HPP_

#include <numeric/config.hpp>
#ifndef __HIP_DEVICE_COMPILE__
#include <cmath>
#endif
#if NUMERIC_ENABLE_HIP
#include <numeric/hip/runtime.hpp>
#endif

namespace numeric::math {

template <typename T> NUMERIC_HOST_DEVICE T min(T a, T b) {
  return a < b ? a : b;
}

template <typename T> NUMERIC_HOST_DEVICE T max(T a, T b) {
  return a < b ? b : a;
}

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE auto div_up(Lhs lhs, Rhs rhs) {
  return (lhs + rhs - 1) / rhs;
}

template <dim_t p, typename Scalar> NUMERIC_HOST_DEVICE Scalar pow(Scalar val) {
  if constexpr (p == 0) {
    return static_cast<Scalar>(1);
  } else if constexpr (p < 0) {
    return pow<-p>(static_cast<Scalar>(1) / val);
  } else {
    const Scalar valpow2 = pow<p / 2>(val);
    if constexpr (p % 2 == 0) {
      return valpow2 * valpow2;
    } else {
      return valpow2 * valpow2 * val;
    }
  }
}

NUMERIC_HOST_DEVICE inline float abs(float x) {
#ifndef __HIP_DEVICE_COMPILE__
  return std::fabs(x);
#else
  return fabsf(x);
#endif
}

NUMERIC_HOST_DEVICE inline double abs(double x) {
#ifndef __HIP_DEVICE_COMPILE__
  return std::fabs(x);
#else
  return fabs(x);
#endif
}

NUMERIC_HOST_DEVICE inline long double abs(long double x) {
#ifndef __HIP_DEVICE_COMPILE__
  return std::fabs(x);
#else
  return fabs(static_cast<double>(x));
#endif
}

NUMERIC_HOST_DEVICE inline float exp(float x) {
#ifndef __HIP_DEVICE_COMPILE__
  return std::exp(x);
#else
  return expf(x);
#endif
}

NUMERIC_HOST_DEVICE inline double exp(double x) {
#ifndef __HIP_DEVICE_COMPILE__
  return std::exp(x);
#else
  return exp(x);
#endif
}

NUMERIC_HOST_DEVICE inline long double exp(long double x) {
#ifndef __HIP_DEVICE_COMPILE__
  return std::exp(x);
#else
  return exp(x);
#endif
}

} // namespace numeric::math

#endif
