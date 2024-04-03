#ifndef NUMERIC_UTILS_ERROR_HPP_
#define NUMERIC_UTILS_ERROR_HPP_

#include <numeric/config.hpp>
#ifndef __HIP_DEVICE_COMPILE__
#include <cstdlib>
#include <fmt/core.h>
#endif

namespace numeric::utils {

namespace internal {

#ifndef __HIP_DEVICE_COMPILE__
template <typename... Ts>
[[noreturn]] void error_impl(const char *file, int line, const char *func,
                             const char *msg, Ts &&...args) {
  const std::string fmtmsg =
      fmt::format(fmt::runtime(msg), std::forward<Ts>(args)...);
  fmt::print(stderr, "file: {}({}) `{}`: {}\n", file, line, func, fmtmsg);
  exit(EXIT_FAILURE);
}
#endif

} // namespace internal

#ifdef __HIP_DEVICE_COMPILE__
#define NUMERIC_ERROR(...) abort()
#else
#define NUMERIC_ERROR(...)                                                     \
  ::numeric::utils::internal::error_impl(__FILE__, __LINE__,                   \
                                         NUMERIC_PRETTY_FUNCTION, __VA_ARGS__)
#endif

#define NUMERIC_ERROR_IF(cond, ...)                                            \
  if (cond) {                                                                  \
    NUMERIC_ERROR(__VA_ARGS__);                                                \
  }

} // namespace numeric::utils

#endif
