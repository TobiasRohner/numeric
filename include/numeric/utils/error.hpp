#ifndef NUMERIC_UTILS_ERROR_HPP_
#define NUMERIC_UTILS_ERROR_HPP_

#include <cstdlib>
#include <fmt/core.h>
#include <numeric/config.hpp>

namespace numeric::utils {

namespace internal {

template <typename... Ts>
[[noreturn]] void error_impl(const char *file, int line, const char *func,
                             const char *msg, Ts &&...args) {
  const std::string fmtmsg =
      fmt::format(fmt::runtime(msg), std::forward<Ts>(args)...);
  fmt::print(stderr, "file: {}({}) `{}`: {}\n", file, line, func, fmtmsg);
  exit(EXIT_FAILURE);
}

} // namespace internal

#define NUMERIC_ERROR(...)                                                     \
  ::numeric::utils::internal::error_impl(__FILE__, __LINE__,                   \
                                         NUMERIC_PRETTY_FUNCTION, __VA_ARGS__)

#define NUMERIC_ERROR_IF(cond, ...)                                            \
  if (cond) {                                                                  \
    NUMERIC_ERROR(__VA_ARGS__);                                                \
  }

} // namespace numeric::utils

#endif
