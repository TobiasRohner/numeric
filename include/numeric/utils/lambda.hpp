#ifndef NUMERIC_UTILS_LAMBDA_HPP_
#define NUMERIC_UTILS_LAMBDA_HPP_

#include <numeric/utils/forward.hpp>

namespace numeric::utils {

template <typename Func> struct Lambda {
  Func f;
  const char *source;

  Lambda(Func f_, const char *src_) : f(f_), source(src_) {}

  template <typename... Args> constexpr auto operator()(Args &&...args) const {
    return f(forward<Args>(args)...);
  }
};

template <typename Func> Lambda<Func> make_lambda(Func f, const char *src) {
  return Lambda<Func>(f, src);
}

} // namespace numeric::utils

#define NUMERIC_LAMBDA(src) ::numeric::utils::make_lambda(src, #src);

#endif
