#ifndef NUMERIC_UTILS_TYPE_NAME_HPP_
#define NUMERIC_UTILS_TYPE_NAME_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>

namespace numeric::utils {

namespace detail {

template <typename T> [[nodiscard]] constexpr std::string_view raw_type_name() {
#ifndef _MSC_VER
  return __PRETTY_FUNCTION__;
#else
  return __FUNCSIG__;
#endif
}

struct TypeNameFormat {
  std::size_t junk_leading = 0;
  std::size_t junk_total = 0;
};

constexpr TypeNameFormat type_name_format = [] {
  TypeNameFormat ret;
  std::string_view sample = raw_type_name<int>();
  ret.junk_leading = sample.find("int");
  ret.junk_total = sample.size() - 3;
  return ret;
}();
static_assert(type_name_format.junk_leading != std::size_t(-1),
              "Unable to determine the type name format on this compiler.");

template <typename T>
static constexpr auto type_name_storage = [] {
  std::array<char, raw_type_name<T>().size() - type_name_format.junk_total + 1>
      ret{};
  std::copy_n(raw_type_name<T>().data() + type_name_format.junk_leading,
              ret.size() - 1, ret.data());
  return ret;
}();

} // namespace detail

template <typename T> [[nodiscard]] constexpr std::string_view type_name() {
  return {detail::type_name_storage<T>.data(),
          detail::type_name_storage<T>.size() - 1};
}

} // namespace numeric::utils

#endif
