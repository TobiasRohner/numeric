#include <gtest/gtest.h>
#include <numeric/utils/tuple.hpp>
#include <string>

TEST(tuple, construction) {
  numeric::utils::Tuple<char, int, std::string> t('a', 42, "Hello World!");
  static_assert(
      numeric::meta::is_same_v<decltype(t.template get<0>()), char &>);
  static_assert(numeric::meta::is_same_v<decltype(t.template get<1>()), int &>);
  static_assert(
      numeric::meta::is_same_v<decltype(t.template get<2>()), std::string &>);
  ASSERT_EQ(t.template get<0>(), 'a');
  ASSERT_EQ(t.template get<1>(), 42);
  ASSERT_EQ(t.template get<2>(), "Hello World!");
}

TEST(tuple, assignment) {
  numeric::utils::Tuple<char, int, std::string> t;
  t.template get<0>() = 'a';
  t.template get<1>() = 42;
  t.template get<2>() = "Hello World!";
  ASSERT_EQ(t.template get<0>(), 'a');
  ASSERT_EQ(t.template get<1>(), 42);
  ASSERT_EQ(t.template get<2>(), "Hello World!");
}

TEST(tuple, structured_binding) {
  numeric::utils::Tuple<char, int, std::string> t('a', 42, "Hello World!");
  const auto &[a, b, c] = t;
  ASSERT_EQ(a, 'a');
  ASSERT_EQ(b, 42);
  ASSERT_EQ(c, "Hello World!");
}
