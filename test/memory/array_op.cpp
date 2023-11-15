#include <gtest/gtest.h>
#include <numeric/math/functions.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>

template <typename T, typename Func> void test_unary(const Func &f) {
  numeric::memory::Layout<3> shape(2, 3, 4);
  numeric::memory::Array<T, 3> array(shape);
  for (size_t i = 0; i < array.size(); ++i) {
    array.raw()[i] = i + 1;
  }
  const auto expr = f(array);
  ASSERT_EQ(array.shape(0), expr.shape(0));
  ASSERT_EQ(array.shape(1), expr.shape(1));
  ASSERT_EQ(array.shape(2), expr.shape(2));
  for (numeric::dim_t i = 0; i < array.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < array.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < array.shape(2); ++k) {
        const auto val = f(array(i, j, k));
        const auto res = expr(i, j, k);
        ASSERT_EQ(val, res);
      }
    }
  }
}

template <typename T1, typename T2, typename Func>
void test_binary(const Func &f) {
  numeric::memory::Layout<3> shape(2, 3, 4);
  numeric::memory::Array<T1, 3> a(shape);
  numeric::memory::Array<T2, 3> b(shape);
  for (size_t i = 0; i < a.size(); ++i) {
    a.raw()[i] = i;
    b.raw()[i] = a.size() - i;
  }
  const auto expr = f(a, b);
  ASSERT_EQ(a.shape(0), expr.shape(0));
  ASSERT_EQ(a.shape(1), expr.shape(1));
  ASSERT_EQ(a.shape(2), expr.shape(2));
  for (numeric::dim_t i = 0; i < a.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < a.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < a.shape(2); ++k) {
        const auto val = f(a(i, j, k), b(i, j, k));
        const auto res = expr(i, j, k);
        ASSERT_EQ(val, res);
      }
    }
  }
}

TEST(array_op, shape) {
  const numeric::memory::Layout<3> shape_a(2, 3, 4);
  const numeric::memory::Layout<3> shape_b(2, 1, 4);
  numeric::memory::Array<int, 3> a(shape_a);
  numeric::memory::Array<int, 3> b(shape_b);
  const numeric::memory::Layout<3> shape_ab = (a + b).layout();
  ASSERT_EQ(shape_ab.shape(0), 2);
  ASSERT_EQ(shape_ab.shape(1), 3);
  ASSERT_EQ(shape_ab.shape(2), 4);
}

TEST(array_op, unary_plus) {
  test_unary<double>([](const auto &val) { return +val; });
}

TEST(array_op, unary_minus) {
  test_unary<double>([](const auto &val) { return -val; });
}

TEST(array_op, unary_bitwise_not) {
  test_unary<int>([](const auto &val) { return ~val; });
}

TEST(array_op, unary_logical_not) {
  test_unary<int>([](const auto &val) { return !val; });
}

TEST(array_op, unary_pow_2) {
  numeric::memory::Layout<3> shape(2, 3, 4);
  numeric::memory::Array<double, 3> array(shape);
  for (size_t i = 0; i < array.size(); ++i) {
    array.raw()[i] = i;
  }
  const auto expr = pow<2>(array);
  ASSERT_EQ(array.shape(0), expr.shape(0));
  ASSERT_EQ(array.shape(1), expr.shape(1));
  ASSERT_EQ(array.shape(2), expr.shape(2));
  for (numeric::dim_t i = 0; i < array.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < array.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < array.shape(2); ++k) {
        const auto val = numeric::math::pow<2>(array(i, j, k));
        const auto res = expr(i, j, k);
        ASSERT_EQ(val, res);
      }
    }
  }
}

TEST(array_op, unary_pow_m2) {
  numeric::memory::Layout<3> shape(2, 3, 4);
  numeric::memory::Array<double, 3> array(shape);
  for (size_t i = 0; i < array.size(); ++i) {
    array.raw()[i] = i;
  }
  const auto expr = pow<-2>(array);
  ASSERT_EQ(array.shape(0), expr.shape(0));
  ASSERT_EQ(array.shape(1), expr.shape(1));
  ASSERT_EQ(array.shape(2), expr.shape(2));
  for (numeric::dim_t i = 0; i < array.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < array.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < array.shape(2); ++k) {
        const auto val = numeric::math::pow<-2>(array(i, j, k));
        const auto res = expr(i, j, k);
        ASSERT_EQ(val, res);
      }
    }
  }
}

TEST(array_op, unary_exp) {
  numeric::memory::Layout<3> shape(2, 3, 4);
  numeric::memory::Array<double, 3> array(shape);
  for (size_t i = 0; i < array.size(); ++i) {
    array.raw()[i] = i;
  }
  const auto expr = exp(array);
  ASSERT_EQ(array.shape(0), expr.shape(0));
  ASSERT_EQ(array.shape(1), expr.shape(1));
  ASSERT_EQ(array.shape(2), expr.shape(2));
  for (numeric::dim_t i = 0; i < array.shape(0); ++i) {
    for (numeric::dim_t j = 0; j < array.shape(1); ++j) {
      for (numeric::dim_t k = 0; k < array.shape(2); ++k) {
        const auto val = numeric::math::exp(array(i, j, k));
        const auto res = expr(i, j, k);
        ASSERT_EQ(val, res);
      }
    }
  }
}

TEST(array_op, binary_plus) {
  test_binary<double, float>(
      [](const auto &a, const auto &b) { return a + b; });
}

TEST(array_op, unary_plus_scalar) {
  test_unary<double>([](const auto &val) { return val + 2; });
}

TEST(array_op, unary_scalar_plus) {
  test_unary<double>([](const auto &val) { return 2 + val; });
}

TEST(array_op, binary_minus) {
  test_binary<double, float>(
      [](const auto &a, const auto &b) { return a - b; });
}

TEST(array_op, unary_minus_scalar) {
  test_unary<double>([](const auto &val) { return val - 2; });
}

TEST(array_op, unary_scalar_minus) {
  test_unary<double>([](const auto &val) { return 2 - val; });
}

TEST(array_op, binary_multiply) {
  test_binary<double, float>(
      [](const auto &a, const auto &b) { return a * b; });
}

TEST(array_op, unary_multiply_scalar) {
  test_unary<double>([](const auto &val) { return val * 2; });
}

TEST(array_op, unary_scalar_multiply) {
  test_unary<double>([](const auto &val) { return 2 * val; });
}

TEST(array_op, binary_divide) {
  test_binary<double, float>(
      [](const auto &a, const auto &b) { return a / b; });
}

TEST(array_op, unary_divide_scalar) {
  test_unary<double>([](const auto &val) { return val / 2; });
}

TEST(array_op, unary_scalar_divide) {
  test_unary<double>([](const auto &val) { return 2 / val; });
}

TEST(array_op, binary_modulo) {
  test_binary<long, int>([](const auto &a, const auto &b) { return a % b; });
}

TEST(array_op, unary_modulo_scalar) {
  test_unary<int>([](const auto &val) { return val % 2; });
}

TEST(array_op, unary_scalar_modulo) {
  test_unary<int>([](const auto &val) { return 2 % val; });
}

TEST(array_op, binary_bitwise_and) {
  test_binary<long, int>([](const auto &a, const auto &b) { return a & b; });
}

TEST(array_op, unary_bitwise_and_scalar) {
  test_unary<int>([](const auto &val) { return val & 2; });
}

TEST(array_op, unary_scalar_bitwise_and) {
  test_unary<int>([](const auto &val) { return 2 & val; });
}

TEST(array_op, binary_bitise_or) {
  test_binary<long, int>([](const auto &a, const auto &b) { return a | b; });
}

TEST(array_op, unary_bitwise_or_scalar) {
  test_unary<int>([](const auto &val) { return val | 2; });
}

TEST(array_op, unary_scalar_bitwise_or) {
  test_unary<int>([](const auto &val) { return 2 | val; });
}

TEST(array_op, binary_bitwise_xor) {
  test_binary<long, int>([](const auto &a, const auto &b) { return a ^ b; });
}

TEST(array_op, unary_bitwise_xor_scalar) {
  test_unary<int>([](const auto &val) { return val ^ 2; });
}

TEST(array_op, unary_scalar_bitwise_xor) {
  test_unary<int>([](const auto &val) { return 2 ^ val; });
}

TEST(array_op, binary_logical_and) {
  test_binary<long, int>([](const auto &a, const auto &b) { return a && b; });
}

TEST(array_op, unary_logical_and_scalar) {
  test_unary<int>([](const auto &val) { return val && 2; });
}

TEST(array_op, unary_scalar_logical_and) {
  test_unary<int>([](const auto &val) { return 2 && val; });
}

TEST(array_op, binary_logical_or) {
  test_binary<long, int>([](const auto &a, const auto &b) { return a || b; });
}

TEST(array_op, unary_logical_or_scalar) {
  test_unary<int>([](const auto &val) { return val || 2; });
}

TEST(array_op, unary_scalar_logical_or) {
  test_unary<int>([](const auto &val) { return 2 || val; });
}

TEST(array_op, binary_equals) {
  test_binary<long, int>([](const auto &a, const auto &b) { return a == b; });
}

TEST(array_op, unary_equals_scalar) {
  test_unary<int>([](const auto &val) { return val == 2; });
}

TEST(array_op, unary_scalar_equals) {
  test_unary<int>([](const auto &val) { return 2 == val; });
}

TEST(array_op, binary_not_equals) {
  test_binary<long, int>([](const auto &a, const auto &b) { return a != b; });
}

TEST(array_op, unary_not_equals_scalar) {
  test_unary<int>([](const auto &val) { return val != 2; });
}

TEST(array_op, unary_scalar_not_equals) {
  test_unary<int>([](const auto &val) { return 2 != val; });
}

TEST(array_op, binary_less) {
  test_binary<long, int>([](const auto &a, const auto &b) { return a < b; });
}

TEST(array_op, unary_less_scalar) {
  test_unary<int>([](const auto &val) { return val < 2; });
}

TEST(array_op, unary_scalar_less) {
  test_unary<int>([](const auto &val) { return 2 < val; });
}

TEST(array_op, binary_greater) {
  test_binary<long, int>([](const auto &a, const auto &b) { return a > b; });
}

TEST(array_op, unary_greater_scalar) {
  test_unary<int>([](const auto &val) { return val > 2; });
}

TEST(array_op, unary_scalar_greater) {
  test_unary<int>([](const auto &val) { return 2 > val; });
}

TEST(array_op, binary_leq) {
  test_binary<long, int>([](const auto &a, const auto &b) { return a <= b; });
}

TEST(array_op, unary_leq_scalar) {
  test_unary<int>([](const auto &val) { return val <= 2; });
}

TEST(array_op, unary_scalar_leq) {
  test_unary<int>([](const auto &val) { return 2 <= val; });
}

TEST(array_op, binary_geq) {
  test_binary<long, int>([](const auto &a, const auto &b) { return a >= b; });
}

TEST(array_op, unary_geq_scalar) {
  test_unary<int>([](const auto &val) { return val >= 2; });
}

TEST(array_op, unary_scalar_geq) {
  test_unary<int>([](const auto &val) { return 2 >= val; });
}
