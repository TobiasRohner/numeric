#include <gtest/gtest.h>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>
#include <numeric/memory/linspace.hpp>
#include <numeric/memory/meshgrid.hpp>

TEST(meshgrid, shape) {
  numeric::memory::Linspace<double> x(0, 1, 10);
  numeric::memory::Linspace<double> y(1, 2, 20);
  numeric::memory::Linspace<double> z(2, 3, 30);
  auto [X, Y, Z] = numeric::memory::meshgrid(x, y, z);
  ASSERT_EQ(X.dim, 3);
  ASSERT_EQ(X.shape(0), 10);
  ASSERT_EQ(X.shape(1), 20);
  ASSERT_EQ(X.shape(2), 30);
  ASSERT_EQ(Y.dim, 3);
  ASSERT_EQ(Y.shape(0), 10);
  ASSERT_EQ(Y.shape(1), 20);
  ASSERT_EQ(Y.shape(2), 30);
  ASSERT_EQ(Z.dim, 3);
  ASSERT_EQ(Z.shape(0), 10);
  ASSERT_EQ(Z.shape(1), 20);
  ASSERT_EQ(Z.shape(2), 30);
}

TEST(meshgrid, elements) {
  numeric::memory::Linspace<double> x(0, 1, 10);
  numeric::memory::Linspace<double> y(1, 2, 20);
  numeric::memory::Linspace<double> z(2, 3, 30);
  const auto [X, Y, Z] = numeric::memory::meshgrid(x, y, z);
  for (numeric::dim_t i = 0; i < 10; ++i) {
    for (numeric::dim_t j = 0; j < 20; ++j) {
      for (numeric::dim_t k = 0; k < 30; ++k) {
        const double Xel = X(i, j, k);
        const double xel = x(i);
        ASSERT_EQ(Xel, xel);
        const double Yel = Y(i, j, k);
        const double yel = y(j);
        ASSERT_EQ(Yel, yel);
        const double Zel = Z(i, j, k);
        const double zel = z(k);
        ASSERT_EQ(Zel, zel);
      }
    }
  }
}

TEST(meshgrid, to_array) {
  numeric::memory::Linspace<double> x(0, 1, 10);
  numeric::memory::Linspace<double> y(1, 2, 20);
  numeric::memory::Linspace<double> z(2, 3, 30);
  const auto [X, Y, Z] = numeric::memory::meshgrid(x, y, z);
  const numeric::memory::Array<double, 3> aX = X;
  const numeric::memory::Array<double, 3> aY = Y;
  const numeric::memory::Array<double, 3> aZ = Z;
  for (numeric::dim_t i = 0; i < 10; ++i) {
    for (numeric::dim_t j = 0; j < 20; ++j) {
      for (numeric::dim_t k = 0; k < 30; ++k) {
        const double Xel = X(i, j, k);
        const double aXel = aX(i, j, k);
        ASSERT_EQ(Xel, aXel);
        const double Yel = Y(i, j, k);
        const double aYel = aY(i, j, k);
        ASSERT_EQ(Yel, aYel);
        const double Zel = Z(i, j, k);
        const double aZel = aZ(i, j, k);
        ASSERT_EQ(Zel, aZel);
      }
    }
  }
}

TEST(meshgrid, broadcast) {
  numeric::memory::Linspace<double> x(0, 1, 10);
  numeric::memory::Linspace<double> y(1, 2, 20);
  numeric::memory::Linspace<double> z(2, 3, 30);
  const auto [X, Y, Z] = numeric::memory::meshgrid(x, y, z);
  const numeric::memory::Shape<4> shape(5, 10, 20, 30);
  const auto brdcX = X.broadcast(shape);
  const auto brdcY = Y.broadcast(shape);
  const auto brdcZ = Z.broadcast(shape);
  for (numeric::dim_t i = 0; i < 5; ++i) {
    for (numeric::dim_t j = 0; j < 10; ++j) {
      for (numeric::dim_t k = 0; k < 20; ++k) {
        for (numeric::dim_t l = 0; l < 30; ++l) {
          const double Xel = X(j, k, l);
          const double brdcXel = brdcX(i, j, k, l);
          ASSERT_EQ(Xel, brdcXel);
          const double Yel = Y(j, k, l);
          const double brdcYel = brdcY(i, j, k, l);
          ASSERT_EQ(Yel, brdcYel);
          const double Zel = Z(j, k, l);
          const double brdcZel = brdcZ(i, j, k, l);
          ASSERT_EQ(Zel, brdcZel);
        }
      }
    }
  }
}

TEST(meshgrid, expr) {
  numeric::memory::Linspace<double> x(0, 1, 10);
  numeric::memory::Linspace<double> y(1, 2, 20);
  numeric::memory::Linspace<double> z(2, 3, 30);
  const auto [X, Y, Z] = numeric::memory::meshgrid(x, y, z);
  const auto f = X * X + Y * Y + Z * Z;
  for (numeric::dim_t i = 0; i < 10; ++i) {
    for (numeric::dim_t j = 0; j < 20; ++j) {
      for (numeric::dim_t k = 0; k < 30; ++k) {
        const double fel = f(i, j, k);
        const double expr = X(i, j, k) * X(i, j, k) + Y(i, j, k) * Y(i, j, k) +
                            Z(i, j, k) * Z(i, j, k);
        ASSERT_EQ(fel, expr);
      }
    }
  }
}

TEST(meshgrid, expr_assign) {
  numeric::memory::Linspace<double> x(0, 1, 10);
  numeric::memory::Linspace<double> y(1, 2, 20);
  numeric::memory::Linspace<double> z(2, 3, 30);
  const auto [X, Y, Z] = numeric::memory::meshgrid(x, y, z);
  const numeric::memory::Array<double, 3> f = X * X + Y * Y + Z * Z;
  for (numeric::dim_t i = 0; i < 10; ++i) {
    for (numeric::dim_t j = 0; j < 20; ++j) {
      for (numeric::dim_t k = 0; k < 30; ++k) {
        const double fel = f(i, j, k);
        const double expr = X(i, j, k) * X(i, j, k) + Y(i, j, k) * Y(i, j, k) +
                            Z(i, j, k) * Z(i, j, k);
        ASSERT_EQ(fel, expr);
      }
    }
  }
}
