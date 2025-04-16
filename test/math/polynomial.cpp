#include <gtest/gtest.h>
#include <numeric/math/polynomial.hpp>
#include <random>

using namespace numeric::math;
using namespace numeric;

// Test the equispaced nodes: they should be x_i = i/order for i=0,...,order.
TEST(NodesEquispacedTest, NodeCalculation) {
  const dim_t order = 5;
  NodesEquispaced nodes(order);
  for (dim_t i = 0; i <= order; ++i) {
    double expected = static_cast<double>(i) / order;
    EXPECT_NEAR(nodes.node<double>(i), expected, 1e-9);
  }
}

// Test the Chebyshev nodes: they should be x_i = cos(i*pi/order) for
// i=0,...,order.
TEST(NodesChebyshevTest, NodeCalculation) {
  const dim_t order = 4;
  NodesChebyshev nodes(order);
  for (dim_t i = 0; i <= order; ++i) {
    double expected = std::cos(i * M_PI / order);
    EXPECT_NEAR(nodes.node<double>(i), expected, 1e-9);
  }
}

// Test Lagrange interpolation evaluation with a known quadratic function
// f(x)=x^2. When using nodes from an interpolator of degree 2, the interpolated
// value should match f(x).
TEST(LagrangeTest, EvalPolynomialExactForQuadratic) {
  const dim_t order = 2;
  NodesEquispaced nodes(order);
  Lagrange<NodesEquispaced> lag(nodes);

  // We compute function values for f(x)=x^2 at the equispaced nodes.
  double y[3];
  for (dim_t i = 0; i <= order; ++i) {
    double xi = lag.node<double>(i);
    y[i] = xi * xi;
  }

  // Test over a set of points in [0,1].
  for (double x = 0.0; x <= 1.0; x += 0.1) {
    double expected = x * x;
    double interp = lag.eval<double>(y, x);
    EXPECT_NEAR(interp, expected, 1e-6);
  }
}

// Test the numerical derivative evaluation: For f(x)=x^2, the derivative is
// f'(x)=2x.
TEST(LagrangeTest, DerivativeForQuadratic) {
  const dim_t order = 2;
  NodesEquispaced nodes(order);
  Lagrange<NodesEquispaced> lag(nodes);

  // Compute f(x)=x^2 at equispaced nodes.
  double y[3];
  for (dim_t i = 0; i <= order; ++i) {
    double xi = lag.node<double>(i);
    y[i] = xi * xi;
  }

  // Compare the computed derivative with the exact derivative 2*x.
  for (double x = 0.0; x <= 1.0; x += 0.1) {
    double expected_deriv = 2 * x;
    double computed_deriv = lag.diff<double>(y, x);
    EXPECT_NEAR(computed_deriv, expected_deriv, 1e-6);
  }
}

// Test that the Lagrange basis functions sum to 1 for any x.
TEST(LagrangeTest, LagrangeBasisFunctionsSumToOne) {
  const dim_t order = 3;
  NodesEquispaced nodes(order);
  Lagrange<NodesEquispaced> lag(nodes);

  for (double x = 0.0; x <= 1.0; x += 0.1) {
    double sum = 0.0;
    for (dim_t i = 0; i <= order; ++i) {
      sum += lag.basis<double>(i, x);
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);
  }
}

// Test the values of individual Lagrange basis functions at the interpolation
// nodes. At x = node(j), basis(j) should be 1 and all others should be 0.
TEST(LagrangeTest, BasisValuesAtNodes) {
  const dim_t order = 3;
  NodesEquispaced nodes(order);
  Lagrange<NodesEquispaced> lag(nodes);

  for (dim_t j = 0; j <= order; ++j) {
    double xj = lag.node<double>(j);
    for (dim_t i = 0; i <= order; ++i) {
      double b = lag.basis<double>(i, xj);
      if (i == j) {
        EXPECT_NEAR(b, 1.0, 1e-9);
      } else {
        EXPECT_NEAR(b, 0.0, 1e-9);
      }
    }
  }
}
