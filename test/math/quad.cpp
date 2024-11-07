#include <cmath>
#include <gtest/gtest.h>
#include <numeric/math/quad/quad_rule.hpp>

TEST(quad, segment) {
  for (numeric::dim_t order = 1; order <= 10; ++order) {
    const auto [points, weights] =
        numeric::math::quad::quad_rule<numeric::mesh::RefElSegment>(order);
    const numeric::dim_t num_points = points.shape(0);
    for (numeric::dim_t a = 0; a <= order; ++a) {
      const auto f = [&](double x) { return std::pow(x, a); };
      double integral = 0;
      for (numeric::dim_t i = 0; i < num_points; ++i) {
        integral += weights(i) * f(points(i, 0));
      }
      const double exact = 1. / (a + 1);
      ASSERT_NEAR(integral, exact, 1e-8);
    }
  }
}

TEST(quad, quad) {
  for (numeric::dim_t order = 1; order <= 10; ++order) {
    const auto [points, weights] =
        numeric::math::quad::quad_rule<numeric::mesh::RefElQuad>(order);
    const numeric::dim_t num_points = points.shape(0);
    for (numeric::dim_t a = 0; a <= order; ++a) {
      for (numeric::dim_t b = 0; b <= order; ++b) {
        const auto f = [&](double x, double y) {
          return std::pow(x, a) * std::pow(y, b);
        };
        double integral = 0;
        for (numeric::dim_t i = 0; i < num_points; ++i) {
          integral += weights(i) * f(points(i, 0), points(i, 1));
        }
        const double exact = 1. / (a + 1) / (b + 1);
        ASSERT_NEAR(integral, exact, 1e-8);
      }
    }
  }
}

TEST(quad, cube) {
  for (numeric::dim_t order = 1; order <= 10; ++order) {
    const auto [points, weights] =
        numeric::math::quad::quad_rule<numeric::mesh::RefElCube>(order);
    const numeric::dim_t num_points = points.shape(0);
    for (numeric::dim_t a = 0; a <= order; ++a) {
      for (numeric::dim_t b = 0; b <= order; ++b) {
        for (numeric::dim_t c = 0; c <= order; ++c) {
          const auto f = [&](double x, double y, double z) {
            return std::pow(x, a) * std::pow(y, b) * std::pow(z, c);
          };
          double integral = 0;
          for (numeric::dim_t i = 0; i < num_points; ++i) {
            integral +=
                weights(i) * f(points(i, 0), points(i, 1), points(i, 2));
          }
          const double exact = 1. / (a + 1) / (b + 1) / (c + 1);
          ASSERT_NEAR(integral, exact, 1e-8);
        }
      }
    }
  }
}
