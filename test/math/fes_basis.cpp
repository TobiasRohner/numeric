#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numeric/math/fes/basis_l2.hpp>

template <typename Element> struct EvalPoints;

template <numeric::dim_t Order> struct EvalPoints<numeric::mesh::Point<Order>> {
  static constexpr numeric::dim_t num_points = 1;
  static constexpr double points[num_points][1] = {{0}};
};

template <numeric::dim_t Order>
struct EvalPoints<numeric::mesh::Segment<Order>> {
  static constexpr numeric::dim_t num_points = 3;
  static constexpr double points[num_points][1] = {{0.}, {0.5}, {1.}};
};

template <numeric::dim_t Order> struct EvalPoints<numeric::mesh::Tria<Order>> {
  static constexpr numeric::dim_t num_points = 4;
  static constexpr double points[num_points][2] = {
      {0., 0.}, {1., 0.}, {0., 1.}, {1. / 3, 1. / 3}};
};

template <numeric::dim_t Order> struct EvalPoints<numeric::mesh::Quad<Order>> {
  static constexpr numeric::dim_t num_points = 5;
  static constexpr double points[num_points][2] = {
      {0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}, {0.5, 0.5}};
};

template <numeric::dim_t Order> struct EvalPoints<numeric::mesh::Tetra<Order>> {
  static constexpr numeric::dim_t num_points = 5;
  static constexpr double points[num_points][3] = {{0., 0., 0.},
                                                   {1., 0., 0.},
                                                   {0., 1., 0.},
                                                   {0., 0., 1.},
                                                   {1. / 3, 1. / 3, 1. / 3}};
};

template <numeric::dim_t Order> struct EvalPoints<numeric::mesh::Cube<Order>> {
  static constexpr numeric::dim_t num_points = 9;
  static constexpr double points[num_points][3] = {
      {0., 0., 0.}, {1., 0., 0.}, {1., 1., 0.}, {0., 1., 0.},   {0., 0., 1.},
      {1., 0., 1.}, {1., 1., 1.}, {0., 1., 0.}, {0.5, 0.5, 0.5}};
};

template <typename Basis, typename Element> static void test_element_sum() {
  // TODO: Replace eval point by better points
  static constexpr numeric::dim_t dim = Element::dim;
  static constexpr numeric::dim_t num_points = EvalPoints<Element>::num_points;
  static constexpr numeric::dim_t num_basis_functs =
      Basis::template num_basis_functions<Element>();

  double basis_functs[num_basis_functs];
  for (numeric::dim_t i = 0; i < num_points; ++i) {
    Basis::template eval<Element>(basis_functs, EvalPoints<Element>::points[i]);
    double sum = 0;
    for (numeric::dim_t j = 0; j < num_basis_functs; ++j) {
      sum += basis_functs[j];
    }
    ASSERT_NEAR(sum, 1, 1e-8);
  }
}

template <typename Basis, typename Element>
static void test_element_gradient() {
  static constexpr numeric::dim_t dim = Element::dim;
  static constexpr numeric::dim_t num_points = EvalPoints<Element>::num_points;
  static constexpr numeric::dim_t num_basis_functs =
      Basis::template num_basis_functions<Element>();
  const double dx = std::sqrt(std::numeric_limits<double>::epsilon());

  double point_l[dim];
  double point_r[dim];
  double basis_functs_l[num_basis_functs];
  double basis_functs_r[num_basis_functs];
  double gradients[num_basis_functs][dim == 0 ? numeric::dim_t(1) : dim];
  for (numeric::dim_t i = 0; i < num_points; ++i) {
    Basis::template gradient<Element>(gradients,
                                      EvalPoints<Element>::points[i]);
    for (numeric::dim_t j = 0; j < dim; ++j) {
      for (numeric::dim_t k = 0; k < dim; ++k) {
        point_l[k] = EvalPoints<Element>::points[i][k];
        point_r[k] = EvalPoints<Element>::points[i][k];
        if (k == j) {
          point_l[k] -= dx / 2;
          point_r[k] += dx / 2;
        }
      }
      Basis::template eval<Element>(basis_functs_l, point_l);
      Basis::template eval<Element>(basis_functs_r, point_r);
      for (numeric::dim_t k = 0; k < num_basis_functs; ++k) {
        const double dfkdxj = (basis_functs_r[k] - basis_functs_l[k]) / dx;
        ASSERT_NEAR(gradients[k][j], dfkdxj, 1e-8);
      }
    }
  }
}

TEST(BasisL2, SumOrder0) {
  using basis_t = numeric::math::fes::BasisL2<0>;
  test_element_sum<basis_t, numeric::mesh::Point<1>>();
  test_element_sum<basis_t, numeric::mesh::Segment<1>>();
  test_element_sum<basis_t, numeric::mesh::Tria<1>>();
  test_element_sum<basis_t, numeric::mesh::Quad<1>>();
  test_element_sum<basis_t, numeric::mesh::Tetra<1>>();
  test_element_sum<basis_t, numeric::mesh::Cube<1>>();
}

TEST(BasisL2, GradientOrder0) {
  using basis_t = numeric::math::fes::BasisL2<0>;
  test_element_gradient<basis_t, numeric::mesh::Point<1>>();
  test_element_gradient<basis_t, numeric::mesh::Segment<1>>();
  test_element_gradient<basis_t, numeric::mesh::Tria<1>>();
  test_element_gradient<basis_t, numeric::mesh::Quad<1>>();
  test_element_gradient<basis_t, numeric::mesh::Tetra<1>>();
  test_element_gradient<basis_t, numeric::mesh::Cube<1>>();
}

TEST(BasisL2, SumOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_sum<basis_t, numeric::mesh::Point<1>>();
  test_element_sum<basis_t, numeric::mesh::Segment<1>>();
  test_element_sum<basis_t, numeric::mesh::Tria<1>>();
  test_element_sum<basis_t, numeric::mesh::Quad<1>>();
  test_element_sum<basis_t, numeric::mesh::Tetra<1>>();
  test_element_sum<basis_t, numeric::mesh::Cube<1>>();
}

TEST(BasisL2, GradientOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_gradient<basis_t, numeric::mesh::Point<1>>();
  test_element_gradient<basis_t, numeric::mesh::Segment<1>>();
  test_element_gradient<basis_t, numeric::mesh::Tria<1>>();
  test_element_gradient<basis_t, numeric::mesh::Quad<1>>();
  test_element_gradient<basis_t, numeric::mesh::Tetra<1>>();
  test_element_gradient<basis_t, numeric::mesh::Cube<1>>();
}
