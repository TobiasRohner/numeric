#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numeric/math/fes/basis_h1.hpp>
#include <numeric/math/fes/basis_l2.hpp>

template <typename Element> struct EvalPoints;

template <> struct EvalPoints<numeric::mesh::RefElPoint> {
  static constexpr numeric::dim_t num_points = 1;
  static constexpr double points[num_points][1] = {{0}};
};

template <> struct EvalPoints<numeric::mesh::RefElSegment> {
  static constexpr numeric::dim_t num_points = 3;
  static constexpr double points[num_points][1] = {{0.}, {0.5}, {1.}};
};

template <> struct EvalPoints<numeric::mesh::RefElTria> {
  static constexpr numeric::dim_t num_points = 4;
  static constexpr double points[num_points][2] = {
      {0., 0.}, {1., 0.}, {0., 1.}, {1. / 3, 1. / 3}};
};

template <> struct EvalPoints<numeric::mesh::RefElQuad> {
  static constexpr numeric::dim_t num_points = 5;
  static constexpr double points[num_points][2] = {
      {0., 0.}, {1., 0.}, {1., 1.}, {0., 1.}, {0.5, 0.5}};
};

template <> struct EvalPoints<numeric::mesh::RefElTetra> {
  static constexpr numeric::dim_t num_points = 5;
  static constexpr double points[num_points][3] = {{0., 0., 0.},
                                                   {1., 0., 0.},
                                                   {0., 1., 0.},
                                                   {0., 0., 1.},
                                                   {1. / 3, 1. / 3, 1. / 3}};
};

template <> struct EvalPoints<numeric::mesh::RefElCube> {
  static constexpr numeric::dim_t num_points = 9;
  static constexpr double points[num_points][3] = {
      {0., 0., 0.}, {1., 0., 0.}, {1., 1., 0.}, {0., 1., 0.},   {0., 0., 1.},
      {1., 0., 1.}, {1., 1., 1.}, {0., 1., 0.}, {0.5, 0.5, 0.5}};
};

template <typename Basis, typename RefEl> static void test_element_sum() {
  // TODO: Replace eval point by better points
  static constexpr numeric::dim_t dim = RefEl::dim;
  static constexpr numeric::dim_t num_points = EvalPoints<RefEl>::num_points;
  static constexpr numeric::dim_t num_basis_functs =
      Basis::template num_basis_functions<RefEl>();

  double basis_functs[num_basis_functs];
  for (numeric::dim_t i = 0; i < num_points; ++i) {
    Basis::template eval<RefEl>(basis_functs, EvalPoints<RefEl>::points[i]);
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

template <typename Basis, typename Element>
static void test_element_num_basis_functions() {
  const numeric::dim_t num_basis_functs =
      Basis::template num_basis_functions<Element>();
  numeric::dim_t sum_of_interior_basis_functs = 0;
  sum_of_interior_basis_functs +=
      Basis::template num_interior_basis_functions<Element>();
  for (numeric::dim_t subelement = 0;
       subelement <
       Element::template num_subelements<numeric::mesh::RefElPoint>();
       ++subelement) {
    sum_of_interior_basis_functs +=
        Basis::template num_basis_functions<Element, numeric::mesh::RefElPoint>(
            subelement);
  }
  for (numeric::dim_t subelement = 0;
       subelement <
       Element::template num_subelements<numeric::mesh::RefElSegment>();
       ++subelement) {
    sum_of_interior_basis_functs += Basis::template num_basis_functions<
        Element, numeric::mesh::RefElSegment>(subelement);
  }
  for (numeric::dim_t subelement = 0;
       subelement <
       Element::template num_subelements<numeric::mesh::RefElTria>();
       ++subelement) {
    sum_of_interior_basis_functs +=
        Basis::template num_basis_functions<Element, numeric::mesh::RefElTria>(
            subelement);
  }
  for (numeric::dim_t subelement = 0;
       subelement <
       Element::template num_subelements<numeric::mesh::RefElQuad>();
       ++subelement) {
    sum_of_interior_basis_functs +=
        Basis::template num_basis_functions<Element, numeric::mesh::RefElQuad>(
            subelement);
  }
  for (numeric::dim_t subelement = 0;
       subelement <
       Element::template num_subelements<numeric::mesh::RefElTetra>();
       ++subelement) {
    sum_of_interior_basis_functs +=
        Basis::template num_basis_functions<Element, numeric::mesh::RefElTetra>(
            subelement);
  }
  for (numeric::dim_t subelement = 0;
       subelement <
       Element::template num_subelements<numeric::mesh::RefElCube>();
       ++subelement) {
    sum_of_interior_basis_functs +=
        Basis::template num_basis_functions<Element, numeric::mesh::RefElCube>(
            subelement);
  }
  ASSERT_EQ(num_basis_functs, sum_of_interior_basis_functs);
}

TEST(BasisL2, PointSumOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_sum<basis_t, numeric::mesh::RefElPoint>();
}

TEST(BasisL2, SegmentSumOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_sum<basis_t, numeric::mesh::RefElSegment>();
}

TEST(BasisL2, TriaSumOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_sum<basis_t, numeric::mesh::RefElTria>();
}

TEST(BasisL2, QuadSumOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_sum<basis_t, numeric::mesh::RefElQuad>();
}

TEST(BasisL2, TetraSumOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_sum<basis_t, numeric::mesh::RefElTetra>();
}

TEST(BasisL2, CubeSumOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_sum<basis_t, numeric::mesh::RefElCube>();
}

TEST(BasisL2, PointGradientOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_gradient<basis_t, numeric::mesh::RefElPoint>();
}

TEST(BasisL2, SegmentGradientOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_gradient<basis_t, numeric::mesh::RefElSegment>();
}

TEST(BasisL2, TriaGradientOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_gradient<basis_t, numeric::mesh::RefElTria>();
}

TEST(BasisL2, QuadGradientOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_gradient<basis_t, numeric::mesh::RefElQuad>();
}

TEST(BasisL2, TetraGradientOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_gradient<basis_t, numeric::mesh::RefElTetra>();
}

TEST(BasisL2, CubeGradientOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_gradient<basis_t, numeric::mesh::RefElCube>();
}

TEST(BasisL2, PointNumBasisFunctsOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElPoint>();
}

TEST(BasisL2, SegmentNumBasisFunctOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElSegment>();
}

TEST(BasisL2, TriaNumBasisFunctOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElTria>();
}

TEST(BasisL2, QuadNumBasisFunctOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElQuad>();
}

TEST(BasisL2, TetraNumBasisFunctOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElTetra>();
}

TEST(BasisL2, CubeNumBasisFunctOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElCube>();
}

TEST(BasisL2, NumBasisFunctsOrder1) {
  using basis_t = numeric::math::fes::BasisL2<1>;
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElPoint>();
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElSegment>();
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElTria>();
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElQuad>();
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElTetra>();
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElCube>();
}

TEST(BasisH1, SumOrder1) {
  using basis_t = numeric::math::fes::BasisH1<1>;
  test_element_sum<basis_t, numeric::mesh::RefElPoint>();
  test_element_sum<basis_t, numeric::mesh::RefElSegment>();
  test_element_sum<basis_t, numeric::mesh::RefElTria>();
  test_element_sum<basis_t, numeric::mesh::RefElQuad>();
  test_element_sum<basis_t, numeric::mesh::RefElTetra>();
  test_element_sum<basis_t, numeric::mesh::RefElCube>();
}

TEST(BasisH1, GradientOrder1) {
  using basis_t = numeric::math::fes::BasisH1<1>;
  test_element_gradient<basis_t, numeric::mesh::RefElPoint>();
  test_element_gradient<basis_t, numeric::mesh::RefElSegment>();
  test_element_gradient<basis_t, numeric::mesh::RefElTria>();
  test_element_gradient<basis_t, numeric::mesh::RefElQuad>();
  test_element_gradient<basis_t, numeric::mesh::RefElTetra>();
  test_element_gradient<basis_t, numeric::mesh::RefElCube>();
}

TEST(BasisH1, NumBasisFunctsOrder1) {
  using basis_t = numeric::math::fes::BasisH1<1>;
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElPoint>();
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElSegment>();
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElTria>();
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElQuad>();
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElTetra>();
  test_element_num_basis_functions<basis_t, numeric::mesh::RefElCube>();
}
