#define NUMERIC_DO_NOT_SPECIALIZE_BASIS_LAGRANGE

#include <gtest/gtest.h>
#include <numeric/math/basis_lagrange.hpp>

template <typename Element, numeric::dim_t order> struct Transformation;

template <> struct Transformation<numeric::mesh::RefElSegment, 1> {
  static constexpr numeric::dim_t reflection[2] = {1, 0};
  static constexpr numeric::dim_t rotation[2] = {1, 0};
};

template <> struct Transformation<numeric::mesh::RefElSegment, 2> {
  static constexpr numeric::dim_t reflection[3] = {1, 0, 2};
  static constexpr numeric::dim_t rotation[3] = {1, 0, 2};
};

template <> struct Transformation<numeric::mesh::RefElSegment, 3> {
  static constexpr numeric::dim_t reflection[4] = {1, 0, 3, 2};
  static constexpr numeric::dim_t rotation[4] = {1, 0, 3, 2};
};

template <> struct Transformation<numeric::mesh::RefElSegment, 4> {
  static constexpr numeric::dim_t reflection[5] = {1, 0, 4, 3, 2};
  static constexpr numeric::dim_t rotation[5] = {1, 0, 4, 3, 2};
};

template <> struct Transformation<numeric::mesh::RefElTria, 1> {
  static constexpr numeric::dim_t reflection[3] = {0, 2, 1};
  static constexpr numeric::dim_t rotation[3] = {1, 2, 0};
};

template <> struct Transformation<numeric::mesh::RefElTria, 2> {
  static constexpr numeric::dim_t reflection[6] = {0, 2, 1, 5, 4, 3};
  static constexpr numeric::dim_t rotation[6] = {1, 2, 0, 4, 5, 3};
};

template <> struct Transformation<numeric::mesh::RefElTria, 3> {
  static constexpr numeric::dim_t reflection[10] = {0, 2, 1, 8, 7,
                                                    6, 5, 4, 3, 9};
  static constexpr numeric::dim_t rotation[10] = {1, 2, 0, 5, 6, 7, 8, 3, 4, 9};
};

template <> struct Transformation<numeric::mesh::RefElTria, 4> {
  static constexpr numeric::dim_t reflection[15] = {0, 2, 1, 11, 10, 9,  8, 7,
                                                    6, 5, 4, 3,  12, 14, 13};
  static constexpr numeric::dim_t rotation[15] = {1,  2, 0, 6, 7,  8,  9, 10,
                                                  11, 3, 4, 5, 13, 14, 12};
};

template <typename Element, numeric::dim_t order>
static numeric::dim_t index_under_group_action_reference(
    numeric::dim_t i,
    const numeric::math::DihedralGroupElement<Element::num_nodes> &action) {
  using DH = numeric::math::DihedralGroupElement<Element::num_nodes>;
  if (action.type == DH::REFLECTION) {
    return index_under_group_action_reference<Element, order>(
        Transformation<Element, order>::reflection[i],
        action / DH::reflection(0));
  } else if (action.n > 0) {
    return index_under_group_action_reference<Element, order>(
        Transformation<Element, order>::rotation[i], action / DH::rotation(1));
  } else {
    return i;
  }
}

template <typename Element, numeric::dim_t order>
static void test_permutation() {
  using basis_t = numeric::math::BasisLagrange<Element, order>;
  using DH = numeric::math::DihedralGroupElement<Element::num_nodes>;
  for (numeric::dim_t i = 0; i < basis_t::num_basis_functions; ++i) {
    for (numeric::dim_t n = 0; n < Element::num_nodes; ++n) {
      ASSERT_EQ((basis_t::node_idx_under_group_action(i, DH::rotation(n))),
                (index_under_group_action_reference<Element, order>(
                    i, DH::rotation(n))));
      ASSERT_EQ((basis_t::node_idx_under_group_action(i, DH::reflection(n))),
                (index_under_group_action_reference<Element, order>(
                    i, DH::reflection(n))));
    }
  }
}

TEST(BasisLagrange, SegmentPermutationOrder1) {
  using ref_el_t = numeric::mesh::RefElSegment;
  static constexpr numeric::dim_t order = 1;
  test_permutation<ref_el_t, order>();
}

TEST(BasisLagrange, SegmentPermutationOrder2) {
  using ref_el_t = numeric::mesh::RefElSegment;
  static constexpr numeric::dim_t order = 2;
  test_permutation<ref_el_t, order>();
}

TEST(BasisLagrange, SegmentPermutationOrder3) {
  using ref_el_t = numeric::mesh::RefElSegment;
  static constexpr numeric::dim_t order = 3;
  test_permutation<ref_el_t, order>();
}

TEST(BasisLagrange, SegmentPermutationOrder4) {
  using ref_el_t = numeric::mesh::RefElSegment;
  static constexpr numeric::dim_t order = 4;
  test_permutation<ref_el_t, order>();
}

TEST(BasisLagrange, TriePermutationOrder1) {
  using ref_el_t = numeric::mesh::RefElTria;
  static constexpr numeric::dim_t order = 1;
  test_permutation<ref_el_t, order>();
}

TEST(BasisLagrange, TriaPermutationOrder2) {
  using ref_el_t = numeric::mesh::RefElTria;
  static constexpr numeric::dim_t order = 2;
  test_permutation<ref_el_t, order>();
}

TEST(BasisLagrange, TriaPermutationOrder3) {
  using ref_el_t = numeric::mesh::RefElTria;
  static constexpr numeric::dim_t order = 3;
  test_permutation<ref_el_t, order>();
}

TEST(BasisLagrange, TriaPermutationOrder4) {
  using ref_el_t = numeric::mesh::RefElTria;
  static constexpr numeric::dim_t order = 4;
  test_permutation<ref_el_t, order>();
}
