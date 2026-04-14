#include <gtest/gtest.h>
#include <numeric/math/dihedral_group.hpp>

using namespace numeric::math;
using namespace numeric;

template <dim_t N> static void test_identity_multiply() {
  using DN = DihedralGroupElement<N>;
  for (dim_t n = 0; n < N; ++n) {
    const DN as = DN::reflection(n);
    const DN ar = DN::rotation(n);
    const DN id = DN::identity();
    ASSERT_EQ(as, as * id);
    ASSERT_EQ(as, id * as);
    ASSERT_EQ(ar, ar * id);
    ASSERT_EQ(ar, id * ar);
  }
}

template <dim_t N> static void test_identity_divide() {
  using DN = DihedralGroupElement<N>;
  for (dim_t n = 0; n < N; ++n) {
    const DN as = DN::reflection(n);
    const DN ar = DN::rotation(n);
    const DN id = DN::identity();
    ASSERT_EQ(as, as / id);
    ASSERT_EQ(ar, ar / id);
  }
}

template <dim_t N> static void test_inverse() {
  using DN = DihedralGroupElement<N>;
  for (dim_t n = 0; n < N; ++n) {
    const DN id = DN::identity();
    const DN as = DN::reflection(n);
    const DN ar = DN::rotation(n);
    ASSERT_EQ(id, as * as.inverse());
    ASSERT_EQ(id, ar * ar.inverse());
  }
}

template <dim_t N> static void test_division() {
  using DN = DihedralGroupElement<N>;
  for (dim_t n1 = 0; n1 < N; ++n1) {
    for (dim_t n2 = 0; n2 < N; ++n2) {
      const DN a1s = DN::reflection(n1);
      const DN a1r = DN::rotation(n1);
      const DN a2s = DN::reflection(n2);
      const DN a2r = DN::rotation(n2);
      ASSERT_EQ(a1s / a2s, a1s * a2s.inverse());
      ASSERT_EQ(a1s / a2r, a1s * a2r.inverse());
      ASSERT_EQ(a1r / a2s, a1r * a2s.inverse());
      ASSERT_EQ(a1r / a2r, a1r * a2r.inverse());
    }
  }
}

TEST(DihedralGroup, IdentityMultiply) {
  test_identity_multiply<1>();
  test_identity_multiply<2>();
  test_identity_multiply<3>();
  test_identity_multiply<4>();
}

TEST(DihedralGroup, IdentityDivide) {
  test_identity_divide<1>();
  test_identity_divide<2>();
  test_identity_divide<3>();
  test_identity_divide<4>();
}

TEST(DihedralGroup, Inverse) {
  test_inverse<1>();
  test_inverse<2>();
  test_inverse<3>();
  test_inverse<4>();
}

TEST(DihedralGroup, Division) {
  test_division<1>();
  test_division<2>();
  test_division<3>();
  test_division<4>();
}
