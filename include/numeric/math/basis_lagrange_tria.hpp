#ifndef NUMERIC_MATH_BASIS_LAGRANGE_TRIA_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_TRIA_HPP_

#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::math {

template <> struct BasisLagrange<mesh::RefElTria, 1> {
  using ref_el_t = mesh::RefElTria;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 3;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    return -coeffs[0] * (x[0] + x[1] - 1) + coeffs[1] * x[0] + coeffs[2] * x[1];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    out[0] = -coeffs[0] + coeffs[1];
    out[1] = -coeffs[0] + coeffs[2];
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      break;
    case 1:
      out[0] = 1;
      out[1] = 0;
      break;
    case 2:
      out[0] = 0;
      out[1] = 1;
      break;
    }
  }
};

template <> struct BasisLagrange<mesh::RefElTria, 2> {
  using ref_el_t = mesh::RefElTria;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 6;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 2 * x[0];
    const Scalar x1 = x[0] + x[1] - 1;
    const Scalar x2 = 2 * x[1];
    const Scalar x3 = x0 - 1;
    return coeffs[0] * x1 * (x2 + x3) + coeffs[1] * x3 * x[0] +
           coeffs[2] * x[1] * (x2 - 1) - 2 * coeffs[3] * x0 * x1 +
           2 * coeffs[4] * x0 * x[1] - 2 * coeffs[5] * x1 * x2;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 4 * x[1];
    const Scalar x1 = 4 * x[0];
    const Scalar x2 = coeffs[0] * (x0 + x1 - 3);
    out[0] = coeffs[1] * (x1 - 1) - 4 * coeffs[3] * (2 * x[0] + x[1] - 1) +
             coeffs[4] * x0 - coeffs[5] * x0 + x2;
    out[1] = coeffs[2] * (x0 - 1) - coeffs[3] * x1 + coeffs[4] * x1 -
             4 * coeffs[5] * (x[0] + 2 * x[1] - 1) + x2;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      break;
    case 1:
      out[0] = 2;
      out[1] = 0;
      break;
    case 2:
      out[0] = 0;
      out[1] = 2;
      break;
    case 3:
      out[0] = 1;
      out[1] = 0;
      break;
    case 4:
      out[0] = 1;
      out[1] = 1;
      break;
    case 5:
      out[0] = 0;
      out[1] = 1;
      break;
    }
  }
};

template <> struct BasisLagrange<mesh::RefElTria, 3> {
  using ref_el_t = mesh::RefElTria;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 10;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] + x[1] - 1;
    const Scalar x1 = x0 * x[0];
    const Scalar x2 = 3 * x[0];
    const Scalar x3 = x2 - 1;
    const Scalar x4 = x3 * x[0];
    const Scalar x5 = (3.0 / 2.0) * x[1];
    const Scalar x6 = 3 * x[1];
    const Scalar x7 = x[1] * (x6 - 1);
    const Scalar x8 = (3.0 / 2.0) * x7;
    const Scalar x9 = (3.0 / 2.0) * x1;
    const Scalar x10 = x2 - 2;
    const Scalar x11 = x10 + x6;
    const Scalar x12 = x0 * x11;
    return -1.0 / 2.0 * coeffs[0] * x12 * (x3 + x6) +
           (1.0 / 2.0) * coeffs[1] * x10 * x4 +
           (1.0 / 2.0) * coeffs[2] * x7 * (x6 - 2) + 3 * coeffs[3] * x11 * x9 -
           3 * coeffs[4] * x3 * x9 + 3 * coeffs[5] * x4 * x5 +
           3 * coeffs[6] * x8 * x[0] - 3 * coeffs[7] * x0 * x8 +
           3 * coeffs[8] * x12 * x5 - 27 * coeffs[9] * x1 * x[1];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 6 * x[0];
    const Scalar x1 = 3 * x[1];
    const Scalar x2 = x1 - 1;
    const Scalar x3 = x2 * x[1];
    const Scalar x4 = (3.0 / 2.0) * coeffs[7];
    const Scalar x5 = x[1] - 1;
    const Scalar x6 = 9 * coeffs[9];
    const Scalar x7 = 6 * x[1];
    const Scalar x8 = -x0 - x7 + 5;
    const Scalar x9 = (3.0 / 2.0) * coeffs[8];
    const Scalar x10 = 3 * x[0];
    const Scalar x11 = x10 - 1;
    const Scalar x12 = x11 * x[0];
    const Scalar x13 = x5 + x[0];
    const Scalar x14 = (3.0 / 2.0) * coeffs[4];
    const Scalar x15 = x10 - 2;
    const Scalar x16 = -x1 - x15;
    const Scalar x17 = -x13;
    const Scalar x18 = x16 * x17;
    const Scalar x19 = -x18;
    const Scalar x20 = (3.0 / 2.0) * coeffs[3];
    const Scalar x21 = -x1 - x11;
    const Scalar x22 =
        (1.0 / 6.0) * coeffs[0] * (x16 * x21 + 3 * x17 * x21 + 3 * x18);
    const Scalar x23 = x1 - 2;
    out[0] = (1.0 / 2.0) * coeffs[1] * (x10 * x11 + x10 * x15 + x11 * x15) +
             (9.0 / 2.0) * coeffs[5] * x[1] * (x0 - 1) +
             (9.0 / 2.0) * coeffs[6] * x2 * x[1] -
             3 * x14 * (x10 * x13 + x11 * x13 + x12) -
             3 * x20 * (x10 * x17 + x16 * x[0] + x19) - 3 * x22 - 3 * x3 * x4 -
             3 * x6 * x[1] * (x5 + 2 * x[0]) - 3 * x8 * x9 * x[1];
    out[1] = (1.0 / 2.0) * coeffs[2] * (x1 * x2 + x1 * x23 + x2 * x23) +
             (9.0 / 2.0) * coeffs[5] * x11 * x[0] +
             (9.0 / 2.0) * coeffs[6] * x[0] * (x7 - 1) - 3 * x12 * x14 -
             3 * x20 * x8 * x[0] - 3 * x22 -
             3 * x4 * (x1 * x13 + x13 * x2 + x3) -
             3 * x6 * x[0] * (x[0] + 2 * x[1] - 1) -
             3 * x9 * (x1 * x17 + x16 * x[1] + x19);
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      break;
    case 1:
      out[0] = 3;
      out[1] = 0;
      break;
    case 2:
      out[0] = 0;
      out[1] = 3;
      break;
    case 3:
      out[0] = 1;
      out[1] = 0;
      break;
    case 4:
      out[0] = 2;
      out[1] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 1;
      break;
    case 6:
      out[0] = 1;
      out[1] = 2;
      break;
    case 7:
      out[0] = 0;
      out[1] = 2;
      break;
    case 8:
      out[0] = 0;
      out[1] = 1;
      break;
    case 9:
      out[0] = 1;
      out[1] = 1;
      break;
    }
  }
};

template <> struct BasisLagrange<mesh::RefElTria, 4> {
  using ref_el_t = mesh::RefElTria;
  static constexpr dim_t order = 4;
  static constexpr dim_t num_basis_functions = 15;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 4 * x[1];
    const Scalar x1 = x[0] + x[1] - 1;
    const Scalar x2 = 4 * x[0];
    const Scalar x3 = x[0] * (x2 - 1);
    const Scalar x4 = x1 * x3;
    const Scalar x5 = x0 - 1;
    const Scalar x6 = x1 * x5;
    const Scalar x7 = x0 * x[0];
    const Scalar x8 = (2.0 / 3.0) * x[1];
    const Scalar x9 = 2 * x[0] - 1;
    const Scalar x10 = x3 * x9;
    const Scalar x11 = (2.0 / 3.0) * x[0];
    const Scalar x12 = 2 * x[1];
    const Scalar x13 = x12 - 1;
    const Scalar x14 = x5 * x[1];
    const Scalar x15 = x13 * x14;
    const Scalar x16 = x2 - 3;
    const Scalar x17 = x0 + x16;
    const Scalar x18 = x1 * x17;
    const Scalar x19 = x6 * x[1];
    const Scalar x20 = (1.0 / 2.0) * x17;
    const Scalar x21 = x18 * (x12 + x9);
    return (1.0 / 3.0) * coeffs[0] * x21 * (x2 + x5) -
           8 * coeffs[10] * x21 * x8 + 8 * coeffs[11] * x19 * x20 +
           8 * coeffs[12] * x18 * x7 - 8 * coeffs[13] * x0 * x4 -
           8 * coeffs[14] * x6 * x7 + (1.0 / 3.0) * coeffs[1] * x10 * x16 +
           (1.0 / 3.0) * coeffs[2] * x15 * (x0 - 3) -
           8 * coeffs[3] * x11 * x21 - 16.0 / 3.0 * coeffs[4] * x4 * x9 +
           8 * coeffs[5] * x20 * x4 + 8 * coeffs[6] * x10 * x8 +
           8 * coeffs[7] * x11 * x15 + 4 * coeffs[8] * x14 * x3 -
           16.0 / 3.0 * coeffs[9] * x13 * x19;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 2 * x[1];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = 4 * x[1];
    const Scalar x3 = x2 - 1;
    const Scalar x4 = 8 * x[0];
    const Scalar x5 = x3 * x[1];
    const Scalar x6 = x1 * x5;
    const Scalar x7 = (2.0 / 3.0) * coeffs[9];
    const Scalar x8 = 2 * x[0];
    const Scalar x9 = x8 - 1;
    const Scalar x10 = 8 * x[1];
    const Scalar x11 = -x10 - x4 + 7;
    const Scalar x12 = (1.0 / 2.0) * coeffs[11];
    const Scalar x13 = 4 * x[0];
    const Scalar x14 = x13 - 1;
    const Scalar x15 = x14 * x[0];
    const Scalar x16 = x[0] + x[1] - 1;
    const Scalar x17 = x13 * x16;
    const Scalar x18 = x14 * x16;
    const Scalar x19 = x13 * x9;
    const Scalar x20 = x14 * x8;
    const Scalar x21 = x14 * x9;
    const Scalar x22 = x13 - 3;
    const Scalar x23 = -x2 - x22;
    const Scalar x24 = x23 * x[0];
    const Scalar x25 = -x16;
    const Scalar x26 = x13 * x25;
    const Scalar x27 = x23 * x25;
    const Scalar x28 = -x27;
    const Scalar x29 = x15 * x9;
    const Scalar x30 = (2.0 / 3.0) * coeffs[4];
    const Scalar x31 = -x0 - x9;
    const Scalar x32 = x23 * x31;
    const Scalar x33 = 2 * x27;
    const Scalar x34 = x25 * x31;
    const Scalar x35 = 4 * x34;
    const Scalar x36 = x32 + x33 + x35;
    const Scalar x37 = (2.0 / 3.0) * coeffs[10];
    const Scalar x38 = (1.0 / 2.0) * coeffs[5];
    const Scalar x39 = x27 * x31;
    const Scalar x40 = -x39;
    const Scalar x41 = (2.0 / 3.0) * coeffs[3];
    const Scalar x42 = -x14 - x2;
    const Scalar x43 = (1.0 / 24.0) * coeffs[0] *
                       (x32 * x42 + x33 * x42 + x35 * x42 + 4 * x39);
    const Scalar x44 = x16 * x2;
    const Scalar x45 = x16 * x3;
    const Scalar x46 = x1 * x2;
    const Scalar x47 = x0 * x3;
    const Scalar x48 = x1 * x3;
    const Scalar x49 = x23 * x[1];
    const Scalar x50 = x2 * x25;
    const Scalar x51 = x2 - 3;
    out[0] = -8 * coeffs[12] * x2 * (x24 + x26 + x28) -
             8 * coeffs[13] * x2 * (x15 + x17 + x18) -
             8 * coeffs[14] * x2 * x3 * (x9 + x[1]) +
             (1.0 / 3.0) * coeffs[1] *
                 (x13 * x21 + x19 * x22 + x20 * x22 + x21 * x22) +
             (16.0 / 3.0) * coeffs[6] * x[1] * (x19 + x20 + x21) +
             (16.0 / 3.0) * coeffs[7] * x1 * x3 * x[1] +
             4 * coeffs[8] * x3 * x[1] * (x4 - 1) - 8 * x11 * x12 * x5 -
             8 * x30 * (x17 * x9 + x18 * x8 + x18 * x9 + x29) -
             8 * x36 * x37 * x[1] -
             8 * x38 * (-x13 * x27 + x14 * x24 + x14 * x26 - x14 * x27) -
             8 * x41 * (x24 * x31 + x26 * x31 + x27 * x8 + x40) - 8 * x43 -
             8 * x6 * x7;
    out[1] = -8 * coeffs[12] * x13 * (x28 + x49 + x50) -
             8 * coeffs[13] * x13 * x14 * (x1 + x[0]) -
             8 * coeffs[14] * x13 * (x44 + x45 + x5) +
             (1.0 / 3.0) * coeffs[2] *
                 (x2 * x48 + x46 * x51 + x47 * x51 + x48 * x51) +
             (16.0 / 3.0) * coeffs[6] * x14 * x9 * x[0] +
             (16.0 / 3.0) * coeffs[7] * x[0] * (x46 + x47 + x48) +
             4 * coeffs[8] * x14 * x[0] * (x10 - 1) - 8 * x11 * x15 * x38 -
             8 * x12 * (-x2 * x27 - x27 * x3 + x3 * x49 + x3 * x50) -
             8 * x29 * x30 - 8 * x36 * x41 * x[0] -
             8 * x37 * (x0 * x27 + x2 * x34 + x32 * x[1] + x40) - 8 * x43 -
             8 * x7 * (x0 * x45 + x1 * x44 + x1 * x45 + x6);
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      break;
    case 1:
      out[0] = 4;
      out[1] = 0;
      break;
    case 2:
      out[0] = 0;
      out[1] = 4;
      break;
    case 3:
      out[0] = 1;
      out[1] = 0;
      break;
    case 4:
      out[0] = 3;
      out[1] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 0;
      break;
    case 6:
      out[0] = 3;
      out[1] = 1;
      break;
    case 7:
      out[0] = 1;
      out[1] = 3;
      break;
    case 8:
      out[0] = 2;
      out[1] = 2;
      break;
    case 9:
      out[0] = 0;
      out[1] = 3;
      break;
    case 10:
      out[0] = 0;
      out[1] = 1;
      break;
    case 11:
      out[0] = 0;
      out[1] = 2;
      break;
    case 12:
      out[0] = 1;
      out[1] = 1;
      break;
    case 13:
      out[0] = 2;
      out[1] = 1;
      break;
    case 14:
      out[0] = 1;
      out[1] = 2;
      break;
    }
  }
};

template <> struct BasisLagrange<mesh::RefElTria, 5> {
  using ref_el_t = mesh::RefElTria;
  static constexpr dim_t order = 5;
  static constexpr dim_t num_basis_functions = 21;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] + x[1] - 1;
    const Scalar x1 = 5 * x[0];
    const Scalar x2 = x1 - 2;
    const Scalar x3 = x1 - 1;
    const Scalar x4 = x3 * x[0];
    const Scalar x5 = x2 * x4;
    const Scalar x6 = 500 * x[1];
    const Scalar x7 = 5 * x[1];
    const Scalar x8 = x[1] * (x7 - 1);
    const Scalar x9 = x8 * (x7 - 2);
    const Scalar x10 = 750 * x4;
    const Scalar x11 = 50 * x9;
    const Scalar x12 = x1 - 3;
    const Scalar x13 = x12 * x5;
    const Scalar x14 = x9 * (x7 - 3);
    const Scalar x15 = 50 * x5;
    const Scalar x16 = x1 - 4;
    const Scalar x17 = x0 * (x16 + x7);
    const Scalar x18 = 25 * x0;
    const Scalar x19 = x17 * (x12 + x7);
    const Scalar x20 = 50 * x19;
    const Scalar x21 = x19 * (x2 + x7);
    const Scalar x22 = 25 * x21;
    return -1.0 / 24.0 * coeffs[0] * x21 * (x3 + x7) +
           (1.0 / 24.0) * coeffs[10] * x11 * x4 -
           1.0 / 24.0 * coeffs[11] * x14 * x18 +
           (1.0 / 24.0) * coeffs[12] * x22 * x[1] +
           (1.0 / 24.0) * coeffs[13] * x11 * x17 -
           1.0 / 24.0 * coeffs[14] * x20 * x8 -
           1.0 / 24.0 * coeffs[15] * x19 * x6 * x[0] -
           1.0 / 24.0 * coeffs[16] * x0 * x5 * x6 -
           125.0 / 6.0 * coeffs[17] * x0 * x9 * x[0] +
           (1.0 / 24.0) * coeffs[18] * x10 * x17 * x[1] -
           1.0 / 24.0 * coeffs[19] * x0 * x10 * x8 +
           (1.0 / 24.0) * coeffs[1] * x13 * x16 +
           (125.0 / 4.0) * coeffs[20] * x17 * x8 * x[0] +
           (1.0 / 24.0) * coeffs[2] * x14 * (x7 - 4) +
           (1.0 / 24.0) * coeffs[3] * x22 * x[0] -
           1.0 / 24.0 * coeffs[4] * x13 * x18 -
           1.0 / 24.0 * coeffs[5] * x20 * x4 +
           (1.0 / 24.0) * coeffs[6] * x15 * x17 +
           (25.0 / 24.0) * coeffs[7] * x13 * x[1] +
           (25.0 / 24.0) * coeffs[8] * x14 * x[0] +
           (1.0 / 24.0) * coeffs[9] * x15 * x8;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 10 * x[0];
    const Scalar x1 = 5 * x[1];
    const Scalar x2 = x1 - 2;
    const Scalar x3 = x1 - 1;
    const Scalar x4 = 25 * coeffs[11];
    const Scalar x5 = x1 - 3;
    const Scalar x6 = x3 * x[1];
    const Scalar x7 = x2 * x6;
    const Scalar x8 = x5 * x7;
    const Scalar x9 = x[1] - 1;
    const Scalar x10 = 10 * x[1];
    const Scalar x11 = -x0 - x10 + 9;
    const Scalar x12 = 5 * x[0];
    const Scalar x13 = x12 - 1;
    const Scalar x14 = x13 * x[0];
    const Scalar x15 = x9 + x[0];
    const Scalar x16 = x12 * x15;
    const Scalar x17 = x13 * x15;
    const Scalar x18 = 750 * coeffs[19];
    const Scalar x19 = x12 - 2;
    const Scalar x20 = x12 * x19;
    const Scalar x21 = x12 * x13;
    const Scalar x22 = x13 * x19;
    const Scalar x23 = x12 - 4;
    const Scalar x24 = x1 + x23;
    const Scalar x25 = -x24;
    const Scalar x26 = x25 * x[0];
    const Scalar x27 = -x15;
    const Scalar x28 = x12 * x27;
    const Scalar x29 = x25 * x27;
    const Scalar x30 = -x29;
    const Scalar x31 = 750 * coeffs[20];
    const Scalar x32 = x14 * x19;
    const Scalar x33 = x16 * x19;
    const Scalar x34 = x12 * x17;
    const Scalar x35 = x17 * x19;
    const Scalar x36 = 500 * x[1];
    const Scalar x37 = x12 - 3;
    const Scalar x38 = x20 * x37;
    const Scalar x39 = x21 * x37;
    const Scalar x40 = x12 * x22;
    const Scalar x41 = x22 * x37;
    const Scalar x42 = -x1 - x37;
    const Scalar x43 = x25 * x42;
    const Scalar x44 = x27 * x42;
    const Scalar x45 = 5 * x44;
    const Scalar x46 = 5 * x29;
    const Scalar x47 = x43 + x45 + x46;
    const Scalar x48 = 50 * coeffs[14];
    const Scalar x49 = x13 * x28;
    const Scalar x50 = x12 * x29;
    const Scalar x51 = x13 * x29;
    const Scalar x52 = 750 * coeffs[18];
    const Scalar x53 = x26 * x42;
    const Scalar x54 = x28 * x42;
    const Scalar x55 = x29 * x42;
    const Scalar x56 = -x55;
    const Scalar x57 = x32 * x37;
    const Scalar x58 = x12 * x35;
    const Scalar x59 = 25 * coeffs[4];
    const Scalar x60 = -x1 - x19;
    const Scalar x61 = x43 * x60;
    const Scalar x62 = x45 * x60;
    const Scalar x63 = x46 * x60;
    const Scalar x64 = 5 * x55;
    const Scalar x65 = x61 + x62 + x63 + x64;
    const Scalar x66 = 25 * coeffs[12];
    const Scalar x67 = x12 * x55;
    const Scalar x68 = 50 * coeffs[5];
    const Scalar x69 = x55 * x60;
    const Scalar x70 = -x69;
    const Scalar x71 = 25 * coeffs[3];
    const Scalar x72 = -x1 - x13;
    const Scalar x73 =
        coeffs[0] * (x61 * x72 + x62 * x72 + x63 * x72 + x64 * x72 + 5 * x69);
    const Scalar x74 = x1 * x15;
    const Scalar x75 = x15 * x3;
    const Scalar x76 = x1 * x2;
    const Scalar x77 = x1 * x3;
    const Scalar x78 = x2 * x3;
    const Scalar x79 = x25 * x[1];
    const Scalar x80 = x1 * x27;
    const Scalar x81 = x2 * x74;
    const Scalar x82 = x1 * x75;
    const Scalar x83 = x2 * x75;
    const Scalar x84 = 500 * x[0];
    const Scalar x85 = x5 * x76;
    const Scalar x86 = x5 * x77;
    const Scalar x87 = x1 * x78;
    const Scalar x88 = x5 * x78;
    const Scalar x89 = x1 * x29;
    const Scalar x90 = x29 * x3;
    const Scalar x91 = x1 * x44;
    const Scalar x92 = x1 * x83;
    const Scalar x93 = x1 - 4;
    const Scalar x94 = x1 * x55;
    out[0] =
        (25.0 / 12.0) * coeffs[10] * x2 * x3 * x[1] * (x0 - 1) -
        25.0 / 12.0 * coeffs[13] * x11 * x7 -
        1.0 / 24.0 * coeffs[15] * x36 * (x50 + x53 + x54 + x56) -
        1.0 / 24.0 * coeffs[16] * x36 * (x32 + x33 + x34 + x35) -
        125.0 / 6.0 * coeffs[17] * x7 * (x9 + 2 * x[0]) +
        (1.0 / 24.0) * coeffs[1] *
            (x12 * x41 + x23 * x38 + x23 * x39 + x23 * x40 + x23 * x41) +
        (25.0 / 12.0) * coeffs[6] *
            (x24 * x32 + x24 * x33 + x24 * x34 + x24 * x35 + x58) +
        (25.0 / 24.0) * coeffs[7] * x[1] * (x38 + x39 + x40 + x41) +
        (25.0 / 24.0) * coeffs[8] * x2 * x3 * x5 * x[1] +
        (25.0 / 12.0) * coeffs[9] * x3 * x[1] * (x20 + x21 + x22) -
        1.0 / 24.0 * x18 * x6 * (x14 + x16 + x17) -
        1.0 / 24.0 * x31 * x6 * (x26 + x28 + x30) - 1.0 / 24.0 * x4 * x8 -
        1.0 / 24.0 * x47 * x48 * x6 -
        1.0 / 24.0 * x52 * x[1] * (x14 * x25 + x49 - x50 - x51) -
        1.0 / 24.0 * x59 * (x33 * x37 + x34 * x37 + x35 * x37 + x57 + x58) -
        1.0 / 24.0 * x65 * x66 * x[1] -
        1.0 / 24.0 * x68 *
            (x12 * x51 + x14 * x43 + x42 * x49 - x42 * x51 - x67) -
        1.0 / 24.0 * x71 * (x50 * x60 + x53 * x60 + x54 * x60 + x67 + x70) -
        1.0 / 24.0 * x73;
    out[1] =
        (25.0 / 12.0) * coeffs[10] * x13 * x[0] * (x76 + x77 + x78) +
        (25.0 / 12.0) * coeffs[13] *
            (x24 * x7 + x24 * x81 + x24 * x82 + x24 * x83 + x92) -
        1.0 / 24.0 * coeffs[15] * x84 * (x43 * x[1] + x56 + x89 + x91) -
        125.0 / 6.0 * coeffs[16] * x32 * (x[0] + 2 * x[1] - 1) -
        1.0 / 24.0 * coeffs[17] * x84 * (x7 + x81 + x82 + x83) +
        (1.0 / 24.0) * coeffs[2] *
            (x1 * x88 + x85 * x93 + x86 * x93 + x87 * x93 + x88 * x93) -
        25.0 / 12.0 * coeffs[6] * x11 * x32 +
        (25.0 / 24.0) * coeffs[7] * x13 * x19 * x37 * x[0] +
        (25.0 / 24.0) * coeffs[8] * x[0] * (x85 + x86 + x87 + x88) +
        (25.0 / 12.0) * coeffs[9] * x13 * x19 * x[0] * (x10 - 1) -
        1.0 / 24.0 * x14 * x18 * (x6 + x74 + x75) -
        1.0 / 24.0 * x14 * x47 * x68 -
        1.0 / 24.0 * x14 * x52 * (x30 + x79 + x80) -
        1.0 / 24.0 * x31 * x[0] * (x3 * x79 + x3 * x80 - x89 - x90) -
        1.0 / 24.0 * x4 * (x5 * x81 + x5 * x82 + x5 * x83 + x8 + x92) -
        1.0 / 24.0 * x48 * (x1 * x90 - x3 * x55 + x43 * x6 + x44 * x77 - x94) -
        1.0 / 24.0 * x57 * x59 - 1.0 / 24.0 * x65 * x71 * x[0] -
        1.0 / 24.0 * x66 * (x60 * x89 + x60 * x91 + x61 * x[1] + x70 + x94) -
        1.0 / 24.0 * x73;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      break;
    case 1:
      out[0] = 5;
      out[1] = 0;
      break;
    case 2:
      out[0] = 0;
      out[1] = 5;
      break;
    case 3:
      out[0] = 1;
      out[1] = 0;
      break;
    case 4:
      out[0] = 4;
      out[1] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 0;
      break;
    case 6:
      out[0] = 3;
      out[1] = 0;
      break;
    case 7:
      out[0] = 4;
      out[1] = 1;
      break;
    case 8:
      out[0] = 1;
      out[1] = 4;
      break;
    case 9:
      out[0] = 3;
      out[1] = 2;
      break;
    case 10:
      out[0] = 2;
      out[1] = 3;
      break;
    case 11:
      out[0] = 0;
      out[1] = 4;
      break;
    case 12:
      out[0] = 0;
      out[1] = 1;
      break;
    case 13:
      out[0] = 0;
      out[1] = 3;
      break;
    case 14:
      out[0] = 0;
      out[1] = 2;
      break;
    case 15:
      out[0] = 1;
      out[1] = 1;
      break;
    case 16:
      out[0] = 3;
      out[1] = 1;
      break;
    case 17:
      out[0] = 1;
      out[1] = 3;
      break;
    case 18:
      out[0] = 2;
      out[1] = 1;
      break;
    case 19:
      out[0] = 2;
      out[1] = 2;
      break;
    case 20:
      out[0] = 1;
      out[1] = 2;
      break;
    }
  }
};

} // namespace numeric::math

#endif
