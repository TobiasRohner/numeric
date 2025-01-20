#ifndef NUMERIC_MATH_BASIS_LAGRANGE_QUAD_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_QUAD_HPP_

#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::math {

template <> struct BasisLagrange<mesh::RefElQuad, 1> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 4;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_ti, const Scalar *x) {
    switch (i) {
    case 0:
      return (x[0] - 1) * (x[1] - 1);
    case 1:
      return -x[0] * (x[1] - 1);
    case 2:
      return x[0] * x[1];
    case 3:
      return -x[1] * (x[0] - 1);
    default:
      return 0;
    }
  }

  template <typename Scalar>
  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    switch (i) {
    case 0:
      out[0] = x[1] - 1;
      out[1] = x[0] - 1;
      break;
    case 1:
      out[0] = 1 - x[1];
      out[1] = -x[0];
      break;
    case 2:
      out[0] = x[1];
      out[1] = x[0];
      break;
    case 3:
      out[0] = -x[1];
      out[1] = 1 - x[0];
      break;
    default:
      break;
    }
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[0] - 1;
    return coeffs[0] * x0 * x1 - coeffs[1] * x0 * x[0] +
           coeffs[2] * x[0] * x[1] - coeffs[3] * x1 * x[1];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[0] - 1;
    out[0] =
        coeffs[0] * x0 - coeffs[1] * x0 + coeffs[2] * x[1] - coeffs[3] * x[1];
    out[1] =
        coeffs[0] * x1 - coeffs[1] * x[0] + coeffs[2] * x[0] - coeffs[3] * x1;
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
      out[0] = 1;
      out[1] = 1;
      break;
    case 3:
      out[0] = 0;
      out[1] = 1;
      break;
    }
  }
}

template <>
struct BasisLagrange<mesh::RefElQuad, 2> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 9;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_ti, const Scalar *x) {
    switch (i) {
    case 0:
      return (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1);
    case 1:
      return x[0] * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1);
    case 2:
      return x[0] * x[1] * (2 * x[0] - 1) * (2 * x[1] - 1);
    case 3:
      return x[1] * (x[0] - 1) * (2 * x[0] - 1) * (2 * x[1] - 1);
    case 4:
      return -4 * x[0] * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1);
    case 5:
      return -4 * x[0] * x[1] * (2 * x[0] - 1) * (x[1] - 1);
    case 6:
      return -4 * x[0] * x[1] * (x[0] - 1) * (2 * x[1] - 1);
    case 7:
      return -4 * x[1] * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1);
    case 8:
      return 16 * x[0] * x[1] * (x[0] - 1) * (x[1] - 1);
    default:
      return 0;
    }
  }

  template <typename Scalar>
  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    switch (i) {
    case 0:
      out[0] = 4 * (x[0] - 3.0 / 4.0) * (x[1] - 1) * (2 * x[1] - 1);
      out[1] = 4 * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 3.0 / 4.0);
      break;
    case 1:
      out[0] = 4 * (x[0] - 1.0 / 4.0) * (x[1] - 1) * (2 * x[1] - 1);
      out[1] = 4 * x[0] * (2 * x[0] - 1) * (x[1] - 3.0 / 4.0);
      break;
    case 2:
      out[0] = 4 * x[1] * (x[0] - 1.0 / 4.0) * (2 * x[1] - 1);
      out[1] = 4 * x[0] * (2 * x[0] - 1) * (x[1] - 1.0 / 4.0);
      break;
    case 3:
      out[0] = 4 * x[1] * (x[0] - 3.0 / 4.0) * (2 * x[1] - 1);
      out[1] = 4 * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1.0 / 4.0);
      break;
    case 4:
      const Scalar x0 = 2 * x[1];
      out[0] = -8 * (x0 - 1) * (x[0] - 1.0 / 2.0) * (x[1] - 1);
      out[1] = -8 * x[0] * (x0 - 3.0 / 2.0) * (x[0] - 1);
      break;
    case 5:
      const Scalar x0 = 2 * x[0];
      out[0] = -8 * x[1] * (x0 - 1.0 / 2.0) * (x[1] - 1);
      out[1] = -8 * x[0] * (x0 - 1) * (x[1] - 1.0 / 2.0);
      break;
    case 6:
      const Scalar x0 = 2 * x[1];
      out[0] = -8 * x[1] * (x0 - 1) * (x[0] - 1.0 / 2.0);
      out[1] = -8 * x[0] * (x0 - 1.0 / 2.0) * (x[0] - 1);
      break;
    case 7:
      const Scalar x0 = 2 * x[0];
      out[0] = -8 * x[1] * (x0 - 3.0 / 2.0) * (x[1] - 1);
      out[1] = -8 * (x0 - 1) * (x[0] - 1) * (x[1] - 1.0 / 2.0);
      break;
    case 8:
      out[0] = 16 * x[1] * (2 * x[0] - 1) * (x[1] - 1);
      out[1] = 16 * x[0] * (x[0] - 1) * (2 * x[1] - 1);
      break;
    default:
      break;
    }
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x1 * x[0];
    const Scalar x3 = 2 * x[0] - 1;
    const Scalar x4 = x3 * x[1];
    const Scalar x5 = 2 * x[1] - 1;
    const Scalar x6 = x0 * x5;
    const Scalar x7 = (1.0 / 4.0) * x4;
    const Scalar x8 = (1.0 / 4.0) * x3;
    return 4 * coeffs[0] * x1 * x6 * x8 + 4 * coeffs[1] * x2 * x5 * x8 +
           4 * coeffs[2] * x5 * x7 * x[0] + 4 * coeffs[3] * x6 * x7 -
           4 * coeffs[4] * x2 * x6 - 4 * coeffs[5] * x2 * x4 -
           4 * coeffs[6] * x6 * x[0] * x[1] - 4 * coeffs[7] * x0 * x1 * x4 +
           16 * coeffs[8] * x0 * x2 * x[1];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = x[1] - 1;
    const Scalar x3 = x2 * x[1];
    const Scalar x4 = x0 - 3;
    const Scalar x5 = 2 * x[0] - 1;
    const Scalar x6 = 4 * x[1];
    const Scalar x7 = 2 * x[1] - 1;
    const Scalar x8 = (1.0 / 4.0) * x7;
    const Scalar x9 = x8 * x[1];
    const Scalar x10 = x5 * x7;
    const Scalar x11 = x2 * x8;
    const Scalar x12 = x6 - 3;
    const Scalar x13 = x[0] - 1;
    const Scalar x14 = x13 * x[0];
    const Scalar x15 = x6 - 1;
    const Scalar x16 = (1.0 / 4.0) * x5;
    const Scalar x17 = x12 * x16;
    const Scalar x18 = x15 * x16;
    out[0] = 4 * coeffs[0] * x11 * x4 + 4 * coeffs[1] * x1 * x11 +
             4 * coeffs[2] * x1 * x9 + 4 * coeffs[3] * x4 * x9 -
             4 * coeffs[4] * x10 * x2 - 4 * coeffs[5] * x1 * x3 -
             4 * coeffs[6] * x10 * x[1] - 4 * coeffs[7] * x3 * x4 +
             4 * coeffs[8] * x2 * x5 * x6;
    out[1] = 4 * coeffs[0] * x13 * x17 + 4 * coeffs[1] * x17 * x[0] +
             4 * coeffs[2] * x18 * x[0] + 4 * coeffs[3] * x13 * x18 -
             4 * coeffs[4] * x12 * x14 - 4 * coeffs[5] * x10 * x[0] -
             4 * coeffs[6] * x14 * x15 - 4 * coeffs[7] * x10 * x13 +
             4 * coeffs[8] * x0 * x13 * x7;
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
      out[0] = 2;
      out[1] = 2;
      break;
    case 3:
      out[0] = 0;
      out[1] = 2;
      break;
    case 4:
      out[0] = 1;
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
      out[1] = 1;
      break;
    case 8:
      out[0] = 1;
      out[1] = 1;
      break;
    }
  }
}

template <>
struct BasisLagrange<mesh::RefElQuad, 3> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 16;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_ti, const Scalar *x) {
    switch (i) {
    case 0:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      return (1.0 / 4.0) * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) *
             (x[0] - 1) * (x[1] - 1);
    case 1:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      return -1.0 / 4.0 * x[0] * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) *
             (x[1] - 1);
    case 2:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      return (1.0 / 4.0) * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1);
    case 3:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      return -1.0 / 4.0 * x[1] * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) *
             (x[0] - 1);
    case 4:
      const Scalar x0 = 3 * x[1];
      return -9.0 / 4.0 * x[0] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (3 * x[0] - 2) * (x[1] - 1);
    case 5:
      const Scalar x0 = 3 * x[1];
      return (9.0 / 4.0) * x[0] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (3 * x[0] - 1) * (x[1] - 1);
    case 6:
      const Scalar x0 = 3 * x[0];
      return (9.0 / 4.0) * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x[1] - 1) *
             (3 * x[1] - 2);
    case 7:
      const Scalar x0 = 3 * x[0];
      return -9.0 / 4.0 * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x[1] - 1) *
             (3 * x[1] - 1);
    case 8:
      const Scalar x0 = 3 * x[1];
      return -9.0 / 4.0 * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (3 * x[0] - 1);
    case 9:
      const Scalar x0 = 3 * x[1];
      return (9.0 / 4.0) * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (3 * x[0] - 2);
    case 10:
      const Scalar x0 = 3 * x[0];
      return (9.0 / 4.0) * x[1] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (x[1] - 1) * (3 * x[1] - 1);
    case 11:
      const Scalar x0 = 3 * x[0];
      return -9.0 / 4.0 * x[1] * (x0 - 2) * (x0 - 1) * (x[0] - 1) * (x[1] - 1) *
             (3 * x[1] - 2);
    case 12:
      return (81.0 / 4.0) * x[0] * x[1] * (x[0] - 1) * (3 * x[0] - 2) *
             (x[1] - 1) * (3 * x[1] - 2);
    case 13:
      return -81.0 / 4.0 * x[0] * x[1] * (x[0] - 1) * (3 * x[0] - 1) *
             (x[1] - 1) * (3 * x[1] - 2);
    case 14:
      return (81.0 / 4.0) * x[0] * x[1] * (x[0] - 1) * (3 * x[0] - 1) *
             (x[1] - 1) * (3 * x[1] - 1);
    case 15:
      return -81.0 / 4.0 * x[0] * x[1] * (x[0] - 1) * (3 * x[0] - 2) *
             (x[1] - 1) * (3 * x[1] - 1);
    default:
      return 0;
    }
  }

  template <typename Scalar>
  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    switch (i) {
    case 0:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 3 * x3;
      const Scalar x5 = x1 - 1;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = 3 * x[1];
      const Scalar x8 = x7 - 2;
      const Scalar x9 = x7 - 1;
      const Scalar x10 = x8 * x9;
      const Scalar x11 = 3 * x0;
      out[0] = (1.0 / 4.0) * x0 * x10 * (x2 * x4 + x4 * x5 + x6);
      out[1] = (1.0 / 4.0) * x3 * x6 * (x10 + x11 * x8 + x11 * x9);
      break;
    case 1:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x1 - 1;
      const Scalar x4 = x2 * x3;
      const Scalar x5 = 3 * x[1];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = 3 * x0;
      out[0] = -1.0 / 4.0 * x0 * x8 * (x1 * x2 + x1 * x3 + x4);
      out[1] = -1.0 / 4.0 * x4 * x[0] * (x6 * x9 + x7 * x9 + x8);
      break;
    case 2:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = 3 * x[1];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      out[0] = (1.0 / 4.0) * x7 * x[1] * (x0 * x1 + x0 * x2 + x3);
      out[1] = (1.0 / 4.0) * x3 * x[0] * (x4 * x5 + x4 * x6 + x7);
      break;
    case 3:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = 3 * x[1];
      const Scalar x7 = x6 - 2;
      const Scalar x8 = x6 - 1;
      const Scalar x9 = x7 * x8;
      out[0] = -1.0 / 4.0 * x9 * x[1] * (x1 * x3 + x3 * x4 + x5);
      out[1] = -1.0 / 4.0 * x2 * x5 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 4:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = x[0] - 1;
      const Scalar x2 = 3 * x[0];
      const Scalar x3 = x2 - 2;
      const Scalar x4 = x3 * x[0];
      const Scalar x5 = 3 * x[1];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = 3 * x0;
      out[0] = -9.0 / 4.0 * x0 * x8 * (x1 * x2 + x1 * x3 + x4);
      out[1] = -9.0 / 4.0 * x1 * x4 * (x6 * x9 + x7 * x9 + x8);
      break;
    case 5:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = x[0] - 1;
      const Scalar x2 = 3 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x3 * x[0];
      const Scalar x5 = 3 * x[1];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = 3 * x0;
      out[0] = (9.0 / 4.0) * x0 * x8 * (x1 * x2 + x1 * x3 + x4);
      out[1] = (9.0 / 4.0) * x1 * x4 * (x6 * x9 + x7 * x9 + x8);
      break;
    case 6:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x1 - 1;
      const Scalar x4 = x2 * x3;
      const Scalar x5 = 3 * x[1];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[1];
      out[0] = (9.0 / 4.0) * x0 * x7 * (x1 * x2 + x1 * x3 + x4);
      out[1] = (9.0 / 4.0) * x4 * x[0] * (x0 * x5 + x0 * x6 + x7);
      break;
    case 7:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x1 - 1;
      const Scalar x4 = x2 * x3;
      const Scalar x5 = 3 * x[1];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[1];
      out[0] = -9.0 / 4.0 * x0 * x7 * (x1 * x2 + x1 * x3 + x4);
      out[1] = -9.0 / 4.0 * x4 * x[0] * (x0 * x5 + x0 * x6 + x7);
      break;
    case 8:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = 3 * x[1];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      out[0] = -9.0 / 4.0 * x7 * x[1] * (x0 * x1 + x0 * x2 + x3);
      out[1] = -9.0 / 4.0 * x0 * x3 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 9:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = 3 * x[1];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      out[0] = (9.0 / 4.0) * x7 * x[1] * (x0 * x1 + x0 * x2 + x3);
      out[1] = (9.0 / 4.0) * x0 * x3 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 10:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 3 * x3;
      const Scalar x5 = x1 - 1;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = 3 * x[1];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = x8 * x[1];
      out[0] = (9.0 / 4.0) * x0 * x9 * (x2 * x4 + x4 * x5 + x6);
      out[1] = (9.0 / 4.0) * x3 * x6 * (x0 * x7 + x0 * x8 + x9);
      break;
    case 11:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 3 * x3;
      const Scalar x5 = x1 - 1;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = 3 * x[1];
      const Scalar x8 = x7 - 2;
      const Scalar x9 = x8 * x[1];
      out[0] = -9.0 / 4.0 * x0 * x9 * (x2 * x4 + x4 * x5 + x6);
      out[1] = -9.0 / 4.0 * x3 * x6 * (x0 * x7 + x0 * x8 + x9);
      break;
    case 12:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = x[0] - 1;
      const Scalar x2 = 3 * x[0];
      const Scalar x3 = x2 - 2;
      const Scalar x4 = x3 * x[0];
      const Scalar x5 = 3 * x[1];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[1];
      out[0] = (81.0 / 4.0) * x0 * x7 * (x1 * x2 + x1 * x3 + x4);
      out[1] = (81.0 / 4.0) * x1 * x4 * (x0 * x5 + x0 * x6 + x7);
      break;
    case 13:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = x[0] - 1;
      const Scalar x2 = 3 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x3 * x[0];
      const Scalar x5 = 3 * x[1];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[1];
      out[0] = -81.0 / 4.0 * x0 * x7 * (x1 * x2 + x1 * x3 + x4);
      out[1] = -81.0 / 4.0 * x1 * x4 * (x0 * x5 + x0 * x6 + x7);
      break;
    case 14:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = x[0] - 1;
      const Scalar x2 = 3 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x3 * x[0];
      const Scalar x5 = 3 * x[1];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[1];
      out[0] = (81.0 / 4.0) * x0 * x7 * (x1 * x2 + x1 * x3 + x4);
      out[1] = (81.0 / 4.0) * x1 * x4 * (x0 * x5 + x0 * x6 + x7);
      break;
    case 15:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = x[0] - 1;
      const Scalar x2 = 3 * x[0];
      const Scalar x3 = x2 - 2;
      const Scalar x4 = x3 * x[0];
      const Scalar x5 = 3 * x[1];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[1];
      out[0] = -81.0 / 4.0 * x0 * x7 * (x1 * x2 + x1 * x3 + x4);
      out[1] = -81.0 / 4.0 * x1 * x4 * (x0 * x5 + x0 * x6 + x7);
      break;
    default:
      break;
    }
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x0 * x1 * x[0] * x[1];
    const Scalar x3 = 3 * x[0];
    const Scalar x4 = x3 - 2;
    const Scalar x5 = 3 * x[1];
    const Scalar x6 = x5 - 2;
    const Scalar x7 = x4 * x6;
    const Scalar x8 = x5 - 1;
    const Scalar x9 = x3 - 1;
    const Scalar x10 = x2 * x9;
    const Scalar x11 = x4 * x8;
    const Scalar x12 = x1 * x7 * x9;
    const Scalar x13 = x1 * x[0];
    const Scalar x14 = x9 * x[1];
    const Scalar x15 = x8 * x[0];
    const Scalar x16 = (1.0 / 9.0) * x0;
    const Scalar x17 = x16 * x[1];
    const Scalar x18 = x17 * x9;
    const Scalar x19 = x7 * x8;
    const Scalar x20 = (1.0 / 81.0) * x14 * x19;
    const Scalar x21 = x13 * x16;
    const Scalar x22 = (1.0 / 81.0) * x12;
    return (81.0 / 4.0) * coeffs[0] * x0 * x22 * x8 +
           (81.0 / 4.0) * coeffs[10] * x1 * x11 * x18 -
           81.0 / 4.0 * coeffs[11] * x12 * x17 +
           (81.0 / 4.0) * coeffs[12] * x2 * x7 -
           81.0 / 4.0 * coeffs[13] * x10 * x6 +
           (81.0 / 4.0) * coeffs[14] * x10 * x8 -
           81.0 / 4.0 * coeffs[15] * x11 * x2 -
           81.0 / 4.0 * coeffs[1] * x15 * x22 +
           (81.0 / 4.0) * coeffs[2] * x20 * x[0] -
           81.0 / 4.0 * coeffs[3] * x0 * x20 -
           81.0 / 4.0 * coeffs[4] * x19 * x21 +
           (81.0 / 4.0) * coeffs[5] * x21 * x6 * x8 * x9 +
           (9.0 / 4.0) * coeffs[6] * x12 * x[0] * x[1] -
           9.0 / 4.0 * coeffs[7] * x11 * x13 * x14 -
           81.0 / 4.0 * coeffs[8] * x15 * x18 * x6 +
           (81.0 / 4.0) * coeffs[9] * x17 * x19 * x[0];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 3 * x[0];
    const Scalar x2 = x0 * x1;
    const Scalar x3 = x1 - 2;
    const Scalar x4 = x3 * x[0];
    const Scalar x5 = x0 * x3;
    const Scalar x6 = x2 + x4 + x5;
    const Scalar x7 = x[1] - 1;
    const Scalar x8 = 3 * x[1];
    const Scalar x9 = x8 - 2;
    const Scalar x10 = x9 * x[1];
    const Scalar x11 = x10 * x7;
    const Scalar x12 = x1 - 1;
    const Scalar x13 = x12 * x[0];
    const Scalar x14 = x0 * x12;
    const Scalar x15 = x13 + x14 + x2;
    const Scalar x16 = x8 - 1;
    const Scalar x17 = x16 * x[1];
    const Scalar x18 = x17 * x7;
    const Scalar x19 = (1.0 / 9.0) * x16;
    const Scalar x20 = x10 * x19;
    const Scalar x21 = x7 * x9;
    const Scalar x22 = x19 * x21;
    const Scalar x23 = x12 * x3;
    const Scalar x24 = x1 * x12 + x1 * x3 + x23;
    const Scalar x25 = (1.0 / 9.0) * x24;
    const Scalar x26 = (1.0 / 81.0) * x16;
    const Scalar x27 = x10 * x26;
    const Scalar x28 = 3 * x14 + x23 + 3 * x5;
    const Scalar x29 = (1.0 / 9.0) * x28;
    const Scalar x30 = x21 * x26;
    const Scalar x31 = x7 * x8;
    const Scalar x32 = x10 + x21 + x31;
    const Scalar x33 = x0 * x32;
    const Scalar x34 = x16 * x7;
    const Scalar x35 = x17 + x31 + x34;
    const Scalar x36 = x0 * x35;
    const Scalar x37 = (1.0 / 9.0) * x12;
    const Scalar x38 = x37 * x4;
    const Scalar x39 = x37 * x5;
    const Scalar x40 = x16 * x9;
    const Scalar x41 = x16 * x8 + x40 + x8 * x9;
    const Scalar x42 = (1.0 / 9.0) * x0;
    const Scalar x43 = x13 * x42;
    const Scalar x44 = x4 * x41;
    const Scalar x45 = (1.0 / 81.0) * x12;
    const Scalar x46 = x45 * x5;
    const Scalar x47 = 3 * x21 + 3 * x34 + x40;
    const Scalar x48 = x4 * x47;
    out[0] = (81.0 / 4.0) * coeffs[0] * x28 * x30 +
             (81.0 / 4.0) * coeffs[10] * x18 * x29 -
             81.0 / 4.0 * coeffs[11] * x11 * x29 +
             (81.0 / 4.0) * coeffs[12] * x11 * x6 -
             81.0 / 4.0 * coeffs[13] * x11 * x15 +
             (81.0 / 4.0) * coeffs[14] * x15 * x18 -
             81.0 / 4.0 * coeffs[15] * x18 * x6 -
             81.0 / 4.0 * coeffs[1] * x24 * x30 +
             (81.0 / 4.0) * coeffs[2] * x24 * x27 -
             81.0 / 4.0 * coeffs[3] * x27 * x28 -
             81.0 / 4.0 * coeffs[4] * x22 * x6 +
             (81.0 / 4.0) * coeffs[5] * x15 * x22 +
             (81.0 / 4.0) * coeffs[6] * x11 * x25 -
             81.0 / 4.0 * coeffs[7] * x18 * x25 -
             81.0 / 4.0 * coeffs[8] * x15 * x20 +
             (81.0 / 4.0) * coeffs[9] * x20 * x6;
    out[1] = (81.0 / 4.0) * coeffs[0] * x46 * x47 +
             (81.0 / 4.0) * coeffs[10] * x35 * x39 -
             81.0 / 4.0 * coeffs[11] * x32 * x39 +
             (81.0 / 4.0) * coeffs[12] * x33 * x4 -
             81.0 / 4.0 * coeffs[13] * x13 * x33 +
             (81.0 / 4.0) * coeffs[14] * x13 * x36 -
             81.0 / 4.0 * coeffs[15] * x36 * x4 -
             81.0 / 4.0 * coeffs[1] * x45 * x48 +
             (81.0 / 4.0) * coeffs[2] * x44 * x45 -
             81.0 / 4.0 * coeffs[3] * x41 * x46 -
             81.0 / 4.0 * coeffs[4] * x42 * x48 +
             (81.0 / 4.0) * coeffs[5] * x43 * x47 +
             (81.0 / 4.0) * coeffs[6] * x32 * x38 -
             81.0 / 4.0 * coeffs[7] * x35 * x38 -
             81.0 / 4.0 * coeffs[8] * x41 * x43 +
             (81.0 / 4.0) * coeffs[9] * x42 * x44;
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
      out[0] = 3;
      out[1] = 3;
      break;
    case 3:
      out[0] = 0;
      out[1] = 3;
      break;
    case 4:
      out[0] = 1;
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
      out[0] = 3;
      out[1] = 2;
      break;
    case 8:
      out[0] = 2;
      out[1] = 3;
      break;
    case 9:
      out[0] = 1;
      out[1] = 3;
      break;
    case 10:
      out[0] = 0;
      out[1] = 2;
      break;
    case 11:
      out[0] = 0;
      out[1] = 1;
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
      out[0] = 2;
      out[1] = 2;
      break;
    case 15:
      out[0] = 1;
      out[1] = 2;
      break;
    }
  }
}

template <>
struct BasisLagrange<mesh::RefElQuad, 4> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 4;
  static constexpr dim_t num_basis_functions = 25;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_ti, const Scalar *x) {
    switch (i) {
    case 0:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return (1.0 / 9.0) * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1);
    case 1:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return (1.0 / 9.0) * x[0] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1);
    case 2:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return (1.0 / 9.0) * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (2 * x[0] - 1) * (2 * x[1] - 1);
    case 3:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return (1.0 / 9.0) * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (2 * x[1] - 1);
    case 4:
      const Scalar x0 = 4 * x[1];
      return -16.0 / 9.0 * x[0] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1) * (2 * x[1] - 1);
    case 5:
      const Scalar x0 = 4 * x[1];
      return -16.0 / 9.0 * x[0] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1);
    case 6:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return (4.0 / 3.0) * x[0] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1);
    case 7:
      const Scalar x0 = 4 * x[0];
      return -16.0 / 9.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (2 * x[0] - 1) *
             (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3);
    case 8:
      const Scalar x0 = 4 * x[0];
      return -16.0 / 9.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (2 * x[0] - 1) *
             (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1);
    case 9:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return (4.0 / 3.0) * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (2 * x[0] - 1) * (x[1] - 1);
    case 10:
      const Scalar x0 = 4 * x[1];
      return -16.0 / 9.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 1) * (2 * x[1] - 1);
    case 11:
      const Scalar x0 = 4 * x[1];
      return -16.0 / 9.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 3) * (2 * x[1] - 1);
    case 12:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return (4.0 / 3.0) * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[1] - 1);
    case 13:
      const Scalar x0 = 4 * x[0];
      return -16.0 / 9.0 * x[1] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1);
    case 14:
      const Scalar x0 = 4 * x[0];
      return -16.0 / 9.0 * x[1] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3);
    case 15:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return (4.0 / 3.0) * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1);
    case 16:
      return (256.0 / 9.0) * x[0] * x[1] * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 3) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3);
    case 17:
      return (256.0 / 9.0) * x[0] * x[1] * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3);
    case 18:
      return (256.0 / 9.0) * x[0] * x[1] * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1);
    case 19:
      return (256.0 / 9.0) * x[0] * x[1] * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 3) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1);
    case 20:
      const Scalar x0 = 4 * x[0];
      return -64.0 / 3.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3);
    case 21:
      const Scalar x0 = 4 * x[1];
      return -64.0 / 3.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1);
    case 22:
      const Scalar x0 = 4 * x[0];
      return -64.0 / 3.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1);
    case 23:
      const Scalar x0 = 4 * x[1];
      return -64.0 / 3.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1);
    case 24:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return 16 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x[0] - 1) * (x[1] - 1);
    default:
      return 0;
    }
  }

  template <typename Scalar>
  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    switch (i) {
    case 0:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 4 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = 2 * x[0] - 1;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = 4 * x3 * x4;
      const Scalar x6 = x1 - 1;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x3 * x7;
      const Scalar x9 = 2 * x[1] - 1;
      const Scalar x10 = 4 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = 4 * x0 * x9;
      out[0] = (1.0 / 9.0) * x0 * x14 * (x2 * x5 + 2 * x4 * x7 + x5 * x6 + x8);
      out[1] =
          (1.0 / 9.0) * x4 * x8 * (2 * x0 * x13 + x11 * x15 + x12 * x15 + x14);
      break;
    case 1:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 4 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = 2 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x1 - 1;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x4 * x7;
      const Scalar x9 = 2 * x[1] - 1;
      const Scalar x10 = 4 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = 4 * x0 * x9;
      out[0] = (1.0 / 9.0) * x0 * x14 * (x2 * x5 + x3 * x7 + x5 * x6 + x8);
      out[1] = (1.0 / 9.0) * x8 * x[0] *
               (2 * x0 * x13 + x11 * x15 + x12 * x15 + x14);
      break;
    case 2:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = 2 * x[1];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = x10 * x9;
      out[0] = (1.0 / 9.0) * x14 * x[1] * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] =
          (1.0 / 9.0) * x7 * x[0] * (x11 * x15 + x12 * x15 + x13 * x8 + x14);
      break;
    case 3:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = 2 * x[1];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = x10 * x9;
      out[0] =
          (1.0 / 9.0) * x14 * x[1] * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = (1.0 / 9.0) * x3 * x7 * (x11 * x15 + x12 * x15 + x13 * x8 + x14);
      break;
    case 4:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 2 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x[0];
      const Scalar x5 = x4 - 3;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = x6 * x[0];
      const Scalar x8 = 2 * x[1] - 1;
      const Scalar x9 = 4 * x[1];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = 4 * x0 * x8;
      out[0] =
          -16.0 / 9.0 * x0 * x13 * (x1 * x3 * x5 + x2 * x3 * x4 + x3 * x6 + x7);
      out[1] =
          -16.0 / 9.0 * x3 * x7 * (2 * x0 * x12 + x10 * x14 + x11 * x14 + x13);
      break;
    case 5:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 2 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x[0];
      const Scalar x5 = x4 - 1;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = x6 * x[0];
      const Scalar x8 = 2 * x[1] - 1;
      const Scalar x9 = 4 * x[1];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = 4 * x0 * x8;
      out[0] =
          -16.0 / 9.0 * x0 * x13 * (x1 * x3 * x5 + x2 * x3 * x4 + x3 * x6 + x7);
      out[1] =
          -16.0 / 9.0 * x3 * x7 * (2 * x0 * x12 + x10 * x14 + x11 * x14 + x13);
      break;
    case 6:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 4 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = x1 * x3;
      const Scalar x5 = x1 - 1;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = x6 * x[0];
      const Scalar x8 = 2 * x[1] - 1;
      const Scalar x9 = 4 * x[1];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = 4 * x0 * x8;
      out[0] = (4.0 / 3.0) * x0 * x13 * (x2 * x4 + x3 * x6 + x4 * x5 + x7);
      out[1] =
          (4.0 / 3.0) * x3 * x7 * (2 * x0 * x12 + x10 * x14 + x11 * x14 + x13);
      break;
    case 7:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 4 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = 2 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x1 - 1;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x4 * x7;
      const Scalar x9 = 2 * x[1];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[1];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[1];
      out[0] = -16.0 / 9.0 * x0 * x14 * (x2 * x5 + x3 * x7 + x5 * x6 + x8);
      out[1] = -16.0 / 9.0 * x8 * x[0] *
               (x0 * x10 * x11 + x0 * x12 * x9 + x0 * x13 + x14);
      break;
    case 8:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 4 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = 2 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x1 - 1;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x4 * x7;
      const Scalar x9 = 2 * x[1];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[1];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[1];
      out[0] = -16.0 / 9.0 * x0 * x14 * (x2 * x5 + x3 * x7 + x5 * x6 + x8);
      out[1] = -16.0 / 9.0 * x8 * x[0] *
               (x0 * x10 * x11 + x0 * x12 * x9 + x0 * x13 + x14);
      break;
    case 9:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 4 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = 2 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x1 - 1;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x4 * x7;
      const Scalar x9 = 4 * x[1];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x0 * x9;
      out[0] = (4.0 / 3.0) * x0 * x13 * (x2 * x5 + x3 * x7 + x5 * x6 + x8);
      out[1] =
          (4.0 / 3.0) * x8 * x[0] * (x0 * x12 + x10 * x14 + x11 * x14 + x13);
      break;
    case 10:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[1];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[1];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = x8 * x9;
      out[0] = -16.0 / 9.0 * x13 * x[1] *
               (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -16.0 / 9.0 * x2 * x6 * (x10 * x14 + x11 * x14 + x12 * x7 + x13);
      break;
    case 11:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[1];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[1];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = x8 * x9;
      out[0] = -16.0 / 9.0 * x13 * x[1] *
               (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -16.0 / 9.0 * x2 * x6 * (x10 * x14 + x11 * x14 + x12 * x7 + x13);
      break;
    case 12:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[1];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[1];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = x8 * x9;
      out[0] = (4.0 / 3.0) * x13 * x[1] * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = (4.0 / 3.0) * x2 * x6 * (x10 * x14 + x11 * x14 + x12 * x7 + x13);
      break;
    case 13:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 4 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = 2 * x[0] - 1;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = 4 * x3 * x4;
      const Scalar x6 = x1 - 1;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x3 * x7;
      const Scalar x9 = 2 * x[1];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[1];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[1];
      out[0] = -16.0 / 9.0 * x0 * x14 * (x2 * x5 + 2 * x4 * x7 + x5 * x6 + x8);
      out[1] = -16.0 / 9.0 * x4 * x8 *
               (x0 * x10 * x11 + x0 * x12 * x9 + x0 * x13 + x14);
      break;
    case 14:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 4 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = 2 * x[0] - 1;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = 4 * x3 * x4;
      const Scalar x6 = x1 - 1;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x3 * x7;
      const Scalar x9 = 2 * x[1];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[1];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[1];
      out[0] = -16.0 / 9.0 * x0 * x14 * (x2 * x5 + 2 * x4 * x7 + x5 * x6 + x8);
      out[1] = -16.0 / 9.0 * x4 * x8 *
               (x0 * x10 * x11 + x0 * x12 * x9 + x0 * x13 + x14);
      break;
    case 15:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 4 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = 2 * x[0] - 1;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = 4 * x3 * x4;
      const Scalar x6 = x1 - 1;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x3 * x7;
      const Scalar x9 = 4 * x[1];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x0 * x9;
      out[0] = (4.0 / 3.0) * x0 * x13 * (x2 * x5 + 2 * x4 * x7 + x5 * x6 + x8);
      out[1] = (4.0 / 3.0) * x4 * x8 * (x0 * x12 + x10 * x14 + x11 * x14 + x13);
      break;
    case 16:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 2 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x[0];
      const Scalar x5 = x4 - 3;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = x6 * x[0];
      const Scalar x8 = 2 * x[1];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[1];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[1];
      out[0] = (256.0 / 9.0) * x0 * x13 *
               (x1 * x3 * x5 + x2 * x3 * x4 + x3 * x6 + x7);
      out[1] = (256.0 / 9.0) * x3 * x7 *
               (x0 * x10 * x9 + x0 * x11 * x8 + x0 * x12 + x13);
      break;
    case 17:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 2 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x[0];
      const Scalar x5 = x4 - 1;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = x6 * x[0];
      const Scalar x8 = 2 * x[1];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[1];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[1];
      out[0] = (256.0 / 9.0) * x0 * x13 *
               (x1 * x3 * x5 + x2 * x3 * x4 + x3 * x6 + x7);
      out[1] = (256.0 / 9.0) * x3 * x7 *
               (x0 * x10 * x9 + x0 * x11 * x8 + x0 * x12 + x13);
      break;
    case 18:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 2 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x[0];
      const Scalar x5 = x4 - 1;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = x6 * x[0];
      const Scalar x8 = 2 * x[1];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[1];
      out[0] = (256.0 / 9.0) * x0 * x13 *
               (x1 * x3 * x5 + x2 * x3 * x4 + x3 * x6 + x7);
      out[1] = (256.0 / 9.0) * x3 * x7 *
               (x0 * x10 * x9 + x0 * x11 * x8 + x0 * x12 + x13);
      break;
    case 19:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 2 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x[0];
      const Scalar x5 = x4 - 3;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = x6 * x[0];
      const Scalar x8 = 2 * x[1];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[1];
      out[0] = (256.0 / 9.0) * x0 * x13 *
               (x1 * x3 * x5 + x2 * x3 * x4 + x3 * x6 + x7);
      out[1] = (256.0 / 9.0) * x3 * x7 *
               (x0 * x10 * x9 + x0 * x11 * x8 + x0 * x12 + x13);
      break;
    case 20:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 4 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = x1 * x3;
      const Scalar x5 = x1 - 1;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = x6 * x[0];
      const Scalar x8 = 2 * x[1];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[1];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[1];
      out[0] = -64.0 / 3.0 * x0 * x13 * (x2 * x4 + x3 * x6 + x4 * x5 + x7);
      out[1] = -64.0 / 3.0 * x3 * x7 *
               (x0 * x10 * x9 + x0 * x11 * x8 + x0 * x12 + x13);
      break;
    case 21:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 2 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x[0];
      const Scalar x5 = x4 - 1;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = x6 * x[0];
      const Scalar x8 = 4 * x[1];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x0 * x8;
      out[0] =
          -64.0 / 3.0 * x0 * x12 * (x1 * x3 * x5 + x2 * x3 * x4 + x3 * x6 + x7);
      out[1] = -64.0 / 3.0 * x3 * x7 * (x0 * x11 + x10 * x13 + x12 + x13 * x9);
      break;
    case 22:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 4 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = x1 * x3;
      const Scalar x5 = x1 - 1;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = x6 * x[0];
      const Scalar x8 = 2 * x[1];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[1];
      out[0] = -64.0 / 3.0 * x0 * x13 * (x2 * x4 + x3 * x6 + x4 * x5 + x7);
      out[1] = -64.0 / 3.0 * x3 * x7 *
               (x0 * x10 * x9 + x0 * x11 * x8 + x0 * x12 + x13);
      break;
    case 23:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 2 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x[0];
      const Scalar x5 = x4 - 3;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = x6 * x[0];
      const Scalar x8 = 4 * x[1];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x0 * x8;
      out[0] =
          -64.0 / 3.0 * x0 * x12 * (x1 * x3 * x5 + x2 * x3 * x4 + x3 * x6 + x7);
      out[1] = -64.0 / 3.0 * x3 * x7 * (x0 * x11 + x10 * x13 + x12 + x13 * x9);
      break;
    case 24:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 4 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = x1 * x3;
      const Scalar x5 = x1 - 1;
      const Scalar x6 = x2 * x5;
      const Scalar x7 = x6 * x[0];
      const Scalar x8 = 4 * x[1];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x0 * x8;
      out[0] = 16 * x0 * x12 * (x2 * x4 + x3 * x6 + x4 * x5 + x7);
      out[1] = 16 * x3 * x7 * (x0 * x11 + x10 * x13 + x12 + x13 * x9);
      break;
    default:
      break;
    }
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 4 * x[1];
    const Scalar x1 = x0 - 3;
    const Scalar x2 = x1 * x[0];
    const Scalar x3 = 2 * x[1] - 1;
    const Scalar x4 = 4 * x[0];
    const Scalar x5 = x4 - 3;
    const Scalar x6 = x[0] - 1;
    const Scalar x7 = x[1] - 1;
    const Scalar x8 = x6 * x7;
    const Scalar x9 = x5 * x8 * x[1];
    const Scalar x10 = x3 * x9;
    const Scalar x11 = 2 * x[0] - 1;
    const Scalar x12 = 256 * x11;
    const Scalar x13 = x4 - 1;
    const Scalar x14 = x1 * x13 * x3 * x[0];
    const Scalar x15 = x0 - 1;
    const Scalar x16 = x11 * x15;
    const Scalar x17 = x16 * x8;
    const Scalar x18 = x13 * x[0];
    const Scalar x19 = x18 * x3;
    const Scalar x20 = 192 * x9;
    const Scalar x21 = x1 * x18;
    const Scalar x22 = x21 * x[1];
    const Scalar x23 = x16 * x2;
    const Scalar x24 = x6 * x[1];
    const Scalar x25 = 16 * x24;
    const Scalar x26 = x3 * x5;
    const Scalar x27 = x14 * x5;
    const Scalar x28 = 12 * x15 * x27;
    const Scalar x29 = 16 * x11;
    const Scalar x30 = x7 * x[1];
    const Scalar x31 = 16 * x16;
    const Scalar x32 = 12 * x16;
    const Scalar x33 = x1 * x11 * x13 * x15 * x3 * x5;
    const Scalar x34 = x33 * x[0];
    const Scalar x35 = x10 * x13;
    const Scalar x36 = 16 * x17;
    return (1.0 / 9.0) * coeffs[0] * x33 * x8 -
           1.0 / 9.0 * coeffs[10] * x14 * x16 * x25 -
           1.0 / 9.0 * coeffs[11] * x23 * x25 * x26 +
           (1.0 / 9.0) * coeffs[12] * x24 * x28 -
           1.0 / 9.0 * coeffs[13] * x31 * x35 -
           1.0 / 9.0 * coeffs[14] * x1 * x29 * x35 +
           (1.0 / 9.0) * coeffs[15] * x1 * x13 * x32 * x9 +
           (1.0 / 9.0) * coeffs[16] * x10 * x12 * x2 +
           (1.0 / 9.0) * coeffs[17] * x12 * x14 * x8 * x[1] +
           (256.0 / 9.0) * coeffs[18] * x17 * x19 * x[1] +
           (256.0 / 9.0) * coeffs[19] * x10 * x16 * x[0] +
           (1.0 / 9.0) * coeffs[1] * x34 * x7 -
           1.0 / 9.0 * coeffs[20] * x14 * x20 -
           64.0 / 3.0 * coeffs[21] * x17 * x22 -
           1.0 / 9.0 * coeffs[22] * x15 * x19 * x20 -
           1.0 / 9.0 * coeffs[23] * x20 * x23 +
           16 * coeffs[24] * x15 * x21 * x9 +
           (1.0 / 9.0) * coeffs[2] * x34 * x[1] +
           (1.0 / 9.0) * coeffs[3] * x24 * x33 -
           1.0 / 9.0 * coeffs[4] * x2 * x26 * x36 -
           1.0 / 9.0 * coeffs[5] * x14 * x36 +
           (1.0 / 9.0) * coeffs[6] * x28 * x8 -
           1.0 / 9.0 * coeffs[7] * x27 * x29 * x30 -
           1.0 / 9.0 * coeffs[8] * x19 * x30 * x31 * x5 +
           (1.0 / 9.0) * coeffs[9] * x22 * x32 * x5 * x7;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 2 * x[0];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = x[0] - 1;
    const Scalar x3 = 4 * x[0];
    const Scalar x4 = x2 * x3;
    const Scalar x5 = x1 * x4;
    const Scalar x6 = x3 - 3;
    const Scalar x7 = x0 * x2;
    const Scalar x8 = x1 * x6;
    const Scalar x9 = x8 * x[0];
    const Scalar x10 = x2 * x8;
    const Scalar x11 = x10 + x5 + x6 * x7 + x9;
    const Scalar x12 = 2 * x[1];
    const Scalar x13 = x12 - 1;
    const Scalar x14 = 4 * x[1];
    const Scalar x15 = x14 - 3;
    const Scalar x16 = x13 * x15;
    const Scalar x17 = x16 * x[1];
    const Scalar x18 = x[1] - 1;
    const Scalar x19 = 256 * x18;
    const Scalar x20 = x17 * x19;
    const Scalar x21 = x3 - 1;
    const Scalar x22 = x1 * x21;
    const Scalar x23 = x22 * x[0];
    const Scalar x24 = x2 * x22;
    const Scalar x25 = x21 * x7 + x23 + x24 + x5;
    const Scalar x26 = x14 - 1;
    const Scalar x27 = x13 * x26;
    const Scalar x28 = x27 * x[1];
    const Scalar x29 = x19 * x28;
    const Scalar x30 = x21 * x6;
    const Scalar x31 = x30 * x[0];
    const Scalar x32 = x2 * x30;
    const Scalar x33 = x21 * x4 + x31 + x32 + x4 * x6;
    const Scalar x34 = 192 * x18;
    const Scalar x35 = x33 * x34;
    const Scalar x36 = x15 * x26;
    const Scalar x37 = x36 * x[1];
    const Scalar x38 = x34 * x37;
    const Scalar x39 = x18 * x37;
    const Scalar x40 = 144 * coeffs[24];
    const Scalar x41 = x16 * x26;
    const Scalar x42 = x41 * x[1];
    const Scalar x43 = 16 * x42;
    const Scalar x44 = 12 * x33;
    const Scalar x45 = x18 * x41;
    const Scalar x46 = 16 * x45;
    const Scalar x47 = x1 * x30;
    const Scalar x48 = x0 * x30 + x22 * x3 + x3 * x8 + x47;
    const Scalar x49 = 16 * x18;
    const Scalar x50 = x48 * x49;
    const Scalar x51 = 12 * coeffs[9];
    const Scalar x52 = 4 * x10 + 4 * x24 + 2 * x32 + x47;
    const Scalar x53 = x49 * x52;
    const Scalar x54 = 12 * coeffs[15];
    const Scalar x55 = x14 * x18;
    const Scalar x56 = x13 * x55;
    const Scalar x57 = x15 * x18;
    const Scalar x58 = x16 * x18;
    const Scalar x59 = x12 * x57 + x17 + x56 + x58;
    const Scalar x60 = 256 * x2;
    const Scalar x61 = x59 * x60;
    const Scalar x62 = x18 * x27;
    const Scalar x63 = x12 * x18 * x26 + x28 + x56 + x62;
    const Scalar x64 = x60 * x63;
    const Scalar x65 = 192 * x2;
    const Scalar x66 = x31 * x65;
    const Scalar x67 = x18 * x36;
    const Scalar x68 = x14 * x57 + x26 * x55 + x37 + x67;
    const Scalar x69 = x65 * x68;
    const Scalar x70 = x2 * x31;
    const Scalar x71 = x47 * x[0];
    const Scalar x72 = 16 * x71;
    const Scalar x73 = 16 * x2;
    const Scalar x74 = x47 * x73;
    const Scalar x75 = x12 * x36 + x14 * x16 + x14 * x27 + x41;
    const Scalar x76 = x73 * x75;
    const Scalar x77 = 12 * x70;
    const Scalar x78 = x47 * x75;
    const Scalar x79 = x41 + 4 * x58 + 4 * x62 + 2 * x67;
    const Scalar x80 = x73 * x79;
    const Scalar x81 = x47 * x79;
    out[0] = (1.0 / 9.0) * coeffs[0] * x45 * x52 -
             1.0 / 9.0 * coeffs[10] * x25 * x43 -
             1.0 / 9.0 * coeffs[11] * x11 * x43 +
             (1.0 / 9.0) * coeffs[12] * x42 * x44 -
             1.0 / 9.0 * coeffs[13] * x28 * x53 -
             1.0 / 9.0 * coeffs[14] * x17 * x53 +
             (1.0 / 9.0) * coeffs[16] * x11 * x20 +
             (1.0 / 9.0) * coeffs[17] * x20 * x25 +
             (1.0 / 9.0) * coeffs[18] * x25 * x29 +
             (1.0 / 9.0) * coeffs[19] * x11 * x29 +
             (1.0 / 9.0) * coeffs[1] * x45 * x48 -
             1.0 / 9.0 * coeffs[20] * x17 * x35 -
             1.0 / 9.0 * coeffs[21] * x25 * x38 -
             1.0 / 9.0 * coeffs[22] * x28 * x35 -
             1.0 / 9.0 * coeffs[23] * x11 * x38 +
             (1.0 / 9.0) * coeffs[2] * x42 * x48 +
             (1.0 / 9.0) * coeffs[3] * x42 * x52 -
             1.0 / 9.0 * coeffs[4] * x11 * x46 -
             1.0 / 9.0 * coeffs[5] * x25 * x46 +
             (1.0 / 9.0) * coeffs[6] * x44 * x45 -
             1.0 / 9.0 * coeffs[7] * x17 * x50 -
             1.0 / 9.0 * coeffs[8] * x28 * x50 + (1.0 / 9.0) * x33 * x39 * x40 +
             (1.0 / 9.0) * x39 * x48 * x51 + (1.0 / 9.0) * x39 * x52 * x54;
    out[1] =
        (1.0 / 9.0) * coeffs[0] * x2 * x81 -
        1.0 / 9.0 * coeffs[10] * x23 * x76 - 1.0 / 9.0 * coeffs[11] * x76 * x9 +
        (1.0 / 9.0) * coeffs[12] * x75 * x77 -
        1.0 / 9.0 * coeffs[13] * x63 * x74 -
        1.0 / 9.0 * coeffs[14] * x59 * x74 +
        (1.0 / 9.0) * coeffs[16] * x61 * x9 +
        (1.0 / 9.0) * coeffs[17] * x23 * x61 +
        (1.0 / 9.0) * coeffs[18] * x23 * x64 +
        (1.0 / 9.0) * coeffs[19] * x64 * x9 +
        (1.0 / 9.0) * coeffs[1] * x81 * x[0] -
        1.0 / 9.0 * coeffs[20] * x59 * x66 -
        1.0 / 9.0 * coeffs[21] * x23 * x69 -
        1.0 / 9.0 * coeffs[22] * x63 * x66 - 1.0 / 9.0 * coeffs[23] * x69 * x9 +
        (1.0 / 9.0) * coeffs[2] * x78 * x[0] +
        (1.0 / 9.0) * coeffs[3] * x2 * x78 - 1.0 / 9.0 * coeffs[4] * x80 * x9 -
        1.0 / 9.0 * coeffs[5] * x23 * x80 +
        (1.0 / 9.0) * coeffs[6] * x77 * x79 -
        1.0 / 9.0 * coeffs[7] * x59 * x72 - 1.0 / 9.0 * coeffs[8] * x63 * x72 +
        (1.0 / 9.0) * x2 * x47 * x54 * x68 + (1.0 / 9.0) * x40 * x68 * x70 +
        (1.0 / 9.0) * x51 * x68 * x71;
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
      out[0] = 4;
      out[1] = 4;
      break;
    case 3:
      out[0] = 0;
      out[1] = 4;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      break;
    case 5:
      out[0] = 3;
      out[1] = 0;
      break;
    case 6:
      out[0] = 2;
      out[1] = 0;
      break;
    case 7:
      out[0] = 4;
      out[1] = 1;
      break;
    case 8:
      out[0] = 4;
      out[1] = 3;
      break;
    case 9:
      out[0] = 4;
      out[1] = 2;
      break;
    case 10:
      out[0] = 3;
      out[1] = 4;
      break;
    case 11:
      out[0] = 1;
      out[1] = 4;
      break;
    case 12:
      out[0] = 2;
      out[1] = 4;
      break;
    case 13:
      out[0] = 0;
      out[1] = 3;
      break;
    case 14:
      out[0] = 0;
      out[1] = 1;
      break;
    case 15:
      out[0] = 0;
      out[1] = 2;
      break;
    case 16:
      out[0] = 1;
      out[1] = 1;
      break;
    case 17:
      out[0] = 3;
      out[1] = 1;
      break;
    case 18:
      out[0] = 3;
      out[1] = 3;
      break;
    case 19:
      out[0] = 1;
      out[1] = 3;
      break;
    case 20:
      out[0] = 2;
      out[1] = 1;
      break;
    case 21:
      out[0] = 3;
      out[1] = 2;
      break;
    case 22:
      out[0] = 2;
      out[1] = 3;
      break;
    case 23:
      out[0] = 1;
      out[1] = 2;
      break;
    case 24:
      out[0] = 2;
      out[1] = 2;
      break;
    }
  }
}

template <>
struct BasisLagrange<mesh::RefElQuad, 5> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 5;
  static constexpr dim_t num_basis_functions = 36;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_ti, const Scalar *x) {
    switch (i) {
    case 0:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (1.0 / 576.0) * (x0 - 4) * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1) *
             (x[1] - 1);
    case 1:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -1.0 / 576.0 * x[0] * (x0 - 4) * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[1] - 1);
    case 2:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (1.0 / 576.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1);
    case 3:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -1.0 / 576.0 * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1);
    case 4:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -25.0 / 576.0 * x[0] * (x0 - 4) * (x0 - 3) * (x0 - 2) * (x1 - 4) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 5:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (25.0 / 576.0) * x[0] * (x0 - 3) * (x0 - 2) * (x0 - 1) * (x1 - 4) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 6:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (25.0 / 288.0) * x[0] * (x0 - 4) * (x0 - 3) * (x0 - 1) * (x1 - 4) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 7:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -25.0 / 288.0 * x[0] * (x0 - 4) * (x0 - 2) * (x0 - 1) * (x1 - 4) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 8:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (25.0 / 576.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x[1] - 1);
    case 9:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -25.0 / 576.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[1] - 1);
    case 10:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -25.0 / 288.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x[1] - 1);
    case 11:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (25.0 / 288.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x[1] - 1);
    case 12:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -25.0 / 576.0 * x[0] * x[1] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1);
    case 13:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (25.0 / 576.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1);
    case 14:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (25.0 / 288.0) * x[0] * x[1] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1);
    case 15:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -25.0 / 288.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1);
    case 16:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (25.0 / 576.0) * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 17:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -25.0 / 576.0 * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x[0] - 1) * (x[1] - 1);
    case 18:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -25.0 / 288.0 * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 19:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (25.0 / 288.0) * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 20:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (625.0 / 576.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x[0] - 1) * (x[1] - 1);
    case 21:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -625.0 / 576.0 * x[0] * x[1] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x[0] - 1) * (x[1] - 1);
    case 22:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (625.0 / 576.0) * x[0] * x[1] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 23:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -625.0 / 576.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 24:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -625.0 / 288.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x[0] - 1) * (x[1] - 1);
    case 25:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (625.0 / 288.0) * x[0] * x[1] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x[0] - 1) * (x[1] - 1);
    case 26:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (625.0 / 288.0) * x[0] * x[1] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 27:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -625.0 / 288.0 * x[0] * x[1] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 28:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -625.0 / 288.0 * x[0] * x[1] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 29:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (625.0 / 288.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 30:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (625.0 / 288.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 31:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -625.0 / 288.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 3) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 32:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (625.0 / 144.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 33:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -625.0 / 144.0 * x[0] * x[1] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 34:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return (625.0 / 144.0) * x[0] * x[1] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    case 35:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      return -625.0 / 144.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 2) * (x1 - 1) * (x[0] - 1) * (x[1] - 1);
    default:
      return 0;
    }
  }

  template <typename Scalar>
  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    switch (i) {
    case 0:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x1 - 2;
      const Scalar x5 = x[0] - 1;
      const Scalar x6 = 5 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x3 * x7;
      const Scalar x9 = x4 * x6 * x7;
      const Scalar x10 = x4 * x8;
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = 5 * x0;
      const Scalar x19 = x12 * x13 * x18;
      out[0] = (1.0 / 576.0) * x0 * x17 *
               (x10 + x2 * x3 * x4 * x6 + x2 * x9 + x3 * x9 + x6 * x8);
      out[1] =
          (1.0 / 576.0) * x10 * x5 *
          (x12 * x14 * x15 * x18 + x14 * x19 + x15 * x19 + x16 * x18 + x17);
      break;
    case 1:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x1 - 2;
      const Scalar x5 = x1 - 1;
      const Scalar x6 = x2 * x3 * x5;
      const Scalar x7 = x1 * x4 * x5;
      const Scalar x8 = x4 * x6;
      const Scalar x9 = 5 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x9 - 3;
      const Scalar x14 = x11 * x12 * x13;
      const Scalar x15 = x10 * x14;
      const Scalar x16 = 5 * x0;
      const Scalar x17 = x10 * x11 * x16;
      out[0] = -1.0 / 576.0 * x0 * x15 *
               (x1 * x2 * x3 * x4 + x1 * x6 + x2 * x7 + x3 * x7 + x8);
      out[1] =
          -1.0 / 576.0 * x8 * x[0] *
          (x10 * x12 * x13 * x16 + x12 * x17 + x13 * x17 + x14 * x16 + x15);
      break;
    case 2:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = 5 * x[1];
      const Scalar x9 = x8 - 2;
      const Scalar x10 = x8 - 1;
      const Scalar x11 = x8 - 4;
      const Scalar x12 = x8 - 3;
      const Scalar x13 = x10 * x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = x10 * x8 * x9;
      out[0] = (1.0 / 576.0) * x14 * x[1] *
               (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] = (1.0 / 576.0) * x7 * x[0] *
               (x11 * x12 * x8 * x9 + x11 * x15 + x12 * x15 + x13 * x8 + x14);
      break;
    case 3:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = 5 * x4;
      const Scalar x6 = x0 - 1;
      const Scalar x7 = x1 * x2 * x6;
      const Scalar x8 = x3 * x5 * x6;
      const Scalar x9 = x3 * x7;
      const Scalar x10 = 5 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = x10 * x11 * x12;
      out[0] = -1.0 / 576.0 * x16 * x[1] *
               (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -1.0 / 576.0 * x4 * x9 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x17 + x14 * x17 + x16);
      break;
    case 4:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x2 * x3;
      const Scalar x5 = x[0] - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x1 - 2;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = x4 * x7;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = 5 * x0;
      const Scalar x19 = x12 * x13 * x18;
      out[0] = -25.0 / 576.0 * x0 * x17 *
               (x10 + x2 * x8 + x3 * x8 + x4 * x6 + x5 * x9);
      out[1] =
          -25.0 / 576.0 * x10 * x5 *
          (x12 * x14 * x15 * x18 + x14 * x19 + x15 * x19 + x16 * x18 + x17);
      break;
    case 5:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = x1 - 2;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = 5 * x0;
      const Scalar x19 = x12 * x13 * x18;
      out[0] = (25.0 / 576.0) * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] =
          (25.0 / 576.0) * x10 * x4 *
          (x12 * x14 * x15 * x18 + x14 * x19 + x15 * x19 + x16 * x18 + x17);
      break;
    case 6:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = 5 * x0;
      const Scalar x19 = x12 * x13 * x18;
      out[0] = (25.0 / 288.0) * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] =
          (25.0 / 288.0) * x10 * x4 *
          (x12 * x14 * x15 * x18 + x14 * x19 + x15 * x19 + x16 * x18 + x17);
      break;
    case 7:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 2;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = 5 * x0;
      const Scalar x19 = x12 * x13 * x18;
      out[0] = -25.0 / 288.0 * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] =
          -25.0 / 288.0 * x10 * x4 *
          (x12 * x14 * x15 * x18 + x14 * x19 + x15 * x19 + x16 * x18 + x17);
      break;
    case 8:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x1 - 2;
      const Scalar x5 = x1 - 1;
      const Scalar x6 = x2 * x3 * x5;
      const Scalar x7 = x1 * x4 * x5;
      const Scalar x8 = x4 * x6;
      const Scalar x9 = 5 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 4;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[1];
      const Scalar x16 = x0 * x9;
      const Scalar x17 = x10 * x16;
      out[0] = (25.0 / 576.0) * x0 * x15 *
               (x1 * x2 * x3 * x4 + x1 * x6 + x2 * x7 + x3 * x7 + x8);
      out[1] = (25.0 / 576.0) * x8 * x[0] *
               (x0 * x14 + x11 * x17 + x12 * x17 + x13 * x16 + x15);
      break;
    case 9:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x1 - 2;
      const Scalar x5 = x1 - 1;
      const Scalar x6 = x2 * x3 * x5;
      const Scalar x7 = x1 * x4 * x5;
      const Scalar x8 = x4 * x6;
      const Scalar x9 = 5 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[1];
      const Scalar x16 = x0 * x9;
      const Scalar x17 = x10 * x16;
      out[0] = -25.0 / 576.0 * x0 * x15 *
               (x1 * x2 * x3 * x4 + x1 * x6 + x2 * x7 + x3 * x7 + x8);
      out[1] = -25.0 / 576.0 * x8 * x[0] *
               (x0 * x14 + x11 * x17 + x12 * x17 + x13 * x16 + x15);
      break;
    case 10:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x1 - 2;
      const Scalar x5 = x1 - 1;
      const Scalar x6 = x2 * x3 * x5;
      const Scalar x7 = x1 * x4 * x5;
      const Scalar x8 = x4 * x6;
      const Scalar x9 = 5 * x[1];
      const Scalar x10 = x9 - 3;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[1];
      const Scalar x16 = x0 * x9;
      const Scalar x17 = x10 * x16;
      out[0] = -25.0 / 288.0 * x0 * x15 *
               (x1 * x2 * x3 * x4 + x1 * x6 + x2 * x7 + x3 * x7 + x8);
      out[1] = -25.0 / 288.0 * x8 * x[0] *
               (x0 * x14 + x11 * x17 + x12 * x17 + x13 * x16 + x15);
      break;
    case 11:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x1 - 2;
      const Scalar x5 = x1 - 1;
      const Scalar x6 = x2 * x3 * x5;
      const Scalar x7 = x1 * x4 * x5;
      const Scalar x8 = x4 * x6;
      const Scalar x9 = 5 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[1];
      const Scalar x16 = x0 * x9;
      const Scalar x17 = x10 * x16;
      out[0] = (25.0 / 288.0) * x0 * x15 *
               (x1 * x2 * x3 * x4 + x1 * x6 + x2 * x7 + x3 * x7 + x8);
      out[1] = (25.0 / 288.0) * x8 * x[0] *
               (x0 * x14 + x11 * x17 + x12 * x17 + x13 * x16 + x15);
      break;
    case 12:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x0 - 2;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x2 * x4;
      const Scalar x6 = x0 - 1;
      const Scalar x7 = x1 * x6;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x8 * x[0];
      const Scalar x10 = 5 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = x10 * x11 * x12;
      out[0] = -25.0 / 576.0 * x16 * x[1] *
               (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -25.0 / 576.0 * x3 * x9 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x17 + x14 * x17 + x16);
      break;
    case 13:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x0 * x4;
      const Scalar x6 = x0 - 2;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = x3 * x6;
      const Scalar x9 = x8 * x[0];
      const Scalar x10 = 5 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = x10 * x11 * x12;
      out[0] = (25.0 / 576.0) * x16 * x[1] *
               (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          (25.0 / 576.0) * x4 * x9 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x17 + x14 * x17 + x16);
      break;
    case 14:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 2;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x2 * x4;
      const Scalar x6 = x0 - 1;
      const Scalar x7 = x1 * x6;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x8 * x[0];
      const Scalar x10 = 5 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = x10 * x11 * x12;
      out[0] = (25.0 / 288.0) * x16 * x[1] *
               (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          (25.0 / 288.0) * x3 * x9 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x17 + x14 * x17 + x16);
      break;
    case 15:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x2 * x4;
      const Scalar x6 = x0 - 1;
      const Scalar x7 = x1 * x6;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x8 * x[0];
      const Scalar x10 = 5 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = x10 * x11 * x12;
      out[0] = -25.0 / 288.0 * x16 * x[1] *
               (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -25.0 / 288.0 * x3 * x9 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x17 + x14 * x17 + x16);
      break;
    case 16:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x1 - 2;
      const Scalar x5 = x[0] - 1;
      const Scalar x6 = 5 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x3 * x7;
      const Scalar x9 = x4 * x6 * x7;
      const Scalar x10 = x4 * x8;
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = (25.0 / 576.0) * x0 * x17 *
               (x10 + x2 * x3 * x4 * x6 + x2 * x9 + x3 * x9 + x6 * x8);
      out[1] = (25.0 / 576.0) * x10 * x5 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 17:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x1 - 2;
      const Scalar x5 = x[0] - 1;
      const Scalar x6 = 5 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x3 * x7;
      const Scalar x9 = x4 * x6 * x7;
      const Scalar x10 = x4 * x8;
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = -25.0 / 576.0 * x0 * x17 *
               (x10 + x2 * x3 * x4 * x6 + x2 * x9 + x3 * x9 + x6 * x8);
      out[1] = -25.0 / 576.0 * x10 * x5 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 18:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x1 - 2;
      const Scalar x5 = x[0] - 1;
      const Scalar x6 = 5 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x3 * x7;
      const Scalar x9 = x4 * x6 * x7;
      const Scalar x10 = x4 * x8;
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = -25.0 / 288.0 * x0 * x17 *
               (x10 + x2 * x3 * x4 * x6 + x2 * x9 + x3 * x9 + x6 * x8);
      out[1] = -25.0 / 288.0 * x10 * x5 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 19:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x1 - 2;
      const Scalar x5 = x[0] - 1;
      const Scalar x6 = 5 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x3 * x7;
      const Scalar x9 = x4 * x6 * x7;
      const Scalar x10 = x4 * x8;
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = (25.0 / 288.0) * x0 * x17 *
               (x10 + x2 * x3 * x4 * x6 + x2 * x9 + x3 * x9 + x6 * x8);
      out[1] = (25.0 / 288.0) * x10 * x5 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 20:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x2 * x3;
      const Scalar x5 = x[0] - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x1 - 2;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = x4 * x7;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = (625.0 / 576.0) * x0 * x17 *
               (x10 + x2 * x8 + x3 * x8 + x4 * x6 + x5 * x9);
      out[1] = (625.0 / 576.0) * x10 * x5 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 21:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = x1 - 2;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = -625.0 / 576.0 * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] = -625.0 / 576.0 * x10 * x4 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 22:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = x1 - 2;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = (625.0 / 576.0) * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] = (625.0 / 576.0) * x10 * x4 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 23:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x2 * x3;
      const Scalar x5 = x[0] - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x1 - 2;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = x4 * x7;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = -625.0 / 576.0 * x0 * x17 *
               (x10 + x2 * x8 + x3 * x8 + x4 * x6 + x5 * x9);
      out[1] = -625.0 / 576.0 * x10 * x5 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 24:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = -625.0 / 288.0 * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] = -625.0 / 288.0 * x10 * x4 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 25:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 2;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = (625.0 / 288.0) * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] = (625.0 / 288.0) * x10 * x4 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 26:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = x1 - 2;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = (625.0 / 288.0) * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] = (625.0 / 288.0) * x10 * x4 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 27:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 3;
      const Scalar x3 = x1 - 2;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = -625.0 / 288.0 * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] = -625.0 / 288.0 * x10 * x4 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 28:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 2;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = -625.0 / 288.0 * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] = -625.0 / 288.0 * x10 * x4 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 29:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = (625.0 / 288.0) * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] = (625.0 / 288.0) * x10 * x4 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 30:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x2 * x3;
      const Scalar x5 = x[0] - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x1 - 2;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = x4 * x7;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = (625.0 / 288.0) * x0 * x17 *
               (x10 + x2 * x8 + x3 * x8 + x4 * x6 + x5 * x9);
      out[1] = (625.0 / 288.0) * x10 * x5 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 31:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x2 * x3;
      const Scalar x5 = x[0] - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x1 - 2;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = x4 * x7;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = -625.0 / 288.0 * x0 * x17 *
               (x10 + x2 * x8 + x3 * x8 + x4 * x6 + x5 * x9);
      out[1] = -625.0 / 288.0 * x10 * x5 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 32:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = (625.0 / 144.0) * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] = (625.0 / 144.0) * x10 * x4 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 33:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 2;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = -625.0 / 144.0 * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] = -625.0 / 144.0 * x10 * x4 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 34:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 2;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = (625.0 / 144.0) * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] = (625.0 / 144.0) * x10 * x4 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    case 35:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = 5 * x[0];
      const Scalar x2 = x1 - 4;
      const Scalar x3 = x1 - 3;
      const Scalar x4 = x[0] - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x3 * x5;
      const Scalar x7 = x1 - 1;
      const Scalar x8 = x2 * x7;
      const Scalar x9 = x3 * x8;
      const Scalar x10 = x9 * x[0];
      const Scalar x11 = 5 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[1];
      const Scalar x18 = x0 * x11;
      const Scalar x19 = x12 * x18;
      out[0] = -625.0 / 144.0 * x0 * x17 *
               (x10 + x2 * x6 + x4 * x9 + x5 * x8 + x6 * x7);
      out[1] = -625.0 / 144.0 * x10 * x4 *
               (x0 * x16 + x13 * x19 + x14 * x19 + x15 * x18 + x17);
      break;
    default:
      break;
    }
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 5 * x[0];
    const Scalar x1 = x0 - 2;
    const Scalar x2 = x[0] - 1;
    const Scalar x3 = x[1] - 1;
    const Scalar x4 = x2 * x3;
    const Scalar x5 = x4 * x[1];
    const Scalar x6 = x1 * x5;
    const Scalar x7 = 5 * x[1];
    const Scalar x8 = x7 - 3;
    const Scalar x9 = 625 * x8;
    const Scalar x10 = x6 * x9;
    const Scalar x11 = x0 - 4;
    const Scalar x12 = x7 - 4;
    const Scalar x13 = x0 - 3;
    const Scalar x14 = x7 - 2;
    const Scalar x15 = x13 * x14;
    const Scalar x16 = x12 * x15;
    const Scalar x17 = x16 * x[0];
    const Scalar x18 = x11 * x17;
    const Scalar x19 = x0 - 1;
    const Scalar x20 = x17 * x19;
    const Scalar x21 = x7 - 1;
    const Scalar x22 = x21 * x6;
    const Scalar x23 = x19 * x22;
    const Scalar x24 = x15 * x9 * x[0];
    const Scalar x25 = x11 * x22;
    const Scalar x26 = x16 * x5;
    const Scalar x27 = x11 * x19;
    const Scalar x28 = 1250 * x8;
    const Scalar x29 = x28 * x[0];
    const Scalar x30 = x27 * x29;
    const Scalar x31 = x14 * x6;
    const Scalar x32 = x12 * x31;
    const Scalar x33 = x12 * x13;
    const Scalar x34 = x29 * x33;
    const Scalar x35 = 1250 * x17;
    const Scalar x36 = x11 * x19 * x21 * x[0];
    const Scalar x37 = x28 * x36;
    const Scalar x38 = x11 * x12 * x19 * x21 * x8 * x[0];
    const Scalar x39 = 2500 * x38;
    const Scalar x40 = 2500 * x36;
    const Scalar x41 = 50 * x38;
    const Scalar x42 = x1 * x41;
    const Scalar x43 = x3 * x[1];
    const Scalar x44 = x2 * x[1];
    const Scalar x45 = 25 * x8;
    const Scalar x46 = x1 * x45;
    const Scalar x47 = x21 * x46;
    const Scalar x48 = x44 * x47;
    const Scalar x49 = x14 * x42;
    const Scalar x50 = x15 * x41;
    const Scalar x51 = x43 * x46;
    const Scalar x52 = x1 * x11 * x12 * x13 * x14 * x19 * x21 * x8;
    const Scalar x53 = x52 * x[0];
    const Scalar x54 = x16 * x27;
    const Scalar x55 = 50 * x22;
    const Scalar x56 = x4 * x47;
    return (1.0 / 576.0) * coeffs[0] * x4 * x52 -
           1.0 / 576.0 * coeffs[10] * x13 * x42 * x43 +
           (25.0 / 288.0) * coeffs[11] * x1 * x16 * x36 * x43 -
           1.0 / 576.0 * coeffs[12] * x20 * x48 +
           (1.0 / 576.0) * coeffs[13] * x18 * x48 +
           (1.0 / 576.0) * coeffs[14] * x44 * x49 -
           1.0 / 576.0 * coeffs[15] * x44 * x50 +
           (1.0 / 576.0) * coeffs[16] * x15 * x22 * x27 * x45 -
           1.0 / 576.0 * coeffs[17] * x45 * x54 * x6 -
           1.0 / 576.0 * coeffs[18] * x54 * x55 +
           (1.0 / 576.0) * coeffs[19] * x27 * x33 * x55 * x8 -
           1.0 / 576.0 * coeffs[1] * x3 * x53 +
           (1.0 / 576.0) * coeffs[20] * x10 * x18 -
           1.0 / 576.0 * coeffs[21] * x10 * x20 +
           (1.0 / 576.0) * coeffs[22] * x23 * x24 -
           1.0 / 576.0 * coeffs[23] * x24 * x25 -
           1.0 / 576.0 * coeffs[24] * x26 * x30 +
           (1.0 / 576.0) * coeffs[25] * x30 * x32 +
           (1.0 / 576.0) * coeffs[26] * x23 * x34 -
           1.0 / 576.0 * coeffs[27] * x23 * x35 -
           1.0 / 576.0 * coeffs[28] * x31 * x37 +
           (1.0 / 576.0) * coeffs[29] * x15 * x37 * x5 +
           (1.0 / 576.0) * coeffs[2] * x53 * x[1] +
           (1.0 / 576.0) * coeffs[30] * x25 * x35 -
           1.0 / 576.0 * coeffs[31] * x25 * x34 +
           (1.0 / 576.0) * coeffs[32] * x13 * x39 * x5 -
           1.0 / 576.0 * coeffs[33] * x39 * x6 +
           (1.0 / 576.0) * coeffs[34] * x32 * x40 -
           1.0 / 576.0 * coeffs[35] * x26 * x40 -
           1.0 / 576.0 * coeffs[3] * x44 * x52 -
           1.0 / 576.0 * coeffs[4] * x18 * x56 +
           (1.0 / 576.0) * coeffs[5] * x20 * x56 +
           (1.0 / 576.0) * coeffs[6] * x4 * x50 -
           1.0 / 576.0 * coeffs[7] * x4 * x49 +
           (1.0 / 576.0) * coeffs[8] * x17 * x27 * x51 -
           1.0 / 576.0 * coeffs[9] * x15 * x36 * x51;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 5 * x[0];
    const Scalar x1 = x0 - 4;
    const Scalar x2 = x0 - 3;
    const Scalar x3 = x[0] - 1;
    const Scalar x4 = x0 * x3;
    const Scalar x5 = x2 * x4;
    const Scalar x6 = x1 * x5;
    const Scalar x7 = x0 - 2;
    const Scalar x8 = x4 * x7;
    const Scalar x9 = x1 * x8;
    const Scalar x10 = x2 * x7;
    const Scalar x11 = x10 * x4;
    const Scalar x12 = x1 * x10;
    const Scalar x13 = x12 * x[0];
    const Scalar x14 = x12 * x3;
    const Scalar x15 = x11 + x13 + x14 + x6 + x9;
    const Scalar x16 = 5 * x[1];
    const Scalar x17 = x16 - 4;
    const Scalar x18 = x16 - 3;
    const Scalar x19 = x16 - 2;
    const Scalar x20 = x18 * x19;
    const Scalar x21 = x17 * x20;
    const Scalar x22 = x21 * x[1];
    const Scalar x23 = x[1] - 1;
    const Scalar x24 = 625 * x23;
    const Scalar x25 = x22 * x24;
    const Scalar x26 = x0 - 1;
    const Scalar x27 = x26 * x5;
    const Scalar x28 = x26 * x8;
    const Scalar x29 = x10 * x26;
    const Scalar x30 = x29 * x[0];
    const Scalar x31 = x29 * x3;
    const Scalar x32 = x11 + x27 + x28 + x30 + x31;
    const Scalar x33 = x16 - 1;
    const Scalar x34 = x20 * x33;
    const Scalar x35 = x34 * x[1];
    const Scalar x36 = x24 * x35;
    const Scalar x37 = x1 * x26;
    const Scalar x38 = x37 * x4;
    const Scalar x39 = x2 * x37;
    const Scalar x40 = x39 * x[0];
    const Scalar x41 = x3 * x39;
    const Scalar x42 = x27 + x38 + x40 + x41 + x6;
    const Scalar x43 = 1250 * x23;
    const Scalar x44 = x22 * x43;
    const Scalar x45 = x37 * x7;
    const Scalar x46 = x45 * x[0];
    const Scalar x47 = x3 * x45;
    const Scalar x48 = x28 + x38 + x46 + x47 + x9;
    const Scalar x49 = x17 * x33;
    const Scalar x50 = x18 * x49;
    const Scalar x51 = x50 * x[1];
    const Scalar x52 = x32 * x43;
    const Scalar x53 = x19 * x49;
    const Scalar x54 = x53 * x[1];
    const Scalar x55 = x35 * x43;
    const Scalar x56 = x15 * x43;
    const Scalar x57 = 2500 * x23;
    const Scalar x58 = x51 * x57;
    const Scalar x59 = x54 * x57;
    const Scalar x60 = x20 * x49;
    const Scalar x61 = x60 * x[1];
    const Scalar x62 = 25 * x61;
    const Scalar x63 = 50 * x61;
    const Scalar x64 = x23 * x60;
    const Scalar x65 = 25 * x64;
    const Scalar x66 = 50 * x64;
    const Scalar x67 = x10 * x37;
    const Scalar x68 = x0 * x12 + x0 * x29 + x0 * x39 + x0 * x45 + x67;
    const Scalar x69 = 50 * x23;
    const Scalar x70 = x68 * x69;
    const Scalar x71 = 25 * x23;
    const Scalar x72 = x68 * x71;
    const Scalar x73 = 5 * x14 + 5 * x31 + 5 * x41 + 5 * x47 + x67;
    const Scalar x74 = x71 * x73;
    const Scalar x75 = x69 * x73;
    const Scalar x76 = x16 * x23;
    const Scalar x77 = x18 * x76;
    const Scalar x78 = x17 * x77;
    const Scalar x79 = x19 * x76;
    const Scalar x80 = x17 * x79;
    const Scalar x81 = x20 * x76;
    const Scalar x82 = x21 * x23;
    const Scalar x83 = x22 + x78 + x80 + x81 + x82;
    const Scalar x84 = 625 * x3;
    const Scalar x85 = x83 * x84;
    const Scalar x86 = x33 * x77;
    const Scalar x87 = x33 * x79;
    const Scalar x88 = x23 * x34;
    const Scalar x89 = x35 + x81 + x86 + x87 + x88;
    const Scalar x90 = x84 * x89;
    const Scalar x91 = 1250 * x3;
    const Scalar x92 = x83 * x91;
    const Scalar x93 = x49 * x76;
    const Scalar x94 = x23 * x50;
    const Scalar x95 = x51 + x78 + x86 + x93 + x94;
    const Scalar x96 = x30 * x91;
    const Scalar x97 = x23 * x53;
    const Scalar x98 = x54 + x80 + x87 + x93 + x97;
    const Scalar x99 = x89 * x91;
    const Scalar x100 = x13 * x91;
    const Scalar x101 = 2500 * x3;
    const Scalar x102 = x101 * x95;
    const Scalar x103 = x101 * x98;
    const Scalar x104 = 50 * x67;
    const Scalar x105 = x104 * x[0];
    const Scalar x106 = 25 * x67 * x[0];
    const Scalar x107 = 25 * x3;
    const Scalar x108 = x107 * x67;
    const Scalar x109 = x104 * x3;
    const Scalar x110 = x16 * x21 + x16 * x34 + x16 * x50 + x16 * x53 + x60;
    const Scalar x111 = x107 * x110;
    const Scalar x112 = 50 * x3;
    const Scalar x113 = x110 * x112;
    const Scalar x114 = x110 * x67;
    const Scalar x115 = x60 + 5 * x82 + 5 * x88 + 5 * x94 + 5 * x97;
    const Scalar x116 = x107 * x115;
    const Scalar x117 = x112 * x115;
    const Scalar x118 = x115 * x67;
    out[0] = (1.0 / 576.0) * coeffs[0] * x64 * x73 -
             1.0 / 576.0 * coeffs[10] * x51 * x70 +
             (1.0 / 576.0) * coeffs[11] * x54 * x70 -
             1.0 / 576.0 * coeffs[12] * x32 * x62 +
             (1.0 / 576.0) * coeffs[13] * x15 * x62 +
             (1.0 / 576.0) * coeffs[14] * x48 * x63 -
             1.0 / 576.0 * coeffs[15] * x42 * x63 +
             (1.0 / 576.0) * coeffs[16] * x35 * x74 -
             1.0 / 576.0 * coeffs[17] * x22 * x74 -
             1.0 / 576.0 * coeffs[18] * x54 * x75 +
             (1.0 / 576.0) * coeffs[19] * x51 * x75 -
             1.0 / 576.0 * coeffs[1] * x64 * x68 +
             (1.0 / 576.0) * coeffs[20] * x15 * x25 -
             1.0 / 576.0 * coeffs[21] * x25 * x32 +
             (1.0 / 576.0) * coeffs[22] * x32 * x36 -
             1.0 / 576.0 * coeffs[23] * x15 * x36 -
             1.0 / 576.0 * coeffs[24] * x42 * x44 +
             (1.0 / 576.0) * coeffs[25] * x44 * x48 +
             (1.0 / 576.0) * coeffs[26] * x51 * x52 -
             1.0 / 576.0 * coeffs[27] * x52 * x54 -
             1.0 / 576.0 * coeffs[28] * x48 * x55 +
             (1.0 / 576.0) * coeffs[29] * x42 * x55 +
             (1.0 / 576.0) * coeffs[2] * x61 * x68 +
             (1.0 / 576.0) * coeffs[30] * x54 * x56 -
             1.0 / 576.0 * coeffs[31] * x51 * x56 +
             (1.0 / 576.0) * coeffs[32] * x42 * x58 -
             1.0 / 576.0 * coeffs[33] * x48 * x58 +
             (1.0 / 576.0) * coeffs[34] * x48 * x59 -
             1.0 / 576.0 * coeffs[35] * x42 * x59 -
             1.0 / 576.0 * coeffs[3] * x61 * x73 -
             1.0 / 576.0 * coeffs[4] * x15 * x65 +
             (1.0 / 576.0) * coeffs[5] * x32 * x65 +
             (1.0 / 576.0) * coeffs[6] * x42 * x66 -
             1.0 / 576.0 * coeffs[7] * x48 * x66 +
             (1.0 / 576.0) * coeffs[8] * x22 * x72 -
             1.0 / 576.0 * coeffs[9] * x35 * x72;
    out[1] = (1.0 / 576.0) * coeffs[0] * x118 * x3 -
             1.0 / 576.0 * coeffs[10] * x105 * x95 +
             (1.0 / 576.0) * coeffs[11] * x105 * x98 -
             1.0 / 576.0 * coeffs[12] * x111 * x30 +
             (1.0 / 576.0) * coeffs[13] * x111 * x13 +
             (1.0 / 576.0) * coeffs[14] * x113 * x46 -
             1.0 / 576.0 * coeffs[15] * x113 * x40 +
             (1.0 / 576.0) * coeffs[16] * x108 * x89 -
             1.0 / 576.0 * coeffs[17] * x108 * x83 -
             1.0 / 576.0 * coeffs[18] * x109 * x98 +
             (1.0 / 576.0) * coeffs[19] * x109 * x95 -
             1.0 / 576.0 * coeffs[1] * x118 * x[0] +
             (1.0 / 576.0) * coeffs[20] * x13 * x85 -
             1.0 / 576.0 * coeffs[21] * x30 * x85 +
             (1.0 / 576.0) * coeffs[22] * x30 * x90 -
             1.0 / 576.0 * coeffs[23] * x13 * x90 -
             1.0 / 576.0 * coeffs[24] * x40 * x92 +
             (1.0 / 576.0) * coeffs[25] * x46 * x92 +
             (1.0 / 576.0) * coeffs[26] * x95 * x96 -
             1.0 / 576.0 * coeffs[27] * x96 * x98 -
             1.0 / 576.0 * coeffs[28] * x46 * x99 +
             (1.0 / 576.0) * coeffs[29] * x40 * x99 +
             (1.0 / 576.0) * coeffs[2] * x114 * x[0] +
             (1.0 / 576.0) * coeffs[30] * x100 * x98 -
             1.0 / 576.0 * coeffs[31] * x100 * x95 +
             (1.0 / 576.0) * coeffs[32] * x102 * x40 -
             1.0 / 576.0 * coeffs[33] * x102 * x46 +
             (1.0 / 576.0) * coeffs[34] * x103 * x46 -
             1.0 / 576.0 * coeffs[35] * x103 * x40 -
             1.0 / 576.0 * coeffs[3] * x114 * x3 -
             1.0 / 576.0 * coeffs[4] * x116 * x13 +
             (1.0 / 576.0) * coeffs[5] * x116 * x30 +
             (1.0 / 576.0) * coeffs[6] * x117 * x40 -
             1.0 / 576.0 * coeffs[7] * x117 * x46 +
             (1.0 / 576.0) * coeffs[8] * x106 * x83 -
             1.0 / 576.0 * coeffs[9] * x106 * x89;
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
      out[0] = 5;
      out[1] = 5;
      break;
    case 3:
      out[0] = 0;
      out[1] = 5;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      break;
    case 5:
      out[0] = 4;
      out[1] = 0;
      break;
    case 6:
      out[0] = 2;
      out[1] = 0;
      break;
    case 7:
      out[0] = 3;
      out[1] = 0;
      break;
    case 8:
      out[0] = 5;
      out[1] = 1;
      break;
    case 9:
      out[0] = 5;
      out[1] = 4;
      break;
    case 10:
      out[0] = 5;
      out[1] = 2;
      break;
    case 11:
      out[0] = 5;
      out[1] = 3;
      break;
    case 12:
      out[0] = 4;
      out[1] = 5;
      break;
    case 13:
      out[0] = 1;
      out[1] = 5;
      break;
    case 14:
      out[0] = 3;
      out[1] = 5;
      break;
    case 15:
      out[0] = 2;
      out[1] = 5;
      break;
    case 16:
      out[0] = 0;
      out[1] = 4;
      break;
    case 17:
      out[0] = 0;
      out[1] = 1;
      break;
    case 18:
      out[0] = 0;
      out[1] = 3;
      break;
    case 19:
      out[0] = 0;
      out[1] = 2;
      break;
    case 20:
      out[0] = 1;
      out[1] = 1;
      break;
    case 21:
      out[0] = 4;
      out[1] = 1;
      break;
    case 22:
      out[0] = 4;
      out[1] = 4;
      break;
    case 23:
      out[0] = 1;
      out[1] = 4;
      break;
    case 24:
      out[0] = 2;
      out[1] = 1;
      break;
    case 25:
      out[0] = 3;
      out[1] = 1;
      break;
    case 26:
      out[0] = 4;
      out[1] = 2;
      break;
    case 27:
      out[0] = 4;
      out[1] = 3;
      break;
    case 28:
      out[0] = 3;
      out[1] = 4;
      break;
    case 29:
      out[0] = 2;
      out[1] = 4;
      break;
    case 30:
      out[0] = 1;
      out[1] = 3;
      break;
    case 31:
      out[0] = 1;
      out[1] = 2;
      break;
    case 32:
      out[0] = 2;
      out[1] = 2;
      break;
    case 33:
      out[0] = 3;
      out[1] = 2;
      break;
    case 34:
      out[0] = 3;
      out[1] = 3;
      break;
    case 35:
      out[0] = 2;
      out[1] = 3;
      break;
    }
  }
}

} // namespace numeric::math

#endif
