#ifndef NUMERIC_MATH_BASIS_LAGRANGE_CUBE_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_CUBE_HPP_

#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::math {

template <> struct BasisLagrange<mesh::RefElCube, 1> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 8;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_ti, const Scalar *x) {
    switch (i) {
    case 0:
      return -(x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 1:
      return x[0] * (x[1] - 1) * (x[2] - 1);
    case 2:
      return -x[0] * x[1] * (x[2] - 1);
    case 3:
      return x[1] * (x[0] - 1) * (x[2] - 1);
    case 4:
      return x[2] * (x[0] - 1) * (x[1] - 1);
    case 5:
      return -x[0] * x[2] * (x[1] - 1);
    case 6:
      return x[0] * x[1] * x[2];
    case 7:
      return -x[1] * x[2] * (x[0] - 1);
    default:
      return 0;
    }
  }

  template <typename Scalar>
  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    switch (i) {
    case 0:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = x[2] - 1;
      const Scalar x2 = x[0] - 1;
      out[0] = -x0 * x1;
      out[1] = -x1 * x2;
      out[2] = -x0 * x2;
      break;
    case 1:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = x[2] - 1;
      out[0] = x0 * x1;
      out[1] = x1 * x[0];
      out[2] = x0 * x[0];
      break;
    case 2:
      const Scalar x0 = x[2] - 1;
      out[0] = -x0 * x[1];
      out[1] = -x0 * x[0];
      out[2] = -x[0] * x[1];
      break;
    case 3:
      const Scalar x0 = x[2] - 1;
      const Scalar x1 = x[0] - 1;
      out[0] = x0 * x[1];
      out[1] = x0 * x1;
      out[2] = x1 * x[1];
      break;
    case 4:
      const Scalar x0 = x[1] - 1;
      const Scalar x1 = x[0] - 1;
      out[0] = x0 * x[2];
      out[1] = x1 * x[2];
      out[2] = x0 * x1;
      break;
    case 5:
      const Scalar x0 = x[1] - 1;
      out[0] = -x0 * x[2];
      out[1] = -x[0] * x[2];
      out[2] = -x0 * x[0];
      break;
    case 6:
      out[0] = x[1] * x[2];
      out[1] = x[0] * x[2];
      out[2] = x[0] * x[1];
      break;
    case 7:
      const Scalar x0 = x[0] - 1;
      out[0] = -x[1] * x[2];
      out[1] = -x0 * x[2];
      out[2] = -x0 * x[1];
      break;
    default:
      break;
    }
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[2] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x[0] - 1;
    return -coeffs[0] * x0 * x1 * x2 + coeffs[1] * x0 * x1 * x[0] -
           coeffs[2] * x0 * x[0] * x[1] + coeffs[3] * x0 * x2 * x[1] +
           coeffs[4] * x1 * x2 * x[2] - coeffs[5] * x1 * x[0] * x[2] +
           coeffs[6] * x[0] * x[1] * x[2] - coeffs[7] * x2 * x[1] * x[2];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = x[2] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x[0] - 1;
    out[0] = -coeffs[0] * x0 * x1 + coeffs[1] * x0 * x1 -
             coeffs[2] * x0 * x[1] + coeffs[3] * x0 * x[1] +
             coeffs[4] * x1 * x[2] - coeffs[5] * x1 * x[2] +
             coeffs[6] * x[1] * x[2] - coeffs[7] * x[1] * x[2];
    out[1] = -coeffs[0] * x0 * x2 + coeffs[1] * x0 * x[0] -
             coeffs[2] * x0 * x[0] + coeffs[3] * x0 * x2 +
             coeffs[4] * x2 * x[2] - coeffs[5] * x[0] * x[2] +
             coeffs[6] * x[0] * x[2] - coeffs[7] * x2 * x[2];
    out[2] = -coeffs[0] * x1 * x2 + coeffs[1] * x1 * x[0] -
             coeffs[2] * x[0] * x[1] + coeffs[3] * x2 * x[1] +
             coeffs[4] * x1 * x2 - coeffs[5] * x1 * x[0] +
             coeffs[6] * x[0] * x[1] - coeffs[7] * x2 * x[1];
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
      break;
    case 1:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 2:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 4:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 5:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 6:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    case 7:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    }
  }
}

template <>
struct BasisLagrange<mesh::RefElCube, 2> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 27;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_ti, const Scalar *x) {
    switch (i) {
    case 0:
      return (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1);
    case 1:
      return x[0] * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1);
    case 2:
      return x[0] * x[1] * (2 * x[0] - 1) * (2 * x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1);
    case 3:
      return x[1] * (x[0] - 1) * (2 * x[0] - 1) * (2 * x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1);
    case 4:
      return x[2] * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (2 * x[2] - 1);
    case 5:
      return x[0] * x[2] * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (2 * x[2] - 1);
    case 6:
      return x[0] * x[1] * x[2] * (2 * x[0] - 1) * (2 * x[1] - 1) *
             (2 * x[2] - 1);
    case 7:
      return x[1] * x[2] * (x[0] - 1) * (2 * x[0] - 1) * (2 * x[1] - 1) *
             (2 * x[2] - 1);
    case 8:
      return -4 * x[0] * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1);
    case 9:
      return -4 * x[0] * x[1] * (2 * x[0] - 1) * (x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1);
    case 10:
      return -4 * x[0] * x[1] * (x[0] - 1) * (2 * x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1);
    case 11:
      return -4 * x[1] * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1);
    case 12:
      return -4 * x[0] * x[2] * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (2 * x[2] - 1);
    case 13:
      return -4 * x[0] * x[1] * x[2] * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[2] - 1);
    case 14:
      return -4 * x[0] * x[1] * x[2] * (x[0] - 1) * (2 * x[1] - 1) *
             (2 * x[2] - 1);
    case 15:
      return -4 * x[1] * x[2] * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[2] - 1);
    case 16:
      return -4 * x[2] * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (x[2] - 1);
    case 17:
      return -4 * x[0] * x[2] * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (x[2] - 1);
    case 18:
      return -4 * x[0] * x[1] * x[2] * (2 * x[0] - 1) * (2 * x[1] - 1) *
             (x[2] - 1);
    case 19:
      return -4 * x[1] * x[2] * (x[0] - 1) * (2 * x[0] - 1) * (2 * x[1] - 1) *
             (x[2] - 1);
    case 20:
      return 16 * x[0] * x[1] * (x[0] - 1) * (x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1);
    case 21:
      return 16 * x[0] * x[1] * x[2] * (x[0] - 1) * (x[1] - 1) * (2 * x[2] - 1);
    case 22:
      return 16 * x[1] * x[2] * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (x[2] - 1);
    case 23:
      return 16 * x[0] * x[1] * x[2] * (2 * x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 24:
      return 16 * x[0] * x[2] * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (x[2] - 1);
    case 25:
      return 16 * x[0] * x[1] * x[2] * (x[0] - 1) * (2 * x[1] - 1) * (x[2] - 1);
    case 26:
      return -64 * x[0] * x[1] * x[2] * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    default:
      return 0;
    }
  }

  template <typename Scalar>
  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    switch (i) {
    case 0:
      const Scalar x0 = (x[2] - 1) * (2 * x[2] - 1);
      const Scalar x1 = (x[1] - 1) * (2 * x[1] - 1);
      const Scalar x2 = (x[0] - 1) * (2 * x[0] - 1);
      out[0] = x0 * x1 * (4 * x[0] - 3);
      out[1] = x0 * x2 * (4 * x[1] - 3);
      out[2] = x1 * x2 * (4 * x[2] - 3);
      break;
    case 1:
      const Scalar x0 = (x[2] - 1) * (2 * x[2] - 1);
      const Scalar x1 = (x[1] - 1) * (2 * x[1] - 1);
      const Scalar x2 = x[0] * (2 * x[0] - 1);
      out[0] = x0 * x1 * (4 * x[0] - 1);
      out[1] = x0 * x2 * (4 * x[1] - 3);
      out[2] = x1 * x2 * (4 * x[2] - 3);
      break;
    case 2:
      const Scalar x0 = (x[2] - 1) * (2 * x[2] - 1);
      const Scalar x1 = x[1] * (2 * x[1] - 1);
      const Scalar x2 = x[0] * (2 * x[0] - 1);
      out[0] = x0 * x1 * (4 * x[0] - 1);
      out[1] = x0 * x2 * (4 * x[1] - 1);
      out[2] = x1 * x2 * (4 * x[2] - 3);
      break;
    case 3:
      const Scalar x0 = (x[2] - 1) * (2 * x[2] - 1);
      const Scalar x1 = x[1] * (2 * x[1] - 1);
      const Scalar x2 = (x[0] - 1) * (2 * x[0] - 1);
      out[0] = x0 * x1 * (4 * x[0] - 3);
      out[1] = x0 * x2 * (4 * x[1] - 1);
      out[2] = x1 * x2 * (4 * x[2] - 3);
      break;
    case 4:
      const Scalar x0 = x[2] * (2 * x[2] - 1);
      const Scalar x1 = (x[1] - 1) * (2 * x[1] - 1);
      const Scalar x2 = (x[0] - 1) * (2 * x[0] - 1);
      out[0] = x0 * x1 * (4 * x[0] - 3);
      out[1] = x0 * x2 * (4 * x[1] - 3);
      out[2] = x1 * x2 * (4 * x[2] - 1);
      break;
    case 5:
      const Scalar x0 = x[2] * (2 * x[2] - 1);
      const Scalar x1 = (x[1] - 1) * (2 * x[1] - 1);
      const Scalar x2 = x[0] * (2 * x[0] - 1);
      out[0] = x0 * x1 * (4 * x[0] - 1);
      out[1] = x0 * x2 * (4 * x[1] - 3);
      out[2] = x1 * x2 * (4 * x[2] - 1);
      break;
    case 6:
      const Scalar x0 = x[2] * (2 * x[2] - 1);
      const Scalar x1 = x[1] * (2 * x[1] - 1);
      const Scalar x2 = x[0] * (2 * x[0] - 1);
      out[0] = x0 * x1 * (4 * x[0] - 1);
      out[1] = x0 * x2 * (4 * x[1] - 1);
      out[2] = x1 * x2 * (4 * x[2] - 1);
      break;
    case 7:
      const Scalar x0 = x[2] * (2 * x[2] - 1);
      const Scalar x1 = x[1] * (2 * x[1] - 1);
      const Scalar x2 = (x[0] - 1) * (2 * x[0] - 1);
      out[0] = x0 * x1 * (4 * x[0] - 3);
      out[1] = x0 * x2 * (4 * x[1] - 1);
      out[2] = x1 * x2 * (4 * x[2] - 1);
      break;
    case 8:
      const Scalar x0 = (x[2] - 1) * (2 * x[2] - 1);
      const Scalar x1 = (x[1] - 1) * (2 * x[1] - 1);
      const Scalar x2 = 16 * x[0] * (x[0] - 1);
      out[0] = -4 * x0 * x1 * (2 * x[0] - 1);
      out[1] = -x0 * x2 * (x[1] - 3.0 / 4.0);
      out[2] = -x1 * x2 * (x[2] - 3.0 / 4.0);
      break;
    case 9:
      const Scalar x0 = (x[2] - 1) * (2 * x[2] - 1);
      const Scalar x1 = 16 * x[1] * (x[1] - 1);
      const Scalar x2 = x[0] * (2 * x[0] - 1);
      out[0] = -x0 * x1 * (x[0] - 1.0 / 4.0);
      out[1] = -4 * x0 * x2 * (2 * x[1] - 1);
      out[2] = -x1 * x2 * (x[2] - 3.0 / 4.0);
      break;
    case 10:
      const Scalar x0 = (x[2] - 1) * (2 * x[2] - 1);
      const Scalar x1 = x[1] * (2 * x[1] - 1);
      const Scalar x2 = 16 * x[0] * (x[0] - 1);
      out[0] = -4 * x0 * x1 * (2 * x[0] - 1);
      out[1] = -x0 * x2 * (x[1] - 1.0 / 4.0);
      out[2] = -x1 * x2 * (x[2] - 3.0 / 4.0);
      break;
    case 11:
      const Scalar x0 = (x[2] - 1) * (2 * x[2] - 1);
      const Scalar x1 = 16 * x[1] * (x[1] - 1);
      const Scalar x2 = (x[0] - 1) * (2 * x[0] - 1);
      out[0] = -x0 * x1 * (x[0] - 3.0 / 4.0);
      out[1] = -4 * x0 * x2 * (2 * x[1] - 1);
      out[2] = -x1 * x2 * (x[2] - 3.0 / 4.0);
      break;
    case 12:
      const Scalar x0 = x[2] * (2 * x[2] - 1);
      const Scalar x1 = (x[1] - 1) * (2 * x[1] - 1);
      const Scalar x2 = 16 * x[0] * (x[0] - 1);
      out[0] = -4 * x0 * x1 * (2 * x[0] - 1);
      out[1] = -x0 * x2 * (x[1] - 3.0 / 4.0);
      out[2] = -x1 * x2 * (x[2] - 1.0 / 4.0);
      break;
    case 13:
      const Scalar x0 = x[2] * (2 * x[2] - 1);
      const Scalar x1 = 16 * x[1] * (x[1] - 1);
      const Scalar x2 = x[0] * (2 * x[0] - 1);
      out[0] = -x0 * x1 * (x[0] - 1.0 / 4.0);
      out[1] = -4 * x0 * x2 * (2 * x[1] - 1);
      out[2] = -x1 * x2 * (x[2] - 1.0 / 4.0);
      break;
    case 14:
      const Scalar x0 = x[2] * (2 * x[2] - 1);
      const Scalar x1 = x[1] * (2 * x[1] - 1);
      const Scalar x2 = 16 * x[0] * (x[0] - 1);
      out[0] = -4 * x0 * x1 * (2 * x[0] - 1);
      out[1] = -x0 * x2 * (x[1] - 1.0 / 4.0);
      out[2] = -x1 * x2 * (x[2] - 1.0 / 4.0);
      break;
    case 15:
      const Scalar x0 = x[2] * (2 * x[2] - 1);
      const Scalar x1 = 16 * x[1] * (x[1] - 1);
      const Scalar x2 = (x[0] - 1) * (2 * x[0] - 1);
      out[0] = -x0 * x1 * (x[0] - 3.0 / 4.0);
      out[1] = -4 * x0 * x2 * (2 * x[1] - 1);
      out[2] = -x1 * x2 * (x[2] - 1.0 / 4.0);
      break;
    case 16:
      const Scalar x0 = (x[1] - 1) * (2 * x[1] - 1);
      const Scalar x1 = 16 * x[2] * (x[2] - 1);
      const Scalar x2 = (x[0] - 1) * (2 * x[0] - 1);
      out[0] = -x0 * x1 * (x[0] - 3.0 / 4.0);
      out[1] = -x1 * x2 * (x[1] - 3.0 / 4.0);
      out[2] = -4 * x0 * x2 * (2 * x[2] - 1);
      break;
    case 17:
      const Scalar x0 = (x[1] - 1) * (2 * x[1] - 1);
      const Scalar x1 = 16 * x[2] * (x[2] - 1);
      const Scalar x2 = x[0] * (2 * x[0] - 1);
      out[0] = -x0 * x1 * (x[0] - 1.0 / 4.0);
      out[1] = -x1 * x2 * (x[1] - 3.0 / 4.0);
      out[2] = -4 * x0 * x2 * (2 * x[2] - 1);
      break;
    case 18:
      const Scalar x0 = x[1] * (2 * x[1] - 1);
      const Scalar x1 = 16 * x[2] * (x[2] - 1);
      const Scalar x2 = x[0] * (2 * x[0] - 1);
      out[0] = -x0 * x1 * (x[0] - 1.0 / 4.0);
      out[1] = -x1 * x2 * (x[1] - 1.0 / 4.0);
      out[2] = -4 * x0 * x2 * (2 * x[2] - 1);
      break;
    case 19:
      const Scalar x0 = x[1] * (2 * x[1] - 1);
      const Scalar x1 = 16 * x[2] * (x[2] - 1);
      const Scalar x2 = (x[0] - 1) * (2 * x[0] - 1);
      out[0] = -x0 * x1 * (x[0] - 3.0 / 4.0);
      out[1] = -x1 * x2 * (x[1] - 1.0 / 4.0);
      out[2] = -4 * x0 * x2 * (2 * x[2] - 1);
      break;
    case 20:
      const Scalar x0 = 2 * x[2];
      const Scalar x1 = 32 * (x0 - 1) * (x[2] - 1);
      const Scalar x2 = x[1] * (x[1] - 1);
      const Scalar x3 = x[0] * (x[0] - 1);
      out[0] = x1 * x2 * (x[0] - 1.0 / 2.0);
      out[1] = x1 * x3 * (x[1] - 1.0 / 2.0);
      out[2] = 32 * x2 * x3 * (x0 - 3.0 / 2.0);
      break;
    case 21:
      const Scalar x0 = 2 * x[2];
      const Scalar x1 = 32 * x[2] * (x0 - 1);
      const Scalar x2 = x[1] * (x[1] - 1);
      const Scalar x3 = x[0] * (x[0] - 1);
      out[0] = x1 * x2 * (x[0] - 1.0 / 2.0);
      out[1] = x1 * x3 * (x[1] - 1.0 / 2.0);
      out[2] = 32 * x2 * x3 * (x0 - 1.0 / 2.0);
      break;
    case 22:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = 32 * x[2] * (x[2] - 1);
      const Scalar x2 = x[1] * (x[1] - 1);
      const Scalar x3 = (x0 - 1) * (x[0] - 1);
      out[0] = x1 * x2 * (x0 - 3.0 / 2.0);
      out[1] = x1 * x3 * (x[1] - 1.0 / 2.0);
      out[2] = 32 * x2 * x3 * (x[2] - 1.0 / 2.0);
      break;
    case 23:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = 32 * x[2] * (x[2] - 1);
      const Scalar x2 = x[1] * (x[1] - 1);
      const Scalar x3 = x[0] * (x0 - 1);
      out[0] = x1 * x2 * (x0 - 1.0 / 2.0);
      out[1] = x1 * x3 * (x[1] - 1.0 / 2.0);
      out[2] = 32 * x2 * x3 * (x[2] - 1.0 / 2.0);
      break;
    case 24:
      const Scalar x0 = 32 * x[2] * (x[2] - 1);
      const Scalar x1 = 2 * x[1];
      const Scalar x2 = (x1 - 1) * (x[1] - 1);
      const Scalar x3 = x[0] * (x[0] - 1);
      out[0] = x0 * x2 * (x[0] - 1.0 / 2.0);
      out[1] = x0 * x3 * (x1 - 3.0 / 2.0);
      out[2] = 32 * x2 * x3 * (x[2] - 1.0 / 2.0);
      break;
    case 25:
      const Scalar x0 = 32 * x[2] * (x[2] - 1);
      const Scalar x1 = 2 * x[1];
      const Scalar x2 = x[1] * (x1 - 1);
      const Scalar x3 = x[0] * (x[0] - 1);
      out[0] = x0 * x2 * (x[0] - 1.0 / 2.0);
      out[1] = x0 * x3 * (x1 - 1.0 / 2.0);
      out[2] = 32 * x2 * x3 * (x[2] - 1.0 / 2.0);
      break;
    case 26:
      const Scalar x0 = 64 * x[2] * (x[2] - 1);
      const Scalar x1 = x[1] * (x[1] - 1);
      const Scalar x2 = x[0] * (x[0] - 1);
      out[0] = -x0 * x1 * (2 * x[0] - 1);
      out[1] = -x0 * x2 * (2 * x[1] - 1);
      out[2] = -64 * x1 * x2 * (2 * x[2] - 1);
      break;
    default:
      break;
    }
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x[2] - 1;
    const Scalar x3 = x0 * x1 * x2;
    const Scalar x4 = x[0] * x[1];
    const Scalar x5 = x4 * x[2];
    const Scalar x6 = 2 * x[2];
    const Scalar x7 = x6 - 1;
    const Scalar x8 = x1 * x7;
    const Scalar x9 = x4 * x6;
    const Scalar x10 = x0 * x9;
    const Scalar x11 = 2 * x[0];
    const Scalar x12 = x11 - 1;
    const Scalar x13 = x12 * x2;
    const Scalar x14 = x1 * x13;
    const Scalar x15 = 2 * x[1] - 1;
    const Scalar x16 = x15 * x2;
    const Scalar x17 = (1.0 / 2.0) * x12;
    const Scalar x18 = (1.0 / 2.0) * x0;
    const Scalar x19 = x15 * x5;
    const Scalar x20 = x19 * x7;
    const Scalar x21 = (1.0 / 2.0) * x13;
    const Scalar x22 = x3 * x[1];
    const Scalar x23 = x22 * x7;
    const Scalar x24 = x15 * x[0];
    const Scalar x25 = x24 * x3;
    const Scalar x26 = x4 * x7;
    const Scalar x27 = x24 * x[2];
    const Scalar x28 = x18 * x8;
    const Scalar x29 = x[1] * x[2];
    const Scalar x30 = x12 * x29;
    const Scalar x31 = x13 * x15;
    const Scalar x32 = x15 * x3;
    const Scalar x33 = (1.0 / 8.0) * x8;
    const Scalar x34 = x12 * x33;
    const Scalar x35 = (1.0 / 8.0) * x7;
    const Scalar x36 = x0 * x35;
    return 8 * coeffs[0] * x12 * x32 * x35 - 8 * coeffs[10] * x16 * x18 * x26 -
           8 * coeffs[11] * x17 * x23 - 8 * coeffs[12] * x27 * x28 -
           8 * coeffs[13] * x17 * x5 * x8 - 8 * coeffs[14] * x18 * x20 -
           8 * coeffs[15] * x28 * x30 - 8 * coeffs[16] * x17 * x32 * x[2] -
           4 * coeffs[17] * x14 * x27 - 8 * coeffs[18] * x19 * x21 -
           8 * coeffs[19] * x18 * x29 * x31 + 8 * coeffs[1] * x13 * x24 * x33 +
           8 * coeffs[20] * x11 * x23 + 8 * coeffs[21] * x10 * x8 +
           8 * coeffs[22] * x12 * x22 * x6 + 8 * coeffs[23] * x14 * x9 +
           8 * coeffs[24] * x25 * x6 + 8 * coeffs[25] * x10 * x16 -
           64 * coeffs[26] * x3 * x5 + coeffs[2] * x26 * x31 +
           8 * coeffs[3] * x31 * x36 * x[1] +
           8 * coeffs[4] * x0 * x15 * x34 * x[2] + 8 * coeffs[5] * x27 * x34 +
           coeffs[6] * x12 * x20 + 8 * coeffs[7] * x15 * x30 * x36 -
           4 * coeffs[8] * x25 * x7 - 8 * coeffs[9] * x21 * x4 * x8;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x0 - 3;
    const Scalar x2 = x[1] - 1;
    const Scalar x3 = x[2] - 1;
    const Scalar x4 = x2 * x3;
    const Scalar x5 = 2 * x[2];
    const Scalar x6 = x5 * x[1];
    const Scalar x7 = x4 * x6;
    const Scalar x8 = x0 - 1;
    const Scalar x9 = x[1] * x[2];
    const Scalar x10 = 2 * x[0];
    const Scalar x11 = x10 - 1;
    const Scalar x12 = x11 * x4;
    const Scalar x13 = 8 * coeffs[26];
    const Scalar x14 = x5 - 1;
    const Scalar x15 = x14 * x2;
    const Scalar x16 = x15 * x9;
    const Scalar x17 = (1.0 / 2.0) * x8;
    const Scalar x18 = (1.0 / 2.0) * x1;
    const Scalar x19 = 2 * x[1];
    const Scalar x20 = x19 - 1;
    const Scalar x21 = x20 * x3;
    const Scalar x22 = x21 * x9;
    const Scalar x23 = x11 * x6;
    const Scalar x24 = x14 * x4;
    const Scalar x25 = coeffs[11] * x[1];
    const Scalar x26 = (1.0 / 2.0) * x11;
    const Scalar x27 = x20 * x26;
    const Scalar x28 = x14 * x9;
    const Scalar x29 = x20 * x[2];
    const Scalar x30 = x29 * x4;
    const Scalar x31 = x12 * x14;
    const Scalar x32 = x20 * x5;
    const Scalar x33 = (1.0 / 8.0) * x8;
    const Scalar x34 = x20 * x28;
    const Scalar x35 = (1.0 / 8.0) * x1;
    const Scalar x36 = coeffs[9] * x[1];
    const Scalar x37 = x14 * x21;
    const Scalar x38 = x37 * x[1];
    const Scalar x39 = coeffs[12] * x[2];
    const Scalar x40 = x15 * x27;
    const Scalar x41 = x15 * x29;
    const Scalar x42 = x20 * x24;
    const Scalar x43 = (1.0 / 2.0) * coeffs[8];
    const Scalar x44 = x20 * x43;
    const Scalar x45 = x[0] - 1;
    const Scalar x46 = x10 * x45;
    const Scalar x47 = coeffs[24] * x46;
    const Scalar x48 = 4 * x[1];
    const Scalar x49 = x48 - 3;
    const Scalar x50 = x49 * x[2];
    const Scalar x51 = x3 * x50;
    const Scalar x52 = coeffs[25] * x46;
    const Scalar x53 = x48 - 1;
    const Scalar x54 = x3 * x53;
    const Scalar x55 = x54 * x[2];
    const Scalar x56 = x21 * x[2];
    const Scalar x57 = x45 * x[0];
    const Scalar x58 = x13 * x57;
    const Scalar x59 = x14 * x57;
    const Scalar x60 = (1.0 / 2.0) * x59;
    const Scalar x61 = x53 * x[2];
    const Scalar x62 = x26 * x[0];
    const Scalar x63 = coeffs[21] * x46;
    const Scalar x64 = coeffs[23] * x10 * x11;
    const Scalar x65 = x14 * x[0];
    const Scalar x66 = x27 * x[2];
    const Scalar x67 = x3 * x45;
    const Scalar x68 = x26 * x67;
    const Scalar x69 = coeffs[20] * x46;
    const Scalar x70 = x11 * x67;
    const Scalar x71 = (1.0 / 8.0) * x11;
    const Scalar x72 = x65 * x71;
    const Scalar x73 = x3 * x49;
    const Scalar x74 = x14 * x45;
    const Scalar x75 = x71 * x74;
    const Scalar x76 = x26 * x37;
    const Scalar x77 = (1.0 / 8.0) * coeffs[0];
    const Scalar x78 = x14 * x70;
    const Scalar x79 = 4 * x[2];
    const Scalar x80 = x79 - 3;
    const Scalar x81 = x2 * x[1];
    const Scalar x82 = x79 - 1;
    const Scalar x83 = x81 * x82;
    const Scalar x84 = x15 * x[1];
    const Scalar x85 = (1.0 / 2.0) * x57;
    const Scalar x86 = x20 * x[1];
    const Scalar x87 = x80 * x86;
    const Scalar x88 = x82 * x85;
    const Scalar x89 = x2 * x80;
    const Scalar x90 = x26 * x45;
    const Scalar x91 = x2 * x20;
    const Scalar x92 = x27 * x[1];
    const Scalar x93 = x11 * x45;
    const Scalar x94 = x71 * x[0];
    const Scalar x95 = x82 * x86;
    const Scalar x96 = x20 * x89;
    const Scalar x97 = x45 * x71;
    const Scalar x98 = x82 * x91;
    out[0] = 8 * coeffs[0] * x35 * x42 - 8 * coeffs[10] * x26 * x38 -
             8 * coeffs[13] * x16 * x17 - 8 * coeffs[14] * x27 * x28 -
             8 * coeffs[15] * x16 * x18 - 8 * coeffs[16] * x18 * x30 -
             8 * coeffs[17] * x17 * x30 - 8 * coeffs[18] * x17 * x22 -
             8 * coeffs[19] * x18 * x22 + 8 * coeffs[1] * x33 * x42 +
             8 * coeffs[20] * x19 * x31 + 8 * coeffs[21] * x15 * x23 +
             8 * coeffs[22] * x1 * x7 + 8 * coeffs[23] * x7 * x8 +
             8 * coeffs[24] * x12 * x32 + 8 * coeffs[25] * x21 * x23 +
             8 * coeffs[2] * x33 * x38 + 8 * coeffs[3] * x35 * x38 +
             8 * coeffs[4] * x35 * x41 + 8 * coeffs[5] * x33 * x41 +
             8 * coeffs[6] * x33 * x34 + 8 * coeffs[7] * x34 * x35 -
             8 * x12 * x13 * x9 - 8 * x17 * x24 * x36 - 8 * x18 * x24 * x25 -
             8 * x31 * x44 - 8 * x39 * x40;
    out[1] = -8 * coeffs[10] * x54 * x60 - 8 * coeffs[11] * x45 * x76 -
             8 * coeffs[13] * x65 * x66 - 8 * coeffs[14] * x60 * x61 -
             8 * coeffs[15] * x66 * x74 - 8 * coeffs[16] * x50 * x68 -
             8 * coeffs[17] * x51 * x62 - 8 * coeffs[18] * x55 * x62 -
             8 * coeffs[19] * x61 * x68 + 8 * coeffs[1] * x72 * x73 +
             8 * coeffs[22] * x32 * x70 + 8 * coeffs[2] * x54 * x72 +
             coeffs[3] * x53 * x78 + 8 * coeffs[4] * x50 * x75 +
             8 * coeffs[5] * x50 * x72 + 8 * coeffs[6] * x61 * x72 +
             8 * coeffs[7] * x61 * x75 - 8 * coeffs[9] * x76 * x[0] +
             8 * x14 * x29 * x63 + 8 * x37 * x69 - 8 * x39 * x49 * x60 -
             8 * x43 * x59 * x73 + 8 * x47 * x51 + 8 * x49 * x77 * x78 +
             8 * x52 * x55 - 8 * x56 * x58 + 8 * x56 * x64;
    out[2] = -8 * coeffs[10] * x85 * x87 - 8 * coeffs[12] * x88 * x91 -
             8 * coeffs[13] * x62 * x83 - 8 * coeffs[14] * x86 * x88 -
             8 * coeffs[15] * x83 * x90 - 8 * coeffs[16] * x40 * x45 -
             8 * coeffs[17] * x40 * x[0] - 8 * coeffs[18] * x65 * x92 -
             8 * coeffs[19] * x74 * x92 + 8 * coeffs[1] * x94 * x96 +
             8 * coeffs[22] * x15 * x19 * x93 + 8 * coeffs[2] * x87 * x94 +
             8 * coeffs[3] * x87 * x97 + 8 * coeffs[4] * x97 * x98 +
             8 * coeffs[5] * x94 * x98 + 8 * coeffs[6] * x94 * x95 +
             8 * coeffs[7] * x95 * x97 + 8 * x14 * x52 * x86 +
             8 * x15 * x20 * x47 - 8 * x25 * x89 * x90 - 8 * x36 * x62 * x89 -
             8 * x44 * x57 * x89 - 8 * x58 * x84 + 8 * x63 * x83 +
             8 * x64 * x84 + 8 * x69 * x80 * x81 + 8 * x77 * x93 * x96;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
      break;
    case 1:
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 2:
      out[0] = 2;
      out[1] = 2;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 4:
      out[0] = 0;
      out[1] = 0;
      out[2] = 2;
      break;
    case 5:
      out[0] = 2;
      out[1] = 0;
      out[2] = 2;
      break;
    case 6:
      out[0] = 2;
      out[1] = 2;
      out[2] = 2;
      break;
    case 7:
      out[0] = 0;
      out[1] = 2;
      out[2] = 2;
      break;
    case 8:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 9:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 10:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 11:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 12:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 13:
      out[0] = 2;
      out[1] = 1;
      out[2] = 2;
      break;
    case 14:
      out[0] = 1;
      out[1] = 2;
      out[2] = 2;
      break;
    case 15:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 16:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 17:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 18:
      out[0] = 2;
      out[1] = 2;
      out[2] = 1;
      break;
    case 19:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 20:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 21:
      out[0] = 1;
      out[1] = 1;
      out[2] = 2;
      break;
    case 22:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 23:
      out[0] = 2;
      out[1] = 1;
      out[2] = 1;
      break;
    case 24:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 25:
      out[0] = 1;
      out[1] = 2;
      out[2] = 1;
      break;
    case 26:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    }
  }
}

template <>
struct BasisLagrange<mesh::RefElCube, 3> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 64;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_ti, const Scalar *x) {
    switch (i) {
    case 0:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      const Scalar x2 = 3 * x[2];
      return -1.0 / 8.0 * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 1:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      const Scalar x2 = 3 * x[2];
      return (1.0 / 8.0) * x[0] * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 2:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      const Scalar x2 = 3 * x[2];
      return -1.0 / 8.0 * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x2 - 2) * (x2 - 1) * (x[2] - 1);
    case 3:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      const Scalar x2 = 3 * x[2];
      return (1.0 / 8.0) * x[1] * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 4:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      const Scalar x2 = 3 * x[2];
      return (1.0 / 8.0) * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 5:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      const Scalar x2 = 3 * x[2];
      return -1.0 / 8.0 * x[0] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x2 - 2) * (x2 - 1) * (x[1] - 1);
    case 6:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      const Scalar x2 = 3 * x[2];
      return (1.0 / 8.0) * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x2 - 2) * (x2 - 1);
    case 7:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      const Scalar x2 = 3 * x[2];
      return -1.0 / 8.0 * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x2 - 2) * (x2 - 1) * (x[0] - 1);
    case 8:
      const Scalar x0 = 3 * x[1];
      const Scalar x1 = 3 * x[2];
      return (9.0 / 8.0) * x[0] * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) *
             (x[0] - 1) * (3 * x[0] - 2) * (x[1] - 1) * (x[2] - 1);
    case 9:
      const Scalar x0 = 3 * x[1];
      const Scalar x1 = 3 * x[2];
      return -9.0 / 8.0 * x[0] * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) *
             (x[0] - 1) * (3 * x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 10:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[2];
      return -9.0 / 8.0 * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[1] - 1) * (3 * x[1] - 2) * (x[2] - 1);
    case 11:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[2];
      return (9.0 / 8.0) * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[1] - 1) * (3 * x[1] - 1) * (x[2] - 1);
    case 12:
      const Scalar x0 = 3 * x[1];
      const Scalar x1 = 3 * x[2];
      return (9.0 / 8.0) * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[0] - 1) * (3 * x[0] - 1) * (x[2] - 1);
    case 13:
      const Scalar x0 = 3 * x[1];
      const Scalar x1 = 3 * x[2];
      return -9.0 / 8.0 * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[0] - 1) * (3 * x[0] - 2) * (x[2] - 1);
    case 14:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[2];
      return -9.0 / 8.0 * x[1] * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) *
             (x[0] - 1) * (x[1] - 1) * (3 * x[1] - 1) * (x[2] - 1);
    case 15:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[2];
      return (9.0 / 8.0) * x[1] * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) *
             (x[0] - 1) * (x[1] - 1) * (3 * x[1] - 2) * (x[2] - 1);
    case 16:
      const Scalar x0 = 3 * x[1];
      const Scalar x1 = 3 * x[2];
      return -9.0 / 8.0 * x[0] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[0] - 1) * (3 * x[0] - 2) * (x[1] - 1);
    case 17:
      const Scalar x0 = 3 * x[1];
      const Scalar x1 = 3 * x[2];
      return (9.0 / 8.0) * x[0] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[0] - 1) * (3 * x[0] - 1) * (x[1] - 1);
    case 18:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[2];
      return (9.0 / 8.0) * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[1] - 1) * (3 * x[1] - 2);
    case 19:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[2];
      return -9.0 / 8.0 * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[1] - 1) * (3 * x[1] - 1);
    case 20:
      const Scalar x0 = 3 * x[1];
      const Scalar x1 = 3 * x[2];
      return -9.0 / 8.0 * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[0] - 1) * (3 * x[0] - 1);
    case 21:
      const Scalar x0 = 3 * x[1];
      const Scalar x1 = 3 * x[2];
      return (9.0 / 8.0) * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[0] - 1) * (3 * x[0] - 2);
    case 22:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[2];
      return (9.0 / 8.0) * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[0] - 1) * (x[1] - 1) * (3 * x[1] - 1);
    case 23:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[2];
      return -9.0 / 8.0 * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[0] - 1) * (x[1] - 1) * (3 * x[1] - 2);
    case 24:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      return (9.0 / 8.0) * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) *
             (x[0] - 1) * (x[1] - 1) * (x[2] - 1) * (3 * x[2] - 2);
    case 25:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      return -9.0 / 8.0 * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) * (x1 - 1) *
             (x[0] - 1) * (x[1] - 1) * (x[2] - 1) * (3 * x[2] - 1);
    case 26:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      return -9.0 / 8.0 * x[0] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[1] - 1) * (x[2] - 1) * (3 * x[2] - 2);
    case 27:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      return (9.0 / 8.0) * x[0] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[1] - 1) * (x[2] - 1) * (3 * x[2] - 1);
    case 28:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      return (9.0 / 8.0) * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[2] - 1) * (3 * x[2] - 2);
    case 29:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      return -9.0 / 8.0 * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[2] - 1) * (3 * x[2] - 1);
    case 30:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      return -9.0 / 8.0 * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[0] - 1) * (x[2] - 1) * (3 * x[2] - 2);
    case 31:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = 3 * x[1];
      return (9.0 / 8.0) * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x1 - 2) *
             (x1 - 1) * (x[0] - 1) * (x[2] - 1) * (3 * x[2] - 1);
    case 32:
      const Scalar x0 = 3 * x[2];
      return -81.0 / 8.0 * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (3 * x[0] - 2) * (x[1] - 1) * (3 * x[1] - 2) * (x[2] - 1);
    case 33:
      const Scalar x0 = 3 * x[2];
      return (81.0 / 8.0) * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (3 * x[0] - 2) * (x[1] - 1) * (3 * x[1] - 1) * (x[2] - 1);
    case 34:
      const Scalar x0 = 3 * x[2];
      return -81.0 / 8.0 * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (3 * x[0] - 1) * (x[1] - 1) * (3 * x[1] - 1) * (x[2] - 1);
    case 35:
      const Scalar x0 = 3 * x[2];
      return (81.0 / 8.0) * x[0] * x[1] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (3 * x[0] - 1) * (x[1] - 1) * (3 * x[1] - 2) * (x[2] - 1);
    case 36:
      const Scalar x0 = 3 * x[2];
      return (81.0 / 8.0) * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) *
             (x[0] - 1) * (3 * x[0] - 2) * (x[1] - 1) * (3 * x[1] - 2);
    case 37:
      const Scalar x0 = 3 * x[2];
      return -81.0 / 8.0 * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) *
             (x[0] - 1) * (3 * x[0] - 1) * (x[1] - 1) * (3 * x[1] - 2);
    case 38:
      const Scalar x0 = 3 * x[2];
      return (81.0 / 8.0) * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) *
             (x[0] - 1) * (3 * x[0] - 1) * (x[1] - 1) * (3 * x[1] - 1);
    case 39:
      const Scalar x0 = 3 * x[2];
      return -81.0 / 8.0 * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) *
             (x[0] - 1) * (3 * x[0] - 2) * (x[1] - 1) * (3 * x[1] - 1);
    case 40:
      const Scalar x0 = 3 * x[0];
      return -81.0 / 8.0 * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (x[1] - 1) * (3 * x[1] - 2) * (x[2] - 1) * (3 * x[2] - 2);
    case 41:
      const Scalar x0 = 3 * x[0];
      return (81.0 / 8.0) * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (x[1] - 1) * (3 * x[1] - 2) * (x[2] - 1) * (3 * x[2] - 1);
    case 42:
      const Scalar x0 = 3 * x[0];
      return -81.0 / 8.0 * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (x[1] - 1) * (3 * x[1] - 1) * (x[2] - 1) * (3 * x[2] - 1);
    case 43:
      const Scalar x0 = 3 * x[0];
      return (81.0 / 8.0) * x[1] * x[2] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (x[1] - 1) * (3 * x[1] - 1) * (x[2] - 1) * (3 * x[2] - 2);
    case 44:
      const Scalar x0 = 3 * x[0];
      return (81.0 / 8.0) * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) *
             (x[1] - 1) * (3 * x[1] - 2) * (x[2] - 1) * (3 * x[2] - 2);
    case 45:
      const Scalar x0 = 3 * x[0];
      return -81.0 / 8.0 * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) *
             (x[1] - 1) * (3 * x[1] - 1) * (x[2] - 1) * (3 * x[2] - 2);
    case 46:
      const Scalar x0 = 3 * x[0];
      return (81.0 / 8.0) * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) *
             (x[1] - 1) * (3 * x[1] - 1) * (x[2] - 1) * (3 * x[2] - 1);
    case 47:
      const Scalar x0 = 3 * x[0];
      return -81.0 / 8.0 * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) *
             (x[1] - 1) * (3 * x[1] - 2) * (x[2] - 1) * (3 * x[2] - 1);
    case 48:
      const Scalar x0 = 3 * x[1];
      return -81.0 / 8.0 * x[0] * x[2] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (3 * x[0] - 2) * (x[1] - 1) * (x[2] - 1) * (3 * x[2] - 2);
    case 49:
      const Scalar x0 = 3 * x[1];
      return (81.0 / 8.0) * x[0] * x[2] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (3 * x[0] - 1) * (x[1] - 1) * (x[2] - 1) * (3 * x[2] - 2);
    case 50:
      const Scalar x0 = 3 * x[1];
      return -81.0 / 8.0 * x[0] * x[2] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (3 * x[0] - 1) * (x[1] - 1) * (x[2] - 1) * (3 * x[2] - 1);
    case 51:
      const Scalar x0 = 3 * x[1];
      return (81.0 / 8.0) * x[0] * x[2] * (x0 - 2) * (x0 - 1) * (x[0] - 1) *
             (3 * x[0] - 2) * (x[1] - 1) * (x[2] - 1) * (3 * x[2] - 1);
    case 52:
      const Scalar x0 = 3 * x[1];
      return -81.0 / 8.0 * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) *
             (x[0] - 1) * (3 * x[0] - 1) * (x[2] - 1) * (3 * x[2] - 2);
    case 53:
      const Scalar x0 = 3 * x[1];
      return (81.0 / 8.0) * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) *
             (x[0] - 1) * (3 * x[0] - 2) * (x[2] - 1) * (3 * x[2] - 2);
    case 54:
      const Scalar x0 = 3 * x[1];
      return -81.0 / 8.0 * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) *
             (x[0] - 1) * (3 * x[0] - 2) * (x[2] - 1) * (3 * x[2] - 1);
    case 55:
      const Scalar x0 = 3 * x[1];
      return (81.0 / 8.0) * x[0] * x[1] * x[2] * (x0 - 2) * (x0 - 1) *
             (x[0] - 1) * (3 * x[0] - 1) * (x[2] - 1) * (3 * x[2] - 1);
    case 56:
      return (729.0 / 8.0) * x[0] * x[1] * x[2] * (x[0] - 1) * (3 * x[0] - 2) *
             (x[1] - 1) * (3 * x[1] - 2) * (x[2] - 1) * (3 * x[2] - 2);
    case 57:
      return -729.0 / 8.0 * x[0] * x[1] * x[2] * (x[0] - 1) * (3 * x[0] - 1) *
             (x[1] - 1) * (3 * x[1] - 2) * (x[2] - 1) * (3 * x[2] - 2);
    case 58:
      return (729.0 / 8.0) * x[0] * x[1] * x[2] * (x[0] - 1) * (3 * x[0] - 1) *
             (x[1] - 1) * (3 * x[1] - 1) * (x[2] - 1) * (3 * x[2] - 2);
    case 59:
      return -729.0 / 8.0 * x[0] * x[1] * x[2] * (x[0] - 1) * (3 * x[0] - 2) *
             (x[1] - 1) * (3 * x[1] - 1) * (x[2] - 1) * (3 * x[2] - 2);
    case 60:
      return -729.0 / 8.0 * x[0] * x[1] * x[2] * (x[0] - 1) * (3 * x[0] - 2) *
             (x[1] - 1) * (3 * x[1] - 2) * (x[2] - 1) * (3 * x[2] - 1);
    case 61:
      return (729.0 / 8.0) * x[0] * x[1] * x[2] * (x[0] - 1) * (3 * x[0] - 1) *
             (x[1] - 1) * (3 * x[1] - 2) * (x[2] - 1) * (3 * x[2] - 1);
    case 62:
      return -729.0 / 8.0 * x[0] * x[1] * x[2] * (x[0] - 1) * (3 * x[0] - 1) *
             (x[1] - 1) * (3 * x[1] - 1) * (x[2] - 1) * (3 * x[2] - 1);
    case 63:
      return (729.0 / 8.0) * x[0] * x[1] * x[2] * (x[0] - 1) * (3 * x[0] - 2) *
             (x[1] - 1) * (3 * x[1] - 1) * (x[2] - 1) * (3 * x[2] - 1);
    default:
      return 0;
    }
  }

  template <typename Scalar>
  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    switch (i) {
    case 0:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x[2] - 1;
      const Scalar x7 = 3 * x[2];
      const Scalar x8 = x7 - 2;
      const Scalar x9 = x7 - 1;
      const Scalar x10 = x8 * x9;
      const Scalar x11 = (1.0 / 8.0) * x10 * x6;
      const Scalar x12 = x[1] - 1;
      const Scalar x13 = 3 * x[1];
      const Scalar x14 = x13 - 2;
      const Scalar x15 = x13 - 1;
      const Scalar x16 = x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = 3 * x12;
      const Scalar x19 = x2 * x5;
      const Scalar x20 = 3 * x6;
      out[0] = -x11 * x17 * (x1 * x3 + x3 * x4 + x5);
      out[1] = -x11 * x19 * (x14 * x18 + x15 * x18 + x16);
      out[2] = -1.0 / 8.0 * x17 * x19 * (x10 + x20 * x8 + x20 * x9);
      break;
    case 1:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = (1.0 / 8.0) * x4 * x8;
      const Scalar x10 = x[1] - 1;
      const Scalar x11 = 3 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x12 * x13;
      const Scalar x15 = x10 * x14;
      const Scalar x16 = 3 * x10;
      const Scalar x17 = x3 * x[0];
      const Scalar x18 = 3 * x4;
      out[0] = x15 * x9 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x17 * x9 * (x12 * x16 + x13 * x16 + x14);
      out[2] = (1.0 / 8.0) * x15 * x17 * (x18 * x6 + x18 * x7 + x8);
      break;
    case 2:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = (1.0 / 8.0) * x4 * x8;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x[1];
      const Scalar x15 = x3 * x[0];
      const Scalar x16 = 3 * x4;
      out[0] = -x14 * x9 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x15 * x9 * (x10 * x11 + x10 * x12 + x13);
      out[2] = -1.0 / 8.0 * x14 * x15 * (x16 * x6 + x16 * x7 + x8);
      break;
    case 3:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x[2] - 1;
      const Scalar x7 = 3 * x[2];
      const Scalar x8 = x7 - 2;
      const Scalar x9 = x7 - 1;
      const Scalar x10 = x8 * x9;
      const Scalar x11 = (1.0 / 8.0) * x10 * x6;
      const Scalar x12 = 3 * x[1];
      const Scalar x13 = x12 - 2;
      const Scalar x14 = x12 - 1;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x15 * x[1];
      const Scalar x17 = x2 * x5;
      const Scalar x18 = 3 * x6;
      out[0] = x11 * x16 * (x1 * x3 + x3 * x4 + x5);
      out[1] = x11 * x17 * (x12 * x13 + x12 * x14 + x15);
      out[2] = (1.0 / 8.0) * x16 * x17 * (x10 + x18 * x8 + x18 * x9);
      break;
    case 4:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = 3 * x[2];
      const Scalar x7 = x6 - 2;
      const Scalar x8 = x6 - 1;
      const Scalar x9 = x7 * x8;
      const Scalar x10 = (1.0 / 8.0) * x9 * x[2];
      const Scalar x11 = x[1] - 1;
      const Scalar x12 = 3 * x[1];
      const Scalar x13 = x12 - 2;
      const Scalar x14 = x12 - 1;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = 3 * x11;
      const Scalar x18 = x2 * x5;
      out[0] = x10 * x16 * (x1 * x3 + x3 * x4 + x5);
      out[1] = x10 * x18 * (x13 * x17 + x14 * x17 + x15);
      out[2] = (1.0 / 8.0) * x16 * x18 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 5:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = 3 * x[2];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = (1.0 / 8.0) * x7 * x[2];
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = 3 * x9;
      const Scalar x16 = x3 * x[0];
      out[0] = -x14 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x16 * x8 * (x11 * x15 + x12 * x15 + x13);
      out[2] = -1.0 / 8.0 * x14 * x16 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 6:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = 3 * x[2];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = (1.0 / 8.0) * x7 * x[2];
      const Scalar x9 = 3 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x3 * x[0];
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (1.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 7:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = 3 * x[2];
      const Scalar x7 = x6 - 2;
      const Scalar x8 = x6 - 1;
      const Scalar x9 = x7 * x8;
      const Scalar x10 = (1.0 / 8.0) * x9 * x[2];
      const Scalar x11 = 3 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x12 * x13;
      const Scalar x15 = x14 * x[1];
      const Scalar x16 = x2 * x5;
      out[0] = -x10 * x15 * (x1 * x3 + x3 * x4 + x5);
      out[1] = -x10 * x16 * (x11 * x12 + x11 * x13 + x14);
      out[2] = -1.0 / 8.0 * x15 * x16 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 8:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = (9.0 / 8.0) * x4 * x8;
      const Scalar x10 = x[1] - 1;
      const Scalar x11 = 3 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x12 * x13;
      const Scalar x15 = x10 * x14;
      const Scalar x16 = 3 * x10;
      const Scalar x17 = x0 * x3;
      const Scalar x18 = 3 * x4;
      out[0] = x15 * x9 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x17 * x9 * (x12 * x16 + x13 * x16 + x14);
      out[2] = (9.0 / 8.0) * x15 * x17 * (x18 * x6 + x18 * x7 + x8);
      break;
    case 9:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = (9.0 / 8.0) * x4 * x8;
      const Scalar x10 = x[1] - 1;
      const Scalar x11 = 3 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x12 * x13;
      const Scalar x15 = x10 * x14;
      const Scalar x16 = 3 * x10;
      const Scalar x17 = x0 * x3;
      const Scalar x18 = 3 * x4;
      out[0] = -x15 * x9 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x17 * x9 * (x12 * x16 + x13 * x16 + x14);
      out[2] = -9.0 / 8.0 * x15 * x17 * (x18 * x6 + x18 * x7 + x8);
      break;
    case 10:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = (9.0 / 8.0) * x4 * x8;
      const Scalar x10 = x[1] - 1;
      const Scalar x11 = 3 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x3 * x[0];
      const Scalar x16 = 3 * x4;
      out[0] = -x14 * x9 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x15 * x9 * (x10 * x11 + x10 * x12 + x13);
      out[2] = -9.0 / 8.0 * x14 * x15 * (x16 * x6 + x16 * x7 + x8);
      break;
    case 11:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = (9.0 / 8.0) * x4 * x8;
      const Scalar x10 = x[1] - 1;
      const Scalar x11 = 3 * x[1];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x3 * x[0];
      const Scalar x16 = 3 * x4;
      out[0] = x14 * x9 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x15 * x9 * (x10 * x11 + x10 * x12 + x13);
      out[2] = (9.0 / 8.0) * x14 * x15 * (x16 * x6 + x16 * x7 + x8);
      break;
    case 12:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = (9.0 / 8.0) * x4 * x8;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x[1];
      const Scalar x15 = x0 * x3;
      const Scalar x16 = 3 * x4;
      out[0] = x14 * x9 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x15 * x9 * (x10 * x11 + x10 * x12 + x13);
      out[2] = (9.0 / 8.0) * x14 * x15 * (x16 * x6 + x16 * x7 + x8);
      break;
    case 13:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = (9.0 / 8.0) * x4 * x8;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x[1];
      const Scalar x15 = x0 * x3;
      const Scalar x16 = 3 * x4;
      out[0] = -x14 * x9 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x15 * x9 * (x10 * x11 + x10 * x12 + x13);
      out[2] = -9.0 / 8.0 * x14 * x15 * (x16 * x6 + x16 * x7 + x8);
      break;
    case 14:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x[2] - 1;
      const Scalar x7 = 3 * x[2];
      const Scalar x8 = x7 - 2;
      const Scalar x9 = x7 - 1;
      const Scalar x10 = x8 * x9;
      const Scalar x11 = (9.0 / 8.0) * x10 * x6;
      const Scalar x12 = x[1] - 1;
      const Scalar x13 = 3 * x[1];
      const Scalar x14 = x13 - 1;
      const Scalar x15 = x14 * x[1];
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x2 * x5;
      const Scalar x18 = 3 * x6;
      out[0] = -x11 * x16 * (x1 * x3 + x3 * x4 + x5);
      out[1] = -x11 * x17 * (x12 * x13 + x12 * x14 + x15);
      out[2] = -9.0 / 8.0 * x16 * x17 * (x10 + x18 * x8 + x18 * x9);
      break;
    case 15:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x[2] - 1;
      const Scalar x7 = 3 * x[2];
      const Scalar x8 = x7 - 2;
      const Scalar x9 = x7 - 1;
      const Scalar x10 = x8 * x9;
      const Scalar x11 = (9.0 / 8.0) * x10 * x6;
      const Scalar x12 = x[1] - 1;
      const Scalar x13 = 3 * x[1];
      const Scalar x14 = x13 - 2;
      const Scalar x15 = x14 * x[1];
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x2 * x5;
      const Scalar x18 = 3 * x6;
      out[0] = x11 * x16 * (x1 * x3 + x3 * x4 + x5);
      out[1] = x11 * x17 * (x12 * x13 + x12 * x14 + x15);
      out[2] = (9.0 / 8.0) * x16 * x17 * (x10 + x18 * x8 + x18 * x9);
      break;
    case 16:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = 3 * x[2];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = (9.0 / 8.0) * x7 * x[2];
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = 3 * x9;
      const Scalar x16 = x0 * x3;
      out[0] = -x14 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x16 * x8 * (x11 * x15 + x12 * x15 + x13);
      out[2] = -9.0 / 8.0 * x14 * x16 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 17:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = 3 * x[2];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = (9.0 / 8.0) * x7 * x[2];
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = 3 * x9;
      const Scalar x16 = x0 * x3;
      out[0] = x14 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x16 * x8 * (x11 * x15 + x12 * x15 + x13);
      out[2] = (9.0 / 8.0) * x14 * x16 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 18:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = 3 * x[2];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = (9.0 / 8.0) * x7 * x[2];
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x3 * x[0];
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (9.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 19:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = 3 * x[2];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = (9.0 / 8.0) * x7 * x[2];
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x3 * x[0];
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -9.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 20:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = 3 * x[2];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = (9.0 / 8.0) * x7 * x[2];
      const Scalar x9 = 3 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x0 * x3;
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -9.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 21:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = 3 * x[2];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = (9.0 / 8.0) * x7 * x[2];
      const Scalar x9 = 3 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x0 * x3;
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (9.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 22:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = 3 * x[2];
      const Scalar x7 = x6 - 2;
      const Scalar x8 = x6 - 1;
      const Scalar x9 = x7 * x8;
      const Scalar x10 = (9.0 / 8.0) * x9 * x[2];
      const Scalar x11 = x[1] - 1;
      const Scalar x12 = 3 * x[1];
      const Scalar x13 = x12 - 1;
      const Scalar x14 = x13 * x[1];
      const Scalar x15 = x11 * x14;
      const Scalar x16 = x2 * x5;
      out[0] = x10 * x15 * (x1 * x3 + x3 * x4 + x5);
      out[1] = x10 * x16 * (x11 * x12 + x11 * x13 + x14);
      out[2] = (9.0 / 8.0) * x15 * x16 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 23:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = 3 * x[2];
      const Scalar x7 = x6 - 2;
      const Scalar x8 = x6 - 1;
      const Scalar x9 = x7 * x8;
      const Scalar x10 = (9.0 / 8.0) * x9 * x[2];
      const Scalar x11 = x[1] - 1;
      const Scalar x12 = 3 * x[1];
      const Scalar x13 = x12 - 2;
      const Scalar x14 = x13 * x[1];
      const Scalar x15 = x11 * x14;
      const Scalar x16 = x2 * x5;
      out[0] = -x10 * x15 * (x1 * x3 + x3 * x4 + x5);
      out[1] = -x10 * x16 * (x11 * x12 + x11 * x13 + x14);
      out[2] = -9.0 / 8.0 * x15 * x16 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 24:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x[2] - 1;
      const Scalar x7 = 3 * x[2];
      const Scalar x8 = x7 - 2;
      const Scalar x9 = x8 * x[2];
      const Scalar x10 = (9.0 / 8.0) * x6 * x9;
      const Scalar x11 = x[1] - 1;
      const Scalar x12 = 3 * x[1];
      const Scalar x13 = x12 - 2;
      const Scalar x14 = x12 - 1;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = 3 * x11;
      const Scalar x18 = x2 * x5;
      out[0] = x10 * x16 * (x1 * x3 + x3 * x4 + x5);
      out[1] = x10 * x18 * (x13 * x17 + x14 * x17 + x15);
      out[2] = (9.0 / 8.0) * x16 * x18 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 25:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x[2] - 1;
      const Scalar x7 = 3 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = x8 * x[2];
      const Scalar x10 = (9.0 / 8.0) * x6 * x9;
      const Scalar x11 = x[1] - 1;
      const Scalar x12 = 3 * x[1];
      const Scalar x13 = x12 - 2;
      const Scalar x14 = x12 - 1;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = 3 * x11;
      const Scalar x18 = x2 * x5;
      out[0] = -x10 * x16 * (x1 * x3 + x3 * x4 + x5);
      out[1] = -x10 * x18 * (x13 * x17 + x14 * x17 + x15);
      out[2] = -9.0 / 8.0 * x16 * x18 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 26:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (9.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = 3 * x9;
      const Scalar x16 = x3 * x[0];
      out[0] = -x14 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x16 * x8 * (x11 * x15 + x12 * x15 + x13);
      out[2] = -9.0 / 8.0 * x14 * x16 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 27:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (9.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = 3 * x9;
      const Scalar x16 = x3 * x[0];
      out[0] = x14 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x16 * x8 * (x11 * x15 + x12 * x15 + x13);
      out[2] = (9.0 / 8.0) * x14 * x16 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 28:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (9.0 / 8.0) * x4 * x7;
      const Scalar x9 = 3 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x3 * x[0];
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (9.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 29:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (9.0 / 8.0) * x4 * x7;
      const Scalar x9 = 3 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x3 * x[0];
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -9.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 30:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x[2] - 1;
      const Scalar x7 = 3 * x[2];
      const Scalar x8 = x7 - 2;
      const Scalar x9 = x8 * x[2];
      const Scalar x10 = (9.0 / 8.0) * x6 * x9;
      const Scalar x11 = 3 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x12 * x13;
      const Scalar x15 = x14 * x[1];
      const Scalar x16 = x2 * x5;
      out[0] = -x10 * x15 * (x1 * x3 + x3 * x4 + x5);
      out[1] = -x10 * x16 * (x11 * x12 + x11 * x13 + x14);
      out[2] = -9.0 / 8.0 * x15 * x16 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 31:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x[2] - 1;
      const Scalar x7 = 3 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = x8 * x[2];
      const Scalar x10 = (9.0 / 8.0) * x6 * x9;
      const Scalar x11 = 3 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x12 * x13;
      const Scalar x15 = x14 * x[1];
      const Scalar x16 = x2 * x5;
      out[0] = x10 * x15 * (x1 * x3 + x3 * x4 + x5);
      out[1] = x10 * x16 * (x11 * x12 + x11 * x13 + x14);
      out[2] = (9.0 / 8.0) * x15 * x16 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 32:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = (81.0 / 8.0) * x4 * x8;
      const Scalar x10 = x[1] - 1;
      const Scalar x11 = 3 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x0 * x3;
      const Scalar x16 = 3 * x4;
      out[0] = -x14 * x9 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x15 * x9 * (x10 * x11 + x10 * x12 + x13);
      out[2] = -81.0 / 8.0 * x14 * x15 * (x16 * x6 + x16 * x7 + x8);
      break;
    case 33:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = (81.0 / 8.0) * x4 * x8;
      const Scalar x10 = x[1] - 1;
      const Scalar x11 = 3 * x[1];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x0 * x3;
      const Scalar x16 = 3 * x4;
      out[0] = x14 * x9 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x15 * x9 * (x10 * x11 + x10 * x12 + x13);
      out[2] = (81.0 / 8.0) * x14 * x15 * (x16 * x6 + x16 * x7 + x8);
      break;
    case 34:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = (81.0 / 8.0) * x4 * x8;
      const Scalar x10 = x[1] - 1;
      const Scalar x11 = 3 * x[1];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x0 * x3;
      const Scalar x16 = 3 * x4;
      out[0] = -x14 * x9 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x15 * x9 * (x10 * x11 + x10 * x12 + x13);
      out[2] = -81.0 / 8.0 * x14 * x15 * (x16 * x6 + x16 * x7 + x8);
      break;
    case 35:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x5 - 1;
      const Scalar x8 = x6 * x7;
      const Scalar x9 = (81.0 / 8.0) * x4 * x8;
      const Scalar x10 = x[1] - 1;
      const Scalar x11 = 3 * x[1];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x0 * x3;
      const Scalar x16 = 3 * x4;
      out[0] = x14 * x9 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x15 * x9 * (x10 * x11 + x10 * x12 + x13);
      out[2] = (81.0 / 8.0) * x14 * x15 * (x16 * x6 + x16 * x7 + x8);
      break;
    case 36:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = 3 * x[2];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = (81.0 / 8.0) * x7 * x[2];
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x0 * x3;
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (81.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 37:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = 3 * x[2];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = (81.0 / 8.0) * x7 * x[2];
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x0 * x3;
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -81.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 38:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = 3 * x[2];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = (81.0 / 8.0) * x7 * x[2];
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x0 * x3;
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (81.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 39:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = 3 * x[2];
      const Scalar x5 = x4 - 2;
      const Scalar x6 = x4 - 1;
      const Scalar x7 = x5 * x6;
      const Scalar x8 = (81.0 / 8.0) * x7 * x[2];
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x0 * x3;
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -81.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 40:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x[2] - 1;
      const Scalar x7 = 3 * x[2];
      const Scalar x8 = x7 - 2;
      const Scalar x9 = x8 * x[2];
      const Scalar x10 = (81.0 / 8.0) * x6 * x9;
      const Scalar x11 = x[1] - 1;
      const Scalar x12 = 3 * x[1];
      const Scalar x13 = x12 - 2;
      const Scalar x14 = x13 * x[1];
      const Scalar x15 = x11 * x14;
      const Scalar x16 = x2 * x5;
      out[0] = -x10 * x15 * (x1 * x3 + x3 * x4 + x5);
      out[1] = -x10 * x16 * (x11 * x12 + x11 * x13 + x14);
      out[2] = -81.0 / 8.0 * x15 * x16 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 41:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x[2] - 1;
      const Scalar x7 = 3 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = x8 * x[2];
      const Scalar x10 = (81.0 / 8.0) * x6 * x9;
      const Scalar x11 = x[1] - 1;
      const Scalar x12 = 3 * x[1];
      const Scalar x13 = x12 - 2;
      const Scalar x14 = x13 * x[1];
      const Scalar x15 = x11 * x14;
      const Scalar x16 = x2 * x5;
      out[0] = x10 * x15 * (x1 * x3 + x3 * x4 + x5);
      out[1] = x10 * x16 * (x11 * x12 + x11 * x13 + x14);
      out[2] = (81.0 / 8.0) * x15 * x16 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 42:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x[2] - 1;
      const Scalar x7 = 3 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = x8 * x[2];
      const Scalar x10 = (81.0 / 8.0) * x6 * x9;
      const Scalar x11 = x[1] - 1;
      const Scalar x12 = 3 * x[1];
      const Scalar x13 = x12 - 1;
      const Scalar x14 = x13 * x[1];
      const Scalar x15 = x11 * x14;
      const Scalar x16 = x2 * x5;
      out[0] = -x10 * x15 * (x1 * x3 + x3 * x4 + x5);
      out[1] = -x10 * x16 * (x11 * x12 + x11 * x13 + x14);
      out[2] = -81.0 / 8.0 * x15 * x16 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 43:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 3 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x[2] - 1;
      const Scalar x7 = 3 * x[2];
      const Scalar x8 = x7 - 2;
      const Scalar x9 = x8 * x[2];
      const Scalar x10 = (81.0 / 8.0) * x6 * x9;
      const Scalar x11 = x[1] - 1;
      const Scalar x12 = 3 * x[1];
      const Scalar x13 = x12 - 1;
      const Scalar x14 = x13 * x[1];
      const Scalar x15 = x11 * x14;
      const Scalar x16 = x2 * x5;
      out[0] = x10 * x15 * (x1 * x3 + x3 * x4 + x5);
      out[1] = x10 * x16 * (x11 * x12 + x11 * x13 + x14);
      out[2] = (81.0 / 8.0) * x15 * x16 * (x6 * x7 + x6 * x8 + x9);
      break;
    case 44:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (81.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x3 * x[0];
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (81.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 45:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (81.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x3 * x[0];
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -81.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 46:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (81.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x3 * x[0];
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (81.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 47:
      const Scalar x0 = 3 * x[0];
      const Scalar x1 = x0 - 2;
      const Scalar x2 = x0 - 1;
      const Scalar x3 = x1 * x2;
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (81.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x3 * x[0];
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -81.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 48:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (81.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = 3 * x9;
      const Scalar x16 = x0 * x3;
      out[0] = -x14 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x16 * x8 * (x11 * x15 + x12 * x15 + x13);
      out[2] = -81.0 / 8.0 * x14 * x16 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 49:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (81.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = 3 * x9;
      const Scalar x16 = x0 * x3;
      out[0] = x14 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x16 * x8 * (x11 * x15 + x12 * x15 + x13);
      out[2] = (81.0 / 8.0) * x14 * x16 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 50:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (81.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = 3 * x9;
      const Scalar x16 = x0 * x3;
      out[0] = -x14 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x16 * x8 * (x11 * x15 + x12 * x15 + x13);
      out[2] = -81.0 / 8.0 * x14 * x16 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 51:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (81.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = 3 * x9;
      const Scalar x16 = x0 * x3;
      out[0] = x14 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x16 * x8 * (x11 * x15 + x12 * x15 + x13);
      out[2] = (81.0 / 8.0) * x14 * x16 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 52:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (81.0 / 8.0) * x4 * x7;
      const Scalar x9 = 3 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x0 * x3;
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -81.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 53:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (81.0 / 8.0) * x4 * x7;
      const Scalar x9 = 3 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x0 * x3;
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (81.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 54:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (81.0 / 8.0) * x4 * x7;
      const Scalar x9 = 3 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x0 * x3;
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -81.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 55:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (81.0 / 8.0) * x4 * x7;
      const Scalar x9 = 3 * x[1];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[1];
      const Scalar x14 = x0 * x3;
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (81.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 56:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (729.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x0 * x3;
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (729.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 57:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (729.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x0 * x3;
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -729.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 58:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (729.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x0 * x3;
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (729.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 59:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 2;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (729.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x0 * x3;
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -729.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 60:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (729.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x0 * x3;
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -729.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 61:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (729.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x0 * x3;
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (729.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 62:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 1;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (729.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x0 * x3;
      out[0] = -x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = -x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = -729.0 / 8.0 * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    case 63:
      const Scalar x0 = x[0] - 1;
      const Scalar x1 = 3 * x[0];
      const Scalar x2 = x1 - 2;
      const Scalar x3 = x2 * x[0];
      const Scalar x4 = x[2] - 1;
      const Scalar x5 = 3 * x[2];
      const Scalar x6 = x5 - 1;
      const Scalar x7 = x6 * x[2];
      const Scalar x8 = (729.0 / 8.0) * x4 * x7;
      const Scalar x9 = x[1] - 1;
      const Scalar x10 = 3 * x[1];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x[1];
      const Scalar x13 = x12 * x9;
      const Scalar x14 = x0 * x3;
      out[0] = x13 * x8 * (x0 * x1 + x0 * x2 + x3);
      out[1] = x14 * x8 * (x10 * x9 + x11 * x9 + x12);
      out[2] = (729.0 / 8.0) * x13 * x14 * (x4 * x5 + x4 * x6 + x7);
      break;
    default:
      break;
    }
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 3 * x[1];
    const Scalar x1 = x0 - 2;
    const Scalar x2 = 3 * x[2];
    const Scalar x3 = x2 - 2;
    const Scalar x4 = 3 * x[0];
    const Scalar x5 = x4 - 1;
    const Scalar x6 = x[0] - 1;
    const Scalar x7 = x[1] - 1;
    const Scalar x8 = x[2] - 1;
    const Scalar x9 = x6 * x7 * x8 * x[0] * x[1] * x[2];
    const Scalar x10 = x5 * x9;
    const Scalar x11 = x4 - 2;
    const Scalar x12 = x11 * x3;
    const Scalar x13 = x0 - 1;
    const Scalar x14 = x2 - 1;
    const Scalar x15 = x1 * x14;
    const Scalar x16 = x11 * x15;
    const Scalar x17 = x13 * x14;
    const Scalar x18 = x7 * x[0] * x[1];
    const Scalar x19 = (1.0 / 9.0) * x18;
    const Scalar x20 = x6 * x[2];
    const Scalar x21 = x20 * x5;
    const Scalar x22 = x21 * x3;
    const Scalar x23 = x17 * x19;
    const Scalar x24 = x5 * x[2];
    const Scalar x25 = x12 * x24;
    const Scalar x26 = x19 * x8;
    const Scalar x27 = x16 * x24;
    const Scalar x28 = x6 * x8;
    const Scalar x29 = x28 * x5;
    const Scalar x30 = x29 * x3;
    const Scalar x31 = x13 * x[0];
    const Scalar x32 = x31 * x[1];
    const Scalar x33 = (1.0 / 9.0) * x32 * x[2];
    const Scalar x34 = (1.0 / 81.0) * x17;
    const Scalar x35 = (1.0 / 81.0) * x32;
    const Scalar x36 = x1 * x12;
    const Scalar x37 = x28 * x36;
    const Scalar x38 = (1.0 / 9.0) * x7;
    const Scalar x39 = x37 * x[1];
    const Scalar x40 = x24 * x39;
    const Scalar x41 = x29 * x[1];
    const Scalar x42 = x38 * x[2];
    const Scalar x43 = x31 * x42;
    const Scalar x44 = (1.0 / 81.0) * x36;
    const Scalar x45 = x14 * x44;
    const Scalar x46 = x5 * x8;
    const Scalar x47 = x44 * x7;
    const Scalar x48 = x17 * x[0];
    const Scalar x49 = x21 * x[1];
    const Scalar x50 = (1.0 / 81.0) * x13;
    const Scalar x51 = x36 * x48;
    const Scalar x52 = (1.0 / 729.0) * x7;
    return -729.0 / 8.0 * coeffs[0] * x17 * x37 * x5 * x52 -
           729.0 / 8.0 * coeffs[10] * x18 * x45 * x46 +
           (9.0 / 8.0) * coeffs[11] * x11 * x13 * x14 * x3 * x5 * x7 * x8 *
               x[0] * x[1] +
           (9.0 / 8.0) * coeffs[12] * x1 * x13 * x14 * x3 * x5 * x6 * x8 *
               x[0] * x[1] -
           729.0 / 8.0 * coeffs[13] * x34 * x39 * x[0] -
           729.0 / 8.0 * coeffs[14] * x12 * x34 * x41 * x7 +
           (9.0 / 8.0) * coeffs[15] * x1 * x11 * x14 * x3 * x5 * x6 * x7 * x8 *
               x[1] -
           729.0 / 8.0 * coeffs[16] * x20 * x47 * x48 +
           (9.0 / 8.0) * coeffs[17] * x1 * x13 * x14 * x3 * x5 * x6 * x7 *
               x[0] * x[2] +
           (9.0 / 8.0) * coeffs[18] * x1 * x11 * x14 * x3 * x5 * x7 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[19] * x18 * x25 * x34 +
           (1.0 / 8.0) * coeffs[1] * x1 * x11 * x13 * x14 * x3 * x5 * x7 * x8 *
               x[0] -
           729.0 / 8.0 * coeffs[20] * x15 * x22 * x35 +
           (9.0 / 8.0) * coeffs[21] * x1 * x11 * x13 * x14 * x3 * x6 * x[0] *
               x[1] * x[2] +
           (9.0 / 8.0) * coeffs[22] * x11 * x13 * x14 * x3 * x5 * x6 * x7 *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[23] * x45 * x49 * x7 +
           (9.0 / 8.0) * coeffs[24] * x1 * x11 * x13 * x3 * x5 * x6 * x7 * x8 *
               x[2] -
           729.0 / 8.0 * coeffs[25] * x16 * x29 * x50 * x7 * x[2] -
           729.0 / 8.0 * coeffs[26] * x24 * x31 * x47 * x8 +
           (9.0 / 8.0) * coeffs[27] * x1 * x11 * x13 * x14 * x5 * x7 * x8 *
               x[0] * x[2] +
           (9.0 / 8.0) * coeffs[28] * x1 * x11 * x13 * x3 * x5 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[29] * x27 * x35 * x8 -
           1.0 / 8.0 * coeffs[2] * x46 * x51 * x[1] -
           729.0 / 8.0 * coeffs[30] * x40 * x50 +
           (9.0 / 8.0) * coeffs[31] * x1 * x11 * x13 * x14 * x5 * x6 * x8 *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[32] * x14 * x19 * x37 +
           (81.0 / 8.0) * coeffs[33] * x11 * x13 * x14 * x3 * x6 * x7 * x8 *
               x[0] * x[1] -
           729.0 / 8.0 * coeffs[34] * x23 * x30 +
           (81.0 / 8.0) * coeffs[35] * x1 * x14 * x3 * x5 * x6 * x7 * x8 *
               x[0] * x[1] +
           (81.0 / 8.0) * coeffs[36] * x1 * x11 * x14 * x3 * x6 * x7 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[37] * x15 * x19 * x22 +
           (81.0 / 8.0) * coeffs[38] * x13 * x14 * x3 * x5 * x6 * x7 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[39] * x12 * x20 * x23 +
           (1.0 / 8.0) * coeffs[3] * x1 * x11 * x13 * x14 * x3 * x5 * x6 * x8 *
               x[1] -
           729.0 / 8.0 * coeffs[40] * x38 * x40 +
           (81.0 / 8.0) * coeffs[41] * x1 * x11 * x14 * x5 * x6 * x7 * x8 *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[42] * x11 * x13 * x14 * x41 * x42 +
           (81.0 / 8.0) * coeffs[43] * x11 * x13 * x3 * x5 * x6 * x7 * x8 *
               x[1] * x[2] +
           (81.0 / 8.0) * coeffs[44] * x1 * x11 * x3 * x5 * x7 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[45] * x13 * x25 * x26 +
           (81.0 / 8.0) * coeffs[46] * x11 * x13 * x14 * x5 * x7 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[47] * x26 * x27 -
           729.0 / 8.0 * coeffs[48] * x37 * x43 +
           (81.0 / 8.0) * coeffs[49] * x1 * x13 * x3 * x5 * x6 * x7 * x8 *
               x[0] * x[2] +
           (1.0 / 8.0) * coeffs[4] * x1 * x11 * x13 * x14 * x3 * x5 * x6 * x7 *
               x[2] -
           729.0 / 8.0 * coeffs[50] * x15 * x29 * x43 +
           (81.0 / 8.0) * coeffs[51] * x1 * x11 * x13 * x14 * x6 * x7 * x8 *
               x[0] * x[2] -
           729.0 / 8.0 * coeffs[52] * x1 * x30 * x33 +
           (81.0 / 8.0) * coeffs[53] * x1 * x11 * x13 * x3 * x6 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[54] * x16 * x28 * x33 +
           (81.0 / 8.0) * coeffs[55] * x1 * x13 * x14 * x5 * x6 * x8 * x[0] *
               x[1] * x[2] +
           (729.0 / 8.0) * coeffs[56] * x1 * x11 * x3 * x6 * x7 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[57] * x1 * x10 * x3 +
           (729.0 / 8.0) * coeffs[58] * x13 * x3 * x5 * x6 * x7 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[59] * x12 * x13 * x9 -
           729.0 / 8.0 * coeffs[5] * x24 * x51 * x52 -
           729.0 / 8.0 * coeffs[60] * x16 * x9 +
           (729.0 / 8.0) * coeffs[61] * x1 * x14 * x5 * x6 * x7 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[62] * x10 * x17 +
           (729.0 / 8.0) * coeffs[63] * x11 * x13 * x14 * x6 * x7 * x8 * x[0] *
               x[1] * x[2] +
           (1.0 / 8.0) * coeffs[6] * x1 * x11 * x13 * x14 * x3 * x5 * x[0] *
               x[1] * x[2] -
           1.0 / 8.0 * coeffs[7] * x17 * x36 * x49 +
           (9.0 / 8.0) * coeffs[8] * x1 * x11 * x13 * x14 * x3 * x6 * x7 * x8 *
               x[0] -
           9.0 / 8.0 * coeffs[9] * x15 * x30 * x31 * x7;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 3 * x[0];
    const Scalar x2 = x0 * x1;
    const Scalar x3 = x1 - 1;
    const Scalar x4 = x3 * x[0];
    const Scalar x5 = x0 * x3;
    const Scalar x6 = x2 + x4 + x5;
    const Scalar x7 = 3 * x[2];
    const Scalar x8 = x7 - 2;
    const Scalar x9 = x8 * x[2];
    const Scalar x10 = x[1] - 1;
    const Scalar x11 = x[2] - 1;
    const Scalar x12 = x10 * x11;
    const Scalar x13 = x12 * x9;
    const Scalar x14 = 3 * x[1];
    const Scalar x15 = x14 - 2;
    const Scalar x16 = x15 * x[1];
    const Scalar x17 = coeffs[57] * x16;
    const Scalar x18 = x1 - 2;
    const Scalar x19 = x18 * x[0];
    const Scalar x20 = x0 * x18;
    const Scalar x21 = x19 + x2 + x20;
    const Scalar x22 = x14 - 1;
    const Scalar x23 = x22 * x[1];
    const Scalar x24 = x21 * x23;
    const Scalar x25 = x16 * x21;
    const Scalar x26 = x7 - 1;
    const Scalar x27 = x26 * x[2];
    const Scalar x28 = x12 * x27;
    const Scalar x29 = coeffs[62] * x23;
    const Scalar x30 = x10 * x26;
    const Scalar x31 = x30 * x9;
    const Scalar x32 = (1.0 / 9.0) * x6;
    const Scalar x33 = x16 * x32;
    const Scalar x34 = x11 * x22;
    const Scalar x35 = x34 * x9;
    const Scalar x36 = (1.0 / 9.0) * x25;
    const Scalar x37 = x27 * x34;
    const Scalar x38 = x22 * x26;
    const Scalar x39 = (1.0 / 81.0) * x38;
    const Scalar x40 = x39 * x9;
    const Scalar x41 = x11 * x8;
    const Scalar x42 = x30 * x41;
    const Scalar x43 = x23 * x42;
    const Scalar x44 = (1.0 / 9.0) * x9;
    const Scalar x45 = x10 * x15;
    const Scalar x46 = x34 * x45;
    const Scalar x47 = x27 * x46;
    const Scalar x48 = x18 * x3;
    const Scalar x49 = x1 * x18 + x1 * x3 + x48;
    const Scalar x50 = (1.0 / 9.0) * x49;
    const Scalar x51 = coeffs[45] * x23;
    const Scalar x52 = coeffs[47] * x16;
    const Scalar x53 = x39 * x41;
    const Scalar x54 = (1.0 / 81.0) * x49;
    const Scalar x55 = x16 * x54;
    const Scalar x56 = 3 * x20 + x48 + 3 * x5;
    const Scalar x57 = (1.0 / 9.0) * x56;
    const Scalar x58 = (1.0 / 81.0) * x56;
    const Scalar x59 = x16 * x58;
    const Scalar x60 = (1.0 / 729.0) * x38;
    const Scalar x61 = x41 * x60;
    const Scalar x62 = coeffs[2] * x16;
    const Scalar x63 = coeffs[5] * x45;
    const Scalar x64 = x60 * x9;
    const Scalar x65 = coeffs[7] * x16;
    const Scalar x66 = coeffs[0] * x45;
    const Scalar x67 = x10 * x14;
    const Scalar x68 = x16 + x45 + x67;
    const Scalar x69 = x0 * x11;
    const Scalar x70 = x69 * x9;
    const Scalar x71 = x4 * x70;
    const Scalar x72 = x10 * x22;
    const Scalar x73 = x23 + x67 + x72;
    const Scalar x74 = x19 * x73;
    const Scalar x75 = x19 * x68;
    const Scalar x76 = x27 * x69;
    const Scalar x77 = x4 * x73;
    const Scalar x78 = x0 * x26;
    const Scalar x79 = x44 * x78;
    const Scalar x80 = x11 * x3;
    const Scalar x81 = (1.0 / 9.0) * x27;
    const Scalar x82 = x26 * x3;
    const Scalar x83 = (1.0 / 81.0) * x82;
    const Scalar x84 = (1.0 / 9.0) * coeffs[32];
    const Scalar x85 = x41 * x75;
    const Scalar x86 = x41 * x78;
    const Scalar x87 = (1.0 / 9.0) * coeffs[34];
    const Scalar x88 = x20 * x80;
    const Scalar x89 = x15 * x22;
    const Scalar x90 = x14 * x15 + x14 * x22 + x89;
    const Scalar x91 = (1.0 / 9.0) * x90;
    const Scalar x92 = x20 * x83;
    const Scalar x93 = x78 * x9;
    const Scalar x94 = (1.0 / 81.0) * x90;
    const Scalar x95 = (1.0 / 81.0) * x19;
    const Scalar x96 = x90 * x95;
    const Scalar x97 = 3 * x45 + 3 * x72 + x89;
    const Scalar x98 = (1.0 / 9.0) * x97;
    const Scalar x99 = x95 * x97;
    const Scalar x100 = (1.0 / 729.0) * x82;
    const Scalar x101 = x100 * x41;
    const Scalar x102 = x100 * x9;
    const Scalar x103 = (1.0 / 81.0) * x97;
    const Scalar x104 = x11 * x7;
    const Scalar x105 = x104 + x41 + x9;
    const Scalar x106 = x0 * x10;
    const Scalar x107 = x106 * x4;
    const Scalar x108 = x105 * x19;
    const Scalar x109 = x106 * x23;
    const Scalar x110 = x106 * x16;
    const Scalar x111 = x11 * x26;
    const Scalar x112 = x104 + x111 + x27;
    const Scalar x113 = x112 * x19;
    const Scalar x114 = x10 * x3;
    const Scalar x115 = (1.0 / 9.0) * x114;
    const Scalar x116 = (1.0 / 9.0) * x16;
    const Scalar x117 = x105 * x116;
    const Scalar x118 = x0 * x22;
    const Scalar x119 = x118 * x4;
    const Scalar x120 = x22 * x3;
    const Scalar x121 = (1.0 / 81.0) * x120;
    const Scalar x122 = x121 * x16;
    const Scalar x123 = x114 * x20;
    const Scalar x124 = (1.0 / 9.0) * x112;
    const Scalar x125 = x118 * x45;
    const Scalar x126 = x125 * x4;
    const Scalar x127 = x26 * x8;
    const Scalar x128 = x127 + x26 * x7 + x7 * x8;
    const Scalar x129 = (1.0 / 9.0) * x128;
    const Scalar x130 = x121 * x45;
    const Scalar x131 = x114 * x95;
    const Scalar x132 = x128 * x16;
    const Scalar x133 = (1.0 / 81.0) * x123;
    const Scalar x134 = 3 * x111 + x127 + 3 * x41;
    const Scalar x135 = x134 * x19;
    const Scalar x136 = x134 * x23;
    const Scalar x137 = x134 * x16;
    const Scalar x138 = (1.0 / 729.0) * x120;
    const Scalar x139 = x138 * x20;
    out[0] =
        -729.0 / 8.0 * coeffs[10] * x42 * x55 +
        (9.0 / 8.0) * coeffs[11] * x10 * x11 * x22 * x26 * x49 * x8 * x[1] +
        (9.0 / 8.0) * coeffs[12] * x11 * x15 * x22 * x26 * x6 * x8 * x[1] -
        729.0 / 8.0 * coeffs[13] * x25 * x53 -
        729.0 / 8.0 * coeffs[14] * x43 * x58 +
        (9.0 / 8.0) * coeffs[15] * x10 * x11 * x15 * x26 * x56 * x8 * x[1] -
        729.0 / 8.0 * coeffs[16] * x21 * x40 * x45 +
        (9.0 / 8.0) * coeffs[17] * x10 * x15 * x22 * x26 * x6 * x8 * x[2] +
        (9.0 / 8.0) * coeffs[18] * x10 * x15 * x26 * x49 * x8 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[19] * x23 * x31 * x54 +
        (1.0 / 8.0) * coeffs[1] * x10 * x11 * x15 * x22 * x26 * x49 * x8 -
        729.0 / 8.0 * coeffs[20] * x16 * x40 * x6 +
        (9.0 / 8.0) * coeffs[21] * x15 * x21 * x22 * x26 * x8 * x[1] * x[2] +
        (9.0 / 8.0) * coeffs[22] * x10 * x22 * x26 * x56 * x8 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[23] * x31 * x59 +
        (9.0 / 8.0) * coeffs[24] * x10 * x11 * x15 * x22 * x56 * x8 * x[2] -
        729.0 / 8.0 * coeffs[25] * x47 * x58 -
        729.0 / 8.0 * coeffs[26] * x46 * x54 * x9 +
        (9.0 / 8.0) * coeffs[27] * x10 * x11 * x15 * x22 * x26 * x49 * x[2] +
        (9.0 / 8.0) * coeffs[28] * x11 * x15 * x22 * x49 * x8 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[29] * x37 * x55 -
        729.0 / 8.0 * coeffs[30] * x35 * x59 +
        (9.0 / 8.0) * coeffs[31] * x11 * x15 * x22 * x26 * x56 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[32] * x36 * x42 +
        (81.0 / 8.0) * coeffs[33] * x10 * x11 * x21 * x22 * x26 * x8 * x[1] -
        729.0 / 8.0 * coeffs[34] * x32 * x43 +
        (81.0 / 8.0) * coeffs[35] * x10 * x11 * x15 * x26 * x6 * x8 * x[1] +
        (81.0 / 8.0) * coeffs[36] * x10 * x15 * x21 * x26 * x8 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[37] * x31 * x33 +
        (81.0 / 8.0) * coeffs[38] * x10 * x22 * x26 * x6 * x8 * x[1] * x[2] -
        81.0 / 8.0 * coeffs[39] * x24 * x31 +
        (1.0 / 8.0) * coeffs[3] * x11 * x15 * x22 * x26 * x56 * x8 * x[1] -
        729.0 / 8.0 * coeffs[40] * x13 * x16 * x57 +
        (81.0 / 8.0) * coeffs[41] * x10 * x11 * x15 * x26 * x56 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[42] * x23 * x28 * x57 +
        (81.0 / 8.0) * coeffs[43] * x10 * x11 * x22 * x56 * x8 * x[1] * x[2] +
        (81.0 / 8.0) * coeffs[44] * x10 * x11 * x15 * x49 * x8 * x[1] * x[2] +
        (81.0 / 8.0) * coeffs[46] * x10 * x11 * x22 * x26 * x49 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[48] * x21 * x44 * x46 +
        (81.0 / 8.0) * coeffs[49] * x10 * x11 * x15 * x22 * x6 * x8 * x[2] +
        (1.0 / 8.0) * coeffs[4] * x10 * x15 * x22 * x26 * x56 * x8 * x[2] -
        729.0 / 8.0 * coeffs[50] * x32 * x47 +
        (81.0 / 8.0) * coeffs[51] * x10 * x11 * x15 * x21 * x22 * x26 * x[2] -
        729.0 / 8.0 * coeffs[52] * x33 * x35 +
        (81.0 / 8.0) * coeffs[53] * x11 * x15 * x21 * x22 * x8 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[54] * x36 * x37 +
        (81.0 / 8.0) * coeffs[55] * x11 * x15 * x22 * x26 * x6 * x[1] * x[2] +
        (729.0 / 8.0) * coeffs[56] * x10 * x11 * x15 * x21 * x8 * x[1] * x[2] +
        (729.0 / 8.0) * coeffs[58] * x10 * x11 * x22 * x6 * x8 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[59] * x13 * x24 -
        729.0 / 8.0 * coeffs[60] * x25 * x28 +
        (729.0 / 8.0) * coeffs[61] * x10 * x11 * x15 * x26 * x6 * x[1] * x[2] +
        (729.0 / 8.0) * coeffs[63] * x10 * x11 * x21 * x22 * x26 * x[1] * x[2] +
        (1.0 / 8.0) * coeffs[6] * x15 * x22 * x26 * x49 * x8 * x[1] * x[2] +
        (9.0 / 8.0) * coeffs[8] * x10 * x11 * x15 * x21 * x22 * x26 * x8 -
        729.0 / 8.0 * coeffs[9] * x45 * x53 * x6 -
        729.0 / 8.0 * x13 * x17 * x6 - 729.0 / 8.0 * x13 * x50 * x51 -
        729.0 / 8.0 * x28 * x29 * x6 - 729.0 / 8.0 * x28 * x50 * x52 -
        729.0 / 8.0 * x49 * x61 * x62 - 729.0 / 8.0 * x49 * x63 * x64 -
        729.0 / 8.0 * x56 * x61 * x66 - 729.0 / 8.0 * x56 * x64 * x65;
    out[1] =
        -729.0 / 8.0 * coeffs[0] * x101 * x20 * x97 -
        729.0 / 8.0 * coeffs[10] * x83 * x85 +
        (9.0 / 8.0) * coeffs[11] * x11 * x18 * x26 * x3 * x73 * x8 * x[0] +
        (9.0 / 8.0) * coeffs[12] * x0 * x11 * x26 * x3 * x8 * x90 * x[0] -
        729.0 / 8.0 * coeffs[13] * x86 * x96 -
        729.0 / 8.0 * coeffs[14] * x41 * x73 * x92 +
        (9.0 / 8.0) * coeffs[15] * x0 * x11 * x18 * x26 * x3 * x68 * x8 -
        729.0 / 8.0 * coeffs[16] * x93 * x99 +
        (9.0 / 8.0) * coeffs[17] * x0 * x26 * x3 * x8 * x97 * x[0] * x[2] +
        (9.0 / 8.0) * coeffs[18] * x18 * x26 * x3 * x68 * x8 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[19] * x74 * x83 * x9 +
        (1.0 / 8.0) * coeffs[1] * x11 * x18 * x26 * x3 * x8 * x97 * x[0] -
        729.0 / 8.0 * coeffs[20] * x4 * x93 * x94 +
        (9.0 / 8.0) * coeffs[21] * x0 * x18 * x26 * x8 * x90 * x[0] * x[2] +
        (9.0 / 8.0) * coeffs[22] * x0 * x18 * x26 * x3 * x73 * x8 * x[2] -
        729.0 / 8.0 * coeffs[23] * x68 * x9 * x92 +
        (9.0 / 8.0) * coeffs[24] * x0 * x11 * x18 * x3 * x8 * x97 * x[2] -
        729.0 / 8.0 * coeffs[25] * x103 * x27 * x88 -
        729.0 / 8.0 * coeffs[26] * x80 * x9 * x99 +
        (9.0 / 8.0) * coeffs[27] * x11 * x18 * x26 * x3 * x97 * x[0] * x[2] +
        (9.0 / 8.0) * coeffs[28] * x11 * x18 * x3 * x8 * x90 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[29] * x27 * x80 * x96 -
        729.0 / 8.0 * coeffs[2] * x101 * x19 * x90 -
        729.0 / 8.0 * coeffs[30] * x88 * x9 * x94 +
        (9.0 / 8.0) * coeffs[31] * x0 * x11 * x18 * x26 * x3 * x90 * x[2] +
        (81.0 / 8.0) * coeffs[33] * x0 * x11 * x18 * x26 * x73 * x8 * x[0] +
        (81.0 / 8.0) * coeffs[35] * x0 * x11 * x26 * x3 * x68 * x8 * x[0] +
        (81.0 / 8.0) * coeffs[36] * x0 * x18 * x26 * x68 * x8 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[37] * x4 * x68 * x79 +
        (81.0 / 8.0) * coeffs[38] * x0 * x26 * x3 * x73 * x8 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[39] * x74 * x79 +
        (1.0 / 8.0) * coeffs[3] * x0 * x11 * x18 * x26 * x3 * x8 * x90 -
        729.0 / 8.0 * coeffs[40] * x44 * x68 * x88 +
        (81.0 / 8.0) * coeffs[41] * x0 * x11 * x18 * x26 * x3 * x68 * x[2] -
        729.0 / 8.0 * coeffs[42] * x73 * x81 * x88 +
        (81.0 / 8.0) * coeffs[43] * x0 * x11 * x18 * x3 * x73 * x8 * x[2] +
        (81.0 / 8.0) * coeffs[44] * x11 * x18 * x3 * x68 * x8 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[45] * x44 * x74 * x80 +
        (81.0 / 8.0) * coeffs[46] * x11 * x18 * x26 * x3 * x73 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[47] * x75 * x80 * x81 -
        729.0 / 8.0 * coeffs[48] * x19 * x70 * x98 +
        (81.0 / 8.0) * coeffs[49] * x0 * x11 * x3 * x8 * x97 * x[0] * x[2] +
        (1.0 / 8.0) * coeffs[4] * x0 * x18 * x26 * x3 * x8 * x97 * x[2] -
        729.0 / 8.0 * coeffs[50] * x4 * x76 * x98 +
        (81.0 / 8.0) * coeffs[51] * x0 * x11 * x18 * x26 * x97 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[52] * x71 * x91 +
        (81.0 / 8.0) * coeffs[53] * x0 * x11 * x18 * x8 * x90 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[54] * x19 * x76 * x91 +
        (81.0 / 8.0) * coeffs[55] * x0 * x11 * x26 * x3 * x90 * x[0] * x[2] +
        (729.0 / 8.0) * coeffs[56] * x0 * x11 * x18 * x68 * x8 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[57] * x68 * x71 +
        (729.0 / 8.0) * coeffs[58] * x0 * x11 * x3 * x73 * x8 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[59] * x70 * x74 -
        729.0 / 8.0 * coeffs[5] * x102 * x19 * x97 -
        729.0 / 8.0 * coeffs[60] * x75 * x76 +
        (729.0 / 8.0) * coeffs[61] * x0 * x11 * x26 * x3 * x68 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[62] * x76 * x77 +
        (729.0 / 8.0) * coeffs[63] * x0 * x11 * x18 * x26 * x73 * x[0] * x[2] +
        (1.0 / 8.0) * coeffs[6] * x18 * x26 * x3 * x8 * x90 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[7] * x102 * x20 * x90 +
        (9.0 / 8.0) * coeffs[8] * x0 * x11 * x18 * x26 * x8 * x97 * x[0] -
        729.0 / 8.0 * coeffs[9] * x103 * x4 * x86 -
        729.0 / 8.0 * x77 * x86 * x87 - 729.0 / 8.0 * x78 * x84 * x85;
    out[2] =
        -729.0 / 8.0 * coeffs[10] * x131 * x137 +
        (9.0 / 8.0) * coeffs[11] * x10 * x134 * x18 * x22 * x3 * x[0] * x[1] +
        (9.0 / 8.0) * coeffs[12] * x0 * x134 * x15 * x22 * x3 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[13] * x118 * x137 * x95 -
        729.0 / 8.0 * coeffs[14] * x133 * x136 +
        (9.0 / 8.0) * coeffs[15] * x0 * x10 * x134 * x15 * x18 * x3 * x[1] -
        729.0 / 8.0 * coeffs[16] * x125 * x128 * x95 +
        (9.0 / 8.0) * coeffs[17] * x0 * x10 * x128 * x15 * x22 * x3 * x[0] +
        (9.0 / 8.0) * coeffs[18] * x10 * x128 * x15 * x18 * x3 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[19] * x128 * x131 * x23 +
        (1.0 / 8.0) * coeffs[1] * x10 * x134 * x15 * x18 * x22 * x3 * x[0] -
        9.0 / 8.0 * coeffs[20] * x119 * x132 +
        (9.0 / 8.0) * coeffs[21] * x0 * x128 * x15 * x18 * x22 * x[0] * x[1] +
        (9.0 / 8.0) * coeffs[22] * x0 * x10 * x128 * x18 * x22 * x3 * x[1] -
        729.0 / 8.0 * coeffs[23] * x132 * x133 +
        (9.0 / 8.0) * coeffs[24] * x0 * x10 * x105 * x15 * x18 * x22 * x3 -
        729.0 / 8.0 * coeffs[25] * x112 * x130 * x20 -
        729.0 / 8.0 * coeffs[26] * x108 * x130 +
        (9.0 / 8.0) * coeffs[27] * x10 * x112 * x15 * x18 * x22 * x3 * x[0] +
        (9.0 / 8.0) * coeffs[28] * x105 * x15 * x18 * x22 * x3 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[29] * x113 * x122 -
        729.0 / 8.0 * coeffs[30] * x105 * x122 * x20 +
        (9.0 / 8.0) * coeffs[31] * x0 * x112 * x15 * x18 * x22 * x3 * x[1] +
        (81.0 / 8.0) * coeffs[33] * x0 * x10 * x134 * x18 * x22 * x[0] * x[1] +
        (81.0 / 8.0) * coeffs[35] * x0 * x10 * x134 * x15 * x3 * x[0] * x[1] +
        (81.0 / 8.0) * coeffs[36] * x0 * x10 * x128 * x15 * x18 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[37] * x110 * x129 * x4 +
        (81.0 / 8.0) * coeffs[38] * x0 * x10 * x128 * x22 * x3 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[39] * x109 * x129 * x19 +
        (1.0 / 8.0) * coeffs[3] * x0 * x134 * x15 * x18 * x22 * x3 * x[1] -
        729.0 / 8.0 * coeffs[40] * x117 * x123 +
        (81.0 / 8.0) * coeffs[41] * x0 * x10 * x112 * x15 * x18 * x3 * x[1] -
        729.0 / 8.0 * coeffs[42] * x123 * x124 * x23 +
        (81.0 / 8.0) * coeffs[43] * x0 * x10 * x105 * x18 * x22 * x3 * x[1] +
        (81.0 / 8.0) * coeffs[44] * x10 * x105 * x15 * x18 * x3 * x[0] * x[1] +
        (81.0 / 8.0) * coeffs[46] * x10 * x112 * x18 * x22 * x3 * x[0] * x[1] -
        81.0 / 8.0 * coeffs[48] * x108 * x125 +
        (81.0 / 8.0) * coeffs[49] * x0 * x10 * x105 * x15 * x22 * x3 * x[0] +
        (1.0 / 8.0) * coeffs[4] * x0 * x10 * x128 * x15 * x18 * x22 * x3 -
        729.0 / 8.0 * coeffs[50] * x124 * x126 +
        (81.0 / 8.0) * coeffs[51] * x0 * x10 * x112 * x15 * x18 * x22 * x[0] -
        729.0 / 8.0 * coeffs[52] * x117 * x119 +
        (81.0 / 8.0) * coeffs[53] * x0 * x105 * x15 * x18 * x22 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[54] * x113 * x116 * x118 +
        (81.0 / 8.0) * coeffs[55] * x0 * x112 * x15 * x22 * x3 * x[0] * x[1] +
        (729.0 / 8.0) * coeffs[56] * x0 * x10 * x105 * x15 * x18 * x[0] * x[1] +
        (729.0 / 8.0) * coeffs[58] * x0 * x10 * x105 * x22 * x3 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[59] * x108 * x109 -
        729.0 / 8.0 * coeffs[60] * x110 * x113 +
        (729.0 / 8.0) * coeffs[61] * x0 * x10 * x112 * x15 * x3 * x[0] * x[1] +
        (729.0 / 8.0) * coeffs[63] * x0 * x10 * x112 * x18 * x22 * x[0] * x[1] +
        (1.0 / 8.0) * coeffs[6] * x128 * x15 * x18 * x22 * x3 * x[0] * x[1] +
        (9.0 / 8.0) * coeffs[8] * x0 * x10 * x134 * x15 * x18 * x22 * x[0] -
        9.0 / 8.0 * coeffs[9] * x126 * x134 - 729.0 / 8.0 * x105 * x107 * x17 -
        729.0 / 8.0 * x107 * x112 * x29 - 729.0 / 8.0 * x107 * x136 * x87 -
        729.0 / 8.0 * x108 * x115 * x51 - 729.0 / 8.0 * x110 * x135 * x84 -
        729.0 / 8.0 * x113 * x115 * x52 -
        729.0 / 8.0 * x128 * x138 * x19 * x63 -
        729.0 / 8.0 * x128 * x139 * x65 - 729.0 / 8.0 * x134 * x139 * x66 -
        729.0 / 8.0 * x135 * x138 * x62;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
      break;
    case 1:
      out[0] = 3;
      out[1] = 0;
      out[2] = 0;
      break;
    case 2:
      out[0] = 3;
      out[1] = 3;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 3;
      out[2] = 0;
      break;
    case 4:
      out[0] = 0;
      out[1] = 0;
      out[2] = 3;
      break;
    case 5:
      out[0] = 3;
      out[1] = 0;
      out[2] = 3;
      break;
    case 6:
      out[0] = 3;
      out[1] = 3;
      out[2] = 3;
      break;
    case 7:
      out[0] = 0;
      out[1] = 3;
      out[2] = 3;
      break;
    case 8:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 9:
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 10:
      out[0] = 3;
      out[1] = 1;
      out[2] = 0;
      break;
    case 11:
      out[0] = 3;
      out[1] = 2;
      out[2] = 0;
      break;
    case 12:
      out[0] = 2;
      out[1] = 3;
      out[2] = 0;
      break;
    case 13:
      out[0] = 1;
      out[1] = 3;
      out[2] = 0;
      break;
    case 14:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 15:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 16:
      out[0] = 1;
      out[1] = 0;
      out[2] = 3;
      break;
    case 17:
      out[0] = 2;
      out[1] = 0;
      out[2] = 3;
      break;
    case 18:
      out[0] = 3;
      out[1] = 1;
      out[2] = 3;
      break;
    case 19:
      out[0] = 3;
      out[1] = 2;
      out[2] = 3;
      break;
    case 20:
      out[0] = 2;
      out[1] = 3;
      out[2] = 3;
      break;
    case 21:
      out[0] = 1;
      out[1] = 3;
      out[2] = 3;
      break;
    case 22:
      out[0] = 0;
      out[1] = 2;
      out[2] = 3;
      break;
    case 23:
      out[0] = 0;
      out[1] = 1;
      out[2] = 3;
      break;
    case 24:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 25:
      out[0] = 0;
      out[1] = 0;
      out[2] = 2;
      break;
    case 26:
      out[0] = 3;
      out[1] = 0;
      out[2] = 1;
      break;
    case 27:
      out[0] = 3;
      out[1] = 0;
      out[2] = 2;
      break;
    case 28:
      out[0] = 3;
      out[1] = 3;
      out[2] = 1;
      break;
    case 29:
      out[0] = 3;
      out[1] = 3;
      out[2] = 2;
      break;
    case 30:
      out[0] = 0;
      out[1] = 3;
      out[2] = 1;
      break;
    case 31:
      out[0] = 0;
      out[1] = 3;
      out[2] = 2;
      break;
    case 32:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 33:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 34:
      out[0] = 2;
      out[1] = 2;
      out[2] = 0;
      break;
    case 35:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 36:
      out[0] = 1;
      out[1] = 1;
      out[2] = 3;
      break;
    case 37:
      out[0] = 2;
      out[1] = 1;
      out[2] = 3;
      break;
    case 38:
      out[0] = 2;
      out[1] = 2;
      out[2] = 3;
      break;
    case 39:
      out[0] = 1;
      out[1] = 2;
      out[2] = 3;
      break;
    case 40:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 41:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 42:
      out[0] = 0;
      out[1] = 2;
      out[2] = 2;
      break;
    case 43:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 44:
      out[0] = 3;
      out[1] = 1;
      out[2] = 1;
      break;
    case 45:
      out[0] = 3;
      out[1] = 2;
      out[2] = 1;
      break;
    case 46:
      out[0] = 3;
      out[1] = 2;
      out[2] = 2;
      break;
    case 47:
      out[0] = 3;
      out[1] = 1;
      out[2] = 2;
      break;
    case 48:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 49:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 50:
      out[0] = 2;
      out[1] = 0;
      out[2] = 2;
      break;
    case 51:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 52:
      out[0] = 2;
      out[1] = 3;
      out[2] = 1;
      break;
    case 53:
      out[0] = 1;
      out[1] = 3;
      out[2] = 1;
      break;
    case 54:
      out[0] = 1;
      out[1] = 3;
      out[2] = 2;
      break;
    case 55:
      out[0] = 2;
      out[1] = 3;
      out[2] = 2;
      break;
    case 56:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    case 57:
      out[0] = 2;
      out[1] = 1;
      out[2] = 1;
      break;
    case 58:
      out[0] = 2;
      out[1] = 2;
      out[2] = 1;
      break;
    case 59:
      out[0] = 1;
      out[1] = 2;
      out[2] = 1;
      break;
    case 60:
      out[0] = 1;
      out[1] = 1;
      out[2] = 2;
      break;
    case 61:
      out[0] = 2;
      out[1] = 1;
      out[2] = 2;
      break;
    case 62:
      out[0] = 2;
      out[1] = 2;
      out[2] = 2;
      break;
    case 63:
      out[0] = 1;
      out[1] = 2;
      out[2] = 2;
      break;
    }
  }
}

template <>
struct BasisLagrange<mesh::RefElCube, 4> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 4;
  static constexpr dim_t num_basis_functions = 125;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_ti, const Scalar *x) {
    switch (i) {
    case 0:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (1.0 / 27.0) * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x2 - 3) * (x2 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 1:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (1.0 / 27.0) * x[0] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x2 - 3) * (x2 - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 2:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (1.0 / 27.0) * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (2 * x[0] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1);
    case 3:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (1.0 / 27.0) * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x2 - 3) * (x2 - 1) * (x[0] - 1) * (2 * x[0] - 1) *
             (2 * x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 4:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (1.0 / 27.0) * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x2 - 3) * (x2 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (2 * x[2] - 1);
    case 5:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (1.0 / 27.0) * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (2 * x[2] - 1);
    case 6:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (1.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (x2 - 3) * (x2 - 1) * (2 * x[0] - 1) *
             (2 * x[1] - 1) * (2 * x[2] - 1);
    case 7:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (1.0 / 27.0) * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (x[0] - 1) * (2 * x[0] - 1) *
             (2 * x[1] - 1) * (2 * x[2] - 1);
    case 8:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[0] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1) *
             (2 * x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 9:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[0] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 10:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (4.0 / 9.0) * x[0] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x2 - 3) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1);
    case 11:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 3) * (x[2] - 1) * (2 * x[2] - 1);
    case 12:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 13:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (4.0 / 9.0) * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1);
    case 14:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) *
             (2 * x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 15:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) *
             (2 * x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 16:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (4.0 / 9.0) * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (x[0] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1);
    case 17:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 18:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 3) * (x[2] - 1) * (2 * x[2] - 1);
    case 19:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (4.0 / 9.0) * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x2 - 3) * (x2 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1);
    case 20:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) *
             (x[1] - 1) * (2 * x[1] - 1) * (2 * x[2] - 1);
    case 21:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) *
             (x[1] - 1) * (2 * x[1] - 1) * (2 * x[2] - 1);
    case 22:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (4.0 / 9.0) * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (2 * x[2] - 1);
    case 23:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 3) * (2 * x[2] - 1);
    case 24:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 1) * (2 * x[2] - 1);
    case 25:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (4.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[2] - 1);
    case 26:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 1) * (2 * x[1] - 1) * (2 * x[2] - 1);
    case 27:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 3) * (2 * x[1] - 1) * (2 * x[2] - 1);
    case 28:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (4.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (x[0] - 1) * (2 * x[1] - 1) *
             (2 * x[2] - 1);
    case 29:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 1) * (2 * x[2] - 1);
    case 30:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -16.0 / 27.0 * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 3) * (2 * x[2] - 1);
    case 31:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (4.0 / 9.0) * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (x[0] - 1) * (2 * x[0] - 1) *
             (x[1] - 1) * (2 * x[2] - 1);
    case 32:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -16.0 / 27.0 * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 33:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -16.0 / 27.0 * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 34:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (4.0 / 9.0) * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) * (x1 - 1) *
             (x2 - 3) * (x2 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (x[2] - 1);
    case 35:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -16.0 / 27.0 * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 36:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -16.0 / 27.0 * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 37:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (4.0 / 9.0) * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (x[2] - 1);
    case 38:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -16.0 / 27.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (2 * x[0] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 39:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -16.0 / 27.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (2 * x[0] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 40:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (4.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (2 * x[0] - 1) * (2 * x[1] - 1) *
             (x[2] - 1);
    case 41:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -16.0 / 27.0 * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 42:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -16.0 / 27.0 * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 43:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (4.0 / 9.0) * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (x[0] - 1) * (2 * x[0] - 1) *
             (2 * x[1] - 1) * (x[2] - 1);
    case 44:
      const Scalar x0 = 4 * x[2];
      return (256.0 / 27.0) * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 3) * (x[2] - 1) * (2 * x[2] - 1);
    case 45:
      const Scalar x0 = 4 * x[2];
      return (256.0 / 27.0) * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 46:
      const Scalar x0 = 4 * x[2];
      return (256.0 / 27.0) * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 47:
      const Scalar x0 = 4 * x[2];
      return (256.0 / 27.0) * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 3) * (x[2] - 1) * (2 * x[2] - 1);
    case 48:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) *
             (x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 49:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 50:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) *
             (x[1] - 1) * (x[2] - 1) * (2 * x[2] - 1);
    case 51:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 3) * (x[2] - 1) * (2 * x[2] - 1);
    case 52:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (16.0 / 3.0) * x[0] * x[1] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1);
    case 53:
      const Scalar x0 = 4 * x[2];
      return (256.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 3) * (2 * x[2] - 1);
    case 54:
      const Scalar x0 = 4 * x[2];
      return (256.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 3) * (2 * x[2] - 1);
    case 55:
      const Scalar x0 = 4 * x[2];
      return (256.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 1) * (2 * x[2] - 1);
    case 56:
      const Scalar x0 = 4 * x[2];
      return (256.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 1) * (2 * x[2] - 1);
    case 57:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 3) * (2 * x[2] - 1);
    case 58:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) *
             (x[1] - 1) * (2 * x[2] - 1);
    case 59:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 1) * (2 * x[2] - 1);
    case 60:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) *
             (x[1] - 1) * (2 * x[2] - 1);
    case 61:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (16.0 / 3.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (x2 - 3) * (x2 - 1) * (x[0] - 1) *
             (x[1] - 1) * (2 * x[2] - 1);
    case 62:
      const Scalar x0 = 4 * x[0];
      return (256.0 / 27.0) * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 63:
      const Scalar x0 = 4 * x[0];
      return (256.0 / 27.0) * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 64:
      const Scalar x0 = 4 * x[0];
      return (256.0 / 27.0) * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 65:
      const Scalar x0 = 4 * x[0];
      return (256.0 / 27.0) * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 66:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 3) * (x[2] - 1);
    case 67:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -64.0 / 9.0 * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1) * (4 * x[2] - 1);
    case 68:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 1) * (x[2] - 1);
    case 69:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -64.0 / 9.0 * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1) * (4 * x[2] - 3);
    case 70:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (16.0 / 3.0) * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (x[0] - 1) * (2 * x[0] - 1) *
             (x[1] - 1) * (x[2] - 1);
    case 71:
      const Scalar x0 = 4 * x[0];
      return (256.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 72:
      const Scalar x0 = 4 * x[0];
      return (256.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 73:
      const Scalar x0 = 4 * x[0];
      return (256.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 74:
      const Scalar x0 = 4 * x[0];
      return (256.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 75:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -64.0 / 9.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (2 * x[0] - 1) * (x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1) * (4 * x[2] - 3);
    case 76:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 1) * (x[2] - 1);
    case 77:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -64.0 / 9.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (2 * x[0] - 1) * (x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1) * (4 * x[2] - 1);
    case 78:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (2 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 3) * (x[2] - 1);
    case 79:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (16.0 / 3.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (x2 - 3) * (x2 - 1) * (2 * x[0] - 1) *
             (x[1] - 1) * (x[2] - 1);
    case 80:
      const Scalar x0 = 4 * x[1];
      return (256.0 / 27.0) * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 81:
      const Scalar x0 = 4 * x[1];
      return (256.0 / 27.0) * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 82:
      const Scalar x0 = 4 * x[1];
      return (256.0 / 27.0) * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 83:
      const Scalar x0 = 4 * x[1];
      return (256.0 / 27.0) * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x[0] - 1) *
             (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 84:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -64.0 / 9.0 * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1) * (4 * x[2] - 3);
    case 85:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) *
             (x[1] - 1) * (2 * x[1] - 1) * (x[2] - 1);
    case 86:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -64.0 / 9.0 * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1) * (4 * x[2] - 1);
    case 87:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) *
             (x[1] - 1) * (2 * x[1] - 1) * (x[2] - 1);
    case 88:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (16.0 / 3.0) * x[0] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (x[2] - 1);
    case 89:
      const Scalar x0 = 4 * x[1];
      return (256.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 90:
      const Scalar x0 = 4 * x[1];
      return (256.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 91:
      const Scalar x0 = 4 * x[1];
      return (256.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 92:
      const Scalar x0 = 4 * x[1];
      return (256.0 / 27.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) * (2 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 93:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -64.0 / 9.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1) * (4 * x[2] - 3);
    case 94:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) *
             (2 * x[1] - 1) * (x[2] - 1);
    case 95:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -64.0 / 9.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1) * (4 * x[2] - 1);
    case 96:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -64.0 / 9.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) *
             (2 * x[1] - 1) * (x[2] - 1);
    case 97:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return (16.0 / 3.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (x2 - 3) * (x2 - 1) * (x[0] - 1) *
             (2 * x[1] - 1) * (x[2] - 1);
    case 98:
      return -4096.0 / 27.0 * x[0] * x[1] * x[2] * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 3) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 99:
      return -4096.0 / 27.0 * x[0] * x[1] * x[2] * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 100:
      return -4096.0 / 27.0 * x[0] * x[1] * x[2] * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 101:
      return -4096.0 / 27.0 * x[0] * x[1] * x[2] * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 3) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 102:
      return -4096.0 / 27.0 * x[0] * x[1] * x[2] * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 3) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 103:
      return -4096.0 / 27.0 * x[0] * x[1] * x[2] * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 104:
      return -4096.0 / 27.0 * x[0] * x[1] * x[2] * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 105:
      return -4096.0 / 27.0 * x[0] * x[1] * x[2] * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 3) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 106:
      const Scalar x0 = 4 * x[0];
      return (1024.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 107:
      const Scalar x0 = 4 * x[1];
      return (1024.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 108:
      const Scalar x0 = 4 * x[0];
      return (1024.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 109:
      const Scalar x0 = 4 * x[1];
      return (1024.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 3);
    case 110:
      const Scalar x0 = 4 * x[0];
      return (1024.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 3) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 111:
      const Scalar x0 = 4 * x[1];
      return (1024.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 112:
      const Scalar x0 = 4 * x[0];
      return (1024.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) * (4 * x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 113:
      const Scalar x0 = 4 * x[1];
      return (1024.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1) *
             (x[2] - 1) * (2 * x[2] - 1) * (4 * x[2] - 1);
    case 114:
      const Scalar x0 = 4 * x[2];
      return (1024.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 3) * (x[2] - 1);
    case 115:
      const Scalar x0 = 4 * x[2];
      return (1024.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 3) * (x[2] - 1);
    case 116:
      const Scalar x0 = 4 * x[2];
      return (1024.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 1) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 1) * (x[2] - 1);
    case 117:
      const Scalar x0 = 4 * x[2];
      return (1024.0 / 9.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x[0] - 1) * (2 * x[0] - 1) * (4 * x[0] - 3) * (x[1] - 1) *
             (2 * x[1] - 1) * (4 * x[1] - 1) * (x[2] - 1);
    case 118:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -256.0 / 3.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1) * (4 * x[2] - 3);
    case 119:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      return -256.0 / 3.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1) *
             (2 * x[2] - 1) * (4 * x[2] - 1);
    case 120:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -256.0 / 3.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 3) * (x[1] - 1) * (x[2] - 1);
    case 121:
      const Scalar x0 = 4 * x[1];
      const Scalar x1 = 4 * x[2];
      return -256.0 / 3.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (x[0] - 1) * (2 * x[0] - 1) *
             (4 * x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 122:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -256.0 / 3.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 3) * (x[2] - 1);
    case 123:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[2];
      return -256.0 / 3.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 1) * (x[0] - 1) * (x[1] - 1) * (2 * x[1] - 1) *
             (4 * x[1] - 1) * (x[2] - 1);
    case 124:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = 4 * x[1];
      const Scalar x2 = 4 * x[2];
      return 64 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 1) * (x1 - 3) *
             (x1 - 1) * (x2 - 3) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) *
             (x[2] - 1);
    default:
      return 0;
    }
  }

  template <typename Scalar>
  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    switch (i) {
    case 0:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2] - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (1.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1] - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x16 * x22;
      const Scalar x24 = 4 * x16 * x17;
      const Scalar x25 = x3 * x7;
      const Scalar x26 = 4 * x8 * x9;
      out[0] = x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = x15 * x25 * (2 * x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = (1.0 / 27.0) * x23 * x25 *
               (x11 * x26 + x12 * x26 + 2 * x13 * x8 + x14);
      break;
    case 1:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2] - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (1.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1] - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x16 * x22;
      const Scalar x24 = 4 * x16 * x17;
      const Scalar x25 = x7 * x[0];
      const Scalar x26 = 4 * x8 * x9;
      out[0] = x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = x15 * x25 * (2 * x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = (1.0 / 27.0) * x23 * x25 *
               (x11 * x26 + x12 * x26 + 2 * x13 * x8 + x14);
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
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2] - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (1.0 / 27.0) * x14 * x8;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x17 * x18;
      const Scalar x25 = x7 * x[0];
      const Scalar x26 = 4 * x8 * x9;
      out[0] = x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = x15 * x25 * (x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = (1.0 / 27.0) * x23 * x25 *
               (x11 * x26 + x12 * x26 + 2 * x13 * x8 + x14);
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
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2] - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (1.0 / 27.0) * x14 * x8;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x17 * x18;
      const Scalar x25 = x3 * x7;
      const Scalar x26 = 4 * x8 * x9;
      out[0] = x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = x15 * x25 * (x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = (1.0 / 27.0) * x23 * x25 *
               (x11 * x26 + x12 * x26 + 2 * x13 * x8 + x14);
      break;
    case 4:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (1.0 / 27.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1] - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x16 * x22;
      const Scalar x24 = 4 * x16 * x17;
      const Scalar x25 = x3 * x7;
      const Scalar x26 = x10 * x9;
      out[0] = x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = x15 * x25 * (2 * x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] =
          (1.0 / 27.0) * x23 * x25 * (x11 * x26 + x12 * x26 + x13 * x8 + x14);
      break;
    case 5:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (1.0 / 27.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1] - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x16 * x22;
      const Scalar x24 = 4 * x16 * x17;
      const Scalar x25 = x7 * x[0];
      const Scalar x26 = x10 * x9;
      out[0] = x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = x15 * x25 * (2 * x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] =
          (1.0 / 27.0) * x23 * x25 * (x11 * x26 + x12 * x26 + x13 * x8 + x14);
      break;
    case 6:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (1.0 / 27.0) * x14 * x[2];
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x17 * x18;
      const Scalar x25 = x7 * x[0];
      const Scalar x26 = x10 * x9;
      out[0] = x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = x15 * x25 * (x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] =
          (1.0 / 27.0) * x23 * x25 * (x11 * x26 + x12 * x26 + x13 * x8 + x14);
      break;
    case 7:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (1.0 / 27.0) * x14 * x[2];
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x17 * x18;
      const Scalar x25 = x3 * x7;
      const Scalar x26 = x10 * x9;
      out[0] = x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = x15 * x25 * (x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] =
          (1.0 / 27.0) * x23 * x25 * (x11 * x26 + x12 * x26 + x13 * x8 + x14);
      break;
    case 8:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (16.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x2 * x6;
      const Scalar x25 = 4 * x7 * x8;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = -16.0 / 27.0 * x22 * x24 *
               (x10 * x25 + x11 * x25 + 2 * x12 * x7 + x13);
      break;
    case 9:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (16.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x2 * x6;
      const Scalar x25 = 4 * x7 * x8;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = -16.0 / 27.0 * x22 * x24 *
               (x10 * x25 + x11 * x25 + 2 * x12 * x7 + x13);
      break;
    case 10:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (4.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x2 * x6;
      const Scalar x25 = 4 * x7 * x8;
      out[0] = x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = (4.0 / 9.0) * x22 * x24 *
               (x10 * x25 + x11 * x25 + 2 * x12 * x7 + x13);
      break;
    case 11:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2] - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (16.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x7 * x[0];
      const Scalar x25 = 4 * x8 * x9;
      out[0] = -x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] =
          -x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] = -16.0 / 27.0 * x23 * x24 *
               (x11 * x25 + x12 * x25 + 2 * x13 * x8 + x14);
      break;
    case 12:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2] - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (16.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 1;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x7 * x[0];
      const Scalar x25 = 4 * x8 * x9;
      out[0] = -x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] =
          -x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] = -16.0 / 27.0 * x23 * x24 *
               (x11 * x25 + x12 * x25 + 2 * x13 * x8 + x14);
      break;
    case 13:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2] - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (4.0 / 9.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x16 * x21;
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x7 * x[0];
      const Scalar x25 = 4 * x8 * x9;
      out[0] = x15 * x22 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = x15 * x24 * (x16 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = (4.0 / 9.0) * x22 * x24 *
               (x11 * x25 + x12 * x25 + 2 * x13 * x8 + x14);
      break;
    case 14:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (16.0 / 27.0) * x13 * x7;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x2 * x6;
      const Scalar x25 = 4 * x7 * x8;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = -16.0 / 27.0 * x22 * x24 *
               (x10 * x25 + x11 * x25 + 2 * x12 * x7 + x13);
      break;
    case 15:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (16.0 / 27.0) * x13 * x7;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x2 * x6;
      const Scalar x25 = 4 * x7 * x8;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = -16.0 / 27.0 * x22 * x24 *
               (x10 * x25 + x11 * x25 + 2 * x12 * x7 + x13);
      break;
    case 16:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (4.0 / 9.0) * x13 * x7;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x2 * x6;
      const Scalar x25 = 4 * x7 * x8;
      out[0] = x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = (4.0 / 9.0) * x22 * x24 *
               (x10 * x25 + x11 * x25 + 2 * x12 * x7 + x13);
      break;
    case 17:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2] - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (16.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 1;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x3 * x7;
      const Scalar x25 = 4 * x8 * x9;
      out[0] = -x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] =
          -x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] = -16.0 / 27.0 * x23 * x24 *
               (x11 * x25 + x12 * x25 + 2 * x13 * x8 + x14);
      break;
    case 18:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2] - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (16.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x3 * x7;
      const Scalar x25 = 4 * x8 * x9;
      out[0] = -x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] =
          -x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] = -16.0 / 27.0 * x23 * x24 *
               (x11 * x25 + x12 * x25 + 2 * x13 * x8 + x14);
      break;
    case 19:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2] - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (4.0 / 9.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x16 * x21;
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x3 * x7;
      const Scalar x25 = 4 * x8 * x9;
      out[0] = x15 * x22 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = x15 * x24 * (x16 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = (4.0 / 9.0) * x22 * x24 *
               (x11 * x25 + x12 * x25 + 2 * x13 * x8 + x14);
      break;
    case 20:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (16.0 / 27.0) * x13 * x[2];
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x2 * x6;
      const Scalar x25 = x8 * x9;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] =
          -16.0 / 27.0 * x22 * x24 * (x10 * x25 + x11 * x25 + x12 * x7 + x13);
      break;
    case 21:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (16.0 / 27.0) * x13 * x[2];
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x2 * x6;
      const Scalar x25 = x8 * x9;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] =
          -16.0 / 27.0 * x22 * x24 * (x10 * x25 + x11 * x25 + x12 * x7 + x13);
      break;
    case 22:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (4.0 / 9.0) * x13 * x[2];
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x2 * x6;
      const Scalar x25 = x8 * x9;
      out[0] = x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] =
          (4.0 / 9.0) * x22 * x24 * (x10 * x25 + x11 * x25 + x12 * x7 + x13);
      break;
    case 23:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (16.0 / 27.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x7 * x[0];
      const Scalar x25 = x10 * x9;
      out[0] = -x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] =
          -x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] =
          -16.0 / 27.0 * x23 * x24 * (x11 * x25 + x12 * x25 + x13 * x8 + x14);
      break;
    case 24:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (16.0 / 27.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 1;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x7 * x[0];
      const Scalar x25 = x10 * x9;
      out[0] = -x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] =
          -x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] =
          -16.0 / 27.0 * x23 * x24 * (x11 * x25 + x12 * x25 + x13 * x8 + x14);
      break;
    case 25:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (4.0 / 9.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x16 * x21;
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x7 * x[0];
      const Scalar x25 = x10 * x9;
      out[0] = x15 * x22 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = x15 * x24 * (x16 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] =
          (4.0 / 9.0) * x22 * x24 * (x11 * x25 + x12 * x25 + x13 * x8 + x14);
      break;
    case 26:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (16.0 / 27.0) * x13 * x[2];
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x2 * x6;
      const Scalar x25 = x8 * x9;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] =
          -16.0 / 27.0 * x22 * x24 * (x10 * x25 + x11 * x25 + x12 * x7 + x13);
      break;
    case 27:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (16.0 / 27.0) * x13 * x[2];
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x2 * x6;
      const Scalar x25 = x8 * x9;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] =
          -16.0 / 27.0 * x22 * x24 * (x10 * x25 + x11 * x25 + x12 * x7 + x13);
      break;
    case 28:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (4.0 / 9.0) * x13 * x[2];
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x2 * x6;
      const Scalar x25 = x8 * x9;
      out[0] = x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] =
          (4.0 / 9.0) * x22 * x24 * (x10 * x25 + x11 * x25 + x12 * x7 + x13);
      break;
    case 29:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (16.0 / 27.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 1;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x3 * x7;
      const Scalar x25 = x10 * x9;
      out[0] = -x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] =
          -x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] =
          -16.0 / 27.0 * x23 * x24 * (x11 * x25 + x12 * x25 + x13 * x8 + x14);
      break;
    case 30:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (16.0 / 27.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x3 * x7;
      const Scalar x25 = x10 * x9;
      out[0] = -x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] =
          -x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] =
          -16.0 / 27.0 * x23 * x24 * (x11 * x25 + x12 * x25 + x13 * x8 + x14);
      break;
    case 31:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x10 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (4.0 / 9.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x16 * x21;
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x3 * x7;
      const Scalar x25 = x10 * x9;
      out[0] = x15 * x22 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = x15 * x24 * (x16 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] =
          (4.0 / 9.0) * x22 * x24 * (x11 * x25 + x12 * x25 + x13 * x8 + x14);
      break;
    case 32:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (16.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1] - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x16 * x22;
      const Scalar x24 = 4 * x16 * x17;
      const Scalar x25 = x3 * x7;
      out[0] = -x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = -x15 * x25 * (2 * x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = -16.0 / 27.0 * x23 * x25 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 33:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (16.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1] - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x16 * x22;
      const Scalar x24 = 4 * x16 * x17;
      const Scalar x25 = x3 * x7;
      out[0] = -x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = -x15 * x25 * (2 * x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = -16.0 / 27.0 * x23 * x25 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 34:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (4.0 / 9.0) * x13 * x8;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x3 * x7;
      const Scalar x25 = x8 * x9;
      out[0] = x14 * x22 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] =
          (4.0 / 9.0) * x22 * x24 * (x10 * x25 + x11 * x25 + x12 * x8 + x13);
      break;
    case 35:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (16.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1] - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x16 * x22;
      const Scalar x24 = 4 * x16 * x17;
      const Scalar x25 = x7 * x[0];
      out[0] = -x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = -x15 * x25 * (2 * x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = -16.0 / 27.0 * x23 * x25 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 36:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (16.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1] - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x16 * x22;
      const Scalar x24 = 4 * x16 * x17;
      const Scalar x25 = x7 * x[0];
      out[0] = -x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = -x15 * x25 * (2 * x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = -16.0 / 27.0 * x23 * x25 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 37:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (4.0 / 9.0) * x13 * x8;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x7 * x[0];
      const Scalar x25 = x8 * x9;
      out[0] = x14 * x22 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] =
          (4.0 / 9.0) * x22 * x24 * (x10 * x25 + x11 * x25 + x12 * x8 + x13);
      break;
    case 38:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (16.0 / 27.0) * x14 * x8;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x17 * x18;
      const Scalar x25 = x7 * x[0];
      out[0] = -x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = -x15 * x25 * (x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = -16.0 / 27.0 * x23 * x25 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 39:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (16.0 / 27.0) * x14 * x8;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x17 * x18;
      const Scalar x25 = x7 * x[0];
      out[0] = -x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = -x15 * x25 * (x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = -16.0 / 27.0 * x23 * x25 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 40:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (4.0 / 9.0) * x13 * x8;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x7 * x[0];
      const Scalar x25 = x8 * x9;
      out[0] = x14 * x22 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] =
          (4.0 / 9.0) * x22 * x24 * (x10 * x25 + x11 * x25 + x12 * x8 + x13);
      break;
    case 41:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (16.0 / 27.0) * x14 * x8;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x17 * x18;
      const Scalar x25 = x3 * x7;
      out[0] = -x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = -x15 * x25 * (x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = -16.0 / 27.0 * x23 * x25 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 42:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (16.0 / 27.0) * x14 * x8;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x18 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x17 * x18;
      const Scalar x25 = x3 * x7;
      out[0] = -x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = -x15 * x25 * (x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = -16.0 / 27.0 * x23 * x25 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 43:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (4.0 / 9.0) * x13 * x8;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x3 * x7;
      const Scalar x25 = x8 * x9;
      out[0] = x14 * x22 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] =
          (4.0 / 9.0) * x22 * x24 * (x10 * x25 + x11 * x25 + x12 * x8 + x13);
      break;
    case 44:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (256.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = 4 * x7 * x8;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = (256.0 / 27.0) * x22 * x23 *
               (x10 * x24 + x11 * x24 + 2 * x12 * x7 + x13);
      break;
    case 45:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (256.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = 4 * x7 * x8;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = (256.0 / 27.0) * x22 * x23 *
               (x10 * x24 + x11 * x24 + 2 * x12 * x7 + x13);
      break;
    case 46:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (256.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = 4 * x7 * x8;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = (256.0 / 27.0) * x22 * x23 *
               (x10 * x24 + x11 * x24 + 2 * x12 * x7 + x13);
      break;
    case 47:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (256.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = 4 * x7 * x8;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = (256.0 / 27.0) * x22 * x23 *
               (x10 * x24 + x11 * x24 + 2 * x12 * x7 + x13);
      break;
    case 48:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (64.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = 4 * x7 * x8;
      out[0] = -x14 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] = -64.0 / 9.0 * x21 * x23 *
               (x10 * x24 + x11 * x24 + 2 * x12 * x7 + x13);
      break;
    case 49:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (64.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = 4 * x7 * x8;
      out[0] = -x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = -64.0 / 9.0 * x22 * x23 *
               (x10 * x24 + x11 * x24 + 2 * x12 * x7 + x13);
      break;
    case 50:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (64.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = 4 * x7 * x8;
      out[0] = -x14 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] = -64.0 / 9.0 * x21 * x23 *
               (x10 * x24 + x11 * x24 + 2 * x12 * x7 + x13);
      break;
    case 51:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (64.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = 4 * x7 * x8;
      out[0] = -x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = -64.0 / 9.0 * x22 * x23 *
               (x10 * x24 + x11 * x24 + 2 * x12 * x7 + x13);
      break;
    case 52:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (16.0 / 3.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = 4 * x7 * x8;
      out[0] = x14 * x21 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] = (16.0 / 3.0) * x21 * x23 *
               (x10 * x24 + x11 * x24 + 2 * x12 * x7 + x13);
      break;
    case 53:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (256.0 / 27.0) * x13 * x[2];
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x8 * x9;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] =
          (256.0 / 27.0) * x22 * x23 * (x10 * x24 + x11 * x24 + x12 * x7 + x13);
      break;
    case 54:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (256.0 / 27.0) * x13 * x[2];
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x8 * x9;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] =
          (256.0 / 27.0) * x22 * x23 * (x10 * x24 + x11 * x24 + x12 * x7 + x13);
      break;
    case 55:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (256.0 / 27.0) * x13 * x[2];
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x8 * x9;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] =
          (256.0 / 27.0) * x22 * x23 * (x10 * x24 + x11 * x24 + x12 * x7 + x13);
      break;
    case 56:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (256.0 / 27.0) * x13 * x[2];
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x8 * x9;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] =
          (256.0 / 27.0) * x22 * x23 * (x10 * x24 + x11 * x24 + x12 * x7 + x13);
      break;
    case 57:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (64.0 / 9.0) * x13 * x[2];
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x8 * x9;
      out[0] = -x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] =
          -64.0 / 9.0 * x22 * x23 * (x10 * x24 + x11 * x24 + x12 * x7 + x13);
      break;
    case 58:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (64.0 / 9.0) * x13 * x[2];
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x8 * x9;
      out[0] = -x14 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] =
          -64.0 / 9.0 * x21 * x23 * (x10 * x24 + x11 * x24 + x12 * x7 + x13);
      break;
    case 59:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (64.0 / 9.0) * x13 * x[2];
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x8 * x9;
      out[0] = -x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] =
          -64.0 / 9.0 * x22 * x23 * (x10 * x24 + x11 * x24 + x12 * x7 + x13);
      break;
    case 60:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (64.0 / 9.0) * x13 * x[2];
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x8 * x9;
      out[0] = -x14 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] =
          -64.0 / 9.0 * x21 * x23 * (x10 * x24 + x11 * x24 + x12 * x7 + x13);
      break;
    case 61:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = 2 * x[2];
      const Scalar x8 = x7 - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x8;
      const Scalar x14 = (16.0 / 3.0) * x13 * x[2];
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x8 * x9;
      out[0] = x14 * x21 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] =
          (16.0 / 3.0) * x21 * x23 * (x10 * x24 + x11 * x24 + x12 * x7 + x13);
      break;
    case 62:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (256.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x3 * x7;
      out[0] = x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] =
          x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] = (256.0 / 27.0) * x23 * x24 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 63:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (256.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x3 * x7;
      out[0] = x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] =
          x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] = (256.0 / 27.0) * x23 * x24 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 64:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (256.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 1;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x3 * x7;
      out[0] = x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] =
          x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] = (256.0 / 27.0) * x23 * x24 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 65:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (256.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 1;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x3 * x7;
      out[0] = x15 * x23 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] =
          x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] = (256.0 / 27.0) * x23 * x24 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 66:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (64.0 / 9.0) * x13 * x8;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x3 * x7;
      const Scalar x24 = x8 * x9;
      out[0] = -x14 * x22 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] =
          -64.0 / 9.0 * x22 * x23 * (x10 * x24 + x11 * x24 + x12 * x8 + x13);
      break;
    case 67:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (64.0 / 9.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x16 * x21;
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x3 * x7;
      out[0] = -x15 * x22 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = -x15 * x24 * (x16 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = -64.0 / 9.0 * x22 * x24 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 68:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (64.0 / 9.0) * x13 * x8;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x3 * x7;
      const Scalar x24 = x8 * x9;
      out[0] = -x14 * x22 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] =
          -64.0 / 9.0 * x22 * x23 * (x10 * x24 + x11 * x24 + x12 * x8 + x13);
      break;
    case 69:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (64.0 / 9.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x16 * x21;
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x3 * x7;
      out[0] = -x15 * x22 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = -x15 * x24 * (x16 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = -64.0 / 9.0 * x22 * x24 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 70:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0] - 1;
      const Scalar x3 = x[0] - 1;
      const Scalar x4 = 4 * x2 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x2 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (16.0 / 3.0) * x13 * x8;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x3 * x7;
      const Scalar x24 = x8 * x9;
      out[0] = x14 * x21 * (x1 * x4 + 2 * x3 * x6 + x4 * x5 + x7);
      out[1] = x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] =
          (16.0 / 3.0) * x21 * x23 * (x10 * x24 + x11 * x24 + x12 * x8 + x13);
      break;
    case 71:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (256.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x7 * x[0];
      out[0] = x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] =
          x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] = (256.0 / 27.0) * x23 * x24 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 72:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (256.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 1;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x7 * x[0];
      out[0] = x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] =
          x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] = (256.0 / 27.0) * x23 * x24 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 73:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (256.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 1;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x7 * x[0];
      out[0] = x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] =
          x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] = (256.0 / 27.0) * x23 * x24 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 74:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (256.0 / 27.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 2 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = 4 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x18 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x22;
      const Scalar x24 = x7 * x[0];
      out[0] = x15 * x23 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] =
          x15 * x24 * (x16 * x17 * x20 + x16 * x18 * x19 + x16 * x21 + x22);
      out[2] = (256.0 / 27.0) * x23 * x24 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 75:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (64.0 / 9.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x16 * x21;
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x7 * x[0];
      out[0] = -x15 * x22 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = -x15 * x24 * (x16 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = -64.0 / 9.0 * x22 * x24 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 76:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (64.0 / 9.0) * x13 * x8;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x7 * x[0];
      const Scalar x24 = x8 * x9;
      out[0] = -x14 * x22 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] =
          -64.0 / 9.0 * x22 * x23 * (x10 * x24 + x11 * x24 + x12 * x8 + x13);
      break;
    case 77:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 2 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = 4 * x[2];
      const Scalar x12 = x11 - 1;
      const Scalar x13 = x10 * x12;
      const Scalar x14 = x13 * x[2];
      const Scalar x15 = (64.0 / 9.0) * x14 * x8;
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x16 * x21;
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x7 * x[0];
      out[0] = -x15 * x22 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = -x15 * x24 * (x16 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = -64.0 / 9.0 * x22 * x24 *
               (x10 * x11 * x8 + x12 * x8 * x9 + x13 * x8 + x14);
      break;
    case 78:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (64.0 / 9.0) * x13 * x8;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x7 * x[0];
      const Scalar x24 = x8 * x9;
      out[0] = -x14 * x22 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] =
          -64.0 / 9.0 * x22 * x23 * (x10 * x24 + x11 * x24 + x12 * x8 + x13);
      break;
    case 79:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = 2 * x[0];
      const Scalar x3 = x2 - 1;
      const Scalar x4 = x0 * x3;
      const Scalar x5 = x0 - 1;
      const Scalar x6 = x1 * x5;
      const Scalar x7 = x3 * x6;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 4 * x[2];
      const Scalar x10 = x9 - 1;
      const Scalar x11 = x9 - 3;
      const Scalar x12 = x10 * x11;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (16.0 / 3.0) * x13 * x8;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x7 * x[0];
      const Scalar x24 = x8 * x9;
      out[0] = x14 * x21 * (x1 * x4 + x2 * x6 + x4 * x5 + x7);
      out[1] = x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] =
          (16.0 / 3.0) * x21 * x23 * (x10 * x24 + x11 * x24 + x12 * x8 + x13);
      break;
    case 80:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (256.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x2 * x6;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = (256.0 / 27.0) * x22 * x24 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 81:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (256.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x2 * x6;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = (256.0 / 27.0) * x22 * x24 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 82:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (256.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x2 * x6;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = (256.0 / 27.0) * x22 * x24 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 83:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (256.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x2 * x6;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = (256.0 / 27.0) * x22 * x24 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 84:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (64.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x2 * x6;
      out[0] = -x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = -x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = -64.0 / 9.0 * x22 * x24 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 85:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (64.0 / 9.0) * x12 * x7;
      const Scalar x14 = x[1] - 1;
      const Scalar x15 = 2 * x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x15 * x19;
      const Scalar x21 = x14 * x20;
      const Scalar x22 = 4 * x14 * x15;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x7 * x8;
      out[0] = -x13 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x13 * x23 * (2 * x14 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] =
          -64.0 / 9.0 * x21 * x23 * (x10 * x24 + x11 * x7 + x12 + x24 * x9);
      break;
    case 86:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (64.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1] - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x15 * x21;
      const Scalar x23 = 4 * x15 * x16;
      const Scalar x24 = x2 * x6;
      out[0] = -x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = -x14 * x24 * (2 * x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = -64.0 / 9.0 * x22 * x24 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 87:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (64.0 / 9.0) * x12 * x7;
      const Scalar x14 = x[1] - 1;
      const Scalar x15 = 2 * x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x15 * x19;
      const Scalar x21 = x14 * x20;
      const Scalar x22 = 4 * x14 * x15;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x7 * x8;
      out[0] = -x13 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x13 * x23 * (2 * x14 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] =
          -64.0 / 9.0 * x21 * x23 * (x10 * x24 + x11 * x7 + x12 + x24 * x9);
      break;
    case 88:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (16.0 / 3.0) * x12 * x7;
      const Scalar x14 = x[1] - 1;
      const Scalar x15 = 2 * x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x15 * x19;
      const Scalar x21 = x14 * x20;
      const Scalar x22 = 4 * x14 * x15;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x7 * x8;
      out[0] = x13 * x21 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = x13 * x23 * (2 * x14 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] =
          (16.0 / 3.0) * x21 * x23 * (x10 * x24 + x11 * x7 + x12 + x24 * x9);
      break;
    case 89:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (256.0 / 27.0) * x13 * x7;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x2 * x6;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = (256.0 / 27.0) * x22 * x24 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 90:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (256.0 / 27.0) * x13 * x7;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x2 * x6;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = (256.0 / 27.0) * x22 * x24 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 91:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (256.0 / 27.0) * x13 * x7;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x2 * x6;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = (256.0 / 27.0) * x22 * x24 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 92:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (256.0 / 27.0) * x13 * x7;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x2 * x6;
      out[0] = x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = (256.0 / 27.0) * x22 * x24 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 93:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (64.0 / 9.0) * x13 * x7;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x2 * x6;
      out[0] = -x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = -x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = -64.0 / 9.0 * x22 * x24 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 94:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (64.0 / 9.0) * x12 * x7;
      const Scalar x14 = 2 * x[1];
      const Scalar x15 = x14 - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x15 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x7 * x8;
      out[0] = -x13 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x13 * x23 * (x14 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] =
          -64.0 / 9.0 * x21 * x23 * (x10 * x24 + x11 * x7 + x12 + x24 * x9);
      break;
    case 95:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (64.0 / 9.0) * x13 * x7;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x17 - 3;
      const Scalar x20 = x18 * x19;
      const Scalar x21 = x16 * x20;
      const Scalar x22 = x21 * x[1];
      const Scalar x23 = x16 * x17;
      const Scalar x24 = x2 * x6;
      out[0] = -x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = -x14 * x24 * (x15 * x20 + x18 * x23 + x19 * x23 + x21);
      out[2] = -64.0 / 9.0 * x22 * x24 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 96:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (64.0 / 9.0) * x12 * x7;
      const Scalar x14 = 2 * x[1];
      const Scalar x15 = x14 - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x15 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x7 * x8;
      out[0] = -x13 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x13 * x23 * (x14 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] =
          -64.0 / 9.0 * x21 * x23 * (x10 * x24 + x11 * x7 + x12 + x24 * x9);
      break;
    case 97:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (16.0 / 3.0) * x12 * x7;
      const Scalar x14 = 2 * x[1];
      const Scalar x15 = x14 - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x15 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      const Scalar x24 = x7 * x8;
      out[0] = x13 * x21 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = x13 * x23 * (x14 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] =
          (16.0 / 3.0) * x21 * x23 * (x10 * x24 + x11 * x7 + x12 + x24 * x9);
      break;
    case 98:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (4096.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = -4096.0 / 27.0 * x22 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 99:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (4096.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = -4096.0 / 27.0 * x22 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 100:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (4096.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = -4096.0 / 27.0 * x22 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 101:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (4096.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = -4096.0 / 27.0 * x22 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 102:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (4096.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = -4096.0 / 27.0 * x22 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 103:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (4096.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = -4096.0 / 27.0 * x22 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 104:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (4096.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = -4096.0 / 27.0 * x22 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 105:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (4096.0 / 27.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      out[0] = -x14 * x22 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          -x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = -4096.0 / 27.0 * x22 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 106:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (1024.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      out[0] = x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] =
          x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = (1024.0 / 9.0) * x22 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 107:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (1024.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      out[0] = x14 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] = (1024.0 / 9.0) * x21 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 108:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (1024.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      out[0] = x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] =
          x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = (1024.0 / 9.0) * x22 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 109:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (1024.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      out[0] = x14 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] = (1024.0 / 9.0) * x21 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 110:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (1024.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      out[0] = x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] =
          x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = (1024.0 / 9.0) * x22 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 111:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (1024.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      out[0] = x14 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] = (1024.0 / 9.0) * x21 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 112:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (1024.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 2 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = 4 * x[1];
      const Scalar x19 = x18 - 1;
      const Scalar x20 = x17 * x19;
      const Scalar x21 = x20 * x[1];
      const Scalar x22 = x15 * x21;
      const Scalar x23 = x2 * x6;
      out[0] = x14 * x22 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] =
          x14 * x23 * (x15 * x16 * x19 + x15 * x17 * x18 + x15 * x20 + x21);
      out[2] = (1024.0 / 9.0) * x22 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 113:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (1024.0 / 9.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      out[0] = x14 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] = (1024.0 / 9.0) * x21 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 114:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (1024.0 / 9.0) * x12 * x7;
      const Scalar x14 = x[1] - 1;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 3;
      const Scalar x19 = x16 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x14 * x20;
      const Scalar x22 = x2 * x6;
      const Scalar x23 = x7 * x8;
      out[0] = x13 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          x13 * x22 * (x14 * x15 * x18 + x14 * x16 * x17 + x14 * x19 + x20);
      out[2] =
          (1024.0 / 9.0) * x21 * x22 * (x10 * x23 + x11 * x7 + x12 + x23 * x9);
      break;
    case 115:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (1024.0 / 9.0) * x12 * x7;
      const Scalar x14 = x[1] - 1;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 3;
      const Scalar x19 = x16 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x14 * x20;
      const Scalar x22 = x2 * x6;
      const Scalar x23 = x7 * x8;
      out[0] = x13 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          x13 * x22 * (x14 * x15 * x18 + x14 * x16 * x17 + x14 * x19 + x20);
      out[2] =
          (1024.0 / 9.0) * x21 * x22 * (x10 * x23 + x11 * x7 + x12 + x23 * x9);
      break;
    case 116:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (1024.0 / 9.0) * x12 * x7;
      const Scalar x14 = x[1] - 1;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x16 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x14 * x20;
      const Scalar x22 = x2 * x6;
      const Scalar x23 = x7 * x8;
      out[0] = x13 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          x13 * x22 * (x14 * x15 * x18 + x14 * x16 * x17 + x14 * x19 + x20);
      out[2] =
          (1024.0 / 9.0) * x21 * x22 * (x10 * x23 + x11 * x7 + x12 + x23 * x9);
      break;
    case 117:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (1024.0 / 9.0) * x12 * x7;
      const Scalar x14 = x[1] - 1;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x16 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x14 * x20;
      const Scalar x22 = x2 * x6;
      const Scalar x23 = x7 * x8;
      out[0] = x13 * x21 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] =
          x13 * x22 * (x14 * x15 * x18 + x14 * x16 * x17 + x14 * x19 + x20);
      out[2] =
          (1024.0 / 9.0) * x21 * x22 * (x10 * x23 + x11 * x7 + x12 + x23 * x9);
      break;
    case 118:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 3;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (256.0 / 3.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      out[0] = -x14 * x21 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = -x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] = -256.0 / 3.0 * x21 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 119:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 2 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = 4 * x[2];
      const Scalar x11 = x10 - 1;
      const Scalar x12 = x11 * x9;
      const Scalar x13 = x12 * x[2];
      const Scalar x14 = (256.0 / 3.0) * x13 * x7;
      const Scalar x15 = x[1] - 1;
      const Scalar x16 = 4 * x[1];
      const Scalar x17 = x16 - 1;
      const Scalar x18 = x16 - 3;
      const Scalar x19 = x17 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x15 * x20;
      const Scalar x22 = x15 * x16;
      const Scalar x23 = x2 * x6;
      out[0] = -x14 * x21 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = -x14 * x23 * (x15 * x19 + x17 * x22 + x18 * x22 + x20);
      out[2] = -256.0 / 3.0 * x21 * x23 *
               (x10 * x7 * x9 + x11 * x7 * x8 + x12 * x7 + x13);
      break;
    case 120:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 3;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (256.0 / 3.0) * x12 * x7;
      const Scalar x14 = x[1] - 1;
      const Scalar x15 = 4 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = x15 - 3;
      const Scalar x18 = x16 * x17;
      const Scalar x19 = x18 * x[1];
      const Scalar x20 = x14 * x19;
      const Scalar x21 = x14 * x15;
      const Scalar x22 = x2 * x6;
      const Scalar x23 = x7 * x8;
      out[0] = -x13 * x20 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x13 * x22 * (x14 * x18 + x16 * x21 + x17 * x21 + x19);
      out[2] =
          -256.0 / 3.0 * x20 * x22 * (x10 * x23 + x11 * x7 + x12 + x23 * x9);
      break;
    case 121:
      const Scalar x0 = 2 * x[0];
      const Scalar x1 = x0 - 1;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = 4 * x[0];
      const Scalar x4 = x3 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (256.0 / 3.0) * x12 * x7;
      const Scalar x14 = x[1] - 1;
      const Scalar x15 = 4 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = x15 - 3;
      const Scalar x18 = x16 * x17;
      const Scalar x19 = x18 * x[1];
      const Scalar x20 = x14 * x19;
      const Scalar x21 = x14 * x15;
      const Scalar x22 = x2 * x6;
      const Scalar x23 = x7 * x8;
      out[0] = -x13 * x20 * (x0 * x2 * x4 + x1 * x2 * x3 + x2 * x5 + x6);
      out[1] = -x13 * x22 * (x14 * x18 + x16 * x21 + x17 * x21 + x19);
      out[2] =
          -256.0 / 3.0 * x20 * x22 * (x10 * x23 + x11 * x7 + x12 + x23 * x9);
      break;
    case 122:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (256.0 / 3.0) * x12 * x7;
      const Scalar x14 = x[1] - 1;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 3;
      const Scalar x19 = x16 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x14 * x20;
      const Scalar x22 = x2 * x6;
      const Scalar x23 = x7 * x8;
      out[0] = -x13 * x21 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] =
          -x13 * x22 * (x14 * x15 * x18 + x14 * x16 * x17 + x14 * x19 + x20);
      out[2] =
          -256.0 / 3.0 * x21 * x22 * (x10 * x23 + x11 * x7 + x12 + x23 * x9);
      break;
    case 123:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = (256.0 / 3.0) * x12 * x7;
      const Scalar x14 = x[1] - 1;
      const Scalar x15 = 2 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = 4 * x[1];
      const Scalar x18 = x17 - 1;
      const Scalar x19 = x16 * x18;
      const Scalar x20 = x19 * x[1];
      const Scalar x21 = x14 * x20;
      const Scalar x22 = x2 * x6;
      const Scalar x23 = x7 * x8;
      out[0] = -x13 * x21 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] =
          -x13 * x22 * (x14 * x15 * x18 + x14 * x16 * x17 + x14 * x19 + x20);
      out[2] =
          -256.0 / 3.0 * x21 * x22 * (x10 * x23 + x11 * x7 + x12 + x23 * x9);
      break;
    case 124:
      const Scalar x0 = 4 * x[0];
      const Scalar x1 = x0 - 3;
      const Scalar x2 = x[0] - 1;
      const Scalar x3 = x0 * x2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x4;
      const Scalar x6 = x5 * x[0];
      const Scalar x7 = x[2] - 1;
      const Scalar x8 = 4 * x[2];
      const Scalar x9 = x8 - 1;
      const Scalar x10 = x8 - 3;
      const Scalar x11 = x10 * x9;
      const Scalar x12 = x11 * x[2];
      const Scalar x13 = 64 * x12 * x7;
      const Scalar x14 = x[1] - 1;
      const Scalar x15 = 4 * x[1];
      const Scalar x16 = x15 - 1;
      const Scalar x17 = x15 - 3;
      const Scalar x18 = x16 * x17;
      const Scalar x19 = x18 * x[1];
      const Scalar x20 = x14 * x19;
      const Scalar x21 = x14 * x15;
      const Scalar x22 = x2 * x6;
      const Scalar x23 = x7 * x8;
      out[0] = x13 * x20 * (x1 * x3 + x2 * x5 + x3 * x4 + x6);
      out[1] = x13 * x22 * (x14 * x18 + x16 * x21 + x17 * x21 + x19);
      out[2] = 64 * x20 * x22 * (x10 * x23 + x11 * x7 + x12 + x23 * x9);
      break;
    default:
      break;
    }
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 2 * x[1] - 1;
    const Scalar x1 = 2 * x[2] - 1;
    const Scalar x2 = 4 * x[0];
    const Scalar x3 = x2 - 1;
    const Scalar x4 = 4 * x[2];
    const Scalar x5 = x4 - 3;
    const Scalar x6 = x0 * x1 * x3 * x5 * x[2];
    const Scalar x7 = 2 * x[0] - 1;
    const Scalar x8 = 4 * x[1];
    const Scalar x9 = x8 - 1;
    const Scalar x10 = x7 * x9;
    const Scalar x11 = x[0] - 1;
    const Scalar x12 = x[1] - 1;
    const Scalar x13 = x[2] - 1;
    const Scalar x14 = x11 * x12 * x13;
    const Scalar x15 = x[0] * x[1];
    const Scalar x16 = x14 * x15;
    const Scalar x17 = x10 * x16;
    const Scalar x18 = 4096 * x17;
    const Scalar x19 = x2 - 3;
    const Scalar x20 = x5 * x[2];
    const Scalar x21 = x19 * x20;
    const Scalar x22 = x0 * x1;
    const Scalar x23 = x18 * x22;
    const Scalar x24 = x8 - 3;
    const Scalar x25 = x16 * x24;
    const Scalar x26 = 4096 * x25 * x7;
    const Scalar x27 = x4 - 1;
    const Scalar x28 = x27 * x[2];
    const Scalar x29 = x19 * x28;
    const Scalar x30 = x22 * x29;
    const Scalar x31 = x28 * x3;
    const Scalar x32 = x22 * x26;
    const Scalar x33 = x19 * x6;
    const Scalar x34 = 3072 * x25;
    const Scalar x35 = 3072 * x20;
    const Scalar x36 = x17 * x24;
    const Scalar x37 = x1 * x36;
    const Scalar x38 = x33 * x9;
    const Scalar x39 = 3072 * x37;
    const Scalar x40 = x3 * x30;
    const Scalar x41 = x3 * x9;
    const Scalar x42 = x16 * x41;
    const Scalar x43 = x21 * x7;
    const Scalar x44 = x25 * x27;
    const Scalar x45 = 3072 * x0;
    const Scalar x46 = x25 * x7;
    const Scalar x47 = x27 * x3;
    const Scalar x48 = x0 * x47;
    const Scalar x49 = x35 * x48;
    const Scalar x50 = 2304 * x21;
    const Scalar x51 = x25 * x50;
    const Scalar x52 = x1 * x41;
    const Scalar x53 = x25 * x52;
    const Scalar x54 = x27 * x50;
    const Scalar x55 = x36 * x47;
    const Scalar x56 = x12 * x[0];
    const Scalar x57 = x24 * x[1];
    const Scalar x58 = x56 * x57;
    const Scalar x59 = x11 * x58;
    const Scalar x60 = 256 * x27;
    const Scalar x61 = x59 * x60;
    const Scalar x62 = x56 * x[1];
    const Scalar x63 = x11 * x27;
    const Scalar x64 = 256 * x10;
    const Scalar x65 = x10 * x21;
    const Scalar x66 = x11 * x65;
    const Scalar x67 = x22 * x66;
    const Scalar x68 = 192 * x27;
    const Scalar x69 = x11 * x68;
    const Scalar x70 = x33 * x58;
    const Scalar x71 = 192 * x1;
    const Scalar x72 = x10 * x71;
    const Scalar x73 = x27 * x71;
    const Scalar x74 = 144 * x27;
    const Scalar x75 = x21 * x74;
    const Scalar x76 = x7 * x70;
    const Scalar x77 = x13 * x62;
    const Scalar x78 = x10 * x33;
    const Scalar x79 = 256 * x78;
    const Scalar x80 = x40 * x64;
    const Scalar x81 = x13 * x58;
    const Scalar x82 = x7 * x81;
    const Scalar x83 = x3 * x72;
    const Scalar x84 = x81 * x83;
    const Scalar x85 = 192 * x48;
    const Scalar x86 = x65 * x85;
    const Scalar x87 = x43 * x85;
    const Scalar x88 = x47 * x81;
    const Scalar x89 = 144 * x65;
    const Scalar x90 = x13 * x15;
    const Scalar x91 = x11 * x90;
    const Scalar x92 = x10 * x24;
    const Scalar x93 = x91 * x92;
    const Scalar x94 = 256 * x93;
    const Scalar x95 = x24 * x91;
    const Scalar x96 = 256 * x22;
    const Scalar x97 = x65 * x96;
    const Scalar x98 = x31 * x96;
    const Scalar x99 = 192 * x24;
    const Scalar x100 = x38 * x99;
    const Scalar x101 = x0 * x65;
    const Scalar x102 = x24 * x90;
    const Scalar x103 = x30 * x41 * x99;
    const Scalar x104 = x20 * x85;
    const Scalar x105 = x41 * x95;
    const Scalar x106 = x0 * x75;
    const Scalar x107 = 16 * x27;
    const Scalar x108 = 12 * x1;
    const Scalar x109 = x108 * x47;
    const Scalar x110 = x107 * x11;
    const Scalar x111 = x110 * x6;
    const Scalar x112 = x107 * x67;
    const Scalar x113 = x15 * x24;
    const Scalar x114 = x38 * x63;
    const Scalar x115 = 16 * x78;
    const Scalar x116 = 16 * x40;
    const Scalar x117 = x10 * x116;
    const Scalar x118 = x48 * x65;
    const Scalar x119 = 12 * x118;
    const Scalar x120 = x19 * x5;
    const Scalar x121 = x120 * x22;
    const Scalar x122 = x121 * x27;
    const Scalar x123 = 256 * x122;
    const Scalar x124 = x22 * x47 * x5;
    const Scalar x125 = 256 * x124;
    const Scalar x126 = x121 * x47;
    const Scalar x127 = x14 * x57;
    const Scalar x128 = x127 * x7;
    const Scalar x129 = 256 * x128;
    const Scalar x130 = x14 * x[1];
    const Scalar x131 = x127 * x83;
    const Scalar x132 = x127 * x47;
    const Scalar x133 = x14 * x[0];
    const Scalar x134 = x133 * x24;
    const Scalar x135 = x10 * x134;
    const Scalar x136 = 256 * x135;
    const Scalar x137 = x134 * x41;
    const Scalar x138 = x0 * x1 * x19 * x24 * x27 * x3 * x5 * x7 * x9;
    const Scalar x139 = x138 * x[2];
    const Scalar x140 = 16 * x126;
    const Scalar x141 = x10 * x140;
    const Scalar x142 = x10 * x108 * x120;
    const Scalar x143 = 16 * x93;
    const Scalar x144 = 12 * x122;
    const Scalar x145 = x24 * x56;
    const Scalar x146 = x56 * x92;
    const Scalar x147 = 12 * x145;
    const Scalar x148 = x110 * x12;
    const Scalar x149 = 16 * x33;
    const Scalar x150 = x13 * x146;
    const Scalar x151 = x13 * x57;
    const Scalar x152 = x11 * x151;
    const Scalar x153 = x13 * x138;
    const Scalar x154 = x153 * x[1];
    const Scalar x155 = x11 * x139;
    const Scalar x156 = x14 * x92;
    const Scalar x157 = 16 * x135;
    return (1.0 / 27.0) * coeffs[0] * x138 * x14 -
           1.0 / 27.0 * coeffs[100] * x18 * x6 -
           1.0 / 27.0 * coeffs[101] * x21 * x23 -
           1.0 / 27.0 * coeffs[102] * x26 * x30 -
           1.0 / 27.0 * coeffs[103] * x31 * x32 -
           1.0 / 27.0 * coeffs[104] * x23 * x31 -
           1.0 / 27.0 * coeffs[105] * x23 * x29 +
           (1.0 / 27.0) * coeffs[106] * x33 * x34 +
           (1.0 / 27.0) * coeffs[107] * x3 * x35 * x37 +
           (1024.0 / 9.0) * coeffs[108] * x16 * x38 +
           (1.0 / 27.0) * coeffs[109] * x21 * x39 +
           (1.0 / 27.0) * coeffs[10] * x137 * x144 +
           (1.0 / 27.0) * coeffs[110] * x34 * x40 +
           (1.0 / 27.0) * coeffs[111] * x31 * x39 +
           (1024.0 / 9.0) * coeffs[112] * x30 * x42 +
           (1.0 / 27.0) * coeffs[113] * x29 * x39 +
           (1.0 / 27.0) * coeffs[114] * x43 * x44 * x45 +
           (1.0 / 27.0) * coeffs[115] * x46 * x49 +
           (1.0 / 27.0) * coeffs[116] * x17 * x49 +
           (1.0 / 27.0) * coeffs[117] * x17 * x21 * x27 * x45 -
           1.0 / 27.0 * coeffs[118] * x51 * x52 -
           256.0 / 3.0 * coeffs[119] * x29 * x53 -
           1.0 / 27.0 * coeffs[11] * x140 * x82 -
           1.0 / 27.0 * coeffs[120] * x36 * x54 -
           256.0 / 3.0 * coeffs[121] * x20 * x55 -
           1.0 / 27.0 * coeffs[122] * x48 * x51 -
           1.0 / 27.0 * coeffs[123] * x0 * x42 * x54 +
           64 * coeffs[124] * x21 * x41 * x44 -
           1.0 / 27.0 * coeffs[12] * x141 * x77 +
           (1.0 / 27.0) * coeffs[13] * x142 * x88 -
           1.0 / 27.0 * coeffs[14] * x124 * x143 -
           1.0 / 27.0 * coeffs[15] * x122 * x143 +
           (1.0 / 27.0) * coeffs[16] * x105 * x144 -
           1.0 / 27.0 * coeffs[17] * x130 * x141 -
           1.0 / 27.0 * coeffs[18] * x128 * x140 +
           (1.0 / 27.0) * coeffs[19] * x132 * x142 +
           (1.0 / 27.0) * coeffs[1] * x153 * x56 -
           1.0 / 27.0 * coeffs[20] * x112 * x145 -
           1.0 / 27.0 * coeffs[21] * x111 * x146 +
           (1.0 / 27.0) * coeffs[22] * x114 * x147 -
           1.0 / 27.0 * coeffs[23] * x107 * x76 -
           1.0 / 27.0 * coeffs[24] * x107 * x62 * x78 +
           (1.0 / 27.0) * coeffs[25] * x109 * x58 * x65 -
           1.0 / 27.0 * coeffs[26] * x111 * x15 * x92 -
           1.0 / 27.0 * coeffs[27] * x112 * x113 +
           (4.0 / 9.0) * coeffs[28] * x113 * x114 -
           1.0 / 27.0 * coeffs[29] * x148 * x78 * x[1] +
           (1.0 / 27.0) * coeffs[2] * x154 * x[0] -
           1.0 / 27.0 * coeffs[30] * x148 * x33 * x57 * x7 +
           (1.0 / 27.0) * coeffs[31] * x109 * x12 * x57 * x66 -
           1.0 / 27.0 * coeffs[32] * x149 * x156 -
           1.0 / 27.0 * coeffs[33] * x116 * x156 +
           (1.0 / 27.0) * coeffs[34] * x119 * x14 * x24 -
           1.0 / 27.0 * coeffs[35] * x149 * x150 -
           1.0 / 27.0 * coeffs[36] * x116 * x150 +
           (1.0 / 27.0) * coeffs[37] * x118 * x13 * x147 -
           1.0 / 27.0 * coeffs[38] * x102 * x115 -
           1.0 / 27.0 * coeffs[39] * x102 * x117 +
           (1.0 / 27.0) * coeffs[3] * x11 * x154 +
           (1.0 / 27.0) * coeffs[40] * x102 * x119 -
           1.0 / 27.0 * coeffs[41] * x115 * x152 -
           1.0 / 27.0 * coeffs[42] * x117 * x152 +
           (4.0 / 9.0) * coeffs[43] * x151 * x48 * x66 +
           (1.0 / 27.0) * coeffs[44] * x123 * x46 +
           (1.0 / 27.0) * coeffs[45] * x123 * x17 +
           (1.0 / 27.0) * coeffs[46] * x125 * x17 +
           (1.0 / 27.0) * coeffs[47] * x125 * x46 -
           1.0 / 27.0 * coeffs[48] * x120 * x36 * x73 -
           1.0 / 27.0 * coeffs[49] * x121 * x42 * x68 +
           (1.0 / 27.0) * coeffs[4] * x12 * x155 -
           1.0 / 27.0 * coeffs[50] * x5 * x55 * x71 -
           64.0 / 9.0 * coeffs[51] * x126 * x25 +
           (1.0 / 27.0) * coeffs[52] * x120 * x53 * x74 +
           (1.0 / 27.0) * coeffs[53] * x22 * x43 * x61 +
           (1.0 / 27.0) * coeffs[54] * x6 * x61 * x7 +
           (1.0 / 27.0) * coeffs[55] * x6 * x62 * x63 * x64 +
           (1.0 / 27.0) * coeffs[56] * x60 * x62 * x67 -
           1.0 / 27.0 * coeffs[57] * x69 * x70 -
           1.0 / 27.0 * coeffs[58] * x20 * x47 * x59 * x72 -
           1.0 / 27.0 * coeffs[59] * x38 * x62 * x69 +
           (1.0 / 27.0) * coeffs[5] * x139 * x56 -
           1.0 / 27.0 * coeffs[60] * x59 * x65 * x73 +
           (1.0 / 27.0) * coeffs[61] * x52 * x59 * x75 +
           (1.0 / 27.0) * coeffs[62] * x129 * x33 +
           (1.0 / 27.0) * coeffs[63] * x129 * x40 +
           (1.0 / 27.0) * coeffs[64] * x130 * x80 +
           (1.0 / 27.0) * coeffs[65] * x130 * x79 -
           1.0 / 27.0 * coeffs[66] * x127 * x87 -
           1.0 / 27.0 * coeffs[67] * x131 * x29 -
           1.0 / 27.0 * coeffs[68] * x130 * x86 -
           1.0 / 27.0 * coeffs[69] * x131 * x21 +
           (1.0 / 27.0) * coeffs[6] * x139 * x15 +
           (1.0 / 27.0) * coeffs[70] * x132 * x89 +
           (256.0 / 27.0) * coeffs[71] * x13 * x76 +
           (1.0 / 27.0) * coeffs[72] * x77 * x79 +
           (1.0 / 27.0) * coeffs[73] * x77 * x80 +
           (256.0 / 27.0) * coeffs[74] * x40 * x82 -
           1.0 / 27.0 * coeffs[75] * x21 * x84 -
           1.0 / 27.0 * coeffs[76] * x77 * x86 -
           1.0 / 27.0 * coeffs[77] * x29 * x84 -
           1.0 / 27.0 * coeffs[78] * x81 * x87 +
           (1.0 / 27.0) * coeffs[79] * x88 * x89 +
           (1.0 / 27.0) * coeffs[7] * x155 * x[1] +
           (1.0 / 27.0) * coeffs[80] * x134 * x97 +
           (1.0 / 27.0) * coeffs[81] * x136 * x6 +
           (1.0 / 27.0) * coeffs[82] * x135 * x98 +
           (1.0 / 27.0) * coeffs[83] * x136 * x30 -
           1.0 / 27.0 * coeffs[84] * x100 * x133 -
           1.0 / 27.0 * coeffs[85] * x104 * x135 -
           1.0 / 27.0 * coeffs[86] * x103 * x133 -
           1.0 / 27.0 * coeffs[87] * x101 * x134 * x68 +
           (1.0 / 27.0) * coeffs[88] * x106 * x137 +
           (1.0 / 27.0) * coeffs[89] * x6 * x94 -
           1.0 / 27.0 * coeffs[8] * x122 * x157 +
           (1.0 / 27.0) * coeffs[90] * x95 * x97 +
           (1.0 / 27.0) * coeffs[91] * x30 * x94 +
           (1.0 / 27.0) * coeffs[92] * x93 * x98 -
           1.0 / 27.0 * coeffs[93] * x100 * x91 -
           1.0 / 27.0 * coeffs[94] * x101 * x102 * x69 -
           1.0 / 27.0 * coeffs[95] * x103 * x91 -
           1.0 / 27.0 * coeffs[96] * x104 * x93 +
           (1.0 / 27.0) * coeffs[97] * x105 * x106 -
           1.0 / 27.0 * coeffs[98] * x21 * x32 -
           1.0 / 27.0 * coeffs[99] * x26 * x6 -
           1.0 / 27.0 * coeffs[9] * x124 * x157;
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
    const Scalar x6 = x3 - 1;
    const Scalar x7 = x0 * x2;
    const Scalar x8 = x1 * x6;
    const Scalar x9 = x8 * x[0];
    const Scalar x10 = x2 * x8;
    const Scalar x11 = x10 + x5 + x6 * x7 + x9;
    const Scalar x12 = 2 * x[1];
    const Scalar x13 = x12 - 1;
    const Scalar x14 = 4 * x[1];
    const Scalar x15 = x14 - 1;
    const Scalar x16 = x13 * x15;
    const Scalar x17 = x16 * x[1];
    const Scalar x18 = x11 * x17;
    const Scalar x19 = 2 * x[2];
    const Scalar x20 = x19 - 1;
    const Scalar x21 = 4 * x[2];
    const Scalar x22 = x21 - 3;
    const Scalar x23 = x20 * x22;
    const Scalar x24 = x23 * x[2];
    const Scalar x25 = x[1] - 1;
    const Scalar x26 = x[2] - 1;
    const Scalar x27 = x25 * x26;
    const Scalar x28 = 4096 * x27;
    const Scalar x29 = x24 * x28;
    const Scalar x30 = x3 - 3;
    const Scalar x31 = x1 * x30;
    const Scalar x32 = x31 * x[0];
    const Scalar x33 = x2 * x31;
    const Scalar x34 = x30 * x7 + x32 + x33 + x5;
    const Scalar x35 = x17 * x34;
    const Scalar x36 = x21 - 1;
    const Scalar x37 = x20 * x36;
    const Scalar x38 = x37 * x[2];
    const Scalar x39 = x28 * x38;
    const Scalar x40 = x14 - 3;
    const Scalar x41 = x13 * x40;
    const Scalar x42 = x41 * x[1];
    const Scalar x43 = x34 * x42;
    const Scalar x44 = x11 * x42;
    const Scalar x45 = x30 * x6;
    const Scalar x46 = x45 * x[0];
    const Scalar x47 = x2 * x45;
    const Scalar x48 = x30 * x4 + x4 * x6 + x46 + x47;
    const Scalar x49 = x24 * x48;
    const Scalar x50 = x27 * x42;
    const Scalar x51 = 3072 * x50;
    const Scalar x52 = x15 * x40;
    const Scalar x53 = x52 * x[1];
    const Scalar x54 = x27 * x53;
    const Scalar x55 = x24 * x54;
    const Scalar x56 = 3072 * x11;
    const Scalar x57 = x27 * x49;
    const Scalar x58 = 3072 * coeffs[108];
    const Scalar x59 = x17 * x58;
    const Scalar x60 = x34 * x54;
    const Scalar x61 = 3072 * x60;
    const Scalar x62 = coeffs[109] * x24;
    const Scalar x63 = x38 * x48;
    const Scalar x64 = x38 * x54;
    const Scalar x65 = x17 * x27;
    const Scalar x66 = 3072 * coeffs[112];
    const Scalar x67 = coeffs[113] * x38;
    const Scalar x68 = x22 * x36;
    const Scalar x69 = x68 * x[2];
    const Scalar x70 = x27 * x69;
    const Scalar x71 = 3072 * x70;
    const Scalar x72 = 2304 * x54;
    const Scalar x73 = x69 * x72;
    const Scalar x74 = x27 * x48;
    const Scalar x75 = x42 * x74;
    const Scalar x76 = 2304 * x69;
    const Scalar x77 = coeffs[122] * x76;
    const Scalar x78 = x17 * x74;
    const Scalar x79 = coeffs[123] * x76;
    const Scalar x80 = 1728 * coeffs[124];
    const Scalar x81 = x69 * x80;
    const Scalar x82 = x25 * x[2];
    const Scalar x83 = x23 * x36;
    const Scalar x84 = 256 * x83;
    const Scalar x85 = x82 * x84;
    const Scalar x86 = coeffs[57] * x42;
    const Scalar x87 = 192 * x83;
    const Scalar x88 = x82 * x87;
    const Scalar x89 = x48 * x88;
    const Scalar x90 = x11 * x87;
    const Scalar x91 = x53 * x82;
    const Scalar x92 = coeffs[59] * x17;
    const Scalar x93 = coeffs[60] * x53;
    const Scalar x94 = x48 * x83;
    const Scalar x95 = 144 * coeffs[61];
    const Scalar x96 = x26 * x[1];
    const Scalar x97 = x11 * x96;
    const Scalar x98 = x16 * x40;
    const Scalar x99 = 256 * x98;
    const Scalar x100 = x24 * x99;
    const Scalar x101 = x34 * x96;
    const Scalar x102 = x38 * x99;
    const Scalar x103 = 192 * x98;
    const Scalar x104 = x103 * x96;
    const Scalar x105 = x104 * x69;
    const Scalar x106 = x103 * x63;
    const Scalar x107 = x69 * x96;
    const Scalar x108 = 144 * x98;
    const Scalar x109 = coeffs[97] * x108;
    const Scalar x110 = x[1] * x[2];
    const Scalar x111 = x83 * x98;
    const Scalar x112 = 16 * x111;
    const Scalar x113 = x11 * x112;
    const Scalar x114 = x112 * x34;
    const Scalar x115 = 12 * x111;
    const Scalar x116 = x115 * x48;
    const Scalar x117 = x27 * x84;
    const Scalar x118 = coeffs[48] * x87;
    const Scalar x119 = coeffs[49] * x87;
    const Scalar x120 = coeffs[51] * x87;
    const Scalar x121 = 144 * x54;
    const Scalar x122 = coeffs[80] * x24;
    const Scalar x123 = x27 * x99;
    const Scalar x124 = x123 * x34;
    const Scalar x125 = x11 * x123;
    const Scalar x126 = coeffs[81] * x24;
    const Scalar x127 = coeffs[82] * x38;
    const Scalar x128 = coeffs[83] * x38;
    const Scalar x129 = coeffs[84] * x103;
    const Scalar x130 = x103 * x70;
    const Scalar x131 = 144 * x69;
    const Scalar x132 = coeffs[88] * x131;
    const Scalar x133 = x30 * x8;
    const Scalar x134 = x0 * x45 + x133 + x3 * x31 + x3 * x8;
    const Scalar x135 = x134 * x50;
    const Scalar x136 = 256 * x135;
    const Scalar x137 = coeffs[71] * x24;
    const Scalar x138 = x134 * x24;
    const Scalar x139 = 256 * x65;
    const Scalar x140 = coeffs[73] * x38;
    const Scalar x141 = coeffs[74] * x38;
    const Scalar x142 = 192 * x134;
    const Scalar x143 = coeffs[76] * x17;
    const Scalar x144 = coeffs[78] * x69;
    const Scalar x145 = x134 * x54;
    const Scalar x146 = coeffs[79] * x131;
    const Scalar x147 = 16 * x83;
    const Scalar x148 = x134 * x147;
    const Scalar x149 = x148 * x82;
    const Scalar x150 = coeffs[23] * x42;
    const Scalar x151 = coeffs[24] * x17;
    const Scalar x152 = 12 * x83;
    const Scalar x153 = 16 * x98;
    const Scalar x154 = x153 * x96;
    const Scalar x155 = x154 * x38;
    const Scalar x156 = 12 * x98;
    const Scalar x157 = x107 * x156;
    const Scalar x158 = x111 * x134;
    const Scalar x159 = x153 * x27;
    const Scalar x160 = x159 * x38;
    const Scalar x161 = x156 * x70;
    const Scalar x162 = 4 * x10 + x133 + 4 * x33 + 2 * x47;
    const Scalar x163 = x162 * x24;
    const Scalar x164 = 256 * x50;
    const Scalar x165 = x162 * x38;
    const Scalar x166 = 192 * x162;
    const Scalar x167 = coeffs[66] * x69;
    const Scalar x168 = x166 * x54;
    const Scalar x169 = coeffs[67] * x38;
    const Scalar x170 = coeffs[68] * x17;
    const Scalar x171 = coeffs[69] * x24;
    const Scalar x172 = coeffs[70] * x69;
    const Scalar x173 = x147 * x162;
    const Scalar x174 = x173 * x82;
    const Scalar x175 = coeffs[29] * x17;
    const Scalar x176 = coeffs[30] * x42;
    const Scalar x177 = x152 * x162;
    const Scalar x178 = x111 * x162;
    const Scalar x179 = x14 * x25;
    const Scalar x180 = x13 * x179;
    const Scalar x181 = x12 * x25;
    const Scalar x182 = x16 * x25;
    const Scalar x183 = x15 * x181 + x17 + x180 + x182;
    const Scalar x184 = x183 * x9;
    const Scalar x185 = x2 * x26;
    const Scalar x186 = 4096 * x185;
    const Scalar x187 = x186 * x24;
    const Scalar x188 = x183 * x32;
    const Scalar x189 = x186 * x38;
    const Scalar x190 = x25 * x41;
    const Scalar x191 = x180 + x181 * x40 + x190 + x42;
    const Scalar x192 = x191 * x32;
    const Scalar x193 = x191 * x9;
    const Scalar x194 = x185 * x46;
    const Scalar x195 = x191 * x194;
    const Scalar x196 = 3072 * x195;
    const Scalar x197 = x25 * x52;
    const Scalar x198 = x15 * x179 + x179 * x40 + x197 + x53;
    const Scalar x199 = x185 * x198;
    const Scalar x200 = x199 * x24;
    const Scalar x201 = 3072 * x9;
    const Scalar x202 = coeffs[107] * x201;
    const Scalar x203 = x183 * x194;
    const Scalar x204 = x199 * x32;
    const Scalar x205 = 3072 * x204;
    const Scalar x206 = x199 * x38;
    const Scalar x207 = coeffs[111] * x201;
    const Scalar x208 = x185 * x69;
    const Scalar x209 = 3072 * x208;
    const Scalar x210 = 2304 * x46;
    const Scalar x211 = coeffs[118] * x210;
    const Scalar x212 = coeffs[119] * x210;
    const Scalar x213 = x199 * x76;
    const Scalar x214 = coeffs[120] * x32;
    const Scalar x215 = coeffs[121] * x9;
    const Scalar x216 = x2 * x[2];
    const Scalar x217 = x216 * x84;
    const Scalar x218 = x216 * x87;
    const Scalar x219 = x218 * x46;
    const Scalar x220 = x87 * x9;
    const Scalar x221 = x198 * x216;
    const Scalar x222 = x46 * x83;
    const Scalar x223 = x26 * x[0];
    const Scalar x224 = x191 * x223;
    const Scalar x225 = 256 * x133;
    const Scalar x226 = x224 * x225;
    const Scalar x227 = x183 * x225;
    const Scalar x228 = x223 * x227;
    const Scalar x229 = 192 * x133;
    const Scalar x230 = x223 * x229;
    const Scalar x231 = x198 * x230;
    const Scalar x232 = x183 * x229;
    const Scalar x233 = x223 * x69;
    const Scalar x234 = x198 * x223;
    const Scalar x235 = x[0] * x[2];
    const Scalar x236 = x133 * x83;
    const Scalar x237 = 16 * x236;
    const Scalar x238 = x191 * x237;
    const Scalar x239 = x183 * x237;
    const Scalar x240 = 12 * x236;
    const Scalar x241 = x185 * x84;
    const Scalar x242 = 144 * x199;
    const Scalar x243 = x185 * x191;
    const Scalar x244 = x225 * x243;
    const Scalar x245 = x185 * x227;
    const Scalar x246 = x199 * x229;
    const Scalar x247 = coeffs[89] * x9;
    const Scalar x248 = x12 * x52 + x14 * x16 + x14 * x41 + x98;
    const Scalar x249 = x185 * x248;
    const Scalar x250 = 256 * x249;
    const Scalar x251 = x24 * x250;
    const Scalar x252 = coeffs[90] * x32;
    const Scalar x253 = x250 * x38;
    const Scalar x254 = coeffs[91] * x32;
    const Scalar x255 = coeffs[92] * x9;
    const Scalar x256 = 192 * x248;
    const Scalar x257 = x194 * x256;
    const Scalar x258 = x208 * x256;
    const Scalar x259 = coeffs[94] * x32;
    const Scalar x260 = coeffs[96] * x9;
    const Scalar x261 = x147 * x9;
    const Scalar x262 = x216 * x248;
    const Scalar x263 = x147 * x32;
    const Scalar x264 = x152 * x46;
    const Scalar x265 = 16 * x133;
    const Scalar x266 = x24 * x265;
    const Scalar x267 = x223 * x248;
    const Scalar x268 = x265 * x38;
    const Scalar x269 = 12 * x133;
    const Scalar x270 = x248 * x269;
    const Scalar x271 = x236 * x248;
    const Scalar x272 = x271 * x[2];
    const Scalar x273 = coeffs[6] * x[0];
    const Scalar x274 = x152 * x194;
    const Scalar x275 = coeffs[7] * x2;
    const Scalar x276 = 4 * x182 + 4 * x190 + 2 * x197 + x98;
    const Scalar x277 = x185 * x276;
    const Scalar x278 = 256 * x277;
    const Scalar x279 = x278 * x32;
    const Scalar x280 = x278 * x9;
    const Scalar x281 = 192 * x276;
    const Scalar x282 = x194 * x281;
    const Scalar x283 = x208 * x281;
    const Scalar x284 = coeffs[85] * x9;
    const Scalar x285 = coeffs[87] * x32;
    const Scalar x286 = x216 * x276;
    const Scalar x287 = x223 * x276;
    const Scalar x288 = x269 * x276;
    const Scalar x289 = x236 * x276;
    const Scalar x290 = x289 * x[2];
    const Scalar x291 = coeffs[5] * x[0];
    const Scalar x292 = coeffs[4] * x2;
    const Scalar x293 = x21 * x26;
    const Scalar x294 = x20 * x293;
    const Scalar x295 = x19 * x26;
    const Scalar x296 = x23 * x26;
    const Scalar x297 = x22 * x295 + x24 + x294 + x296;
    const Scalar x298 = x17 * x297;
    const Scalar x299 = x2 * x25;
    const Scalar x300 = 4096 * x299;
    const Scalar x301 = x300 * x9;
    const Scalar x302 = x300 * x32;
    const Scalar x303 = x26 * x37;
    const Scalar x304 = x294 + x295 * x36 + x303 + x38;
    const Scalar x305 = x304 * x42;
    const Scalar x306 = x17 * x304;
    const Scalar x307 = x297 * x42;
    const Scalar x308 = x299 * x46;
    const Scalar x309 = 3072 * x308;
    const Scalar x310 = x299 * x53;
    const Scalar x311 = x297 * x310;
    const Scalar x312 = x297 * x308;
    const Scalar x313 = 3072 * x32;
    const Scalar x314 = x304 * x310;
    const Scalar x315 = x26 * x68;
    const Scalar x316 = x22 * x293 + x293 * x36 + x315 + x69;
    const Scalar x317 = x299 * x42;
    const Scalar x318 = x317 * x32;
    const Scalar x319 = x201 * x316;
    const Scalar x320 = x17 * x299;
    const Scalar x321 = x310 * x316;
    const Scalar x322 = 2304 * x321;
    const Scalar x323 = x210 * x316;
    const Scalar x324 = x25 * x[0];
    const Scalar x325 = x225 * x324;
    const Scalar x326 = x297 * x324;
    const Scalar x327 = x229 * x53;
    const Scalar x328 = x229 * x316;
    const Scalar x329 = x324 * x328;
    const Scalar x330 = x304 * x324;
    const Scalar x331 = x324 * x53;
    const Scalar x332 = 144 * x133;
    const Scalar x333 = x2 * x[1];
    const Scalar x334 = x333 * x99;
    const Scalar x335 = x297 * x334;
    const Scalar x336 = x304 * x334;
    const Scalar x337 = x333 * x46;
    const Scalar x338 = x103 * x316;
    const Scalar x339 = x333 * x338;
    const Scalar x340 = x103 * x304;
    const Scalar x341 = x[0] * x[1];
    const Scalar x342 = x133 * x98;
    const Scalar x343 = 16 * x342;
    const Scalar x344 = x297 * x343;
    const Scalar x345 = x304 * x343;
    const Scalar x346 = 12 * x316 * x342;
    const Scalar x347 = x225 * x299;
    const Scalar x348 = x299 * x32;
    const Scalar x349 = x348 * x99;
    const Scalar x350 = x297 * x299;
    const Scalar x351 = x9 * x99;
    const Scalar x352 = x299 * x338;
    const Scalar x353 = x19 * x68 + x21 * x23 + x21 * x37 + x83;
    const Scalar x354 = 256 * x353;
    const Scalar x355 = x317 * x9;
    const Scalar x356 = x320 * x354;
    const Scalar x357 = 192 * x353;
    const Scalar x358 = x308 * x357;
    const Scalar x359 = x310 * x46;
    const Scalar x360 = x265 * x353;
    const Scalar x361 = x324 * x360;
    const Scalar x362 = x269 * x353;
    const Scalar x363 = x153 * x9;
    const Scalar x364 = x353 * x363;
    const Scalar x365 = x153 * x32;
    const Scalar x366 = x156 * x353;
    const Scalar x367 = x342 * x353;
    const Scalar x368 = x367 * x[1];
    const Scalar x369 = x153 * x348;
    const Scalar x370 = x299 * x360;
    const Scalar x371 = x25 * x367;
    const Scalar x372 = 4 * x296 + 4 * x303 + 2 * x315 + x83;
    const Scalar x373 = 256 * x372;
    const Scalar x374 = x320 * x373;
    const Scalar x375 = 192 * x372;
    const Scalar x376 = x310 * x375;
    const Scalar x377 = x375 * x46;
    const Scalar x378 = x265 * x372;
    const Scalar x379 = x324 * x378;
    const Scalar x380 = x269 * x372;
    const Scalar x381 = x333 * x372;
    const Scalar x382 = x156 * x372;
    const Scalar x383 = x342 * x372;
    const Scalar x384 = x383 * x[0];
    out[0] = (1.0 / 27.0) * coeffs[0] * x178 * x27 -
             1.0 / 27.0 * coeffs[100] * x18 * x29 -
             1.0 / 27.0 * coeffs[101] * x29 * x35 -
             1.0 / 27.0 * coeffs[102] * x39 * x43 -
             1.0 / 27.0 * coeffs[103] * x39 * x44 -
             1.0 / 27.0 * coeffs[104] * x18 * x39 -
             1.0 / 27.0 * coeffs[105] * x35 * x39 +
             (1.0 / 27.0) * coeffs[106] * x49 * x51 +
             (1.0 / 27.0) * coeffs[107] * x55 * x56 +
             (1.0 / 27.0) * coeffs[10] * x115 * x74 +
             (1.0 / 27.0) * coeffs[110] * x51 * x63 +
             (1.0 / 27.0) * coeffs[111] * x56 * x64 +
             (1.0 / 27.0) * coeffs[114] * x43 * x71 +
             (1.0 / 27.0) * coeffs[115] * x44 * x71 +
             (1.0 / 27.0) * coeffs[116] * x18 * x71 +
             (1.0 / 27.0) * coeffs[117] * x35 * x71 -
             1.0 / 27.0 * coeffs[118] * x49 * x72 -
             1.0 / 27.0 * coeffs[119] * x63 * x72 -
             1.0 / 27.0 * coeffs[11] * x135 * x147 -
             1.0 / 27.0 * coeffs[120] * x34 * x73 -
             1.0 / 27.0 * coeffs[121] * x11 * x73 -
             1.0 / 27.0 * coeffs[12] * x148 * x65 +
             (1.0 / 27.0) * coeffs[13] * x145 * x152 -
             1.0 / 27.0 * coeffs[14] * x112 * x97 -
             1.0 / 27.0 * coeffs[15] * x114 * x96 +
             (1.0 / 27.0) * coeffs[16] * x116 * x96 -
             1.0 / 27.0 * coeffs[17] * x173 * x65 -
             1.0 / 27.0 * coeffs[18] * x173 * x50 +
             (1.0 / 27.0) * coeffs[19] * x177 * x54 +
             (1.0 / 27.0) * coeffs[1] * x158 * x27 -
             1.0 / 27.0 * coeffs[20] * x114 * x82 -
             1.0 / 27.0 * coeffs[21] * x113 * x82 +
             (1.0 / 27.0) * coeffs[22] * x116 * x82 +
             (1.0 / 27.0) * coeffs[25] * x134 * x152 * x91 -
             1.0 / 27.0 * coeffs[26] * x110 * x113 -
             1.0 / 27.0 * coeffs[27] * x110 * x114 +
             (1.0 / 27.0) * coeffs[28] * x110 * x116 +
             (1.0 / 27.0) * coeffs[2] * x158 * x96 +
             (1.0 / 27.0) * coeffs[31] * x177 * x91 -
             1.0 / 27.0 * coeffs[32] * x159 * x163 -
             1.0 / 27.0 * coeffs[33] * x160 * x162 +
             (1.0 / 27.0) * coeffs[34] * x161 * x162 -
             1.0 / 27.0 * coeffs[35] * x138 * x159 -
             1.0 / 27.0 * coeffs[36] * x134 * x160 +
             (1.0 / 27.0) * coeffs[37] * x134 * x161 -
             1.0 / 27.0 * coeffs[38] * x138 * x154 -
             1.0 / 27.0 * coeffs[39] * x134 * x155 +
             (1.0 / 27.0) * coeffs[3] * x178 * x96 +
             (1.0 / 27.0) * coeffs[40] * x134 * x157 -
             1.0 / 27.0 * coeffs[41] * x154 * x163 -
             1.0 / 27.0 * coeffs[42] * x155 * x162 +
             (1.0 / 27.0) * coeffs[43] * x157 * x162 +
             (1.0 / 27.0) * coeffs[44] * x117 * x43 +
             (1.0 / 27.0) * coeffs[45] * x117 * x35 +
             (1.0 / 27.0) * coeffs[46] * x117 * x18 +
             (1.0 / 27.0) * coeffs[47] * x117 * x44 +
             (1.0 / 27.0) * coeffs[4] * x178 * x82 -
             1.0 / 27.0 * coeffs[50] * x54 * x90 +
             (1.0 / 27.0) * coeffs[52] * x121 * x94 +
             (1.0 / 27.0) * coeffs[53] * x43 * x85 +
             (1.0 / 27.0) * coeffs[54] * x44 * x85 +
             (1.0 / 27.0) * coeffs[55] * x18 * x85 +
             (1.0 / 27.0) * coeffs[56] * x35 * x85 -
             1.0 / 27.0 * coeffs[58] * x90 * x91 +
             (1.0 / 27.0) * coeffs[5] * x158 * x82 +
             (1.0 / 27.0) * coeffs[62] * x163 * x164 +
             (1.0 / 27.0) * coeffs[63] * x164 * x165 +
             (1.0 / 27.0) * coeffs[64] * x139 * x165 +
             (1.0 / 27.0) * coeffs[65] * x139 * x163 +
             (1.0 / 27.0) * coeffs[6] * x110 * x158 +
             (1.0 / 27.0) * coeffs[72] * x138 * x139 -
             1.0 / 27.0 * coeffs[75] * x142 * x55 -
             1.0 / 27.0 * coeffs[77] * x142 * x64 +
             (1.0 / 27.0) * coeffs[7] * x110 * x178 -
             1.0 / 27.0 * coeffs[85] * x11 * x130 -
             1.0 / 27.0 * coeffs[86] * x106 * x27 -
             1.0 / 27.0 * coeffs[87] * x130 * x34 +
             (1.0 / 27.0) * coeffs[89] * x100 * x97 -
             1.0 / 27.0 * coeffs[8] * x114 * x27 +
             (1.0 / 27.0) * coeffs[90] * x100 * x101 +
             (1.0 / 27.0) * coeffs[91] * x101 * x102 +
             (1.0 / 27.0) * coeffs[92] * x102 * x97 -
             1.0 / 27.0 * coeffs[93] * x104 * x49 -
             1.0 / 27.0 * coeffs[94] * x105 * x34 -
             1.0 / 27.0 * coeffs[95] * x106 * x96 -
             1.0 / 27.0 * coeffs[96] * x105 * x11 -
             1.0 / 27.0 * coeffs[98] * x29 * x43 -
             1.0 / 27.0 * coeffs[99] * x29 * x44 -
             1.0 / 27.0 * coeffs[9] * x113 * x27 +
             (1.0 / 27.0) * x107 * x109 * x48 - 1.0 / 27.0 * x118 * x60 -
             1.0 / 27.0 * x119 * x78 - 1.0 / 27.0 * x120 * x75 +
             (1.0 / 27.0) * x121 * x162 * x172 + (1.0 / 27.0) * x122 * x124 +
             (1.0 / 27.0) * x124 * x128 + (1.0 / 27.0) * x125 * x126 +
             (1.0 / 27.0) * x125 * x127 - 1.0 / 27.0 * x129 * x57 +
             (1.0 / 27.0) * x132 * x74 * x98 +
             (1.0 / 27.0) * x134 * x139 * x140 + (1.0 / 27.0) * x136 * x137 +
             (1.0 / 27.0) * x136 * x141 - 1.0 / 27.0 * x142 * x143 * x70 -
             1.0 / 27.0 * x142 * x144 * x50 + (1.0 / 27.0) * x145 * x146 -
             1.0 / 27.0 * x149 * x150 - 1.0 / 27.0 * x149 * x151 -
             1.0 / 27.0 * x166 * x167 * x50 - 1.0 / 27.0 * x166 * x170 * x70 -
             1.0 / 27.0 * x168 * x169 - 1.0 / 27.0 * x168 * x171 -
             1.0 / 27.0 * x174 * x175 - 1.0 / 27.0 * x174 * x176 -
             1.0 / 27.0 * x34 * x88 * x93 + (1.0 / 27.0) * x48 * x54 * x81 +
             (1.0 / 27.0) * x57 * x59 + (1.0 / 27.0) * x61 * x62 +
             (1.0 / 27.0) * x61 * x67 + (1.0 / 27.0) * x63 * x65 * x66 -
             1.0 / 27.0 * x75 * x77 - 1.0 / 27.0 * x78 * x79 -
             1.0 / 27.0 * x86 * x89 - 1.0 / 27.0 * x89 * x92 +
             (1.0 / 27.0) * x91 * x94 * x95;
    out[1] = (1.0 / 27.0) * coeffs[0] * x185 * x289 -
             1.0 / 27.0 * coeffs[100] * x184 * x187 -
             1.0 / 27.0 * coeffs[101] * x187 * x188 -
             1.0 / 27.0 * coeffs[102] * x189 * x192 -
             1.0 / 27.0 * coeffs[103] * x189 * x193 -
             1.0 / 27.0 * coeffs[104] * x184 * x189 -
             1.0 / 27.0 * coeffs[105] * x188 * x189 +
             (1.0 / 27.0) * coeffs[106] * x196 * x24 +
             (1.0 / 27.0) * coeffs[10] * x274 * x276 +
             (1.0 / 27.0) * coeffs[110] * x196 * x38 +
             (1.0 / 27.0) * coeffs[114] * x192 * x209 +
             (1.0 / 27.0) * coeffs[115] * x193 * x209 +
             (1.0 / 27.0) * coeffs[116] * x184 * x209 +
             (1.0 / 27.0) * coeffs[117] * x188 * x209 -
             1.0 / 27.0 * coeffs[11] * x224 * x237 -
             1.0 / 27.0 * coeffs[12] * x223 * x239 +
             (1.0 / 27.0) * coeffs[13] * x234 * x240 -
             1.0 / 27.0 * coeffs[14] * x249 * x261 -
             1.0 / 27.0 * coeffs[15] * x249 * x263 +
             (1.0 / 27.0) * coeffs[16] * x248 * x274 -
             1.0 / 27.0 * coeffs[17] * x185 * x239 -
             1.0 / 27.0 * coeffs[18] * x237 * x243 +
             (1.0 / 27.0) * coeffs[19] * x199 * x240 +
             (1.0 / 27.0) * coeffs[1] * x223 * x289 -
             1.0 / 27.0 * coeffs[20] * x263 * x286 -
             1.0 / 27.0 * coeffs[21] * x261 * x286 +
             (1.0 / 27.0) * coeffs[22] * x264 * x286 -
             1.0 / 27.0 * coeffs[23] * x235 * x238 -
             1.0 / 27.0 * coeffs[24] * x235 * x239 +
             (1.0 / 27.0) * coeffs[25] * x198 * x235 * x240 -
             1.0 / 27.0 * coeffs[26] * x261 * x262 -
             1.0 / 27.0 * coeffs[27] * x262 * x263 +
             (1.0 / 27.0) * coeffs[28] * x262 * x264 -
             1.0 / 27.0 * coeffs[29] * x216 * x239 +
             (1.0 / 27.0) * coeffs[2] * x223 * x271 -
             1.0 / 27.0 * coeffs[30] * x216 * x238 +
             (1.0 / 27.0) * coeffs[31] * x221 * x240 -
             1.0 / 27.0 * coeffs[32] * x266 * x277 -
             1.0 / 27.0 * coeffs[33] * x268 * x277 +
             (1.0 / 27.0) * coeffs[34] * x208 * x288 -
             1.0 / 27.0 * coeffs[35] * x266 * x287 -
             1.0 / 27.0 * coeffs[36] * x268 * x287 +
             (1.0 / 27.0) * coeffs[37] * x233 * x288 -
             1.0 / 27.0 * coeffs[38] * x266 * x267 -
             1.0 / 27.0 * coeffs[39] * x267 * x268 +
             (1.0 / 27.0) * coeffs[3] * x185 * x271 +
             (1.0 / 27.0) * coeffs[40] * x233 * x270 -
             1.0 / 27.0 * coeffs[41] * x249 * x266 -
             1.0 / 27.0 * coeffs[42] * x249 * x268 +
             (1.0 / 27.0) * coeffs[43] * x208 * x270 +
             (1.0 / 27.0) * coeffs[44] * x192 * x241 +
             (1.0 / 27.0) * coeffs[45] * x188 * x241 +
             (1.0 / 27.0) * coeffs[46] * x184 * x241 +
             (1.0 / 27.0) * coeffs[47] * x193 * x241 -
             1.0 / 27.0 * coeffs[50] * x199 * x220 +
             (1.0 / 27.0) * coeffs[52] * x222 * x242 +
             (1.0 / 27.0) * coeffs[53] * x192 * x217 +
             (1.0 / 27.0) * coeffs[54] * x193 * x217 +
             (1.0 / 27.0) * coeffs[55] * x184 * x217 +
             (1.0 / 27.0) * coeffs[56] * x188 * x217 -
             1.0 / 27.0 * coeffs[57] * x191 * x219 -
             1.0 / 27.0 * coeffs[58] * x220 * x221 -
             1.0 / 27.0 * coeffs[59] * x183 * x219 -
             1.0 / 27.0 * coeffs[60] * x198 * x218 * x32 +
             (1.0 / 27.0) * coeffs[62] * x24 * x244 +
             (1.0 / 27.0) * coeffs[63] * x244 * x38 +
             (1.0 / 27.0) * coeffs[64] * x245 * x38 +
             (1.0 / 27.0) * coeffs[65] * x24 * x245 -
             1.0 / 27.0 * coeffs[68] * x208 * x232 +
             (1.0 / 27.0) * coeffs[72] * x228 * x24 -
             1.0 / 27.0 * coeffs[75] * x231 * x24 -
             1.0 / 27.0 * coeffs[76] * x232 * x233 -
             1.0 / 27.0 * coeffs[77] * x231 * x38 -
             1.0 / 27.0 * coeffs[84] * x24 * x282 -
             1.0 / 27.0 * coeffs[86] * x282 * x38 -
             1.0 / 27.0 * coeffs[8] * x263 * x277 -
             1.0 / 27.0 * coeffs[93] * x24 * x257 -
             1.0 / 27.0 * coeffs[95] * x257 * x38 +
             (1.0 / 27.0) * coeffs[97] * x131 * x194 * x248 -
             1.0 / 27.0 * coeffs[98] * x187 * x192 -
             1.0 / 27.0 * coeffs[99] * x187 * x193 -
             1.0 / 27.0 * coeffs[9] * x261 * x277 - 1.0 / 27.0 * x118 * x204 -
             1.0 / 27.0 * x119 * x203 - 1.0 / 27.0 * x120 * x195 +
             (1.0 / 27.0) * x122 * x279 + (1.0 / 27.0) * x126 * x280 +
             (1.0 / 27.0) * x127 * x280 + (1.0 / 27.0) * x128 * x279 +
             (1.0 / 27.0) * x132 * x194 * x276 +
             (1.0 / 27.0) * x133 * x146 * x234 +
             (1.0 / 27.0) * x133 * x172 * x242 + (1.0 / 27.0) * x137 * x226 +
             (1.0 / 27.0) * x140 * x228 + (1.0 / 27.0) * x141 * x226 -
             1.0 / 27.0 * x144 * x191 * x230 - 1.0 / 27.0 * x167 * x229 * x243 -
             1.0 / 27.0 * x169 * x246 - 1.0 / 27.0 * x171 * x246 -
             1.0 / 27.0 * x195 * x77 + (1.0 / 27.0) * x199 * x46 * x81 +
             (1.0 / 27.0) * x200 * x202 - 1.0 / 27.0 * x200 * x211 +
             (1.0 / 27.0) * x203 * x24 * x58 + (1.0 / 27.0) * x203 * x38 * x66 -
             1.0 / 27.0 * x203 * x79 + (1.0 / 27.0) * x205 * x62 +
             (1.0 / 27.0) * x205 * x67 + (1.0 / 27.0) * x206 * x207 -
             1.0 / 27.0 * x206 * x212 - 1.0 / 27.0 * x213 * x214 -
             1.0 / 27.0 * x213 * x215 + (1.0 / 27.0) * x221 * x222 * x95 +
             (1.0 / 27.0) * x247 * x251 + (1.0 / 27.0) * x251 * x252 +
             (1.0 / 27.0) * x253 * x254 + (1.0 / 27.0) * x253 * x255 -
             1.0 / 27.0 * x258 * x259 - 1.0 / 27.0 * x258 * x260 +
             (1.0 / 27.0) * x272 * x273 + (1.0 / 27.0) * x272 * x275 -
             1.0 / 27.0 * x283 * x284 - 1.0 / 27.0 * x283 * x285 +
             (1.0 / 27.0) * x290 * x291 + (1.0 / 27.0) * x290 * x292;
    out[2] = (1.0 / 27.0) * coeffs[0] * x299 * x383 -
             1.0 / 27.0 * coeffs[100] * x298 * x301 -
             1.0 / 27.0 * coeffs[101] * x298 * x302 -
             1.0 / 27.0 * coeffs[102] * x302 * x305 -
             1.0 / 27.0 * coeffs[103] * x301 * x305 -
             1.0 / 27.0 * coeffs[104] * x301 * x306 -
             1.0 / 27.0 * coeffs[105] * x302 * x306 +
             (1.0 / 27.0) * coeffs[106] * x307 * x309 +
             (1.0 / 27.0) * coeffs[109] * x311 * x313 +
             (1.0 / 27.0) * coeffs[10] * x308 * x382 +
             (1.0 / 27.0) * coeffs[110] * x305 * x309 +
             (1.0 / 27.0) * coeffs[113] * x313 * x314 +
             (1024.0 / 9.0) * coeffs[114] * x316 * x318 +
             (1.0 / 27.0) * coeffs[115] * x317 * x319 +
             (1.0 / 27.0) * coeffs[116] * x319 * x320 +
             (1.0 / 27.0) * coeffs[117] * x313 * x316 * x320 -
             1.0 / 27.0 * coeffs[11] * x379 * x42 -
             1.0 / 27.0 * coeffs[122] * x317 * x323 -
             1.0 / 27.0 * coeffs[123] * x320 * x323 -
             1.0 / 27.0 * coeffs[12] * x17 * x379 +
             (1.0 / 27.0) * coeffs[13] * x331 * x380 -
             1.0 / 27.0 * coeffs[14] * x363 * x381 -
             1.0 / 27.0 * coeffs[15] * x365 * x381 +
             (1.0 / 27.0) * coeffs[16] * x337 * x382 -
             1.0 / 27.0 * coeffs[17] * x320 * x378 -
             1.0 / 27.0 * coeffs[18] * x317 * x378 +
             (1.0 / 27.0) * coeffs[19] * x310 * x380 +
             (1.0 / 27.0) * coeffs[1] * x25 * x384 -
             1.0 / 27.0 * coeffs[20] * x353 * x369 -
             1.0 / 27.0 * coeffs[21] * x299 * x364 +
             (1.0 / 27.0) * coeffs[22] * x308 * x366 +
             (1.0 / 27.0) * coeffs[25] * x331 * x362 -
             1.0 / 27.0 * coeffs[26] * x333 * x364 -
             1.0 / 27.0 * coeffs[27] * x333 * x353 * x365 +
             (1.0 / 27.0) * coeffs[28] * x337 * x366 +
             (1.0 / 27.0) * coeffs[2] * x384 * x[1] +
             (1.0 / 27.0) * coeffs[31] * x310 * x362 -
             1.0 / 27.0 * coeffs[32] * x343 * x350 -
             1.0 / 27.0 * coeffs[33] * x299 * x345 +
             (1.0 / 27.0) * coeffs[34] * x299 * x346 -
             1.0 / 27.0 * coeffs[35] * x326 * x343 -
             1.0 / 27.0 * coeffs[36] * x330 * x343 +
             (1.0 / 27.0) * coeffs[37] * x324 * x346 -
             1.0 / 27.0 * coeffs[38] * x341 * x344 -
             1.0 / 27.0 * coeffs[39] * x341 * x345 +
             (1.0 / 27.0) * coeffs[3] * x333 * x383 +
             (1.0 / 27.0) * coeffs[40] * x341 * x346 -
             1.0 / 27.0 * coeffs[41] * x333 * x344 -
             1.0 / 27.0 * coeffs[42] * x333 * x345 +
             (1.0 / 27.0) * coeffs[43] * x333 * x346 +
             (1.0 / 27.0) * coeffs[44] * x318 * x373 +
             (1.0 / 27.0) * coeffs[45] * x32 * x374 +
             (1.0 / 27.0) * coeffs[46] * x374 * x9 +
             (1.0 / 27.0) * coeffs[47] * x355 * x373 -
             1.0 / 27.0 * coeffs[48] * x32 * x376 -
             1.0 / 27.0 * coeffs[49] * x320 * x377 -
             1.0 / 27.0 * coeffs[50] * x376 * x9 -
             1.0 / 27.0 * coeffs[51] * x317 * x377 +
             (16.0 / 3.0) * coeffs[52] * x359 * x372 +
             (1.0 / 27.0) * coeffs[53] * x318 * x354 +
             (1.0 / 27.0) * coeffs[54] * x354 * x355 +
             (1.0 / 27.0) * coeffs[55] * x356 * x9 +
             (1.0 / 27.0) * coeffs[56] * x32 * x356 -
             1.0 / 27.0 * coeffs[58] * x310 * x357 * x9 +
             (1.0 / 27.0) * coeffs[62] * x307 * x347 +
             (1.0 / 27.0) * coeffs[63] * x305 * x347 +
             (1.0 / 27.0) * coeffs[64] * x306 * x347 +
             (1.0 / 27.0) * coeffs[65] * x298 * x347 -
             1.0 / 27.0 * coeffs[66] * x317 * x328 -
             1.0 / 27.0 * coeffs[67] * x229 * x314 -
             1.0 / 27.0 * coeffs[69] * x229 * x311 +
             (1.0 / 27.0) * coeffs[70] * x321 * x332 +
             (1.0 / 27.0) * coeffs[71] * x307 * x325 +
             (1.0 / 27.0) * coeffs[72] * x298 * x325 +
             (1.0 / 27.0) * coeffs[73] * x306 * x325 +
             (1.0 / 27.0) * coeffs[74] * x305 * x325 -
             1.0 / 27.0 * coeffs[75] * x326 * x327 -
             1.0 / 27.0 * coeffs[77] * x327 * x330 -
             1.0 / 27.0 * coeffs[78] * x329 * x42 +
             (1.0 / 27.0) * coeffs[79] * x316 * x331 * x332 +
             (1.0 / 27.0) * coeffs[80] * x297 * x349 +
             (1.0 / 27.0) * coeffs[81] * x350 * x351 +
             (1.0 / 27.0) * coeffs[82] * x299 * x304 * x351 +
             (1.0 / 27.0) * coeffs[83] * x304 * x349 -
             1.0 / 27.0 * coeffs[86] * x308 * x340 +
             (1.0 / 27.0) * coeffs[88] * x108 * x308 * x316 -
             1.0 / 27.0 * coeffs[8] * x369 * x372 -
             1.0 / 27.0 * coeffs[93] * x103 * x297 * x337 -
             1.0 / 27.0 * coeffs[95] * x337 * x340 -
             1.0 / 27.0 * coeffs[98] * x302 * x307 -
             1.0 / 27.0 * coeffs[99] * x301 * x307 -
             1.0 / 27.0 * coeffs[9] * x299 * x363 * x372 +
             (1.0 / 27.0) * x109 * x316 * x337 - 1.0 / 27.0 * x129 * x312 -
             1.0 / 27.0 * x143 * x329 - 1.0 / 27.0 * x150 * x361 -
             1.0 / 27.0 * x151 * x361 - 1.0 / 27.0 * x170 * x299 * x328 -
             1.0 / 27.0 * x175 * x370 - 1.0 / 27.0 * x176 * x370 +
             (1.0 / 27.0) * x202 * x311 + (1.0 / 27.0) * x207 * x314 -
             1.0 / 27.0 * x211 * x311 - 1.0 / 27.0 * x212 * x314 -
             1.0 / 27.0 * x214 * x322 - 1.0 / 27.0 * x215 * x322 +
             (1.0 / 27.0) * x247 * x335 + (1.0 / 27.0) * x252 * x335 +
             (1.0 / 27.0) * x254 * x336 + (1.0 / 27.0) * x255 * x336 -
             1.0 / 27.0 * x259 * x339 - 1.0 / 27.0 * x260 * x339 +
             (1.0 / 27.0) * x273 * x368 + (1.0 / 27.0) * x275 * x368 -
             1.0 / 27.0 * x284 * x352 - 1.0 / 27.0 * x285 * x352 +
             (1.0 / 27.0) * x291 * x371 + (1.0 / 27.0) * x292 * x371 +
             (1.0 / 27.0) * x306 * x308 * x66 + (1.0 / 27.0) * x312 * x59 +
             (1.0 / 27.0) * x321 * x46 * x80 - 1.0 / 27.0 * x348 * x357 * x93 +
             (1.0 / 27.0) * x353 * x359 * x95 - 1.0 / 27.0 * x358 * x86 -
             1.0 / 27.0 * x358 * x92;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
      break;
    case 1:
      out[0] = 4;
      out[1] = 0;
      out[2] = 0;
      break;
    case 2:
      out[0] = 4;
      out[1] = 4;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 4;
      out[2] = 0;
      break;
    case 4:
      out[0] = 0;
      out[1] = 0;
      out[2] = 4;
      break;
    case 5:
      out[0] = 4;
      out[1] = 0;
      out[2] = 4;
      break;
    case 6:
      out[0] = 4;
      out[1] = 4;
      out[2] = 4;
      break;
    case 7:
      out[0] = 0;
      out[1] = 4;
      out[2] = 4;
      break;
    case 8:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 9:
      out[0] = 3;
      out[1] = 0;
      out[2] = 0;
      break;
    case 10:
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 11:
      out[0] = 4;
      out[1] = 1;
      out[2] = 0;
      break;
    case 12:
      out[0] = 4;
      out[1] = 3;
      out[2] = 0;
      break;
    case 13:
      out[0] = 4;
      out[1] = 2;
      out[2] = 0;
      break;
    case 14:
      out[0] = 3;
      out[1] = 4;
      out[2] = 0;
      break;
    case 15:
      out[0] = 1;
      out[1] = 4;
      out[2] = 0;
      break;
    case 16:
      out[0] = 2;
      out[1] = 4;
      out[2] = 0;
      break;
    case 17:
      out[0] = 0;
      out[1] = 3;
      out[2] = 0;
      break;
    case 18:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 19:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 20:
      out[0] = 1;
      out[1] = 0;
      out[2] = 4;
      break;
    case 21:
      out[0] = 3;
      out[1] = 0;
      out[2] = 4;
      break;
    case 22:
      out[0] = 2;
      out[1] = 0;
      out[2] = 4;
      break;
    case 23:
      out[0] = 4;
      out[1] = 1;
      out[2] = 4;
      break;
    case 24:
      out[0] = 4;
      out[1] = 3;
      out[2] = 4;
      break;
    case 25:
      out[0] = 4;
      out[1] = 2;
      out[2] = 4;
      break;
    case 26:
      out[0] = 3;
      out[1] = 4;
      out[2] = 4;
      break;
    case 27:
      out[0] = 1;
      out[1] = 4;
      out[2] = 4;
      break;
    case 28:
      out[0] = 2;
      out[1] = 4;
      out[2] = 4;
      break;
    case 29:
      out[0] = 0;
      out[1] = 3;
      out[2] = 4;
      break;
    case 30:
      out[0] = 0;
      out[1] = 1;
      out[2] = 4;
      break;
    case 31:
      out[0] = 0;
      out[1] = 2;
      out[2] = 4;
      break;
    case 32:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 33:
      out[0] = 0;
      out[1] = 0;
      out[2] = 3;
      break;
    case 34:
      out[0] = 0;
      out[1] = 0;
      out[2] = 2;
      break;
    case 35:
      out[0] = 4;
      out[1] = 0;
      out[2] = 1;
      break;
    case 36:
      out[0] = 4;
      out[1] = 0;
      out[2] = 3;
      break;
    case 37:
      out[0] = 4;
      out[1] = 0;
      out[2] = 2;
      break;
    case 38:
      out[0] = 4;
      out[1] = 4;
      out[2] = 1;
      break;
    case 39:
      out[0] = 4;
      out[1] = 4;
      out[2] = 3;
      break;
    case 40:
      out[0] = 4;
      out[1] = 4;
      out[2] = 2;
      break;
    case 41:
      out[0] = 0;
      out[1] = 4;
      out[2] = 1;
      break;
    case 42:
      out[0] = 0;
      out[1] = 4;
      out[2] = 3;
      break;
    case 43:
      out[0] = 0;
      out[1] = 4;
      out[2] = 2;
      break;
    case 44:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 45:
      out[0] = 1;
      out[1] = 3;
      out[2] = 0;
      break;
    case 46:
      out[0] = 3;
      out[1] = 3;
      out[2] = 0;
      break;
    case 47:
      out[0] = 3;
      out[1] = 1;
      out[2] = 0;
      break;
    case 48:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 49:
      out[0] = 2;
      out[1] = 3;
      out[2] = 0;
      break;
    case 50:
      out[0] = 3;
      out[1] = 2;
      out[2] = 0;
      break;
    case 51:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 52:
      out[0] = 2;
      out[1] = 2;
      out[2] = 0;
      break;
    case 53:
      out[0] = 1;
      out[1] = 1;
      out[2] = 4;
      break;
    case 54:
      out[0] = 3;
      out[1] = 1;
      out[2] = 4;
      break;
    case 55:
      out[0] = 3;
      out[1] = 3;
      out[2] = 4;
      break;
    case 56:
      out[0] = 1;
      out[1] = 3;
      out[2] = 4;
      break;
    case 57:
      out[0] = 2;
      out[1] = 1;
      out[2] = 4;
      break;
    case 58:
      out[0] = 3;
      out[1] = 2;
      out[2] = 4;
      break;
    case 59:
      out[0] = 2;
      out[1] = 3;
      out[2] = 4;
      break;
    case 60:
      out[0] = 1;
      out[1] = 2;
      out[2] = 4;
      break;
    case 61:
      out[0] = 2;
      out[1] = 2;
      out[2] = 4;
      break;
    case 62:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 63:
      out[0] = 0;
      out[1] = 1;
      out[2] = 3;
      break;
    case 64:
      out[0] = 0;
      out[1] = 3;
      out[2] = 3;
      break;
    case 65:
      out[0] = 0;
      out[1] = 3;
      out[2] = 1;
      break;
    case 66:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 67:
      out[0] = 0;
      out[1] = 2;
      out[2] = 3;
      break;
    case 68:
      out[0] = 0;
      out[1] = 3;
      out[2] = 2;
      break;
    case 69:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 70:
      out[0] = 0;
      out[1] = 2;
      out[2] = 2;
      break;
    case 71:
      out[0] = 4;
      out[1] = 1;
      out[2] = 1;
      break;
    case 72:
      out[0] = 4;
      out[1] = 3;
      out[2] = 1;
      break;
    case 73:
      out[0] = 4;
      out[1] = 3;
      out[2] = 3;
      break;
    case 74:
      out[0] = 4;
      out[1] = 1;
      out[2] = 3;
      break;
    case 75:
      out[0] = 4;
      out[1] = 2;
      out[2] = 1;
      break;
    case 76:
      out[0] = 4;
      out[1] = 3;
      out[2] = 2;
      break;
    case 77:
      out[0] = 4;
      out[1] = 2;
      out[2] = 3;
      break;
    case 78:
      out[0] = 4;
      out[1] = 1;
      out[2] = 2;
      break;
    case 79:
      out[0] = 4;
      out[1] = 2;
      out[2] = 2;
      break;
    case 80:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 81:
      out[0] = 3;
      out[1] = 0;
      out[2] = 1;
      break;
    case 82:
      out[0] = 3;
      out[1] = 0;
      out[2] = 3;
      break;
    case 83:
      out[0] = 1;
      out[1] = 0;
      out[2] = 3;
      break;
    case 84:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 85:
      out[0] = 3;
      out[1] = 0;
      out[2] = 2;
      break;
    case 86:
      out[0] = 2;
      out[1] = 0;
      out[2] = 3;
      break;
    case 87:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 88:
      out[0] = 2;
      out[1] = 0;
      out[2] = 2;
      break;
    case 89:
      out[0] = 3;
      out[1] = 4;
      out[2] = 1;
      break;
    case 90:
      out[0] = 1;
      out[1] = 4;
      out[2] = 1;
      break;
    case 91:
      out[0] = 1;
      out[1] = 4;
      out[2] = 3;
      break;
    case 92:
      out[0] = 3;
      out[1] = 4;
      out[2] = 3;
      break;
    case 93:
      out[0] = 2;
      out[1] = 4;
      out[2] = 1;
      break;
    case 94:
      out[0] = 1;
      out[1] = 4;
      out[2] = 2;
      break;
    case 95:
      out[0] = 2;
      out[1] = 4;
      out[2] = 3;
      break;
    case 96:
      out[0] = 3;
      out[1] = 4;
      out[2] = 2;
      break;
    case 97:
      out[0] = 2;
      out[1] = 4;
      out[2] = 2;
      break;
    case 98:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    case 99:
      out[0] = 3;
      out[1] = 1;
      out[2] = 1;
      break;
    case 100:
      out[0] = 3;
      out[1] = 3;
      out[2] = 1;
      break;
    case 101:
      out[0] = 1;
      out[1] = 3;
      out[2] = 1;
      break;
    case 102:
      out[0] = 1;
      out[1] = 1;
      out[2] = 3;
      break;
    case 103:
      out[0] = 3;
      out[1] = 1;
      out[2] = 3;
      break;
    case 104:
      out[0] = 3;
      out[1] = 3;
      out[2] = 3;
      break;
    case 105:
      out[0] = 1;
      out[1] = 3;
      out[2] = 3;
      break;
    case 106:
      out[0] = 2;
      out[1] = 1;
      out[2] = 1;
      break;
    case 107:
      out[0] = 3;
      out[1] = 2;
      out[2] = 1;
      break;
    case 108:
      out[0] = 2;
      out[1] = 3;
      out[2] = 1;
      break;
    case 109:
      out[0] = 1;
      out[1] = 2;
      out[2] = 1;
      break;
    case 110:
      out[0] = 2;
      out[1] = 1;
      out[2] = 3;
      break;
    case 111:
      out[0] = 3;
      out[1] = 2;
      out[2] = 3;
      break;
    case 112:
      out[0] = 2;
      out[1] = 3;
      out[2] = 3;
      break;
    case 113:
      out[0] = 1;
      out[1] = 2;
      out[2] = 3;
      break;
    case 114:
      out[0] = 1;
      out[1] = 1;
      out[2] = 2;
      break;
    case 115:
      out[0] = 3;
      out[1] = 1;
      out[2] = 2;
      break;
    case 116:
      out[0] = 3;
      out[1] = 3;
      out[2] = 2;
      break;
    case 117:
      out[0] = 1;
      out[1] = 3;
      out[2] = 2;
      break;
    case 118:
      out[0] = 2;
      out[1] = 2;
      out[2] = 1;
      break;
    case 119:
      out[0] = 2;
      out[1] = 2;
      out[2] = 3;
      break;
    case 120:
      out[0] = 1;
      out[1] = 2;
      out[2] = 2;
      break;
    case 121:
      out[0] = 3;
      out[1] = 2;
      out[2] = 2;
      break;
    case 122:
      out[0] = 2;
      out[1] = 1;
      out[2] = 2;
      break;
    case 123:
      out[0] = 2;
      out[1] = 3;
      out[2] = 2;
      break;
    case 124:
      out[0] = 2;
      out[1] = 2;
      out[2] = 2;
      break;
    }
  }
}

template <>
struct BasisLagrange<mesh::RefElCube, 5> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 5;
  static constexpr dim_t num_basis_functions = 216;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_ti, const Scalar *x) {
    switch (i) {
    case 0:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -1.0 / 13824.0 * (x0 - 4) * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 1:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (1.0 / 13824.0) * x[0] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 2:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -1.0 / 13824.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[2] - 1);
    case 3:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (1.0 / 13824.0) * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 4:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (1.0 / 13824.0) * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 5:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -1.0 / 13824.0 * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[1] - 1);
    case 6:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (1.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) *
             (x2 - 4) * (x2 - 3) * (x2 - 2) * (x2 - 1);
    case 7:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -1.0 / 13824.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[0] - 1);
    case 8:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 13824.0) * x[0] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 9:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 13824.0 * x[0] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 10:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 6912.0 * x[0] * (x0 - 4) * (x0 - 3) * (x0 - 1) * (x1 - 4) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 11:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 6912.0) * x[0] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 12:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 13824.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 13:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 13824.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 14:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 6912.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 15:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 6912.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 16:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 13824.0) * x[0] * x[1] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 17:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 13824.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 18:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 6912.0 * x[0] * x[1] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 19:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 6912.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 20:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 13824.0 * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 21:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 13824.0) * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 22:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 6912.0) * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 23:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 6912.0 * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 24:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 13824.0 * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 25:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 13824.0) * x[0] * x[2] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 26:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 6912.0) * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 27:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 6912.0 * x[0] * x[2] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 28:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[1] - 1);
    case 29:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[1] - 1);
    case 30:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[1] - 1);
    case 31:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[1] - 1);
    case 32:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[0] - 1);
    case 33:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[0] - 1);
    case 34:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[0] - 1);
    case 35:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[0] - 1);
    case 36:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 13824.0) * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 37:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 13824.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 38:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 6912.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 39:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 6912.0) * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 40:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 13824.0) * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 41:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 13824.0 * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 42:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 6912.0 * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 43:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 6912.0) * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 44:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 13824.0 * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x[1] - 1) * (x[2] - 1);
    case 45:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 13824.0) * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 46:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 6912.0) * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 47:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 6912.0 * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 48:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) *
             (x2 - 4) * (x2 - 3) * (x2 - 2) * (x[2] - 1);
    case 49:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) *
             (x2 - 3) * (x2 - 2) * (x2 - 1) * (x[2] - 1);
    case 50:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) *
             (x2 - 4) * (x2 - 3) * (x2 - 1) * (x[2] - 1);
    case 51:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) *
             (x2 - 4) * (x2 - 2) * (x2 - 1) * (x[2] - 1);
    case 52:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 13824.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x[0] - 1) * (x[2] - 1);
    case 53:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 13824.0) * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 54:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (25.0 / 6912.0) * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 55:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -25.0 / 6912.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 56:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 13824.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 57:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 13824.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 58:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 13824.0 * x[0] * x[1] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 59:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 13824.0) * x[0] * x[1] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 60:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 61:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 62:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 63:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 64:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 65:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 66:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 67:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 68:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 3456.0 * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 69:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 3456.0) * x[0] * x[1] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 70:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 3456.0 * x[0] * x[1] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 71:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 3456.0) * x[0] * x[1] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 72:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 73:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 74:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 75:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 76:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 77:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 78:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 79:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 80:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 81:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 82:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 83:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 84:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 85:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 86:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 87:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[1] - 1);
    case 88:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 13824.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 89:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 13824.0) * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 90:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 13824.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 91:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 13824.0) * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 92:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 93:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 94:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 95:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 96:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 97:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 98:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 99:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 100:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 3456.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 101:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 3456.0) * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 102:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 3456.0 * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 103:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 3456.0) * x[1] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 104:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x[1] - 1) * (x[2] - 1);
    case 105:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x[1] - 1) * (x[2] - 1);
    case 106:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 107:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 108:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x[1] - 1) * (x[2] - 1);
    case 109:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x[1] - 1) * (x[2] - 1);
    case 110:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 111:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 112:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 113:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 114:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 115:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) *
             (x2 - 3) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 116:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 117:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 118:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 119:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) *
             (x2 - 2) * (x2 - 1) * (x[1] - 1) * (x[2] - 1);
    case 120:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 13824.0 * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 121:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 13824.0) * x[0] * x[2] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 122:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 13824.0 * x[0] * x[2] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 123:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 13824.0) * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 124:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 125:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[2] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 126:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[2] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 127:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[2] * (x0 - 3) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 128:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[2] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 129:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 130:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 131:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 2) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 132:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 3456.0 * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 133:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 3456.0) * x[0] * x[2] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 134:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 3456.0 * x[0] * x[2] * (x0 - 4) * (x0 - 2) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 135:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 3456.0) * x[0] * x[2] * (x0 - 4) * (x0 - 3) * (x0 - 1) *
             (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 136:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x[0] - 1) * (x[2] - 1);
    case 137:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x[0] - 1) * (x[2] - 1);
    case 138:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 139:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 140:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x[0] - 1) * (x[2] - 1);
    case 141:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 2) * (x[0] - 1) * (x[2] - 1);
    case 142:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 143:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 144:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 145:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 146:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 147:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 148:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 149:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 3) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 150:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 151:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) *
             (x2 - 2) * (x2 - 1) * (x[0] - 1) * (x[2] - 1);
    case 152:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 153:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 154:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 155:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 156:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 157:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 158:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 13824.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 159:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 13824.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 160:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 161:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 162:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 163:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 164:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 165:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 166:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 167:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 168:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 169:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 170:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 171:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 172:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 173:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 174:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 175:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 176:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 177:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 178:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 179:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 180:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 181:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 182:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 6912.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 183:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 6912.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 184:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 185:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 186:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 187:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 2) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 188:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 189:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 190:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 191:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 3) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 192:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 193:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 194:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 195:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 2) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 196:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 197:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 198:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 199:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 3) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 200:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 201:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 202:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 203:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 2) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 204:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 205:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 206:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 3456.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 207:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 3456.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 3) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 208:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 1728.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 209:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 1728.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 210:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 1728.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 211:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 1728.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 3) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 212:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 1728.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 213:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 1728.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 3) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 214:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return (15625.0 / 1728.0) * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 2) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    case 215:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = 5 * x[1];
      const Scalar x2 = 5 * x[2];
      return -15625.0 / 1728.0 * x[0] * x[1] * x[2] * (x0 - 4) * (x0 - 3) *
             (x0 - 1) * (x1 - 4) * (x1 - 2) * (x1 - 1) * (x2 - 4) * (x2 - 2) *
             (x2 - 1) * (x[0] - 1) * (x[1] - 1) * (x[2] - 1);
    default:
      return 0;
    }
  }

  template <typename Scalar>
  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    switch (i) {
    case 0:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (1.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] =
          -1.0 / 13824.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 1:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x9 - 3;
      const Scalar x14 = x11 * x12 * x13;
      const Scalar x15 = x10 * x14;
      const Scalar x16 = (1.0 / 13824.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x18 - 3;
      const Scalar x23 = x20 * x21 * x22;
      const Scalar x24 = x19 * x23;
      const Scalar x25 = x17 * x24;
      const Scalar x26 = 5 * x17;
      const Scalar x27 = x19 * x20 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = 5 * x8;
      const Scalar x30 = x10 * x11 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 *
          (x19 * x21 * x22 * x26 + x21 * x27 + x22 * x27 + x23 * x26 + x24);
      out[2] =
          (1.0 / 13824.0) * x25 * x28 *
          (x10 * x12 * x13 * x29 + x12 * x30 + x13 * x30 + x14 * x29 + x15);
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
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x9 - 3;
      const Scalar x14 = x11 * x12 * x13;
      const Scalar x15 = x10 * x14;
      const Scalar x16 = (1.0 / 13824.0) * x15 * x8;
      const Scalar x17 = 5 * x[1];
      const Scalar x18 = x17 - 2;
      const Scalar x19 = x17 - 1;
      const Scalar x20 = x17 - 4;
      const Scalar x21 = x17 - 3;
      const Scalar x22 = x19 * x20 * x21;
      const Scalar x23 = x18 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x18 * x19;
      const Scalar x26 = x7 * x[0];
      const Scalar x27 = 5 * x8;
      const Scalar x28 = x10 * x11 * x27;
      out[0] =
          -x16 * x24 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x26 *
          (x17 * x18 * x20 * x21 + x17 * x22 + x20 * x25 + x21 * x25 + x23);
      out[2] =
          -1.0 / 13824.0 * x24 * x26 *
          (x10 * x12 * x13 * x27 + x12 * x28 + x13 * x28 + x14 * x27 + x15);
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (1.0 / 13824.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x4 * x9;
      const Scalar x29 = 5 * x10;
      const Scalar x30 = x12 * x13 * x29;
      out[0] =
          x18 * x26 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] =
          (1.0 / 13824.0) * x26 * x28 *
          (x12 * x14 * x15 * x29 + x14 * x30 + x15 * x30 + x16 * x29 + x17);
      break;
    case 4:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (1.0 / 13824.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x18 * x25;
      const Scalar x27 = 5 * x18;
      const Scalar x28 = x20 * x21 * x27;
      const Scalar x29 = x4 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] =
          x17 * x26 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x17 * x29 *
          (x20 * x22 * x23 * x27 + x22 * x28 + x23 * x28 + x24 * x27 + x25);
      out[2] =
          (1.0 / 13824.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 5:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = 5 * x[2];
      const Scalar x9 = x8 - 2;
      const Scalar x10 = x8 - 1;
      const Scalar x11 = x8 - 4;
      const Scalar x12 = x8 - 3;
      const Scalar x13 = x10 * x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (1.0 / 13824.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 5 * x[1];
      const Scalar x18 = x17 - 2;
      const Scalar x19 = x17 - 1;
      const Scalar x20 = x17 - 4;
      const Scalar x21 = x17 - 3;
      const Scalar x22 = x19 * x20 * x21;
      const Scalar x23 = x18 * x22;
      const Scalar x24 = x16 * x23;
      const Scalar x25 = 5 * x16;
      const Scalar x26 = x18 * x19 * x25;
      const Scalar x27 = x7 * x[0];
      const Scalar x28 = x10 * x8 * x9;
      out[0] =
          -x15 * x24 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x15 * x27 *
          (x18 * x20 * x21 * x25 + x20 * x26 + x21 * x26 + x22 * x25 + x23);
      out[2] = -1.0 / 13824.0 * x24 * x27 *
               (x11 * x12 * x8 * x9 + x11 * x28 + x12 * x28 + x13 * x8 + x14);
      break;
    case 6:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = 5 * x[2];
      const Scalar x9 = x8 - 2;
      const Scalar x10 = x8 - 1;
      const Scalar x11 = x8 - 4;
      const Scalar x12 = x8 - 3;
      const Scalar x13 = x10 * x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (1.0 / 13824.0) * x14 * x[2];
      const Scalar x16 = 5 * x[1];
      const Scalar x17 = x16 - 2;
      const Scalar x18 = x16 - 1;
      const Scalar x19 = x16 - 4;
      const Scalar x20 = x16 - 3;
      const Scalar x21 = x18 * x19 * x20;
      const Scalar x22 = x17 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x16 * x17 * x18;
      const Scalar x25 = x7 * x[0];
      const Scalar x26 = x10 * x8 * x9;
      out[0] =
          x15 * x23 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x15 * x25 *
          (x16 * x17 * x19 * x20 + x16 * x21 + x19 * x24 + x20 * x24 + x22);
      out[2] = (1.0 / 13824.0) * x23 * x25 *
               (x11 * x12 * x8 * x9 + x11 * x26 + x12 * x26 + x13 * x8 + x14);
      break;
    case 7:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (1.0 / 13824.0) * x16 * x[2];
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x18 - 3;
      const Scalar x23 = x20 * x21 * x22;
      const Scalar x24 = x19 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x19 * x20;
      const Scalar x27 = x4 * x9;
      const Scalar x28 = x10 * x11 * x12;
      out[0] =
          -x17 * x25 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x17 * x27 *
          (x18 * x19 * x21 * x22 + x18 * x23 + x21 * x26 + x22 * x26 + x24);
      out[2] =
          -1.0 / 13824.0 * x25 * x27 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x28 + x14 * x28 + x16);
      break;
    case 8:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (25.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] =
          (25.0 / 13824.0) * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 9:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (25.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] =
          -25.0 / 13824.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 10:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (25.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] =
          -25.0 / 6912.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 11:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (25.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] =
          (25.0 / 6912.0) * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 12:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x9 - 3;
      const Scalar x14 = x11 * x12 * x13;
      const Scalar x15 = x10 * x14;
      const Scalar x16 = (25.0 / 13824.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 4;
      const Scalar x21 = x18 - 3;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = 5 * x8;
      const Scalar x30 = x10 * x11 * x29;
      out[0] =
          -x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] =
          -25.0 / 13824.0 * x25 * x28 *
          (x10 * x12 * x13 * x29 + x12 * x30 + x13 * x30 + x14 * x29 + x15);
      break;
    case 13:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x9 - 3;
      const Scalar x14 = x11 * x12 * x13;
      const Scalar x15 = x10 * x14;
      const Scalar x16 = (25.0 / 13824.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 3;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = 5 * x8;
      const Scalar x30 = x10 * x11 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] =
          (25.0 / 13824.0) * x25 * x28 *
          (x10 * x12 * x13 * x29 + x12 * x30 + x13 * x30 + x14 * x29 + x15);
      break;
    case 14:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x9 - 3;
      const Scalar x14 = x11 * x12 * x13;
      const Scalar x15 = x10 * x14;
      const Scalar x16 = (25.0 / 6912.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = 5 * x8;
      const Scalar x30 = x10 * x11 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] =
          (25.0 / 6912.0) * x25 * x28 *
          (x10 * x12 * x13 * x29 + x12 * x30 + x13 * x30 + x14 * x29 + x15);
      break;
    case 15:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x9 - 3;
      const Scalar x14 = x11 * x12 * x13;
      const Scalar x15 = x10 * x14;
      const Scalar x16 = (25.0 / 6912.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = 5 * x8;
      const Scalar x30 = x10 * x11 * x29;
      out[0] =
          -x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] =
          -25.0 / 6912.0 * x25 * x28 *
          (x10 * x12 * x13 * x29 + x12 * x30 + x13 * x30 + x14 * x29 + x15);
      break;
    case 16:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (25.0 / 13824.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = 5 * x10;
      const Scalar x30 = x12 * x13 * x29;
      out[0] = x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] =
          (25.0 / 13824.0) * x26 * x28 *
          (x12 * x14 * x15 * x29 + x14 * x30 + x15 * x30 + x16 * x29 + x17);
      break;
    case 17:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (25.0 / 13824.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x4 * x9;
      const Scalar x29 = 5 * x10;
      const Scalar x30 = x12 * x13 * x29;
      out[0] = -x18 * x26 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] =
          -25.0 / 13824.0 * x26 * x28 *
          (x12 * x14 * x15 * x29 + x14 * x30 + x15 * x30 + x16 * x29 + x17);
      break;
    case 18:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (25.0 / 6912.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = 5 * x10;
      const Scalar x30 = x12 * x13 * x29;
      out[0] = -x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] =
          -25.0 / 6912.0 * x26 * x28 *
          (x12 * x14 * x15 * x29 + x14 * x30 + x15 * x30 + x16 * x29 + x17);
      break;
    case 19:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (25.0 / 6912.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = 5 * x10;
      const Scalar x30 = x12 * x13 * x29;
      out[0] = x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] =
          (25.0 / 6912.0) * x26 * x28 *
          (x12 * x14 * x15 * x29 + x14 * x30 + x15 * x30 + x16 * x29 + x17);
      break;
    case 20:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (25.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          -25.0 / 13824.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 21:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (25.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] =
          x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          (25.0 / 13824.0) * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 22:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (25.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] =
          x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          (25.0 / 6912.0) * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 23:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (25.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          -25.0 / 6912.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 24:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (25.0 / 13824.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x18 * x25;
      const Scalar x27 = 5 * x18;
      const Scalar x28 = x20 * x21 * x27;
      const Scalar x29 = x4 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = -x17 * x26 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x17 * x29 *
          (x20 * x22 * x23 * x27 + x22 * x28 + x23 * x28 + x24 * x27 + x25);
      out[2] =
          -25.0 / 13824.0 * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 25:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (25.0 / 13824.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x18 * x25;
      const Scalar x27 = 5 * x18;
      const Scalar x28 = x20 * x21 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x17 * x29 *
          (x20 * x22 * x23 * x27 + x22 * x28 + x23 * x28 + x24 * x27 + x25);
      out[2] =
          (25.0 / 13824.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 26:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (25.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x18 * x25;
      const Scalar x27 = 5 * x18;
      const Scalar x28 = x20 * x21 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x17 * x29 *
          (x20 * x22 * x23 * x27 + x22 * x28 + x23 * x28 + x24 * x27 + x25);
      out[2] =
          (25.0 / 6912.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 27:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (25.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x18 * x25;
      const Scalar x27 = 5 * x18;
      const Scalar x28 = x20 * x21 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = -x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x17 * x29 *
          (x20 * x22 * x23 * x27 + x22 * x28 + x23 * x28 + x24 * x27 + x25);
      out[2] =
          -25.0 / 6912.0 * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 28:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = 5 * x[2];
      const Scalar x9 = x8 - 2;
      const Scalar x10 = x8 - 1;
      const Scalar x11 = x8 - 4;
      const Scalar x12 = x8 - 3;
      const Scalar x13 = x10 * x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (25.0 / 13824.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 5 * x[1];
      const Scalar x18 = x17 - 2;
      const Scalar x19 = x17 - 4;
      const Scalar x20 = x17 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x18 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x16 * x23;
      const Scalar x25 = x16 * x17;
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x7 * x[0];
      const Scalar x28 = x10 * x8 * x9;
      out[0] =
          x15 * x24 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x15 * x27 * (x16 * x22 + x19 * x26 + x20 * x26 + x21 * x25 + x23);
      out[2] = (25.0 / 13824.0) * x24 * x27 *
               (x11 * x12 * x8 * x9 + x11 * x28 + x12 * x28 + x13 * x8 + x14);
      break;
    case 29:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = 5 * x[2];
      const Scalar x9 = x8 - 2;
      const Scalar x10 = x8 - 1;
      const Scalar x11 = x8 - 4;
      const Scalar x12 = x8 - 3;
      const Scalar x13 = x10 * x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (25.0 / 13824.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 5 * x[1];
      const Scalar x18 = x17 - 2;
      const Scalar x19 = x17 - 1;
      const Scalar x20 = x17 - 3;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x18 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x16 * x23;
      const Scalar x25 = x16 * x17;
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x7 * x[0];
      const Scalar x28 = x10 * x8 * x9;
      out[0] =
          -x15 * x24 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x15 * x27 * (x16 * x22 + x19 * x26 + x20 * x26 + x21 * x25 + x23);
      out[2] = -25.0 / 13824.0 * x24 * x27 *
               (x11 * x12 * x8 * x9 + x11 * x28 + x12 * x28 + x13 * x8 + x14);
      break;
    case 30:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = 5 * x[2];
      const Scalar x9 = x8 - 2;
      const Scalar x10 = x8 - 1;
      const Scalar x11 = x8 - 4;
      const Scalar x12 = x8 - 3;
      const Scalar x13 = x10 * x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (25.0 / 6912.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 5 * x[1];
      const Scalar x18 = x17 - 3;
      const Scalar x19 = x17 - 1;
      const Scalar x20 = x17 - 4;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x18 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x16 * x23;
      const Scalar x25 = x16 * x17;
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x7 * x[0];
      const Scalar x28 = x10 * x8 * x9;
      out[0] =
          -x15 * x24 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x15 * x27 * (x16 * x22 + x19 * x26 + x20 * x26 + x21 * x25 + x23);
      out[2] = -25.0 / 6912.0 * x24 * x27 *
               (x11 * x12 * x8 * x9 + x11 * x28 + x12 * x28 + x13 * x8 + x14);
      break;
    case 31:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = 5 * x[2];
      const Scalar x9 = x8 - 2;
      const Scalar x10 = x8 - 1;
      const Scalar x11 = x8 - 4;
      const Scalar x12 = x8 - 3;
      const Scalar x13 = x10 * x11 * x12;
      const Scalar x14 = x13 * x9;
      const Scalar x15 = (25.0 / 6912.0) * x14 * x[2];
      const Scalar x16 = x[1] - 1;
      const Scalar x17 = 5 * x[1];
      const Scalar x18 = x17 - 2;
      const Scalar x19 = x17 - 1;
      const Scalar x20 = x17 - 4;
      const Scalar x21 = x19 * x20;
      const Scalar x22 = x18 * x21;
      const Scalar x23 = x22 * x[1];
      const Scalar x24 = x16 * x23;
      const Scalar x25 = x16 * x17;
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x7 * x[0];
      const Scalar x28 = x10 * x8 * x9;
      out[0] =
          x15 * x24 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x15 * x27 * (x16 * x22 + x19 * x26 + x20 * x26 + x21 * x25 + x23);
      out[2] = (25.0 / 6912.0) * x24 * x27 *
               (x11 * x12 * x8 * x9 + x11 * x28 + x12 * x28 + x13 * x8 + x14);
      break;
    case 32:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (25.0 / 13824.0) * x16 * x[2];
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x18 - 3;
      const Scalar x23 = x20 * x21 * x22;
      const Scalar x24 = x19 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x19 * x20;
      const Scalar x27 = x3 * x9;
      const Scalar x28 = x10 * x11 * x12;
      out[0] = -x17 * x25 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x17 * x27 *
          (x18 * x19 * x21 * x22 + x18 * x23 + x21 * x26 + x22 * x26 + x24);
      out[2] =
          -25.0 / 13824.0 * x25 * x27 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x28 + x14 * x28 + x16);
      break;
    case 33:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (25.0 / 13824.0) * x16 * x[2];
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x18 - 3;
      const Scalar x23 = x20 * x21 * x22;
      const Scalar x24 = x19 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x19 * x20;
      const Scalar x27 = x4 * x9;
      const Scalar x28 = x10 * x11 * x12;
      out[0] = x17 * x25 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x17 * x27 *
          (x18 * x19 * x21 * x22 + x18 * x23 + x21 * x26 + x22 * x26 + x24);
      out[2] =
          (25.0 / 13824.0) * x25 * x27 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x28 + x14 * x28 + x16);
      break;
    case 34:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (25.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x18 - 3;
      const Scalar x23 = x20 * x21 * x22;
      const Scalar x24 = x19 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x19 * x20;
      const Scalar x27 = x3 * x9;
      const Scalar x28 = x10 * x11 * x12;
      out[0] = x17 * x25 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x17 * x27 *
          (x18 * x19 * x21 * x22 + x18 * x23 + x21 * x26 + x22 * x26 + x24);
      out[2] =
          (25.0 / 6912.0) * x25 * x27 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x28 + x14 * x28 + x16);
      break;
    case 35:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (25.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x18 - 3;
      const Scalar x23 = x20 * x21 * x22;
      const Scalar x24 = x19 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x19 * x20;
      const Scalar x27 = x3 * x9;
      const Scalar x28 = x10 * x11 * x12;
      out[0] = -x17 * x25 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x17 * x27 *
          (x18 * x19 * x21 * x22 + x18 * x23 + x21 * x26 + x22 * x26 + x24);
      out[2] =
          -25.0 / 6912.0 * x25 * x27 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x28 + x14 * x28 + x16);
      break;
    case 36:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (25.0 / 13824.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 3;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x4 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] =
          x17 * x26 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          (25.0 / 13824.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 37:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (25.0 / 13824.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 4;
      const Scalar x22 = x19 - 3;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x4 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] =
          -x17 * x26 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          -25.0 / 13824.0 * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 38:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (25.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x4 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] =
          -x17 * x26 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          -25.0 / 6912.0 * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 39:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (25.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x4 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] =
          x17 * x26 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          (25.0 / 6912.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 40:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (25.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = (25.0 / 13824.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 41:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (25.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = -25.0 / 13824.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 42:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (25.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = -25.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 43:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (25.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = (25.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 44:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 4;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (25.0 / 13824.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x18 - 3;
      const Scalar x23 = x20 * x21 * x22;
      const Scalar x24 = x19 * x23;
      const Scalar x25 = x17 * x24;
      const Scalar x26 = 5 * x17;
      const Scalar x27 = x19 * x20 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          -x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x28 *
          (x19 * x21 * x22 * x26 + x21 * x27 + x22 * x27 + x23 * x26 + x24);
      out[2] = -25.0 / 13824.0 * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 45:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (25.0 / 13824.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x18 - 3;
      const Scalar x23 = x20 * x21 * x22;
      const Scalar x24 = x19 * x23;
      const Scalar x25 = x17 * x24;
      const Scalar x26 = 5 * x17;
      const Scalar x27 = x19 * x20 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 *
          (x19 * x21 * x22 * x26 + x21 * x27 + x22 * x27 + x23 * x26 + x24);
      out[2] = (25.0 / 13824.0) * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 46:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 3;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (25.0 / 6912.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x18 - 3;
      const Scalar x23 = x20 * x21 * x22;
      const Scalar x24 = x19 * x23;
      const Scalar x25 = x17 * x24;
      const Scalar x26 = 5 * x17;
      const Scalar x27 = x19 * x20 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 *
          (x19 * x21 * x22 * x26 + x21 * x27 + x22 * x27 + x23 * x26 + x24);
      out[2] = (25.0 / 6912.0) * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 47:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (25.0 / 6912.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x18 - 3;
      const Scalar x23 = x20 * x21 * x22;
      const Scalar x24 = x19 * x23;
      const Scalar x25 = x17 * x24;
      const Scalar x26 = 5 * x17;
      const Scalar x27 = x19 * x20 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          -x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x28 *
          (x19 * x21 * x22 * x26 + x21 * x27 + x22 * x27 + x23 * x26 + x24);
      out[2] = -25.0 / 6912.0 * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 48:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 4;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (25.0 / 13824.0) * x15 * x8;
      const Scalar x17 = 5 * x[1];
      const Scalar x18 = x17 - 2;
      const Scalar x19 = x17 - 1;
      const Scalar x20 = x17 - 4;
      const Scalar x21 = x17 - 3;
      const Scalar x22 = x19 * x20 * x21;
      const Scalar x23 = x18 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x18 * x19;
      const Scalar x26 = x7 * x[0];
      const Scalar x27 = x8 * x9;
      const Scalar x28 = x10 * x27;
      out[0] =
          x16 * x24 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x26 *
          (x17 * x18 * x20 * x21 + x17 * x22 + x20 * x25 + x21 * x25 + x23);
      out[2] = (25.0 / 13824.0) * x24 * x26 *
               (x11 * x28 + x12 * x28 + x13 * x27 + x14 * x8 + x15);
      break;
    case 49:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (25.0 / 13824.0) * x15 * x8;
      const Scalar x17 = 5 * x[1];
      const Scalar x18 = x17 - 2;
      const Scalar x19 = x17 - 1;
      const Scalar x20 = x17 - 4;
      const Scalar x21 = x17 - 3;
      const Scalar x22 = x19 * x20 * x21;
      const Scalar x23 = x18 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x18 * x19;
      const Scalar x26 = x7 * x[0];
      const Scalar x27 = x8 * x9;
      const Scalar x28 = x10 * x27;
      out[0] =
          -x16 * x24 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x26 *
          (x17 * x18 * x20 * x21 + x17 * x22 + x20 * x25 + x21 * x25 + x23);
      out[2] = -25.0 / 13824.0 * x24 * x26 *
               (x11 * x28 + x12 * x28 + x13 * x27 + x14 * x8 + x15);
      break;
    case 50:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 3;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (25.0 / 6912.0) * x15 * x8;
      const Scalar x17 = 5 * x[1];
      const Scalar x18 = x17 - 2;
      const Scalar x19 = x17 - 1;
      const Scalar x20 = x17 - 4;
      const Scalar x21 = x17 - 3;
      const Scalar x22 = x19 * x20 * x21;
      const Scalar x23 = x18 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x18 * x19;
      const Scalar x26 = x7 * x[0];
      const Scalar x27 = x8 * x9;
      const Scalar x28 = x10 * x27;
      out[0] =
          -x16 * x24 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x26 *
          (x17 * x18 * x20 * x21 + x17 * x22 + x20 * x25 + x21 * x25 + x23);
      out[2] = -25.0 / 6912.0 * x24 * x26 *
               (x11 * x28 + x12 * x28 + x13 * x27 + x14 * x8 + x15);
      break;
    case 51:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (25.0 / 6912.0) * x15 * x8;
      const Scalar x17 = 5 * x[1];
      const Scalar x18 = x17 - 2;
      const Scalar x19 = x17 - 1;
      const Scalar x20 = x17 - 4;
      const Scalar x21 = x17 - 3;
      const Scalar x22 = x19 * x20 * x21;
      const Scalar x23 = x18 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x18 * x19;
      const Scalar x26 = x7 * x[0];
      const Scalar x27 = x8 * x9;
      const Scalar x28 = x10 * x27;
      out[0] =
          x16 * x24 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x26 *
          (x17 * x18 * x20 * x21 + x17 * x22 + x20 * x25 + x21 * x25 + x23);
      out[2] = (25.0 / 6912.0) * x24 * x26 *
               (x11 * x28 + x12 * x28 + x13 * x27 + x14 * x8 + x15);
      break;
    case 52:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (25.0 / 13824.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x4 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] =
          -x18 * x26 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = -25.0 / 13824.0 * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 53:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (25.0 / 13824.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x4 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] =
          x18 * x26 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = (25.0 / 13824.0) * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 54:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (25.0 / 6912.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x4 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] =
          x18 * x26 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = (25.0 / 6912.0) * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 55:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (25.0 / 6912.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x4 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] =
          -x18 * x26 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = -25.0 / 6912.0 * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 56:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = -x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          -625.0 / 13824.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 57:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          (625.0 / 13824.0) * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 58:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          -625.0 / 13824.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 59:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          (625.0 / 13824.0) * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 60:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          (625.0 / 6912.0) * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 61:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = -x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          -625.0 / 6912.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 62:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          -625.0 / 6912.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 63:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          (625.0 / 6912.0) * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 64:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          (625.0 / 6912.0) * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 65:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          -625.0 / 6912.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 66:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          -625.0 / 6912.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 67:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          (625.0 / 6912.0) * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 68:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          -625.0 / 3456.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 69:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          (625.0 / 3456.0) * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 70:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          -625.0 / 3456.0 * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 71:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x11 - 3;
      const Scalar x16 = x13 * x14 * x15;
      const Scalar x17 = x12 * x16;
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = 5 * x10;
      const Scalar x32 = x12 * x13 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] =
          (625.0 / 3456.0) * x27 * x30 *
          (x12 * x14 * x15 * x31 + x14 * x32 + x15 * x32 + x16 * x31 + x17);
      break;
    case 72:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 13824.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 4;
      const Scalar x22 = x19 - 3;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x4 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = x17 * x26 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          (625.0 / 13824.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 73:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 13824.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 4;
      const Scalar x22 = x19 - 3;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = -x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          -625.0 / 13824.0 * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 74:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 13824.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 3;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          (625.0 / 13824.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 75:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 13824.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 3;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x4 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = -x17 * x26 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          -625.0 / 13824.0 * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 76:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 4;
      const Scalar x22 = x19 - 3;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = -x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          -625.0 / 6912.0 * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 77:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 4;
      const Scalar x22 = x19 - 3;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          (625.0 / 6912.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 78:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          (625.0 / 6912.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 79:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = -x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          -625.0 / 6912.0 * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 80:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 3;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = -x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          -625.0 / 6912.0 * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 81:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 3;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          (625.0 / 6912.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 82:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x4 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = x17 * x26 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          (625.0 / 6912.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 83:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 6912.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x4 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = -x17 * x26 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          -625.0 / 6912.0 * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 84:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 3456.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          (625.0 / 3456.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 85:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 3456.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 3;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = -x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          -625.0 / 3456.0 * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 86:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 3456.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          (625.0 / 3456.0) * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 87:
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
      const Scalar x10 = 5 * x[2];
      const Scalar x11 = x10 - 2;
      const Scalar x12 = x10 - 1;
      const Scalar x13 = x10 - 4;
      const Scalar x14 = x10 - 3;
      const Scalar x15 = x12 * x13 * x14;
      const Scalar x16 = x11 * x15;
      const Scalar x17 = (625.0 / 3456.0) * x16 * x[2];
      const Scalar x18 = x[1] - 1;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x21 * x22;
      const Scalar x24 = x20 * x23;
      const Scalar x25 = x24 * x[1];
      const Scalar x26 = x18 * x25;
      const Scalar x27 = x18 * x19;
      const Scalar x28 = x20 * x27;
      const Scalar x29 = x3 * x9;
      const Scalar x30 = x10 * x11 * x12;
      out[0] = -x17 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x17 * x29 * (x18 * x24 + x21 * x28 + x22 * x28 + x23 * x27 + x25);
      out[2] =
          -625.0 / 3456.0 * x26 * x29 *
          (x10 * x11 * x13 * x14 + x10 * x15 + x13 * x30 + x14 * x30 + x16);
      break;
    case 88:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -625.0 / 13824.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 89:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (625.0 / 13824.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 90:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -625.0 / 13824.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 91:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (625.0 / 13824.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 92:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 93:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 94:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 95:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 96:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 97:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 98:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 99:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 100:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 101:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 102:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          -x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 103:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] =
          x18 * x27 * (x1 * x2 * x3 * x5 + x1 * x8 + x2 * x8 + x5 * x7 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 104:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 4;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 13824.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 4;
      const Scalar x21 = x18 - 3;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = (625.0 / 13824.0) * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 105:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 4;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 13824.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 3;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          -x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = -625.0 / 13824.0 * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 106:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 13824.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 3;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = (625.0 / 13824.0) * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 107:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 13824.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 4;
      const Scalar x21 = x18 - 3;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          -x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = -625.0 / 13824.0 * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 108:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 4;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 6912.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          -x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = -625.0 / 6912.0 * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 109:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 4;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 6912.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = (625.0 / 6912.0) * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 110:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 3;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 6912.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 3;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = (625.0 / 6912.0) * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 111:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 6912.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 3;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          -x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = -625.0 / 6912.0 * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 112:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 6912.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          -x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = -625.0 / 6912.0 * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 113:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 3;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 6912.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = (625.0 / 6912.0) * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 114:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 6912.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 4;
      const Scalar x21 = x18 - 3;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = (625.0 / 6912.0) * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 115:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 3;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 6912.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 4;
      const Scalar x21 = x18 - 3;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          -x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = -625.0 / 6912.0 * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 116:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 3;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 3456.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = (625.0 / 3456.0) * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 117:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 3;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 3456.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          -x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = -625.0 / 3456.0 * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 118:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 3456.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 2;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = (625.0 / 3456.0) * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 119:
      const Scalar x0 = 5 * x[0];
      const Scalar x1 = x0 - 4;
      const Scalar x2 = x0 - 3;
      const Scalar x3 = x0 - 2;
      const Scalar x4 = x0 - 1;
      const Scalar x5 = x1 * x2 * x4;
      const Scalar x6 = x0 * x3 * x4;
      const Scalar x7 = x3 * x5;
      const Scalar x8 = x[2] - 1;
      const Scalar x9 = 5 * x[2];
      const Scalar x10 = x9 - 2;
      const Scalar x11 = x9 - 1;
      const Scalar x12 = x9 - 4;
      const Scalar x13 = x11 * x12;
      const Scalar x14 = x10 * x13;
      const Scalar x15 = x14 * x[2];
      const Scalar x16 = (625.0 / 3456.0) * x15 * x8;
      const Scalar x17 = x[1] - 1;
      const Scalar x18 = 5 * x[1];
      const Scalar x19 = x18 - 3;
      const Scalar x20 = x18 - 1;
      const Scalar x21 = x18 - 4;
      const Scalar x22 = x20 * x21;
      const Scalar x23 = x19 * x22;
      const Scalar x24 = x23 * x[1];
      const Scalar x25 = x17 * x24;
      const Scalar x26 = x17 * x18;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x7 * x[0];
      const Scalar x29 = x8 * x9;
      const Scalar x30 = x10 * x29;
      out[0] =
          -x16 * x25 * (x0 * x1 * x2 * x3 + x0 * x5 + x1 * x6 + x2 * x6 + x7);
      out[1] =
          -x16 * x28 * (x17 * x23 + x20 * x27 + x21 * x27 + x22 * x26 + x24);
      out[2] = -625.0 / 3456.0 * x25 * x28 *
               (x11 * x30 + x12 * x30 + x13 * x29 + x14 * x8 + x15);
      break;
    case 120:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = -625.0 / 13824.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 121:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = (625.0 / 13824.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 122:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = -625.0 / 13824.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 123:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = (625.0 / 13824.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 124:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = (625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 125:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = -625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 126:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = -625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 127:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = (625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 128:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = (625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 129:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = -625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 130:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = -625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 131:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = (625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 132:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = -625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 133:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = (625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 134:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = -625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 135:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x20 - 3;
      const Scalar x25 = x22 * x23 * x24;
      const Scalar x26 = x21 * x25;
      const Scalar x27 = x19 * x26;
      const Scalar x28 = 5 * x19;
      const Scalar x29 = x21 * x22 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 *
          (x21 * x23 * x24 * x28 + x23 * x29 + x24 * x29 + x25 * x28 + x26);
      out[2] = (625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 136:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = -x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = -625.0 / 13824.0 * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 137:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x4 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = x18 * x26 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = (625.0 / 13824.0) * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 138:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x4 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = -x18 * x26 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = -625.0 / 13824.0 * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 139:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = (625.0 / 13824.0) * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 140:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = (625.0 / 6912.0) * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 141:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = -x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = -625.0 / 6912.0 * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 142:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x4 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = -x18 * x26 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = -625.0 / 6912.0 * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 143:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x4 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = x18 * x26 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = (625.0 / 6912.0) * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 144:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = (625.0 / 6912.0) * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 145:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = -x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = -625.0 / 6912.0 * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 146:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = -x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = -625.0 / 6912.0 * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 147:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = (625.0 / 6912.0) * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 148:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = -x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = -625.0 / 3456.0 * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 149:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = (625.0 / 3456.0) * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 150:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = -x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = -625.0 / 3456.0 * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 151:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = 5 * x[1];
      const Scalar x20 = x19 - 2;
      const Scalar x21 = x19 - 1;
      const Scalar x22 = x19 - 4;
      const Scalar x23 = x19 - 3;
      const Scalar x24 = x21 * x22 * x23;
      const Scalar x25 = x20 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x20 * x21;
      const Scalar x28 = x3 * x9;
      const Scalar x29 = x10 * x11;
      const Scalar x30 = x12 * x29;
      out[0] = x18 * x26 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x28 *
          (x19 * x20 * x22 * x23 + x19 * x24 + x22 * x27 + x23 * x27 + x25);
      out[2] = (625.0 / 3456.0) * x26 * x28 *
               (x10 * x16 + x13 * x30 + x14 * x30 + x15 * x29 + x17);
      break;
    case 152:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 13824.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 153:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 13824.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 154:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 13824.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 155:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 13824.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 156:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 13824.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 157:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 13824.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 158:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 13824.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 159:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 13824.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 13824.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 160:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 161:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 162:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 163:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 164:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 165:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 166:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 167:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 168:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 169:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 170:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 171:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 172:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 173:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 174:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 175:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 176:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 177:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 178:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 179:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 180:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 181:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 182:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 6912.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 183:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 6912.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 6912.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 184:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 185:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 186:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 187:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 4;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 188:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 189:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 190:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 191:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 3;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 192:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 193:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 194:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 195:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x4 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x7 + x2 * x7 + x3 * x5 + x4 * x8 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 196:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 197:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 198:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 199:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 200:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 201:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 202:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 203:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 4;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 204:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 205:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 206:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 3456.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 207:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 3456.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 3;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 3456.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 208:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 1728.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 1728.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 209:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 1728.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 1728.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 210:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 1728.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 1728.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 211:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 3;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 1728.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 1728.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 212:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 1728.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 1728.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 213:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 1728.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 3;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 1728.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 214:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 1728.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = (15625.0 / 1728.0) * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    case 215:
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
      const Scalar x10 = x[2] - 1;
      const Scalar x11 = 5 * x[2];
      const Scalar x12 = x11 - 2;
      const Scalar x13 = x11 - 1;
      const Scalar x14 = x11 - 4;
      const Scalar x15 = x13 * x14;
      const Scalar x16 = x12 * x15;
      const Scalar x17 = x16 * x[2];
      const Scalar x18 = (15625.0 / 1728.0) * x10 * x17;
      const Scalar x19 = x[1] - 1;
      const Scalar x20 = 5 * x[1];
      const Scalar x21 = x20 - 2;
      const Scalar x22 = x20 - 1;
      const Scalar x23 = x20 - 4;
      const Scalar x24 = x22 * x23;
      const Scalar x25 = x21 * x24;
      const Scalar x26 = x25 * x[1];
      const Scalar x27 = x19 * x26;
      const Scalar x28 = x19 * x20;
      const Scalar x29 = x21 * x28;
      const Scalar x30 = x3 * x9;
      const Scalar x31 = x10 * x11;
      const Scalar x32 = x12 * x31;
      out[0] = -x18 * x27 * (x1 * x5 + x3 * x8 + x4 * x7 + x5 * x6 + x9);
      out[1] =
          -x18 * x30 * (x19 * x25 + x22 * x29 + x23 * x29 + x24 * x28 + x26);
      out[2] = -15625.0 / 1728.0 * x27 * x30 *
               (x10 * x16 + x13 * x32 + x14 * x32 + x15 * x31 + x17);
      break;
    default:
      break;
    }
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x[2] - 1;
    const Scalar x3 = 5 * x[0];
    const Scalar x4 = x3 - 4;
    const Scalar x5 = x3 - 3;
    const Scalar x6 = x3 - 2;
    const Scalar x7 = 5 * x[1];
    const Scalar x8 = x7 - 4;
    const Scalar x9 = x7 - 3;
    const Scalar x10 = x7 - 2;
    const Scalar x11 = 5 * x[2];
    const Scalar x12 = x11 - 4;
    const Scalar x13 = x11 - 3;
    const Scalar x14 = x11 - 2;
    const Scalar x15 = x3 - 1;
    const Scalar x16 = x0 * x15;
    const Scalar x17 = x5 * x8;
    const Scalar x18 = x13 * x9;
    const Scalar x19 = x1 * x2 * x[0];
    const Scalar x20 = x[1] * x[2];
    const Scalar x21 = x19 * x20;
    const Scalar x22 = x18 * x21;
    const Scalar x23 = x14 * x22;
    const Scalar x24 = x12 * x6;
    const Scalar x25 = x10 * x24;
    const Scalar x26 = x23 * x25;
    const Scalar x27 = x7 - 1;
    const Scalar x28 = 15625 * x5;
    const Scalar x29 = x0 * x4 * x8;
    const Scalar x30 = x10 * x29;
    const Scalar x31 = x11 - 1;
    const Scalar x32 = x23 * x6;
    const Scalar x33 = x15 * x27;
    const Scalar x34 = x0 * x31;
    const Scalar x35 = x10 * x32;
    const Scalar x36 = x15 * x29;
    const Scalar x37 = x10 * x12;
    const Scalar x38 = 31250 * x5;
    const Scalar x39 = x23 * x38;
    const Scalar x40 = x13 * x6;
    const Scalar x41 = x10 * x21;
    const Scalar x42 = x12 * x41;
    const Scalar x43 = x40 * x42;
    const Scalar x44 = x14 * x8;
    const Scalar x45 = x33 * x44;
    const Scalar x46 = 31250 * x10;
    const Scalar x47 = x24 * x33;
    const Scalar x48 = x23 * x47;
    const Scalar x49 = x4 * x48;
    const Scalar x50 = x27 * x29;
    const Scalar x51 = x24 * x50;
    const Scalar x52 = x15 * x31;
    const Scalar x53 = x29 * x52;
    const Scalar x54 = x34 * x39;
    const Scalar x55 = x6 * x8;
    const Scalar x56 = x33 * x4;
    const Scalar x57 = x10 * x56;
    const Scalar x58 = x14 * x41;
    const Scalar x59 = x40 * x58;
    const Scalar x60 = x31 * x59;
    const Scalar x61 = x24 * x31;
    const Scalar x62 = x10 * x22;
    const Scalar x63 = x61 * x62;
    const Scalar x64 = x14 * x9;
    const Scalar x65 = x42 * x64;
    const Scalar x66 = x38 * x65;
    const Scalar x67 = x34 * x47;
    const Scalar x68 = x34 * x4;
    const Scalar x69 = x13 * x14;
    const Scalar x70 = x33 * x42;
    const Scalar x71 = 62500 * x5;
    const Scalar x72 = x29 * x71;
    const Scalar x73 = 62500 * x29;
    const Scalar x74 = x0 * x15 * x27 * x31 * x4 * x8;
    const Scalar x75 = x21 * x64;
    const Scalar x76 = x27 * x72;
    const Scalar x77 = x31 * x43;
    const Scalar x78 = x22 * x8;
    const Scalar x79 = x34 * x71;
    const Scalar x80 = x6 * x70;
    const Scalar x81 = x64 * x80;
    const Scalar x82 = x0 * x12 * x15 * x27 * x31 * x4 * x5 * x8;
    const Scalar x83 = 125000 * x82;
    const Scalar x84 = 125000 * x74;
    const Scalar x85 = 625 * x5;
    const Scalar x86 = x10 * x85;
    const Scalar x87 = 1250 * x17;
    const Scalar x88 = x31 * x4;
    const Scalar x89 = 1250 * x5;
    const Scalar x90 = x61 * x89;
    const Scalar x91 = x10 * x4;
    const Scalar x92 = x15 * x91;
    const Scalar x93 = 2500 * x17;
    const Scalar x94 = x47 * x88;
    const Scalar x95 = x2 * x[1];
    const Scalar x96 = x95 * x[2];
    const Scalar x97 = x96 * x[0];
    const Scalar x98 = x10 * x97;
    const Scalar x99 = x18 * x98;
    const Scalar x100 = x0 * x85;
    const Scalar x101 = x44 * x47;
    const Scalar x102 = x31 * x50;
    const Scalar x103 = x14 * x18;
    const Scalar x104 = x103 * x85;
    const Scalar x105 = x103 * x89;
    const Scalar x106 = x33 * x37;
    const Scalar x107 = x50 * x90;
    const Scalar x108 = x14 * x74;
    const Scalar x109 = x108 * x6;
    const Scalar x110 = x67 * x87;
    const Scalar x111 = 2500 * x18;
    const Scalar x112 = x111 * x[0];
    const Scalar x113 = x25 * x74;
    const Scalar x114 = 2500 * x64;
    const Scalar x115 = x10 * x82;
    const Scalar x116 = x115 * x[0];
    const Scalar x117 = x18 * x44;
    const Scalar x118 = x1 * x20;
    const Scalar x119 = x118 * x[0];
    const Scalar x120 = x117 * x119;
    const Scalar x121 = x103 * x119;
    const Scalar x122 = x119 * x37;
    const Scalar x123 = x40 * x89;
    const Scalar x124 = x34 * x45;
    const Scalar x125 = 1250 * x103;
    const Scalar x126 = x67 * x91;
    const Scalar x127 = x111 * x82;
    const Scalar x128 = x1 * x[2];
    const Scalar x129 = x128 * x95;
    const Scalar x130 = x129 * x6;
    const Scalar x131 = 2500 * x14;
    const Scalar x132 = x19 * x[2];
    const Scalar x133 = x10 * x132;
    const Scalar x134 = x133 * x18;
    const Scalar x135 = x30 * x47;
    const Scalar x136 = x108 * x89;
    const Scalar x137 = x133 * x64;
    const Scalar x138 = 25 * x5;
    const Scalar x139 = x10 * x138;
    const Scalar x140 = 50 * x5;
    const Scalar x141 = x139 * x[0];
    const Scalar x142 = x117 * x67;
    const Scalar x143 = 50 * x103;
    const Scalar x144 = 50 * x17 * x94;
    const Scalar x145 = x19 * x[1];
    const Scalar x146 = x145 * x61;
    const Scalar x147 = x145 * x67;
    const Scalar x148 = x145 * x37;
    const Scalar x149 = x123 * x14;
    const Scalar x150 = x148 * x56;
    const Scalar x151 = x10 * x36;
    const Scalar x152 = x103 * x95;
    const Scalar x153 = x141 * x50;
    const Scalar x154 = x143 * x95;
    const Scalar x155 = x113 * x[0];
    const Scalar x156 = x103 * x61;
    const Scalar x157 = 50 * x115;
    const Scalar x158 = x157 * x6;
    const Scalar x159 =
        x10 * x12 * x13 * x14 * x15 * x27 * x31 * x4 * x5 * x6 * x8 * x9;
    const Scalar x160 = x159 * x[0];
    const Scalar x161 = x0 * x159;
    const Scalar x162 = x128 * x18 * x2;
    return -1.0 / 13824.0 * coeffs[0] * x1 * x161 * x2 -
           1.0 / 13824.0 * coeffs[100] * x127 * x130 +
           (625.0 / 3456.0) * coeffs[101] * x0 * x1 * x12 * x14 * x15 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[102] * x115 * x130 * x131 +
           (625.0 / 3456.0) * coeffs[103] * x0 * x1 * x10 * x12 * x13 * x15 *
               x2 * x27 * x31 * x4 * x5 * x6 * x8 * x[1] * x[2] +
           (625.0 / 13824.0) * coeffs[104] * x1 * x10 * x12 * x13 * x14 * x15 *
               x2 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[105] * x49 * x86 +
           (625.0 / 13824.0) * coeffs[106] * x1 * x10 * x13 * x14 * x15 * x2 *
               x27 * x31 * x4 * x5 * x6 * x9 * x[0] * x[1] * x[2] -
           625.0 / 13824.0 * coeffs[107] * x17 * x35 * x4 * x52 -
           1.0 / 13824.0 * coeffs[108] * x49 * x87 +
           (625.0 / 6912.0) * coeffs[109] * x1 * x10 * x12 * x13 * x14 * x15 *
               x2 * x27 * x4 * x5 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[10] * x115 * x143 * x19 +
           (625.0 / 6912.0) * coeffs[110] * x1 * x10 * x12 * x13 * x15 * x2 *
               x27 * x31 * x4 * x5 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[111] * x81 * x88 * x89 -
           1.0 / 13824.0 * coeffs[112] * x56 * x60 * x87 +
           (625.0 / 6912.0) * coeffs[113] * x1 * x13 * x14 * x15 * x2 * x27 *
               x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (625.0 / 6912.0) * coeffs[114] * x1 * x10 * x12 * x14 * x15 * x2 *
               x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[115] * x78 * x90 * x92 +
           (625.0 / 3456.0) * coeffs[116] * x1 * x12 * x13 * x15 * x2 * x27 *
               x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[117] * x56 * x77 * x93 +
           (625.0 / 3456.0) * coeffs[118] * x1 * x10 * x12 * x14 * x15 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[119] * x75 * x93 * x94 +
           (25.0 / 6912.0) * coeffs[11] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x27 * x31 * x4 * x6 * x8 * x9 * x[0] -
           1.0 / 13824.0 * coeffs[120] * x104 * x133 * x51 +
           (625.0 / 13824.0) * coeffs[121] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x27 * x5 * x6 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[122] * x124 * x134 * x6 * x85 +
           (625.0 / 13824.0) * coeffs[123] * x0 * x1 * x10 * x13 * x14 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[2] +
           (625.0 / 6912.0) * coeffs[124] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x27 * x4 * x5 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[125] * x125 * x132 * x135 -
           1.0 / 13824.0 * coeffs[126] * x110 * x134 +
           (625.0 / 6912.0) * coeffs[127] * x0 * x1 * x10 * x12 * x14 * x15 *
               x2 * x27 * x31 * x5 * x6 * x8 * x9 * x[0] * x[2] +
           (625.0 / 6912.0) * coeffs[128] * x0 * x1 * x10 * x13 * x14 * x15 *
               x2 * x27 * x31 * x4 * x6 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[129] * x134 * x136 -
           1.0 / 13824.0 * coeffs[12] * x117 * x138 * x146 * x92 -
           1.0 / 13824.0 * coeffs[130] * x107 * x137 +
           (625.0 / 6912.0) * coeffs[131] * x0 * x1 * x10 * x12 * x13 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[132] * x127 * x133 +
           (625.0 / 3456.0) * coeffs[133] * x0 * x1 * x10 * x12 * x13 * x15 *
               x2 * x27 * x31 * x4 * x6 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[134] * x113 * x114 * x132 +
           (625.0 / 3456.0) * coeffs[135] * x0 * x1 * x10 * x12 * x14 * x15 *
               x2 * x27 * x31 * x4 * x5 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[136] * x100 * x101 * x99 +
           (625.0 / 13824.0) * coeffs[137] * x0 * x10 * x12 * x13 * x14 * x2 *
               x27 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[138] * x102 * x104 * x6 * x98 +
           (625.0 / 13824.0) * coeffs[139] * x0 * x10 * x13 * x14 * x15 * x2 *
               x27 * x31 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (25.0 / 13824.0) * coeffs[13] * x1 * x10 * x12 * x13 * x14 * x15 *
               x2 * x27 * x31 * x4 * x5 * x6 * x9 * x[0] * x[1] +
           (625.0 / 6912.0) * coeffs[140] * x0 * x10 * x12 * x13 * x14 * x15 *
               x2 * x27 * x4 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[141] * x105 * x106 * x29 * x97 -
           1.0 / 13824.0 * coeffs[142] * x107 * x99 +
           (625.0 / 6912.0) * coeffs[143] * x0 * x10 * x12 * x14 * x2 * x27 *
               x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (625.0 / 6912.0) * coeffs[144] * x0 * x10 * x13 * x14 * x15 * x2 *
               x27 * x31 * x4 * x5 * x8 * x9 * x[0] * x[1] * x[2] -
           625.0 / 6912.0 * coeffs[145] * x109 * x99 -
           1.0 / 13824.0 * coeffs[146] * x110 * x64 * x98 +
           (625.0 / 6912.0) * coeffs[147] * x0 * x10 * x12 * x13 * x15 * x2 *
               x27 * x31 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[148] * x112 * x113 * x96 +
           (625.0 / 3456.0) * coeffs[149] * x0 * x10 * x12 * x13 * x15 * x2 *
               x27 * x31 * x4 * x5 * x8 * x9 * x[0] * x[1] * x[2] +
           (25.0 / 6912.0) * coeffs[14] * x1 * x12 * x13 * x14 * x15 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[150] * x114 * x116 * x96 +
           (625.0 / 3456.0) * coeffs[151] * x0 * x10 * x12 * x14 * x15 * x2 *
               x27 * x31 * x4 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (15625.0 / 13824.0) * coeffs[152] * x0 * x1 * x10 * x12 * x13 * x14 *
               x2 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           15625.0 / 13824.0 * coeffs[153] * x16 * x17 * x26 +
           (15625.0 / 13824.0) * coeffs[154] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x27 * x5 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[155] * x0 * x26 * x27 * x28 * x4 -
           1.0 / 13824.0 * coeffs[156] * x28 * x30 * x31 * x32 +
           (15625.0 / 13824.0) * coeffs[157] * x0 * x1 * x10 * x13 * x14 * x15 *
               x2 * x31 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[158] * x28 * x33 * x34 * x35 +
           (15625.0 / 13824.0) * coeffs[159] * x0 * x1 * x10 * x13 * x14 * x2 *
               x27 * x31 * x4 * x5 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[15] * x140 * x150 * x31 * x40 * x44 -
           1.0 / 13824.0 * coeffs[160] * x36 * x37 * x39 +
           (15625.0 / 6912.0) * coeffs[161] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x4 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (15625.0 / 6912.0) * coeffs[162] * x0 * x1 * x12 * x13 * x14 * x15 *
               x2 * x27 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[163] * x0 * x38 * x43 * x45 -
           1.0 / 13824.0 * coeffs[164] * x0 * x46 * x49 +
           (15625.0 / 6912.0) * coeffs[165] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x27 * x4 * x5 * x9 * x[0] * x[1] * x[2] +
           (15625.0 / 6912.0) * coeffs[166] * x0 * x1 * x10 * x12 * x13 * x14 *
               x2 * x27 * x4 * x5 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[167] * x39 * x51 +
           (15625.0 / 6912.0) * coeffs[168] * x0 * x1 * x10 * x13 * x14 * x15 *
               x2 * x31 * x4 * x5 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[169] * x32 * x46 * x53 +
           (25.0 / 13824.0) * coeffs[16] * x0 * x10 * x12 * x13 * x14 * x15 *
               x2 * x27 * x31 * x5 * x6 * x8 * x9 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[170] * x33 * x54 * x55 +
           (15625.0 / 6912.0) * coeffs[171] * x0 * x1 * x10 * x13 * x14 * x15 *
               x2 * x27 * x31 * x5 * x6 * x8 * x[0] * x[1] * x[2] +
           (15625.0 / 6912.0) * coeffs[172] * x0 * x1 * x10 * x13 * x14 * x15 *
               x2 * x27 * x31 * x4 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[173] * x54 * x57 -
           1.0 / 13824.0 * coeffs[174] * x38 * x50 * x60 +
           (15625.0 / 6912.0) * coeffs[175] * x0 * x1 * x13 * x14 * x2 * x27 *
               x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[176] * x29 * x38 * x63 +
           (15625.0 / 6912.0) * coeffs[177] * x0 * x1 * x10 * x12 * x14 * x2 *
               x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (15625.0 / 6912.0) * coeffs[178] * x0 * x1 * x10 * x12 * x13 * x15 *
               x2 * x31 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[179] * x15 * x34 * x55 * x66 -
           1.0 / 13824.0 * coeffs[17] * x152 * x153 * x61 -
           1.0 / 13824.0 * coeffs[180] * x38 * x62 * x67 +
           (15625.0 / 6912.0) * coeffs[181] * x0 * x1 * x10 * x12 * x14 * x15 *
               x2 * x27 * x31 * x5 * x6 * x9 * x[0] * x[1] * x[2] +
           (15625.0 / 6912.0) * coeffs[182] * x0 * x1 * x10 * x12 * x13 * x2 *
               x27 * x31 * x4 * x5 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[183] * x27 * x6 * x66 * x68 +
           (15625.0 / 3456.0) * coeffs[184] * x0 * x1 * x12 * x13 * x14 * x15 *
               x2 * x27 * x4 * x5 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[185] * x69 * x70 * x72 +
           (15625.0 / 3456.0) * coeffs[186] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x27 * x4 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[187] * x48 * x73 -
           1.0 / 13824.0 * coeffs[188] * x23 * x71 * x74 +
           (15625.0 / 3456.0) * coeffs[189] * x0 * x1 * x13 * x14 * x15 * x2 *
               x27 * x31 * x4 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[18] * x154 * x155 -
           15625.0 / 3456.0 * coeffs[190] * x59 * x74 +
           (15625.0 / 3456.0) * coeffs[191] * x0 * x1 * x10 * x13 * x14 * x15 *
               x2 * x27 * x31 * x4 * x5 * x8 * x[0] * x[1] * x[2] +
           (15625.0 / 3456.0) * coeffs[192] * x0 * x1 * x12 * x13 * x2 * x27 *
               x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[193] * x61 * x75 * x76 +
           (15625.0 / 3456.0) * coeffs[194] * x0 * x1 * x10 * x12 * x14 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[195] * x76 * x77 -
           1.0 / 13824.0 * coeffs[196] * x67 * x71 * x78 +
           (15625.0 / 3456.0) * coeffs[197] * x0 * x1 * x10 * x12 * x13 * x15 *
               x2 * x27 * x31 * x5 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[198] * x44 * x79 * x80 +
           (15625.0 / 3456.0) * coeffs[199] * x0 * x1 * x12 * x14 * x15 * x2 *
               x27 * x31 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (25.0 / 6912.0) * coeffs[19] * x0 * x10 * x12 * x13 * x14 * x15 *
               x2 * x27 * x31 * x4 * x5 * x8 * x9 * x[0] * x[1] +
           (1.0 / 13824.0) * coeffs[1] * x1 * x10 * x12 * x13 * x14 * x15 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[0] +
           (15625.0 / 3456.0) * coeffs[200] * x0 * x1 * x10 * x12 * x13 * x15 *
               x2 * x31 * x4 * x5 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[201] * x15 * x63 * x73 +
           (15625.0 / 3456.0) * coeffs[202] * x0 * x1 * x10 * x12 * x14 * x15 *
               x2 * x31 * x4 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[203] * x52 * x65 * x72 +
           (15625.0 / 3456.0) * coeffs[204] * x0 * x1 * x10 * x12 * x13 * x15 *
               x2 * x27 * x31 * x4 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[205] * x12 * x56 * x62 * x79 +
           (15625.0 / 3456.0) * coeffs[206] * x0 * x1 * x10 * x12 * x14 * x15 *
               x2 * x27 * x31 * x4 * x5 * x9 * x[0] * x[1] * x[2] -
           15625.0 / 3456.0 * coeffs[207] * x68 * x81 -
           1.0 / 13824.0 * coeffs[208] * x22 * x83 +
           (15625.0 / 1728.0) * coeffs[209] * x0 * x1 * x12 * x13 * x15 * x2 *
               x27 * x31 * x4 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[20] * x1 * x126 * x138 * x152 -
           1.0 / 13824.0 * coeffs[210] * x43 * x84 +
           (15625.0 / 1728.0) * coeffs[211] * x0 * x1 * x10 * x12 * x13 * x15 *
               x2 * x27 * x31 * x4 * x5 * x8 * x[0] * x[1] * x[2] +
           (15625.0 / 1728.0) * coeffs[212] * x0 * x1 * x12 * x14 * x15 * x2 *
               x27 * x31 * x4 * x5 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[213] * x24 * x75 * x84 +
           (15625.0 / 1728.0) * coeffs[214] * x0 * x1 * x10 * x12 * x14 * x15 *
               x2 * x27 * x31 * x4 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[215] * x58 * x83 +
           (25.0 / 13824.0) * coeffs[21] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x31 * x4 * x5 * x6 * x8 * x9 * x[1] +
           (25.0 / 6912.0) * coeffs[22] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x27 * x31 * x4 * x5 * x6 * x8 * x[1] -
           1.0 / 13824.0 * coeffs[23] * x1 * x154 * x6 * x82 -
           1.0 / 13824.0 * coeffs[24] * x128 * x153 * x156 +
           (25.0 / 13824.0) * coeffs[25] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x27 * x31 * x5 * x6 * x8 * x9 * x[0] * x[2] +
           (25.0 / 6912.0) * coeffs[26] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x27 * x31 * x4 * x5 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[27] * x128 * x143 * x155 +
           (25.0 / 13824.0) * coeffs[28] * x1 * x10 * x12 * x13 * x14 * x15 *
               x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[29] * x121 * x139 * x94 -
           1.0 / 13824.0 * coeffs[2] * x160 * x95 -
           1.0 / 13824.0 * coeffs[30] * x120 * x140 * x94 +
           (25.0 / 6912.0) * coeffs[31] * x1 * x10 * x12 * x13 * x14 * x15 *
               x27 * x31 * x4 * x5 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[32] * x141 * x142 * x20 +
           (25.0 / 13824.0) * coeffs[33] * x0 * x10 * x12 * x13 * x14 * x27 *
               x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (25.0 / 6912.0) * coeffs[34] * x0 * x10 * x12 * x13 * x14 * x15 *
               x27 * x31 * x4 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[35] * x116 * x143 * x20 +
           (25.0 / 13824.0) * coeffs[36] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x27 * x31 * x4 * x5 * x6 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[37] * x118 * x138 * x151 * x156 -
           1.0 / 13824.0 * coeffs[38] * x118 * x14 * x157 * x40 +
           (25.0 / 6912.0) * coeffs[39] * x0 * x1 * x12 * x13 * x14 * x15 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[1] * x[2] +
           (1.0 / 13824.0) * coeffs[3] * x0 * x10 * x12 * x13 * x14 * x15 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[1] +
           (25.0 / 13824.0) * coeffs[40] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x27 * x4 * x5 * x6 * x8 * x9 * x[2] -
           1.0 / 13824.0 * coeffs[41] * x109 * x139 * x162 -
           1.0 / 13824.0 * coeffs[42] * x158 * x162 +
           (25.0 / 6912.0) * coeffs[43] * x0 * x1 * x10 * x12 * x14 * x15 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[2] -
           1.0 / 13824.0 * coeffs[44] * x101 * x134 * x138 * x4 +
           (25.0 / 13824.0) * coeffs[45] * x1 * x10 * x13 * x14 * x15 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[2] +
           (25.0 / 6912.0) * coeffs[46] * x1 * x10 * x12 * x13 * x15 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[47] * x137 * x144 +
           (25.0 / 13824.0) * coeffs[48] * x10 * x12 * x13 * x14 * x15 * x2 *
               x27 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[49] * x117 * x138 * x31 * x57 * x6 * x97 +
           (1.0 / 13824.0) * coeffs[4] * x0 * x1 * x10 * x12 * x13 * x14 * x15 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[2] -
           1.0 / 13824.0 * coeffs[50] * x144 * x99 +
           (25.0 / 6912.0) * coeffs[51] * x10 * x12 * x14 * x15 * x2 * x27 *
               x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[52] * x103 * x135 * x138 * x96 +
           (25.0 / 13824.0) * coeffs[53] * x0 * x10 * x13 * x14 * x15 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[1] * x[2] +
           (25.0 / 6912.0) * coeffs[54] * x0 * x10 * x12 * x13 * x15 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[55] * x158 * x64 * x96 -
           1.0 / 13824.0 * coeffs[56] * x104 * x146 * x30 +
           (625.0 / 13824.0) * coeffs[57] * x0 * x1 * x10 * x12 * x13 * x14 *
               x2 * x27 * x31 * x4 * x5 * x6 * x9 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[58] * x103 * x147 * x86 +
           (625.0 / 13824.0) * coeffs[59] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x31 * x5 * x6 * x8 * x9 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[5] * x128 * x160 +
           (625.0 / 6912.0) * coeffs[60] * x0 * x1 * x12 * x13 * x14 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[61] * x102 * x148 * x149 -
           1.0 / 13824.0 * coeffs[62] * x105 * x150 * x34 +
           (625.0 / 6912.0) * coeffs[63] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x27 * x31 * x4 * x6 * x9 * x[0] * x[1] +
           (625.0 / 6912.0) * coeffs[64] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x27 * x31 * x5 * x6 * x8 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[65] * x117 * x147 * x89 -
           1.0 / 13824.0 * coeffs[66] * x125 * x146 * x151 +
           (625.0 / 6912.0) * coeffs[67] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x31 * x4 * x5 * x8 * x9 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[68] * x127 * x14 * x145 +
           (625.0 / 3456.0) * coeffs[69] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x27 * x31 * x4 * x5 * x8 * x[0] * x[1] +
           (1.0 / 13824.0) * coeffs[6] * x10 * x12 * x13 * x14 * x15 * x27 *
               x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[70] * x131 * x148 * x40 * x74 +
           (625.0 / 3456.0) * coeffs[71] * x0 * x1 * x12 * x13 * x14 * x15 *
               x2 * x27 * x31 * x4 * x6 * x8 * x9 * x[0] * x[1] +
           (625.0 / 13824.0) * coeffs[72] * x0 * x1 * x10 * x12 * x13 * x14 *
               x31 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[73] * x120 * x16 * x61 * x86 +
           (625.0 / 13824.0) * coeffs[74] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x27 * x31 * x5 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[75] * x100 * x121 * x27 * x61 * x91 -
           1.0 / 13824.0 * coeffs[76] * x105 * x122 * x53 +
           (625.0 / 6912.0) * coeffs[77] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x31 * x4 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (625.0 / 6912.0) * coeffs[78] * x0 * x1 * x12 * x13 * x14 * x15 *
               x27 * x31 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[79] * x122 * x123 * x124 -
           1.0 / 13824.0 * coeffs[7] * x161 * x20 -
           1.0 / 13824.0 * coeffs[80] * x119 * x125 * x126 +
           (625.0 / 6912.0) * coeffs[81] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x27 * x31 * x4 * x5 * x9 * x[0] * x[1] * x[2] +
           (625.0 / 6912.0) * coeffs[82] * x0 * x1 * x10 * x12 * x13 * x14 *
               x27 * x31 * x4 * x5 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[83] * x107 * x121 +
           (625.0 / 3456.0) * coeffs[84] * x0 * x1 * x12 * x13 * x14 * x15 *
               x27 * x31 * x4 * x5 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[85] * x108 * x112 * x118 * x24 +
           (625.0 / 3456.0) * coeffs[86] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x27 * x31 * x4 * x6 * x8 * x[0] * x[1] * x[2] -
           625.0 / 3456.0 * coeffs[87] * x116 * x118 * x69 -
           1.0 / 13824.0 * coeffs[88] * x104 * x129 * x25 * x36 +
           (625.0 / 13824.0) * coeffs[89] * x0 * x1 * x10 * x13 * x14 * x15 *
               x2 * x31 * x4 * x5 * x6 * x8 * x9 * x[1] * x[2] +
           (25.0 / 13824.0) * coeffs[8] * x0 * x1 * x10 * x12 * x13 * x14 * x2 *
               x27 * x31 * x4 * x5 * x6 * x8 * x9 * x[0] -
           1.0 / 13824.0 * coeffs[90] * x104 * x130 * x34 * x57 +
           (625.0 / 13824.0) * coeffs[91] * x0 * x1 * x10 * x12 * x13 * x14 *
               x15 * x2 * x27 * x4 * x5 * x6 * x9 * x[1] * x[2] +
           (625.0 / 6912.0) * coeffs[92] * x0 * x1 * x10 * x12 * x13 * x15 *
               x2 * x31 * x4 * x5 * x6 * x8 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[93] * x129 * x151 * x64 * x90 -
           1.0 / 13824.0 * coeffs[94] * x130 * x136 * x18 +
           (625.0 / 6912.0) * coeffs[95] * x0 * x1 * x10 * x13 * x14 * x15 *
               x2 * x27 * x31 * x4 * x5 * x6 * x8 * x[1] * x[2] +
           (625.0 / 6912.0) * coeffs[96] * x0 * x1 * x10 * x12 * x14 * x15 *
               x2 * x27 * x31 * x4 * x5 * x6 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[97] * x126 * x129 * x18 * x89 -
           1.0 / 13824.0 * coeffs[98] * x106 * x129 * x149 * x29 +
           (625.0 / 6912.0) * coeffs[99] * x0 * x1 * x12 * x13 * x14 * x15 *
               x2 * x27 * x4 * x5 * x6 * x8 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[9] * x139 * x142 * x19;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[2] - 1;
    const Scalar x2 = 5 * x[1];
    const Scalar x3 = x2 - 4;
    const Scalar x4 = x2 - 3;
    const Scalar x5 = x2 - 2;
    const Scalar x6 = 5 * x[2];
    const Scalar x7 = x6 - 4;
    const Scalar x8 = x6 - 3;
    const Scalar x9 = x6 - 2;
    const Scalar x10 = 5 * x[0];
    const Scalar x11 = x10 - 4;
    const Scalar x12 = x10 - 3;
    const Scalar x13 = x[0] - 1;
    const Scalar x14 = x10 * x13;
    const Scalar x15 = x12 * x14;
    const Scalar x16 = x11 * x15;
    const Scalar x17 = x10 - 2;
    const Scalar x18 = x14 * x17;
    const Scalar x19 = x11 * x18;
    const Scalar x20 = x12 * x17;
    const Scalar x21 = x14 * x20;
    const Scalar x22 = x11 * x20;
    const Scalar x23 = x22 * x[0];
    const Scalar x24 = x13 * x22;
    const Scalar x25 = x16 + x19 + x21 + x23 + x24;
    const Scalar x26 = x10 - 1;
    const Scalar x27 = x15 * x26;
    const Scalar x28 = x18 * x26;
    const Scalar x29 = x20 * x26;
    const Scalar x30 = x29 * x[0];
    const Scalar x31 = x13 * x29;
    const Scalar x32 = x21 + x27 + x28 + x30 + x31;
    const Scalar x33 = x4 * x5;
    const Scalar x34 = x3 * x33;
    const Scalar x35 = x34 * x[1];
    const Scalar x36 = x0 * x1;
    const Scalar x37 = x35 * x36;
    const Scalar x38 = x8 * x9;
    const Scalar x39 = x38 * x7;
    const Scalar x40 = x39 * x[2];
    const Scalar x41 = 15625 * x40;
    const Scalar x42 = coeffs[153] * x41;
    const Scalar x43 = x2 - 1;
    const Scalar x44 = x33 * x43;
    const Scalar x45 = x44 * x[1];
    const Scalar x46 = x36 * x45;
    const Scalar x47 = coeffs[155] * x41;
    const Scalar x48 = x25 * x37;
    const Scalar x49 = x6 - 1;
    const Scalar x50 = x38 * x49;
    const Scalar x51 = x50 * x[2];
    const Scalar x52 = 15625 * x51;
    const Scalar x53 = coeffs[156] * x52;
    const Scalar x54 = x32 * x46;
    const Scalar x55 = coeffs[158] * x52;
    const Scalar x56 = x11 * x26;
    const Scalar x57 = x14 * x56;
    const Scalar x58 = x12 * x56;
    const Scalar x59 = x58 * x[0];
    const Scalar x60 = x13 * x58;
    const Scalar x61 = x16 + x27 + x57 + x59 + x60;
    const Scalar x62 = x36 * x61;
    const Scalar x63 = x35 * x62;
    const Scalar x64 = 31250 * x40;
    const Scalar x65 = coeffs[160] * x64;
    const Scalar x66 = x17 * x56;
    const Scalar x67 = x66 * x[0];
    const Scalar x68 = x13 * x66;
    const Scalar x69 = x19 + x28 + x57 + x67 + x68;
    const Scalar x70 = x3 * x43;
    const Scalar x71 = x5 * x70;
    const Scalar x72 = x71 * x[1];
    const Scalar x73 = x36 * x72;
    const Scalar x74 = x32 * x73;
    const Scalar x75 = coeffs[163] * x64;
    const Scalar x76 = x36 * x69;
    const Scalar x77 = x45 * x76;
    const Scalar x78 = coeffs[164] * x64;
    const Scalar x79 = x4 * x70;
    const Scalar x80 = x79 * x[1];
    const Scalar x81 = x36 * x80;
    const Scalar x82 = x25 * x81;
    const Scalar x83 = coeffs[167] * x64;
    const Scalar x84 = x35 * x76;
    const Scalar x85 = 31250 * x51;
    const Scalar x86 = coeffs[169] * x85;
    const Scalar x87 = x32 * x81;
    const Scalar x88 = coeffs[170] * x85;
    const Scalar x89 = x45 * x62;
    const Scalar x90 = coeffs[173] * x85;
    const Scalar x91 = x25 * x73;
    const Scalar x92 = coeffs[174] * x85;
    const Scalar x93 = x49 * x7;
    const Scalar x94 = x8 * x93;
    const Scalar x95 = x94 * x[2];
    const Scalar x96 = x25 * x95;
    const Scalar x97 = 31250 * x37;
    const Scalar x98 = x9 * x93;
    const Scalar x99 = x98 * x[2];
    const Scalar x100 = x32 * x99;
    const Scalar x101 = x32 * x95;
    const Scalar x102 = 31250 * x46;
    const Scalar x103 = x25 * x99;
    const Scalar x104 = x62 * x72;
    const Scalar x105 = 62500 * x40;
    const Scalar x106 = coeffs[185] * x105;
    const Scalar x107 = x76 * x80;
    const Scalar x108 = coeffs[187] * x105;
    const Scalar x109 = x62 * x80;
    const Scalar x110 = 62500 * x51;
    const Scalar x111 = coeffs[188] * x110;
    const Scalar x112 = x72 * x76;
    const Scalar x113 = coeffs[190] * x110;
    const Scalar x114 = 62500 * x99;
    const Scalar x115 = coeffs[193] * x114;
    const Scalar x116 = 62500 * x95;
    const Scalar x117 = coeffs[195] * x116;
    const Scalar x118 = coeffs[196] * x116;
    const Scalar x119 = coeffs[198] * x114;
    const Scalar x120 = coeffs[201] * x116;
    const Scalar x121 = coeffs[203] * x114;
    const Scalar x122 = coeffs[205] * x116;
    const Scalar x123 = coeffs[207] * x114;
    const Scalar x124 = 125000 * x95;
    const Scalar x125 = coeffs[208] * x124;
    const Scalar x126 = 125000 * x99;
    const Scalar x127 = coeffs[213] * x126;
    const Scalar x128 = coeffs[215] * x126;
    const Scalar x129 = coeffs[136] * x40;
    const Scalar x130 = 625 * x32;
    const Scalar x131 = x33 * x70;
    const Scalar x132 = x1 * x[1];
    const Scalar x133 = x131 * x132;
    const Scalar x134 = 625 * x25;
    const Scalar x135 = x133 * x51;
    const Scalar x136 = x132 * x61;
    const Scalar x137 = 1250 * x131;
    const Scalar x138 = x137 * x40;
    const Scalar x139 = x132 * x137;
    const Scalar x140 = x132 * x69;
    const Scalar x141 = x137 * x51;
    const Scalar x142 = 2500 * x131;
    const Scalar x143 = x142 * x95;
    const Scalar x144 = x142 * x99;
    const Scalar x145 = coeffs[73] * x35;
    const Scalar x146 = x38 * x93;
    const Scalar x147 = x0 * x[2];
    const Scalar x148 = x146 * x147;
    const Scalar x149 = x148 * x45;
    const Scalar x150 = coeffs[76] * x35;
    const Scalar x151 = 1250 * x146;
    const Scalar x152 = x147 * x151;
    const Scalar x153 = coeffs[79] * x72;
    const Scalar x154 = coeffs[80] * x45;
    const Scalar x155 = coeffs[83] * x80;
    const Scalar x156 = coeffs[85] * x80;
    const Scalar x157 = 2500 * x146;
    const Scalar x158 = x147 * x157;
    const Scalar x159 = coeffs[87] * x72;
    const Scalar x160 = x131 * x36;
    const Scalar x161 = x160 * x40;
    const Scalar x162 = x160 * x51;
    const Scalar x163 = x137 * x36;
    const Scalar x164 = x[1] * x[2];
    const Scalar x165 = x131 * x146;
    const Scalar x166 = 25 * x165;
    const Scalar x167 = x166 * x32;
    const Scalar x168 = 50 * x165;
    const Scalar x169 = 625 * x146;
    const Scalar x170 = coeffs[56] * x169;
    const Scalar x171 = coeffs[58] * x169;
    const Scalar x172 = coeffs[61] * x151;
    const Scalar x173 = coeffs[62] * x151;
    const Scalar x174 = coeffs[65] * x151;
    const Scalar x175 = coeffs[66] * x151;
    const Scalar x176 = coeffs[68] * x157;
    const Scalar x177 = coeffs[70] * x157;
    const Scalar x178 = x166 * x25;
    const Scalar x179 = x20 * x56;
    const Scalar x180 = x10 * x22 + x10 * x29 + x10 * x58 + x10 * x66 + x179;
    const Scalar x181 = 625 * x180;
    const Scalar x182 = 1250 * x180;
    const Scalar x183 = coeffs[111] * x99;
    const Scalar x184 = coeffs[115] * x95;
    const Scalar x185 = 2500 * x180;
    const Scalar x186 = 25 * x180;
    const Scalar x187 = 50 * x148;
    const Scalar x188 = coeffs[30] * x80;
    const Scalar x189 = 50 * x133;
    const Scalar x190 = coeffs[50] * x95;
    const Scalar x191 = 25 * x146;
    const Scalar x192 = 50 * x146;
    const Scalar x193 = 50 * x160;
    const Scalar x194 = coeffs[47] * x99;
    const Scalar x195 = x165 * x180;
    const Scalar x196 = x179 + 5 * x24 + 5 * x31 + 5 * x60 + 5 * x68;
    const Scalar x197 = 2500 * x196;
    const Scalar x198 = coeffs[100] * x95;
    const Scalar x199 = coeffs[102] * x99;
    const Scalar x200 = 625 * x196;
    const Scalar x201 = coeffs[88] * x40;
    const Scalar x202 = coeffs[90] * x51;
    const Scalar x203 = 1250 * x196;
    const Scalar x204 = coeffs[93] * x99;
    const Scalar x205 = coeffs[97] * x95;
    const Scalar x206 = 25 * x196;
    const Scalar x207 = coeffs[38] * x72;
    const Scalar x208 = coeffs[55] * x99;
    const Scalar x209 = x165 * x196;
    const Scalar x210 = coeffs[42] * x95;
    const Scalar x211 = x0 * x2;
    const Scalar x212 = x211 * x4;
    const Scalar x213 = x212 * x3;
    const Scalar x214 = x211 * x5;
    const Scalar x215 = x214 * x3;
    const Scalar x216 = x211 * x33;
    const Scalar x217 = x0 * x34;
    const Scalar x218 = x213 + x215 + x216 + x217 + x35;
    const Scalar x219 = x1 * x13;
    const Scalar x220 = x218 * x219;
    const Scalar x221 = x212 * x43;
    const Scalar x222 = x214 * x43;
    const Scalar x223 = x0 * x44;
    const Scalar x224 = x216 + x221 + x222 + x223 + x45;
    const Scalar x225 = x219 * x224;
    const Scalar x226 = x220 * x23;
    const Scalar x227 = x225 * x30;
    const Scalar x228 = x220 * x59;
    const Scalar x229 = x211 * x70;
    const Scalar x230 = x0 * x79;
    const Scalar x231 = x213 + x221 + x229 + x230 + x80;
    const Scalar x232 = x0 * x71;
    const Scalar x233 = x215 + x222 + x229 + x232 + x72;
    const Scalar x234 = x219 * x233;
    const Scalar x235 = x234 * x30;
    const Scalar x236 = x225 * x67;
    const Scalar x237 = x219 * x231;
    const Scalar x238 = x23 * x237;
    const Scalar x239 = x220 * x67;
    const Scalar x240 = x237 * x30;
    const Scalar x241 = x225 * x59;
    const Scalar x242 = x23 * x234;
    const Scalar x243 = 31250 * x220;
    const Scalar x244 = x23 * x95;
    const Scalar x245 = x30 * x99;
    const Scalar x246 = 31250 * x225;
    const Scalar x247 = x30 * x95;
    const Scalar x248 = x23 * x99;
    const Scalar x249 = x234 * x59;
    const Scalar x250 = x237 * x67;
    const Scalar x251 = x237 * x59;
    const Scalar x252 = x234 * x67;
    const Scalar x253 = coeffs[210] * x67;
    const Scalar x254 = x1 * x[0];
    const Scalar x255 = x179 * x254;
    const Scalar x256 = 625 * x255;
    const Scalar x257 = 1250 * x40;
    const Scalar x258 = 1250 * x255;
    const Scalar x259 = 2500 * x179;
    const Scalar x260 = x254 * x259;
    const Scalar x261 = x13 * x30;
    const Scalar x262 = x218 * x[2];
    const Scalar x263 = x13 * x23;
    const Scalar x264 = x224 * x[2];
    const Scalar x265 = x13 * x59;
    const Scalar x266 = x151 * x[2];
    const Scalar x267 = x13 * x266;
    const Scalar x268 = x13 * x67;
    const Scalar x269 = x157 * x[2];
    const Scalar x270 = x146 * x179;
    const Scalar x271 = x270 * x[0];
    const Scalar x272 = 50 * x[2];
    const Scalar x273 = 625 * x179;
    const Scalar x274 = 1250 * x179;
    const Scalar x275 = 25 * x1;
    const Scalar x276 = 50 * x1;
    const Scalar x277 = x13 * x270;
    const Scalar x278 = x131 + x2 * x34 + x2 * x44 + x2 * x71 + x2 * x79;
    const Scalar x279 = x1 * x278;
    const Scalar x280 = 625 * x13;
    const Scalar x281 = x280 * x30;
    const Scalar x282 = x279 * x51;
    const Scalar x283 = x23 * x280;
    const Scalar x284 = coeffs[138] * x283;
    const Scalar x285 = 1250 * x279;
    const Scalar x286 = coeffs[142] * x13;
    const Scalar x287 = coeffs[145] * x268;
    const Scalar x288 = coeffs[148] * x268;
    const Scalar x289 = 2500 * x279;
    const Scalar x290 = coeffs[150] * x265;
    const Scalar x291 = x278 * x[2];
    const Scalar x292 = x191 * x261;
    const Scalar x293 = x192 * x265;
    const Scalar x294 = 25 * x179;
    const Scalar x295 = x294 * x[0];
    const Scalar x296 = 50 * x179;
    const Scalar x297 = x279 * x296;
    const Scalar x298 = x191 * x263;
    const Scalar x299 = x192 * x268;
    const Scalar x300 = x13 * x294;
    const Scalar x301 = x131 + 5 * x217 + 5 * x223 + 5 * x230 + 5 * x232;
    const Scalar x302 = x1 * x301;
    const Scalar x303 = x302 * x40;
    const Scalar x304 = coeffs[120] * x283;
    const Scalar x305 = x302 * x51;
    const Scalar x306 = coeffs[122] * x281;
    const Scalar x307 = 1250 * x302;
    const Scalar x308 = coeffs[126] * x13;
    const Scalar x309 = coeffs[132] * x265;
    const Scalar x310 = 2500 * x302;
    const Scalar x311 = coeffs[134] * x268;
    const Scalar x312 = x301 * x[2];
    const Scalar x313 = x296 * x302;
    const Scalar x314 = x1 * x6;
    const Scalar x315 = x314 * x8;
    const Scalar x316 = x315 * x7;
    const Scalar x317 = x314 * x9;
    const Scalar x318 = x317 * x7;
    const Scalar x319 = x314 * x38;
    const Scalar x320 = x1 * x39;
    const Scalar x321 = x316 + x318 + x319 + x320 + x40;
    const Scalar x322 = x0 * x35;
    const Scalar x323 = 15625 * x321;
    const Scalar x324 = x0 * x45;
    const Scalar x325 = x315 * x49;
    const Scalar x326 = x317 * x49;
    const Scalar x327 = x1 * x50;
    const Scalar x328 = x319 + x325 + x326 + x327 + x51;
    const Scalar x329 = 15625 * x328;
    const Scalar x330 = x0 * x265;
    const Scalar x331 = 31250 * x321;
    const Scalar x332 = x0 * x72;
    const Scalar x333 = x0 * x268;
    const Scalar x334 = x0 * x80;
    const Scalar x335 = 31250 * x328;
    const Scalar x336 = 31250 * x322;
    const Scalar x337 = x314 * x93;
    const Scalar x338 = x1 * x94;
    const Scalar x339 = x316 + x325 + x337 + x338 + x95;
    const Scalar x340 = x1 * x98;
    const Scalar x341 = x318 + x326 + x337 + x340 + x99;
    const Scalar x342 = x261 * x341;
    const Scalar x343 = 31250 * x324;
    const Scalar x344 = x263 * x341;
    const Scalar x345 = x330 * x72;
    const Scalar x346 = 62500 * x321;
    const Scalar x347 = x268 * x334;
    const Scalar x348 = x330 * x80;
    const Scalar x349 = 62500 * x328;
    const Scalar x350 = 62500 * x263;
    const Scalar x351 = x334 * x341;
    const Scalar x352 = x332 * x339;
    const Scalar x353 = 62500 * x261;
    const Scalar x354 = x334 * x339;
    const Scalar x355 = 62500 * x35;
    const Scalar x356 = 62500 * x45;
    const Scalar x357 = 125000 * x339;
    const Scalar x358 = x13 * x332;
    const Scalar x359 = 125000 * x341;
    const Scalar x360 = x273 * x[0];
    const Scalar x361 = x274 * x[0];
    const Scalar x362 = x259 * x[0];
    const Scalar x363 = x131 * x[1];
    const Scalar x364 = x137 * x321;
    const Scalar x365 = x137 * x[1];
    const Scalar x366 = x137 * x328;
    const Scalar x367 = x142 * x[1];
    const Scalar x368 = x0 * x131;
    const Scalar x369 = x0 * x137;
    const Scalar x370 = x0 * x142;
    const Scalar x371 = x131 * x179;
    const Scalar x372 = x371 * x[0];
    const Scalar x373 = 25 * x[1];
    const Scalar x374 = 50 * x[1];
    const Scalar x375 = x179 * x280;
    const Scalar x376 = x13 * x274;
    const Scalar x377 = 25 * x0;
    const Scalar x378 = 50 * x0;
    const Scalar x379 = x13 * x371;
    const Scalar x380 = x146 + x39 * x6 + x50 * x6 + x6 * x94 + x6 * x98;
    const Scalar x381 = x0 * x380;
    const Scalar x382 = x381 * x45;
    const Scalar x383 = 1250 * x381;
    const Scalar x384 = 2500 * x381;
    const Scalar x385 = x296 * x381;
    const Scalar x386 = x380 * x[1];
    const Scalar x387 = 25 * x131;
    const Scalar x388 = x261 * x387;
    const Scalar x389 = 50 * x131;
    const Scalar x390 = x265 * x389;
    const Scalar x391 = x263 * x387;
    const Scalar x392 = x268 * x389;
    const Scalar x393 = x146 + 5 * x320 + 5 * x327 + 5 * x338 + 5 * x340;
    const Scalar x394 = x0 * x393;
    const Scalar x395 = x35 * x394;
    const Scalar x396 = x394 * x45;
    const Scalar x397 = 1250 * x394;
    const Scalar x398 = 2500 * x394;
    const Scalar x399 = x296 * x394;
    const Scalar x400 = x393 * x[1];
    out[0] =
        -1.0 / 13824.0 * coeffs[0] * x209 * x36 +
        (625.0 / 3456.0) * coeffs[101] * x0 * x1 * x196 * x3 * x4 * x43 * x49 *
            x7 * x9 * x[1] * x[2] +
        (625.0 / 3456.0) * coeffs[103] * x0 * x1 * x196 * x3 * x43 * x49 * x5 *
            x7 * x8 * x[1] * x[2] +
        (625.0 / 13824.0) * coeffs[104] * x0 * x1 * x180 * x3 * x4 * x5 * x7 *
            x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[105] * x181 * x40 * x46 +
        (625.0 / 13824.0) * coeffs[106] * x0 * x1 * x180 * x4 * x43 * x49 * x5 *
            x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[107] * x181 * x37 * x51 -
        1.0 / 13824.0 * coeffs[108] * x182 * x40 * x81 +
        (625.0 / 6912.0) * coeffs[109] * x0 * x1 * x180 * x3 * x43 * x5 * x7 *
            x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[10] * x168 * x62 +
        (625.0 / 6912.0) * coeffs[110] * x0 * x1 * x180 * x4 * x43 * x49 * x5 *
            x7 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[112] * x182 * x51 * x73 +
        (625.0 / 6912.0) * coeffs[113] * x0 * x1 * x180 * x3 * x4 * x43 * x49 *
            x8 * x9 * x[1] * x[2] +
        (625.0 / 6912.0) * coeffs[114] * x0 * x1 * x180 * x3 * x4 * x49 * x5 *
            x7 * x9 * x[1] * x[2] +
        (625.0 / 3456.0) * coeffs[116] * x0 * x1 * x180 * x3 * x4 * x43 * x49 *
            x7 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[117] * x185 * x73 * x95 +
        (625.0 / 3456.0) * coeffs[118] * x0 * x1 * x180 * x3 * x43 * x49 * x5 *
            x7 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[119] * x185 * x81 * x99 +
        (25.0 / 6912.0) * coeffs[11] * x0 * x1 * x3 * x4 * x43 * x49 * x5 *
            x69 * x7 * x8 * x9 -
        1.0 / 13824.0 * coeffs[120] * x134 * x161 +
        (625.0 / 13824.0) * coeffs[121] * x0 * x1 * x3 * x32 * x4 * x43 * x5 *
            x7 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[122] * x130 * x162 +
        (625.0 / 13824.0) * coeffs[123] * x0 * x1 * x25 * x3 * x4 * x43 * x49 *
            x5 * x8 * x9 * x[2] +
        (625.0 / 6912.0) * coeffs[124] * x0 * x1 * x3 * x4 * x43 * x5 * x61 *
            x7 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[125] * x138 * x76 -
        1.0 / 13824.0 * coeffs[126] * x101 * x163 +
        (625.0 / 6912.0) * coeffs[127] * x0 * x1 * x3 * x32 * x4 * x43 * x49 *
            x5 * x7 * x9 * x[2] +
        (625.0 / 6912.0) * coeffs[128] * x0 * x1 * x3 * x4 * x43 * x49 * x5 *
            x69 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[129] * x141 * x62 -
        1.0 / 13824.0 * coeffs[12] * x180 * x191 * x37 -
        1.0 / 13824.0 * coeffs[130] * x103 * x163 +
        (625.0 / 6912.0) * coeffs[131] * x0 * x1 * x25 * x3 * x4 * x43 * x49 *
            x5 * x7 * x8 * x[2] -
        1.0 / 13824.0 * coeffs[132] * x143 * x62 +
        (625.0 / 3456.0) * coeffs[133] * x0 * x1 * x3 * x4 * x43 * x49 * x5 *
            x69 * x7 * x8 * x[2] -
        1.0 / 13824.0 * coeffs[134] * x144 * x76 +
        (625.0 / 3456.0) * coeffs[135] * x0 * x1 * x3 * x4 * x43 * x49 * x5 *
            x61 * x7 * x9 * x[2] +
        (625.0 / 13824.0) * coeffs[137] * x1 * x25 * x3 * x4 * x43 * x5 * x7 *
            x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[138] * x134 * x135 +
        (625.0 / 13824.0) * coeffs[139] * x1 * x3 * x32 * x4 * x43 * x49 * x5 *
            x8 * x9 * x[1] * x[2] +
        (25.0 / 13824.0) * coeffs[13] * x0 * x1 * x180 * x4 * x43 * x49 * x5 *
            x7 * x8 * x9 * x[1] +
        (625.0 / 6912.0) * coeffs[140] * x1 * x3 * x4 * x43 * x5 * x69 * x7 *
            x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[141] * x136 * x138 -
        1.0 / 13824.0 * coeffs[142] * x139 * x96 +
        (625.0 / 6912.0) * coeffs[143] * x1 * x25 * x3 * x4 * x43 * x49 * x5 *
            x7 * x9 * x[1] * x[2] +
        (625.0 / 6912.0) * coeffs[144] * x1 * x3 * x4 * x43 * x49 * x5 * x61 *
            x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[145] * x140 * x141 -
        1.0 / 13824.0 * coeffs[146] * x100 * x139 +
        (625.0 / 6912.0) * coeffs[147] * x1 * x3 * x32 * x4 * x43 * x49 * x5 *
            x7 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[148] * x140 * x143 +
        (625.0 / 3456.0) * coeffs[149] * x1 * x3 * x4 * x43 * x49 * x5 * x61 *
            x7 * x8 * x[1] * x[2] +
        (25.0 / 6912.0) * coeffs[14] * x0 * x1 * x180 * x3 * x4 * x43 * x49 *
            x7 * x8 * x9 * x[1] -
        1.0 / 13824.0 * coeffs[150] * x136 * x144 +
        (625.0 / 3456.0) * coeffs[151] * x1 * x3 * x4 * x43 * x49 * x5 * x69 *
            x7 * x9 * x[1] * x[2] +
        (15625.0 / 13824.0) * coeffs[152] * x0 * x1 * x25 * x3 * x4 * x5 * x7 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 13824.0) * coeffs[154] * x0 * x1 * x32 * x4 * x43 * x5 * x7 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 13824.0) * coeffs[157] * x0 * x1 * x3 * x32 * x4 * x49 * x5 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 13824.0) * coeffs[159] * x0 * x1 * x25 * x4 * x43 * x49 *
            x5 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[15] * x180 * x192 * x73 +
        (15625.0 / 6912.0) * coeffs[161] * x0 * x1 * x3 * x4 * x5 * x69 * x7 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[162] * x0 * x1 * x3 * x32 * x4 * x43 * x7 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[165] * x0 * x1 * x4 * x43 * x5 * x61 * x7 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[166] * x0 * x1 * x25 * x3 *
            x43 * x5 * x7 * x8 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[168] * x0 * x1 * x3 * x4 * x49 * x5 *
            x61 * x8 * x9 * x[1] * x[2] +
        (25.0 / 13824.0) * coeffs[16] * x1 * x3 * x32 * x4 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[1] +
        (15625.0 / 6912.0) * coeffs[171] * x0 * x1 * x3 * x32 * x43 *
            x49 * x5 * x8 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[172] * x0 * x1 * x4 * x43 * x49 * x5 *
            x69 * x8 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[175] * x0 * x1 * x25 * x3 * x4 * x43 *
            x49 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[176] * x96 * x97 +
        (15625.0 / 6912.0) * coeffs[177] * x0 * x1 * x25 * x3 * x4 *
            x49 * x5 * x7 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[178] * x0 * x1 * x3 * x32 * x4 *
            x49 * x5 * x7 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[179] * x100 * x97 -
        1.0 / 13824.0 * coeffs[17] * x132 * x178 -
        1.0 / 13824.0 * coeffs[180] * x101 * x102 +
        (15625.0 / 6912.0) * coeffs[181] * x0 * x1 * x32 * x4 * x43 *
            x49 * x5 * x7 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[182] * x0 * x1 * x25 * x4 * x43 *
            x49 * x5 * x7 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[183] * x102 * x103 +
        (15625.0 / 3456.0) * coeffs[184] * x0 * x1 * x3 * x4 * x43 *
            x61 * x7 * x8 * x9 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[186] * x0 * x1 * x3 * x43 * x5 *
            x69 * x7 * x8 * x9 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[189] * x0 * x1 * x3 * x4 * x43 * x49 *
            x69 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[18] * x140 * x168 +
        (15625.0 / 3456.0) * coeffs[191] * x0 * x1 * x3 * x43 * x49 * x5 *
            x61 * x8 * x9 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[192] * x0 * x1 * x25 * x3 * x4 * x43 *
            x49 * x7 * x8 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[194] * x0 * x1 * x25 * x3 * x43 *
            x49 * x5 * x7 * x9 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[197] * x0 * x1 * x3 * x32 * x43 *
            x49 * x5 * x7 * x8 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[199] * x0 * x1 * x3 * x32 * x4 * x43 *
            x49 * x7 * x9 * x[1] * x[2] +
        (25.0 / 6912.0) * coeffs[19] * x1 * x3 * x4 * x43 * x49 * x5 *
            x61 * x7 * x8 * x9 * x[1] +
        (1.0 / 13824.0) * coeffs[1] * x0 * x1 * x180 * x3 * x4 * x43 *
            x49 * x5 * x7 * x8 * x9 +
        (15625.0 / 3456.0) * coeffs[200] * x0 * x1 * x3 * x4 * x49 * x5 *
            x61 * x7 * x8 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[202] * x0 * x1 * x3 * x4 * x49 * x5 *
            x69 * x7 * x9 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[204] * x0 * x1 * x4 * x43 * x49 * x5 *
            x69 * x7 * x8 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[206] * x0 * x1 * x4 * x43 * x49 * x5 *
            x61 * x7 * x9 * x[1] * x[2] +
        (15625.0 / 1728.0) * coeffs[209] * x0 * x1 * x3 * x4 * x43 * x49 *
            x69 * x7 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[20] * x191 * x196 * x46 -
        1.0 / 13824.0 * coeffs[210] * x124 * x72 * x76 +
        (15625.0 / 1728.0) * coeffs[211] * x0 * x1 * x3 * x43 * x49 * x5 *
            x61 * x7 * x8 * x[1] * x[2] +
        (15625.0 / 1728.0) * coeffs[212] * x0 * x1 * x3 * x4 * x43 * x49 *
            x61 * x7 * x9 * x[1] * x[2] +
        (15625.0 / 1728.0) * coeffs[214] * x0 * x1 * x3 * x43 * x49 * x5 *
            x69 * x7 * x9 * x[1] * x[2] +
        (25.0 / 13824.0) * coeffs[21] * x0 * x1 * x196 * x3 * x4 *
            x49 * x5 * x7 * x8 * x9 * x[1] +
        (25.0 / 6912.0) * coeffs[22] * x0 * x1 * x196 * x3 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[1] -
        1.0 / 13824.0 * coeffs[23] * x192 * x196 * x81 -
        1.0 / 13824.0 * coeffs[24] * x147 * x178 +
        (25.0 / 13824.0) * coeffs[25] * x0 * x3 * x32 * x4 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[2] +
        (25.0 / 6912.0) * coeffs[26] * x0 * x3 * x4 * x43 * x49 * x5 *
            x61 * x7 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[27] * x147 * x168 * x69 +
        (25.0 / 13824.0) * coeffs[28] * x0 * x180 * x3 * x4 *
            x49 * x5 * x7 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[29] * x149 * x186 -
        1.0 / 13824.0 * coeffs[2] * x132 * x195 +
        (25.0 / 6912.0) * coeffs[31] * x0 * x180 * x3 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[32] * x164 * x167 +
        (25.0 / 13824.0) * coeffs[33] * x25 * x3 * x4 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[1] * x[2] +
        (25.0 / 6912.0) * coeffs[34] * x3 * x4 * x43 * x49 * x5 *
            x69 * x7 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[35] * x164 * x168 * x61 +
        (25.0 / 13824.0) * coeffs[36] * x0 * x196 * x4 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[37] * x148 * x206 * x35 +
        (25.0 / 6912.0) * coeffs[39] * x0 * x196 * x3 * x4 * x43 *
            x49 * x7 * x8 * x9 * x[1] * x[2] +
        (1.0 / 13824.0) * coeffs[3] * x1 * x196 * x3 * x4 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[1] +
        (25.0 / 13824.0) * coeffs[40] * x0 * x1 * x196 * x3 * x4 *
            x43 * x5 * x7 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[41] * x162 * x206 +
        (25.0 / 6912.0) * coeffs[43] * x0 * x1 * x196 * x3 * x4 * x43 *
            x49 * x5 * x7 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[44] * x161 * x186 +
        (25.0 / 13824.0) * coeffs[45] * x0 * x1 * x180 * x3 * x4 * x43 *
            x49 * x5 * x8 * x9 * x[2] +
        (25.0 / 6912.0) * coeffs[46] * x0 * x1 * x180 * x3 * x4 * x43 *
            x49 * x5 * x7 * x8 * x[2] +
        (25.0 / 13824.0) * coeffs[48] * x1 * x180 * x3 * x4 *
            x43 * x5 * x7 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[49] * x135 * x186 +
        (1.0 / 13824.0) * coeffs[4] * x0 * x196 * x3 * x4 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[2] +
        (25.0 / 6912.0) * coeffs[51] * x1 * x180 * x3 * x4 * x43 *
            x49 * x5 * x7 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[52] * x133 * x206 * x40 +
        (25.0 / 13824.0) * coeffs[53] * x1 * x196 * x3 * x4 * x43 *
            x49 * x5 * x8 * x9 * x[1] * x[2] +
        (25.0 / 6912.0) * coeffs[54] * x1 * x196 * x3 * x4 * x43 *
            x49 * x5 * x7 * x8 * x[1] * x[2] +
        (625.0 / 13824.0) * coeffs[57] * x0 * x1 * x25 * x4 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[1] +
        (625.0 / 13824.0) * coeffs[59] * x0 * x1 * x3 * x32 * x4 *
            x49 * x5 * x7 * x8 * x9 * x[1] -
        1.0 / 13824.0 * coeffs[5] * x147 * x195 +
        (625.0 / 6912.0) * coeffs[60] * x0 * x1 * x25 * x3 * x4 * x43 *
            x49 * x7 * x8 * x9 * x[1] +
        (625.0 / 6912.0) * coeffs[63] * x0 * x1 * x4 * x43 * x49 * x5 *
            x69 * x7 * x8 * x9 * x[1] +
        (625.0 / 6912.0) * coeffs[64] * x0 * x1 * x3 * x32 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[1] +
        (625.0 / 6912.0) * coeffs[67] * x0 * x1 * x3 * x4 * x49 * x5 *
            x61 * x7 * x8 * x9 * x[1] +
        (625.0 / 3456.0) * coeffs[69] * x0 * x1 * x3 * x43 * x49 * x5 *
            x61 * x7 * x8 * x9 * x[1] +
        (1.0 / 13824.0) * coeffs[6] * x180 * x3 * x4 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[1] * x[2] +
        (625.0 / 3456.0) * coeffs[71] * x0 * x1 * x3 * x4 * x43 * x49 *
            x69 * x7 * x8 * x9 * x[1] +
        (625.0 / 13824.0) * coeffs[72] * x0 * x25 * x3 * x4 *
            x49 * x5 * x7 * x8 * x9 * x[1] * x[2] +
        (625.0 / 13824.0) * coeffs[74] * x0 * x32 * x4 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[75] * x134 * x149 +
        (625.0 / 6912.0) * coeffs[77] * x0 * x3 * x4 * x49 * x5 *
            x69 * x7 * x8 * x9 * x[1] * x[2] +
        (625.0 / 6912.0) * coeffs[78] * x0 * x3 * x32 * x4 * x43 *
            x49 * x7 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[7] * x164 * x209 +
        (625.0 / 6912.0) * coeffs[81] * x0 * x4 * x43 * x49 * x5 *
            x61 * x7 * x8 * x9 * x[1] * x[2] +
        (625.0 / 6912.0) * coeffs[82] * x0 * x25 * x3 * x43 *
            x49 * x5 * x7 * x8 * x9 * x[1] * x[2] +
        (625.0 / 3456.0) * coeffs[84] * x0 * x3 * x4 * x43 * x49 *
            x61 * x7 * x8 * x9 * x[1] * x[2] +
        (625.0 / 3456.0) * coeffs[86] * x0 * x3 * x43 * x49 * x5 *
            x69 * x7 * x8 * x9 * x[1] * x[2] +
        (625.0 / 13824.0) * coeffs[89] * x0 * x1 * x196 * x3 * x4 *
            x49 * x5 * x8 * x9 * x[1] * x[2] +
        (25.0 / 13824.0) * coeffs[8] * x0 * x1 * x25 * x3 * x4 * x43 *
            x49 * x5 * x7 * x8 * x9 +
        (625.0 / 13824.0) * coeffs[91] * x0 * x1 * x196 * x4 *
            x43 * x5 * x7 * x8 * x9 * x[1] * x[2] +
        (625.0 / 6912.0) * coeffs[92] * x0 * x1 * x196 * x3 * x4 *
            x49 * x5 * x7 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[94] * x203 * x51 * x81 +
        (625.0 / 6912.0) * coeffs[95] * x0 * x1 * x196 * x3 * x43 *
            x49 * x5 * x8 * x9 * x[1] * x[2] +
        (625.0 / 6912.0) * coeffs[96] * x0 * x1 * x196 * x4 * x43 *
            x49 * x5 * x7 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[98] * x203 * x40 * x73 +
        (625.0 / 6912.0) * coeffs[99] * x0 * x1 * x196 * x3 * x4 *
            x43 * x7 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[9] * x167 * x36 - 1.0 / 13824.0 * x104 * x106 -
        1.0 / 13824.0 * x104 * x128 - 1.0 / 13824.0 * x107 * x108 -
        1.0 / 13824.0 * x107 * x127 - 1.0 / 13824.0 * x109 * x111 -
        1.0 / 13824.0 * x109 * x125 - 1.0 / 13824.0 * x109 * x176 -
        1.0 / 13824.0 * x112 * x113 - 1.0 / 13824.0 * x112 * x177 -
        1.0 / 13824.0 * x115 * x82 - 1.0 / 13824.0 * x117 * x91 -
        1.0 / 13824.0 * x118 * x87 - 1.0 / 13824.0 * x119 * x74 -
        1.0 / 13824.0 * x120 * x84 - 1.0 / 13824.0 * x121 * x63 -
        1.0 / 13824.0 * x122 * x89 - 1.0 / 13824.0 * x123 * x77 -
        1.0 / 13824.0 * x129 * x130 * x133 -
        1.0 / 13824.0 * x130 * x145 * x148 - 1.0 / 13824.0 * x150 * x152 * x61 -
        1.0 / 13824.0 * x152 * x153 * x32 - 1.0 / 13824.0 * x152 * x154 * x69 -
        1.0 / 13824.0 * x152 * x155 * x25 - 1.0 / 13824.0 * x156 * x158 * x69 -
        1.0 / 13824.0 * x158 * x159 * x61 - 1.0 / 13824.0 * x170 * x48 -
        1.0 / 13824.0 * x171 * x54 - 1.0 / 13824.0 * x172 * x91 -
        1.0 / 13824.0 * x173 * x89 - 1.0 / 13824.0 * x174 * x87 -
        1.0 / 13824.0 * x175 * x84 - 1.0 / 13824.0 * x180 * x187 * x188 -
        1.0 / 13824.0 * x180 * x189 * x190 -
        1.0 / 13824.0 * x180 * x193 * x194 - 1.0 / 13824.0 * x182 * x183 * x46 -
        1.0 / 13824.0 * x182 * x184 * x37 - 1.0 / 13824.0 * x187 * x196 * x207 -
        1.0 / 13824.0 * x189 * x196 * x208 -
        1.0 / 13824.0 * x193 * x196 * x210 - 1.0 / 13824.0 * x197 * x198 * x81 -
        1.0 / 13824.0 * x197 * x199 * x73 - 1.0 / 13824.0 * x200 * x201 * x37 -
        1.0 / 13824.0 * x200 * x202 * x46 - 1.0 / 13824.0 * x203 * x204 * x37 -
        1.0 / 13824.0 * x203 * x205 * x46 - 1.0 / 13824.0 * x25 * x46 * x47 -
        1.0 / 13824.0 * x32 * x37 * x42 - 1.0 / 13824.0 * x48 * x53 -
        1.0 / 13824.0 * x54 * x55 - 1.0 / 13824.0 * x63 * x65 -
        1.0 / 13824.0 * x74 * x75 - 1.0 / 13824.0 * x77 * x78 -
        1.0 / 13824.0 * x82 * x83 - 1.0 / 13824.0 * x84 * x86 -
        1.0 / 13824.0 * x87 * x88 - 1.0 / 13824.0 * x89 * x90 -
        1.0 / 13824.0 * x91 * x92;
    out[1] =
        -1.0 / 13824.0 * coeffs[0] * x277 * x302 +
        (625.0 / 3456.0) * coeffs[101] * x1 * x11 * x12 * x13 * x17 * x231 *
            x26 * x49 * x7 * x9 * x[2] +
        (625.0 / 3456.0) * coeffs[103] * x1 * x11 * x12 * x13 * x17 * x233 *
            x26 * x49 * x7 * x8 * x[2] +
        (625.0 / 13824.0) * coeffs[104] * x1 * x11 * x12 * x17 * x218 * x26 *
            x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[105] * x224 * x256 * x40 +
        (625.0 / 13824.0) * coeffs[106] * x1 * x11 * x12 * x17 * x224 * x26 *
            x49 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[107] * x218 * x256 * x51 -
        1.0 / 13824.0 * coeffs[108] * x231 * x255 * x257 +
        (625.0 / 6912.0) * coeffs[109] * x1 * x11 * x12 * x17 * x233 * x26 *
            x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[10] * x293 * x302 +
        (625.0 / 6912.0) * coeffs[110] * x1 * x11 * x12 * x17 * x224 * x26 *
            x49 * x7 * x8 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[112] * x233 * x258 * x51 +
        (625.0 / 6912.0) * coeffs[113] * x1 * x11 * x12 * x17 * x231 * x26 *
            x49 * x8 * x9 * x[0] * x[2] +
        (625.0 / 6912.0) * coeffs[114] * x1 * x11 * x12 * x17 * x218 * x26 *
            x49 * x7 * x9 * x[0] * x[2] +
        (625.0 / 3456.0) * coeffs[116] * x1 * x11 * x12 * x17 * x231 * x26 *
            x49 * x7 * x8 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[117] * x233 * x260 * x95 +
        (625.0 / 3456.0) * coeffs[118] * x1 * x11 * x12 * x17 * x233 * x26 *
            x49 * x7 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[119] * x231 * x260 * x99 +
        (25.0 / 6912.0) * coeffs[11] * x1 * x11 * x13 * x17 * x26 * x301 * x49 *
            x7 * x8 * x9 * x[0] +
        (625.0 / 13824.0) * coeffs[121] * x1 * x12 * x13 * x17 * x26 * x301 *
            x7 * x8 * x9 * x[0] * x[2] +
        (625.0 / 13824.0) * coeffs[123] * x1 * x11 * x12 * x13 * x17 * x301 *
            x49 * x8 * x9 * x[0] * x[2] +
        (625.0 / 6912.0) * coeffs[124] * x1 * x11 * x12 * x13 * x26 * x301 *
            x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[125] * x257 * x268 * x302 +
        (625.0 / 6912.0) * coeffs[127] * x1 * x12 * x13 * x17 * x26 * x301 *
            x49 * x7 * x9 * x[0] * x[2] +
        (625.0 / 6912.0) * coeffs[128] * x1 * x11 * x13 * x17 * x26 * x301 *
            x49 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[129] * x265 * x307 * x51 -
        1.0 / 13824.0 * coeffs[12] * x218 * x271 * x275 -
        1.0 / 13824.0 * coeffs[130] * x13 * x248 * x307 +
        (625.0 / 6912.0) * coeffs[131] * x1 * x11 * x12 * x13 * x17 * x301 *
            x49 * x7 * x8 * x[0] * x[2] +
        (625.0 / 3456.0) * coeffs[133] * x1 * x11 * x13 * x17 * x26 * x301 *
            x49 * x7 * x8 * x[0] * x[2] +
        (625.0 / 3456.0) * coeffs[135] * x1 * x11 * x12 * x13 * x26 * x301 *
            x49 * x7 * x9 * x[0] * x[2] +
        (625.0 / 13824.0) * coeffs[137] * x1 * x11 * x12 * x13 * x17 * x278 *
            x7 * x8 * x9 * x[0] * x[2] +
        (625.0 / 13824.0) * coeffs[139] * x1 * x12 * x13 * x17 * x26 * x278 *
            x49 * x8 * x9 * x[0] * x[2] +
        (25.0 / 13824.0) * coeffs[13] * x1 * x11 * x12 * x17 * x224 * x26 *
            x49 * x7 * x8 * x9 * x[0] +
        (625.0 / 6912.0) * coeffs[140] * x1 * x11 * x13 * x17 * x26 * x278 *
            x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[141] * x257 * x265 * x279 +
        (625.0 / 6912.0) * coeffs[143] * x1 * x11 * x12 * x13 * x17 * x278 *
            x49 * x7 * x9 * x[0] * x[2] +
        (625.0 / 6912.0) * coeffs[144] * x1 * x11 * x12 * x13 * x26 * x278 *
            x49 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[146] * x13 * x245 * x285 +
        (625.0 / 6912.0) * coeffs[147] * x1 * x12 * x13 * x17 * x26 * x278 *
            x49 * x7 * x8 * x[0] * x[2] +
        (625.0 / 3456.0) * coeffs[149] * x1 * x11 * x12 * x13 * x26 * x278 *
            x49 * x7 * x8 * x[0] * x[2] +
        (25.0 / 6912.0) * coeffs[14] * x1 * x11 * x12 * x17 * x231 * x26 * x49 *
            x7 * x8 * x9 * x[0] +
        (625.0 / 3456.0) * coeffs[151] * x1 * x11 * x13 * x17 * x26 * x278 *
            x49 * x7 * x9 * x[0] * x[2] +
        (15625.0 / 13824.0) * coeffs[152] * x1 * x11 * x12 * x13 * x17 * x218 *
            x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 13824.0) * coeffs[154] * x1 * x12 * x13 * x17 * x224 * x26 *
            x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 13824.0) * coeffs[157] * x1 * x12 * x13 * x17 * x218 * x26 *
            x49 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 13824.0) * coeffs[159] * x1 * x11 * x12 * x13 * x17 * x224 *
            x49 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[15] * x233 * x271 * x276 +
        (15625.0 / 6912.0) * coeffs[161] * x1 * x11 * x13 * x17 * x218 * x26 *
            x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[162] * x1 * x12 * x13 * x17 * x231 * x26 *
            x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[165] * x1 * x11 * x12 * x13 * x224 * x26 *
            x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[166] * x1 * x11 * x12 * x13 * x17 * x233 *
            x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[168] * x1 * x11 * x12 * x13 * x218 * x26 *
            x49 * x8 * x9 * x[0] * x[2] +
        (25.0 / 13824.0) * coeffs[16] * x1 * x12 * x13 * x17 * x26 * x278 *
            x49 * x7 * x8 * x9 * x[0] +
        (15625.0 / 6912.0) * coeffs[171] * x1 * x12 * x13 * x17 * x233 * x26 *
            x49 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[172] * x1 * x11 * x13 * x17 * x224 * x26 *
            x49 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[175] * x1 * x11 * x12 * x13 * x17 * x231 *
            x49 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[176] * x243 * x244 +
        (15625.0 / 6912.0) * coeffs[177] * x1 * x11 * x12 * x13 * x17 * x218 *
            x49 * x7 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[178] * x1 * x12 * x13 * x17 * x218 * x26 *
            x49 * x7 * x8 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[179] * x243 * x245 -
        1.0 / 13824.0 * coeffs[17] * x279 * x298 -
        1.0 / 13824.0 * coeffs[180] * x246 * x247 +
        (15625.0 / 6912.0) * coeffs[181] * x1 * x12 * x13 * x17 * x224 * x26 *
            x49 * x7 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[182] * x1 * x11 * x12 * x13 * x17 * x224 *
            x49 * x7 * x8 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[183] * x246 * x248 +
        (15625.0 / 3456.0) * coeffs[184] * x1 * x11 * x12 * x13 * x231 *
            x26 * x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[186] * x1 * x11 * x13 * x17 * x233 *
            x26 * x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[189] * x1 * x11 * x13 * x17 * x231 * x26 *
            x49 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[18] * x279 * x299 +
        (15625.0 / 3456.0) * coeffs[191] * x1 * x11 * x12 * x13 * x233 * x26 *
            x49 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[192] * x1 * x11 * x12 * x13 * x17 * x231 *
            x49 * x7 * x8 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[194] * x1 * x11 * x12 * x13 * x17 * x233 *
            x49 * x7 * x9 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[197] * x1 * x12 * x13 * x17 * x233 * x26 *
            x49 * x7 * x8 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[199] * x1 * x12 * x13 * x17 * x231 * x26 *
            x49 * x7 * x9 * x[0] * x[2] +
        (25.0 / 6912.0) * coeffs[19] * x1 * x11 * x12 * x13 * x26 * x278 *
            x49 * x7 * x8 * x9 * x[0] +
        (1.0 / 13824.0) * coeffs[1] * x1 * x11 * x12 * x17 * x26 * x301 *
            x49 * x7 * x8 * x9 * x[0] +
        (15625.0 / 3456.0) * coeffs[200] * x1 * x11 * x12 * x13 * x218 * x26 *
            x49 * x7 * x8 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[202] * x1 * x11 * x13 * x17 * x218 * x26 *
            x49 * x7 * x9 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[204] * x1 * x11 * x13 * x17 * x224 * x26 *
            x49 * x7 * x8 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[206] * x1 * x11 * x12 * x13 * x224 * x26 *
            x49 * x7 * x9 * x[0] * x[2] +
        (15625.0 / 1728.0) * coeffs[209] * x1 * x11 * x13 * x17 * x231 * x26 *
            x49 * x7 * x8 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[20] * x224 * x275 * x277 +
        (15625.0 / 1728.0) * coeffs[211] * x1 * x11 * x12 * x13 * x233 * x26 *
            x49 * x7 * x8 * x[0] * x[2] +
        (15625.0 / 1728.0) * coeffs[212] * x1 * x11 * x12 * x13 * x231 * x26 *
            x49 * x7 * x9 * x[0] * x[2] +
        (15625.0 / 1728.0) * coeffs[214] * x1 * x11 * x13 * x17 * x233 * x26 *
            x49 * x7 * x9 * x[0] * x[2] +
        (25.0 / 13824.0) * coeffs[21] * x1 * x11 * x12 * x13 * x17 * x218 *
            x26 * x49 * x7 * x8 * x9 +
        (25.0 / 6912.0) * coeffs[22] * x1 * x11 * x12 * x13 * x17 * x233 * x26 *
            x49 * x7 * x8 * x9 -
        1.0 / 13824.0 * coeffs[23] * x231 * x276 * x277 -
        1.0 / 13824.0 * coeffs[24] * x298 * x312 +
        (25.0 / 13824.0) * coeffs[25] * x12 * x13 * x17 * x26 * x301 *
            x49 * x7 * x8 * x9 * x[0] * x[2] +
        (25.0 / 6912.0) * coeffs[26] * x11 * x12 * x13 * x26 * x301 *
            x49 * x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[27] * x299 * x312 +
        (25.0 / 13824.0) * coeffs[28] * x11 * x12 * x17 * x218 * x26 *
            x49 * x7 * x8 * x9 * x[0] * x[2] -
        25.0 / 13824.0 * coeffs[29] * x264 * x271 -
        1.0 / 13824.0 * coeffs[2] * x271 * x279 -
        1.0 / 13824.0 * coeffs[30] * x231 * x271 * x272 +
        (25.0 / 6912.0) * coeffs[31] * x11 * x12 * x17 * x233 * x26 *
            x49 * x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[32] * x291 * x292 +
        (25.0 / 13824.0) * coeffs[33] * x11 * x12 * x13 * x17 * x278 *
            x49 * x7 * x8 * x9 * x[0] * x[2] +
        (25.0 / 6912.0) * coeffs[34] * x11 * x13 * x17 * x26 * x278 *
            x49 * x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[35] * x291 * x293 +
        (25.0 / 13824.0) * coeffs[36] * x11 * x12 * x13 * x17 * x224 * x26 *
            x49 * x7 * x8 * x9 * x[2] -
        25.0 / 13824.0 * coeffs[37] * x262 * x277 -
        1.0 / 13824.0 * coeffs[38] * x233 * x272 * x277 +
        (25.0 / 6912.0) * coeffs[39] * x11 * x12 * x13 * x17 * x231 * x26 *
            x49 * x7 * x8 * x9 * x[2] +
        (1.0 / 13824.0) * coeffs[3] * x1 * x11 * x12 * x13 * x17 * x26 * x278 *
            x49 * x7 * x8 * x9 +
        (25.0 / 13824.0) * coeffs[40] * x1 * x11 * x12 * x13 * x17 * x26 *
            x301 * x7 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[41] * x300 * x305 +
        (25.0 / 6912.0) * coeffs[43] * x1 * x11 * x12 * x13 * x17 * x26 * x301 *
            x49 * x7 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[44] * x295 * x303 +
        (25.0 / 13824.0) * coeffs[45] * x1 * x11 * x12 * x17 * x26 * x301 *
            x49 * x8 * x9 * x[0] * x[2] +
        (25.0 / 6912.0) * coeffs[46] * x1 * x11 * x12 * x17 * x26 * x301 *
            x49 * x7 * x8 * x[0] * x[2] +
        (25.0 / 13824.0) * coeffs[48] * x1 * x11 * x12 * x17 * x26 *
            x278 * x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[49] * x282 * x295 +
        (1.0 / 13824.0) * coeffs[4] * x11 * x12 * x13 * x17 * x26 * x301 *
            x49 * x7 * x8 * x9 * x[2] +
        (25.0 / 6912.0) * coeffs[51] * x1 * x11 * x12 * x17 * x26 * x278 *
            x49 * x7 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[52] * x279 * x300 * x40 +
        (25.0 / 13824.0) * coeffs[53] * x1 * x11 * x12 * x13 * x17 * x26 *
            x278 * x49 * x8 * x9 * x[2] +
        (25.0 / 6912.0) * coeffs[54] * x1 * x11 * x12 * x13 * x17 * x26 * x278 *
            x49 * x7 * x8 * x[2] +
        (625.0 / 13824.0) * coeffs[57] * x1 * x11 * x12 * x13 * x17 * x224 *
            x49 * x7 * x8 * x9 * x[0] +
        (625.0 / 13824.0) * coeffs[59] * x1 * x12 * x13 * x17 * x218 * x26 *
            x49 * x7 * x8 * x9 * x[0] -
        1.0 / 13824.0 * coeffs[5] * x271 * x312 +
        (625.0 / 6912.0) * coeffs[60] * x1 * x11 * x12 * x13 * x17 * x231 *
            x49 * x7 * x8 * x9 * x[0] +
        (625.0 / 6912.0) * coeffs[63] * x1 * x11 * x13 * x17 * x224 * x26 *
            x49 * x7 * x8 * x9 * x[0] +
        (625.0 / 6912.0) * coeffs[64] * x1 * x12 * x13 * x17 * x233 * x26 *
            x49 * x7 * x8 * x9 * x[0] +
        (625.0 / 6912.0) * coeffs[67] * x1 * x11 * x12 * x13 * x218 * x26 *
            x49 * x7 * x8 * x9 * x[0] +
        (625.0 / 3456.0) * coeffs[69] * x1 * x11 * x12 * x13 * x233 * x26 *
            x49 * x7 * x8 * x9 * x[0] +
        (1.0 / 13824.0) * coeffs[6] * x11 * x12 * x17 * x26 * x278 *
            x49 * x7 * x8 * x9 * x[0] * x[2] +
        (625.0 / 3456.0) * coeffs[71] * x1 * x11 * x13 * x17 * x231 * x26 *
            x49 * x7 * x8 * x9 * x[0] +
        (625.0 / 13824.0) * coeffs[72] * x11 * x12 * x13 * x17 * x218 *
            x49 * x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[73] * x169 * x261 * x262 +
        (625.0 / 13824.0) * coeffs[74] * x12 * x13 * x17 * x224 * x26 *
            x49 * x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[75] * x169 * x263 * x264 -
        1.0 / 13824.0 * coeffs[76] * x218 * x265 * x266 +
        (625.0 / 6912.0) * coeffs[77] * x11 * x13 * x17 * x218 * x26 *
            x49 * x7 * x8 * x9 * x[0] * x[2] +
        (625.0 / 6912.0) * coeffs[78] * x12 * x13 * x17 * x231 * x26 *
            x49 * x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[79] * x233 * x267 * x30 -
        1.0 / 13824.0 * coeffs[7] * x277 * x291 -
        1.0 / 13824.0 * coeffs[80] * x224 * x266 * x268 +
        (625.0 / 6912.0) * coeffs[81] * x11 * x12 * x13 * x224 * x26 *
            x49 * x7 * x8 * x9 * x[0] * x[2] +
        (625.0 / 6912.0) * coeffs[82] * x11 * x12 * x13 * x17 * x233 *
            x49 * x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[83] * x23 * x231 * x267 +
        (625.0 / 3456.0) * coeffs[84] * x11 * x12 * x13 * x231 * x26 *
            x49 * x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[85] * x231 * x268 * x269 +
        (625.0 / 3456.0) * coeffs[86] * x11 * x13 * x17 * x233 * x26 *
            x49 * x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[87] * x233 * x265 * x269 +
        (625.0 / 13824.0) * coeffs[89] * x1 * x11 * x12 * x13 * x17 * x218 *
            x26 * x49 * x8 * x9 * x[2] +
        (25.0 / 13824.0) * coeffs[8] * x1 * x11 * x12 * x13 * x17 * x301 *
            x49 * x7 * x8 * x9 * x[0] +
        (625.0 / 13824.0) * coeffs[91] * x1 * x11 * x12 * x13 * x17 * x224 *
            x26 * x7 * x8 * x9 * x[2] +
        (625.0 / 6912.0) * coeffs[92] * x1 * x11 * x12 * x13 * x17 * x218 *
            x26 * x49 * x7 * x8 * x[2] -
        1.0 / 13824.0 * coeffs[94] * x237 * x274 * x51 +
        (625.0 / 6912.0) * coeffs[95] * x1 * x11 * x12 * x13 * x17 * x233 *
            x26 * x49 * x8 * x9 * x[2] +
        (625.0 / 6912.0) * coeffs[96] * x1 * x11 * x12 * x13 * x17 * x224 *
            x26 * x49 * x7 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[98] * x179 * x234 * x257 +
        (625.0 / 6912.0) * coeffs[99] * x1 * x11 * x12 * x13 * x17 * x231 *
            x26 * x7 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[9] * x292 * x302 - 1.0 / 13824.0 * x106 * x249 -
        1.0 / 13824.0 * x108 * x250 - 1.0 / 13824.0 * x111 * x251 -
        1.0 / 13824.0 * x113 * x252 - 1.0 / 13824.0 * x115 * x238 -
        1.0 / 13824.0 * x117 * x242 - 1.0 / 13824.0 * x118 * x240 -
        1.0 / 13824.0 * x119 * x235 - 1.0 / 13824.0 * x120 * x239 -
        1.0 / 13824.0 * x121 * x228 - 1.0 / 13824.0 * x122 * x241 -
        1.0 / 13824.0 * x123 * x236 - 1.0 / 13824.0 * x124 * x234 * x253 -
        1.0 / 13824.0 * x125 * x251 - 1.0 / 13824.0 * x127 * x250 -
        1.0 / 13824.0 * x128 * x249 - 1.0 / 13824.0 * x129 * x279 * x281 -
        1.0 / 13824.0 * x13 * x208 * x297 - 1.0 / 13824.0 * x13 * x210 * x313 -
        1.0 / 13824.0 * x170 * x226 - 1.0 / 13824.0 * x171 * x227 -
        1.0 / 13824.0 * x172 * x242 - 1.0 / 13824.0 * x173 * x241 -
        1.0 / 13824.0 * x174 * x240 - 1.0 / 13824.0 * x175 * x239 -
        1.0 / 13824.0 * x176 * x251 - 1.0 / 13824.0 * x177 * x252 -
        1.0 / 13824.0 * x183 * x224 * x258 -
        1.0 / 13824.0 * x184 * x218 * x258 -
        1.0 / 13824.0 * x190 * x297 * x[0] -
        1.0 / 13824.0 * x194 * x313 * x[0] -
        1.0 / 13824.0 * x198 * x237 * x259 -
        1.0 / 13824.0 * x199 * x234 * x259 -
        1.0 / 13824.0 * x201 * x220 * x273 -
        1.0 / 13824.0 * x202 * x225 * x273 -
        1.0 / 13824.0 * x204 * x220 * x274 -
        1.0 / 13824.0 * x205 * x225 * x274 - 1.0 / 13824.0 * x220 * x30 * x42 -
        1.0 / 13824.0 * x225 * x23 * x47 - 1.0 / 13824.0 * x226 * x53 -
        1.0 / 13824.0 * x227 * x55 - 1.0 / 13824.0 * x228 * x65 -
        1.0 / 13824.0 * x235 * x75 - 1.0 / 13824.0 * x236 * x78 -
        1.0 / 13824.0 * x238 * x83 - 1.0 / 13824.0 * x239 * x86 -
        1.0 / 13824.0 * x240 * x88 - 1.0 / 13824.0 * x241 * x90 -
        1.0 / 13824.0 * x242 * x92 - 1.0 / 13824.0 * x244 * x285 * x286 -
        1.0 / 13824.0 * x247 * x307 * x308 - 1.0 / 13824.0 * x282 * x284 -
        1.0 / 13824.0 * x285 * x287 * x51 - 1.0 / 13824.0 * x288 * x289 * x95 -
        1.0 / 13824.0 * x289 * x290 * x99 - 1.0 / 13824.0 * x303 * x304 -
        1.0 / 13824.0 * x305 * x306 - 1.0 / 13824.0 * x309 * x310 * x95 -
        1.0 / 13824.0 * x310 * x311 * x99;
    out[2] =
        -1.0 / 13824.0 * coeffs[0] * x379 * x394 -
        1.0 / 13824.0 * coeffs[100] * x13 * x259 * x354 +
        (625.0 / 3456.0) * coeffs[101] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x341 * x4 * x43 * x[1] -
        1.0 / 13824.0 * coeffs[102] * x259 * x341 * x358 +
        (625.0 / 3456.0) * coeffs[103] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x339 * x43 * x5 * x[1] +
        (625.0 / 13824.0) * coeffs[104] * x0 * x11 * x12 * x17 * x26 * x3 *
            x321 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[105] * x321 * x324 * x360 +
        (625.0 / 13824.0) * coeffs[106] * x0 * x11 * x12 * x17 * x26 * x328 *
            x4 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[107] * x322 * x328 * x360 -
        1.0 / 13824.0 * coeffs[108] * x321 * x334 * x361 +
        (625.0 / 6912.0) * coeffs[109] * x0 * x11 * x12 * x17 * x26 * x3 *
            x321 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[10] * x390 * x394 +
        (625.0 / 6912.0) * coeffs[110] * x0 * x11 * x12 * x17 * x26 * x339 *
            x4 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[111] * x324 * x341 * x361 -
        1.0 / 13824.0 * coeffs[112] * x328 * x332 * x361 +
        (625.0 / 6912.0) * coeffs[113] * x0 * x11 * x12 * x17 * x26 * x3 *
            x328 * x4 * x43 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[114] * x0 * x11 * x12 * x17 * x26 * x3 *
            x341 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[115] * x322 * x339 * x361 +
        (625.0 / 3456.0) * coeffs[116] * x0 * x11 * x12 * x17 * x26 * x3 *
            x339 * x4 * x43 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[117] * x352 * x362 +
        (625.0 / 3456.0) * coeffs[118] * x0 * x11 * x12 * x17 * x26 * x3 *
            x341 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[119] * x351 * x362 +
        (25.0 / 6912.0) * coeffs[11] * x0 * x11 * x13 * x17 * x26 * x3 * x393 *
            x4 * x43 * x5 * x[0] +
        (625.0 / 13824.0) * coeffs[121] * x0 * x12 * x13 * x17 * x26 * x3 *
            x321 * x4 * x43 * x5 * x[0] +
        (625.0 / 13824.0) * coeffs[123] * x0 * x11 * x12 * x13 * x17 * x3 *
            x328 * x4 * x43 * x5 * x[0] +
        (625.0 / 6912.0) * coeffs[124] * x0 * x11 * x12 * x13 * x26 * x3 *
            x321 * x4 * x43 * x5 * x[0] -
        1.0 / 13824.0 * coeffs[125] * x333 * x364 +
        (625.0 / 6912.0) * coeffs[127] * x0 * x12 * x13 * x17 * x26 * x3 *
            x341 * x4 * x43 * x5 * x[0] +
        (625.0 / 6912.0) * coeffs[128] * x0 * x11 * x13 * x17 * x26 * x3 *
            x328 * x4 * x43 * x5 * x[0] -
        1.0 / 13824.0 * coeffs[129] * x330 * x366 -
        1.0 / 13824.0 * coeffs[12] * x295 * x395 -
        1.0 / 13824.0 * coeffs[130] * x344 * x369 +
        (625.0 / 6912.0) * coeffs[131] * x0 * x11 * x12 * x13 * x17 * x3 *
            x339 * x4 * x43 * x5 * x[0] +
        (625.0 / 3456.0) * coeffs[133] * x0 * x11 * x13 * x17 * x26 * x3 *
            x339 * x4 * x43 * x5 * x[0] +
        (625.0 / 3456.0) * coeffs[135] * x0 * x11 * x12 * x13 * x26 * x3 *
            x341 * x4 * x43 * x5 * x[0] -
        1.0 / 13824.0 * coeffs[136] * x281 * x321 * x363 +
        (625.0 / 13824.0) * coeffs[137] * x11 * x12 * x13 * x17 * x3 * x321 *
            x4 * x43 * x5 * x[0] * x[1] +
        (625.0 / 13824.0) * coeffs[139] * x12 * x13 * x17 * x26 * x3 * x328 *
            x4 * x43 * x5 * x[0] * x[1] +
        (25.0 / 13824.0) * coeffs[13] * x0 * x11 * x12 * x17 * x26 * x393 * x4 *
            x43 * x5 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[140] * x11 * x13 * x17 * x26 * x3 * x321 *
            x4 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[141] * x265 * x364 * x[1] +
        (625.0 / 6912.0) * coeffs[143] * x11 * x12 * x13 * x17 * x3 * x341 *
            x4 * x43 * x5 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[144] * x11 * x12 * x13 * x26 * x3 * x328 *
            x4 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[146] * x342 * x365 +
        (625.0 / 6912.0) * coeffs[147] * x12 * x13 * x17 * x26 * x3 * x339 *
            x4 * x43 * x5 * x[0] * x[1] +
        (625.0 / 3456.0) * coeffs[149] * x11 * x12 * x13 * x26 * x3 * x339 *
            x4 * x43 * x5 * x[0] * x[1] +
        (25.0 / 6912.0) * coeffs[14] * x0 * x11 * x12 * x17 * x26 * x3 * x393 *
            x4 * x43 * x[0] * x[1] +
        (625.0 / 3456.0) * coeffs[151] * x11 * x13 * x17 * x26 * x3 * x341 *
            x4 * x43 * x5 * x[0] * x[1] +
        (15625.0 / 13824.0) * coeffs[152] * x0 * x11 * x12 * x13 * x17 * x3 *
            x321 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[153] * x261 * x322 * x323 +
        (15625.0 / 13824.0) * coeffs[154] * x0 * x12 * x13 * x17 * x26 * x321 *
            x4 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[155] * x263 * x323 * x324 -
        1.0 / 13824.0 * coeffs[156] * x263 * x322 * x329 +
        (15625.0 / 13824.0) * coeffs[157] * x0 * x12 * x13 * x17 * x26 * x3 *
            x328 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[158] * x261 * x324 * x329 +
        (15625.0 / 13824.0) * coeffs[159] * x0 * x11 * x12 * x13 * x17 * x328 *
            x4 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[15] * x399 * x72 * x[0] -
        1.0 / 13824.0 * coeffs[160] * x330 * x331 * x35 +
        (15625.0 / 6912.0) * coeffs[161] * x0 * x11 * x13 * x17 * x26 * x3 *
            x321 * x4 * x5 * x[0] * x[1] +
        (15625.0 / 6912.0) * coeffs[162] * x0 * x12 * x13 * x17 * x26 * x3 *
            x321 * x4 * x43 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[163] * x261 * x331 * x332 -
        1.0 / 13824.0 * coeffs[164] * x331 * x333 * x45 +
        (15625.0 / 6912.0) * coeffs[165] * x0 * x11 * x12 * x13 * x26 * x321 *
            x4 * x43 * x5 * x[0] * x[1] +
        (15625.0 / 6912.0) * coeffs[166] * x0 * x11 * x12 * x13 * x17 * x3 *
            x321 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[167] * x263 * x331 * x334 +
        (15625.0 / 6912.0) * coeffs[168] * x0 * x11 * x12 * x13 * x26 * x3 *
            x328 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[169] * x333 * x335 * x35 +
        (25.0 / 13824.0) * coeffs[16] * x12 * x13 * x17 * x26 * x3 * x393 * x4 *
            x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[170] * x261 * x334 * x335 +
        (15625.0 / 6912.0) * coeffs[171] * x0 * x12 * x13 * x17 * x26 * x3 *
            x328 * x43 * x5 * x[0] * x[1] +
        (15625.0 / 6912.0) * coeffs[172] * x0 * x11 * x13 * x17 * x26 *
            x328 * x4 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[173] * x330 * x335 * x45 -
        1.0 / 13824.0 * coeffs[174] * x263 * x332 * x335 +
        (15625.0 / 6912.0) * coeffs[175] * x0 * x11 * x12 * x13 * x17 * x3 *
            x328 * x4 * x43 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[176] * x263 * x336 * x339 +
        (15625.0 / 6912.0) * coeffs[177] * x0 * x11 * x12 * x13 * x17 * x3 *
            x341 * x4 * x5 * x[0] * x[1] +
        (15625.0 / 6912.0) * coeffs[178] * x0 * x12 * x13 * x17 * x26 * x3 *
            x339 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[179] * x336 * x342 -
        1.0 / 13824.0 * coeffs[17] * x391 * x400 -
        1.0 / 13824.0 * coeffs[180] * x261 * x339 * x343 +
        (15625.0 / 6912.0) * coeffs[181] * x0 * x12 * x13 * x17 * x26 *
            x341 * x4 * x43 * x5 * x[0] * x[1] +
        (15625.0 / 6912.0) * coeffs[182] * x0 * x11 * x12 * x13 * x17 *
            x339 * x4 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[183] * x343 * x344 +
        (15625.0 / 3456.0) * coeffs[184] * x0 * x11 * x12 * x13 * x26 * x3 *
            x321 * x4 * x43 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[185] * x345 * x346 +
        (15625.0 / 3456.0) * coeffs[186] * x0 * x11 * x13 * x17 * x26 * x3 *
            x321 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[187] * x346 * x347 -
        1.0 / 13824.0 * coeffs[188] * x348 * x349 +
        (15625.0 / 3456.0) * coeffs[189] * x0 * x11 * x13 * x17 * x26 * x3 *
            x328 * x4 * x43 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[18] * x392 * x400 -
        1.0 / 13824.0 * coeffs[190] * x268 * x332 * x349 +
        (15625.0 / 3456.0) * coeffs[191] * x0 * x11 * x12 * x13 * x26 * x3 *
            x328 * x43 * x5 * x[0] * x[1] +
        (15625.0 / 3456.0) * coeffs[192] * x0 * x11 * x12 * x13 * x17 * x3 *
            x339 * x4 * x43 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[193] * x350 * x351 +
        (15625.0 / 3456.0) * coeffs[194] * x0 * x11 * x12 * x13 * x17 * x3 *
            x341 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[195] * x350 * x352 -
        1.0 / 13824.0 * coeffs[196] * x353 * x354 +
        (15625.0 / 3456.0) * coeffs[197] * x0 * x12 * x13 * x17 * x26 * x3 *
            x339 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[198] * x332 * x341 * x353 +
        (15625.0 / 3456.0) * coeffs[199] * x0 * x12 * x13 * x17 * x26 * x3 *
            x341 * x4 * x43 * x[0] * x[1] +
        (25.0 / 6912.0) * coeffs[19] * x11 * x12 * x13 * x26 * x3 * x393 * x4 *
            x43 * x5 * x[0] * x[1] +
        (1.0 / 13824.0) * coeffs[1] * x0 * x11 * x12 * x17 * x26 * x3 *
            x393 * x4 * x43 * x5 * x[0] +
        (15625.0 / 3456.0) * coeffs[200] * x0 * x11 * x12 * x13 * x26 * x3 *
            x339 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[201] * x333 * x339 * x355 +
        (15625.0 / 3456.0) * coeffs[202] * x0 * x11 * x13 * x17 * x26 * x3 *
            x341 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[203] * x330 * x341 * x355 +
        (15625.0 / 3456.0) * coeffs[204] * x0 * x11 * x13 * x17 * x26 *
            x339 * x4 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[205] * x330 * x339 * x356 +
        (15625.0 / 3456.0) * coeffs[206] * x0 * x11 * x12 * x13 * x26 *
            x341 * x4 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[207] * x333 * x341 * x356 -
        1.0 / 13824.0 * coeffs[208] * x348 * x357 +
        (15625.0 / 1728.0) * coeffs[209] * x0 * x11 * x13 * x17 * x26 * x3 *
            x339 * x4 * x43 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[20] * x300 * x396 +
        (15625.0 / 1728.0) * coeffs[211] * x0 * x11 * x12 * x13 * x26 * x3 *
            x339 * x43 * x5 * x[0] * x[1] +
        (15625.0 / 1728.0) * coeffs[212] * x0 * x11 * x12 * x13 * x26 * x3 *
            x341 * x4 * x43 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[213] * x347 * x359 +
        (15625.0 / 1728.0) * coeffs[214] * x0 * x11 * x13 * x17 * x26 * x3 *
            x341 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[215] * x345 * x359 +
        (25.0 / 13824.0) * coeffs[21] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x393 * x4 * x5 * x[1] +
        (25.0 / 6912.0) * coeffs[22] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x393 * x43 * x5 * x[1] -
        1.0 / 13824.0 * coeffs[23] * x13 * x399 * x80 -
        1.0 / 13824.0 * coeffs[24] * x381 * x391 +
        (25.0 / 13824.0) * coeffs[25] * x0 * x12 * x13 * x17 * x26 * x3 *
            x380 * x4 * x43 * x5 * x[0] +
        (25.0 / 6912.0) * coeffs[26] * x0 * x11 * x12 * x13 * x26 * x3 *
            x380 * x4 * x43 * x5 * x[0] -
        1.0 / 13824.0 * coeffs[27] * x381 * x392 +
        (25.0 / 13824.0) * coeffs[28] * x0 * x11 * x12 * x17 * x26 * x3 *
            x380 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[29] * x295 * x382 -
        1.0 / 13824.0 * coeffs[2] * x372 * x400 +
        (25.0 / 6912.0) * coeffs[31] * x0 * x11 * x12 * x17 * x26 * x3 * x380 *
            x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[32] * x386 * x388 +
        (25.0 / 13824.0) * coeffs[33] * x11 * x12 * x13 * x17 * x3 * x380 * x4 *
            x43 * x5 * x[0] * x[1] +
        (25.0 / 6912.0) * coeffs[34] * x11 * x13 * x17 * x26 * x3 * x380 * x4 *
            x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[35] * x386 * x390 +
        (25.0 / 13824.0) * coeffs[36] * x0 * x11 * x12 * x13 * x17 * x26 *
            x380 * x4 * x43 * x5 * x[1] -
        1.0 / 13824.0 * coeffs[37] * x300 * x35 * x381 +
        (25.0 / 6912.0) * coeffs[39] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x380 * x4 * x43 * x[1] +
        (1.0 / 13824.0) * coeffs[3] * x11 * x12 * x13 * x17 * x26 * x3 *
            x393 * x4 * x43 * x5 * x[1] +
        (25.0 / 13824.0) * coeffs[40] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x321 * x4 * x43 * x5 -
        1.0 / 13824.0 * coeffs[41] * x328 * x377 * x379 -
        1.0 / 13824.0 * coeffs[42] * x339 * x378 * x379 +
        (25.0 / 6912.0) * coeffs[43] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x341 * x4 * x43 * x5 -
        1.0 / 13824.0 * coeffs[44] * x321 * x372 * x377 +
        (25.0 / 13824.0) * coeffs[45] * x0 * x11 * x12 * x17 * x26 * x3 *
            x328 * x4 * x43 * x5 * x[0] +
        (25.0 / 6912.0) * coeffs[46] * x0 * x11 * x12 * x17 * x26 * x3 *
            x339 * x4 * x43 * x5 * x[0] -
        1.0 / 13824.0 * coeffs[47] * x341 * x372 * x378 +
        (25.0 / 13824.0) * coeffs[48] * x11 * x12 * x17 * x26 * x3 * x321 * x4 *
            x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[49] * x328 * x372 * x373 +
        (1.0 / 13824.0) * coeffs[4] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x380 * x4 * x43 * x5 -
        1.0 / 13824.0 * coeffs[50] * x339 * x372 * x374 +
        (25.0 / 6912.0) * coeffs[51] * x11 * x12 * x17 * x26 * x3 * x341 * x4 *
            x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[52] * x321 * x373 * x379 +
        (25.0 / 13824.0) * coeffs[53] * x11 * x12 * x13 * x17 * x26 * x3 *
            x328 * x4 * x43 * x5 * x[1] +
        (25.0 / 6912.0) * coeffs[54] * x11 * x12 * x13 * x17 * x26 * x3 *
            x339 * x4 * x43 * x5 * x[1] -
        1.0 / 13824.0 * coeffs[55] * x341 * x374 * x379 -
        1.0 / 13824.0 * coeffs[56] * x283 * x395 +
        (625.0 / 13824.0) * coeffs[57] * x0 * x11 * x12 * x13 * x17 *
            x393 * x4 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[58] * x281 * x396 +
        (625.0 / 13824.0) * coeffs[59] * x0 * x12 * x13 * x17 * x26 * x3 *
            x393 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[5] * x372 * x381 +
        (625.0 / 6912.0) * coeffs[60] * x0 * x11 * x12 * x13 * x17 * x3 *
            x393 * x4 * x43 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[61] * x263 * x397 * x72 -
        1.0 / 13824.0 * coeffs[62] * x265 * x397 * x45 +
        (625.0 / 6912.0) * coeffs[63] * x0 * x11 * x13 * x17 * x26 * x393 * x4 *
            x43 * x5 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[64] * x0 * x12 * x13 * x17 * x26 * x3 * x393 *
            x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[65] * x261 * x397 * x80 -
        1.0 / 13824.0 * coeffs[66] * x268 * x35 * x397 +
        (625.0 / 6912.0) * coeffs[67] * x0 * x11 * x12 * x13 * x26 * x3 *
            x393 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[68] * x265 * x398 * x80 +
        (625.0 / 3456.0) * coeffs[69] * x0 * x11 * x12 * x13 * x26 * x3 * x393 *
            x43 * x5 * x[0] * x[1] +
        (1.0 / 13824.0) * coeffs[6] * x11 * x12 * x17 * x26 * x3 * x380 * x4 *
            x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[70] * x268 * x398 * x72 +
        (625.0 / 3456.0) * coeffs[71] * x0 * x11 * x13 * x17 * x26 * x3 *
            x393 * x4 * x43 * x[0] * x[1] +
        (625.0 / 13824.0) * coeffs[72] * x0 * x11 * x12 * x13 * x17 * x3 *
            x380 * x4 * x5 * x[0] * x[1] +
        (625.0 / 13824.0) * coeffs[74] * x0 * x12 * x13 * x17 * x26 *
            x380 * x4 * x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[75] * x283 * x382 +
        (625.0 / 6912.0) * coeffs[77] * x0 * x11 * x13 * x17 * x26 * x3 *
            x380 * x4 * x5 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[78] * x0 * x12 * x13 * x17 * x26 * x3 *
            x380 * x4 * x43 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[7] * x379 * x386 +
        (625.0 / 6912.0) * coeffs[81] * x0 * x11 * x12 * x13 * x26 * x380 * x4 *
            x43 * x5 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[82] * x0 * x11 * x12 * x13 * x17 * x3 * x380 *
            x43 * x5 * x[0] * x[1] +
        (625.0 / 3456.0) * coeffs[84] * x0 * x11 * x12 * x13 * x26 * x3 *
            x380 * x4 * x43 * x[0] * x[1] +
        (625.0 / 3456.0) * coeffs[86] * x0 * x11 * x13 * x17 * x26 * x3 * x380 *
            x43 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[88] * x321 * x322 * x375 +
        (625.0 / 13824.0) * coeffs[89] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x328 * x4 * x5 * x[1] +
        (25.0 / 13824.0) * coeffs[8] * x0 * x11 * x12 * x13 * x17 * x3 *
            x393 * x4 * x43 * x5 * x[0] -
        1.0 / 13824.0 * coeffs[90] * x324 * x328 * x375 +
        (625.0 / 13824.0) * coeffs[91] * x0 * x11 * x12 * x13 * x17 * x26 *
            x321 * x4 * x43 * x5 * x[1] +
        (625.0 / 6912.0) * coeffs[92] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x339 * x4 * x5 * x[1] -
        1.0 / 13824.0 * coeffs[93] * x322 * x341 * x376 -
        1.0 / 13824.0 * coeffs[94] * x328 * x334 * x376 +
        (625.0 / 6912.0) * coeffs[95] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x328 * x43 * x5 * x[1] +
        (625.0 / 6912.0) * coeffs[96] * x0 * x11 * x12 * x13 * x17 * x26 *
            x341 * x4 * x43 * x5 * x[1] -
        1.0 / 13824.0 * coeffs[97] * x324 * x339 * x376 -
        1.0 / 13824.0 * coeffs[98] * x274 * x321 * x358 +
        (625.0 / 6912.0) * coeffs[99] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x321 * x4 * x43 * x[1] -
        1.0 / 13824.0 * coeffs[9] * x388 * x394 -
        1.0 / 13824.0 * x13 * x207 * x385 - 1.0 / 13824.0 * x145 * x281 * x381 -
        1.0 / 13824.0 * x150 * x265 * x383 -
        1.0 / 13824.0 * x153 * x261 * x383 -
        1.0 / 13824.0 * x154 * x268 * x383 -
        1.0 / 13824.0 * x155 * x263 * x383 -
        1.0 / 13824.0 * x156 * x268 * x384 -
        1.0 / 13824.0 * x159 * x265 * x384 -
        1.0 / 13824.0 * x188 * x385 * x[0] -
        1.0 / 13824.0 * x23 * x286 * x339 * x365 -
        1.0 / 13824.0 * x253 * x357 * x358 -
        1.0 / 13824.0 * x284 * x328 * x363 -
        1.0 / 13824.0 * x287 * x366 * x[1] -
        1.0 / 13824.0 * x288 * x339 * x367 -
        1.0 / 13824.0 * x290 * x341 * x367 -
        1.0 / 13824.0 * x30 * x308 * x339 * x369 -
        1.0 / 13824.0 * x304 * x321 * x368 -
        1.0 / 13824.0 * x306 * x328 * x368 -
        1.0 / 13824.0 * x309 * x339 * x370 - 1.0 / 13824.0 * x311 * x341 * x370;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
      break;
    case 1:
      out[0] = 5;
      out[1] = 0;
      out[2] = 0;
      break;
    case 2:
      out[0] = 5;
      out[1] = 5;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 5;
      out[2] = 0;
      break;
    case 4:
      out[0] = 0;
      out[1] = 0;
      out[2] = 5;
      break;
    case 5:
      out[0] = 5;
      out[1] = 0;
      out[2] = 5;
      break;
    case 6:
      out[0] = 5;
      out[1] = 5;
      out[2] = 5;
      break;
    case 7:
      out[0] = 0;
      out[1] = 5;
      out[2] = 5;
      break;
    case 8:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 9:
      out[0] = 4;
      out[1] = 0;
      out[2] = 0;
      break;
    case 10:
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 11:
      out[0] = 3;
      out[1] = 0;
      out[2] = 0;
      break;
    case 12:
      out[0] = 5;
      out[1] = 1;
      out[2] = 0;
      break;
    case 13:
      out[0] = 5;
      out[1] = 4;
      out[2] = 0;
      break;
    case 14:
      out[0] = 5;
      out[1] = 2;
      out[2] = 0;
      break;
    case 15:
      out[0] = 5;
      out[1] = 3;
      out[2] = 0;
      break;
    case 16:
      out[0] = 4;
      out[1] = 5;
      out[2] = 0;
      break;
    case 17:
      out[0] = 1;
      out[1] = 5;
      out[2] = 0;
      break;
    case 18:
      out[0] = 3;
      out[1] = 5;
      out[2] = 0;
      break;
    case 19:
      out[0] = 2;
      out[1] = 5;
      out[2] = 0;
      break;
    case 20:
      out[0] = 0;
      out[1] = 4;
      out[2] = 0;
      break;
    case 21:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 22:
      out[0] = 0;
      out[1] = 3;
      out[2] = 0;
      break;
    case 23:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 24:
      out[0] = 1;
      out[1] = 0;
      out[2] = 5;
      break;
    case 25:
      out[0] = 4;
      out[1] = 0;
      out[2] = 5;
      break;
    case 26:
      out[0] = 2;
      out[1] = 0;
      out[2] = 5;
      break;
    case 27:
      out[0] = 3;
      out[1] = 0;
      out[2] = 5;
      break;
    case 28:
      out[0] = 5;
      out[1] = 1;
      out[2] = 5;
      break;
    case 29:
      out[0] = 5;
      out[1] = 4;
      out[2] = 5;
      break;
    case 30:
      out[0] = 5;
      out[1] = 2;
      out[2] = 5;
      break;
    case 31:
      out[0] = 5;
      out[1] = 3;
      out[2] = 5;
      break;
    case 32:
      out[0] = 4;
      out[1] = 5;
      out[2] = 5;
      break;
    case 33:
      out[0] = 1;
      out[1] = 5;
      out[2] = 5;
      break;
    case 34:
      out[0] = 3;
      out[1] = 5;
      out[2] = 5;
      break;
    case 35:
      out[0] = 2;
      out[1] = 5;
      out[2] = 5;
      break;
    case 36:
      out[0] = 0;
      out[1] = 4;
      out[2] = 5;
      break;
    case 37:
      out[0] = 0;
      out[1] = 1;
      out[2] = 5;
      break;
    case 38:
      out[0] = 0;
      out[1] = 3;
      out[2] = 5;
      break;
    case 39:
      out[0] = 0;
      out[1] = 2;
      out[2] = 5;
      break;
    case 40:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 41:
      out[0] = 0;
      out[1] = 0;
      out[2] = 4;
      break;
    case 42:
      out[0] = 0;
      out[1] = 0;
      out[2] = 2;
      break;
    case 43:
      out[0] = 0;
      out[1] = 0;
      out[2] = 3;
      break;
    case 44:
      out[0] = 5;
      out[1] = 0;
      out[2] = 1;
      break;
    case 45:
      out[0] = 5;
      out[1] = 0;
      out[2] = 4;
      break;
    case 46:
      out[0] = 5;
      out[1] = 0;
      out[2] = 2;
      break;
    case 47:
      out[0] = 5;
      out[1] = 0;
      out[2] = 3;
      break;
    case 48:
      out[0] = 5;
      out[1] = 5;
      out[2] = 1;
      break;
    case 49:
      out[0] = 5;
      out[1] = 5;
      out[2] = 4;
      break;
    case 50:
      out[0] = 5;
      out[1] = 5;
      out[2] = 2;
      break;
    case 51:
      out[0] = 5;
      out[1] = 5;
      out[2] = 3;
      break;
    case 52:
      out[0] = 0;
      out[1] = 5;
      out[2] = 1;
      break;
    case 53:
      out[0] = 0;
      out[1] = 5;
      out[2] = 4;
      break;
    case 54:
      out[0] = 0;
      out[1] = 5;
      out[2] = 2;
      break;
    case 55:
      out[0] = 0;
      out[1] = 5;
      out[2] = 3;
      break;
    case 56:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 57:
      out[0] = 1;
      out[1] = 4;
      out[2] = 0;
      break;
    case 58:
      out[0] = 4;
      out[1] = 4;
      out[2] = 0;
      break;
    case 59:
      out[0] = 4;
      out[1] = 1;
      out[2] = 0;
      break;
    case 60:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 61:
      out[0] = 1;
      out[1] = 3;
      out[2] = 0;
      break;
    case 62:
      out[0] = 2;
      out[1] = 4;
      out[2] = 0;
      break;
    case 63:
      out[0] = 3;
      out[1] = 4;
      out[2] = 0;
      break;
    case 64:
      out[0] = 4;
      out[1] = 3;
      out[2] = 0;
      break;
    case 65:
      out[0] = 4;
      out[1] = 2;
      out[2] = 0;
      break;
    case 66:
      out[0] = 3;
      out[1] = 1;
      out[2] = 0;
      break;
    case 67:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 68:
      out[0] = 2;
      out[1] = 2;
      out[2] = 0;
      break;
    case 69:
      out[0] = 2;
      out[1] = 3;
      out[2] = 0;
      break;
    case 70:
      out[0] = 3;
      out[1] = 3;
      out[2] = 0;
      break;
    case 71:
      out[0] = 3;
      out[1] = 2;
      out[2] = 0;
      break;
    case 72:
      out[0] = 1;
      out[1] = 1;
      out[2] = 5;
      break;
    case 73:
      out[0] = 4;
      out[1] = 1;
      out[2] = 5;
      break;
    case 74:
      out[0] = 4;
      out[1] = 4;
      out[2] = 5;
      break;
    case 75:
      out[0] = 1;
      out[1] = 4;
      out[2] = 5;
      break;
    case 76:
      out[0] = 2;
      out[1] = 1;
      out[2] = 5;
      break;
    case 77:
      out[0] = 3;
      out[1] = 1;
      out[2] = 5;
      break;
    case 78:
      out[0] = 4;
      out[1] = 2;
      out[2] = 5;
      break;
    case 79:
      out[0] = 4;
      out[1] = 3;
      out[2] = 5;
      break;
    case 80:
      out[0] = 3;
      out[1] = 4;
      out[2] = 5;
      break;
    case 81:
      out[0] = 2;
      out[1] = 4;
      out[2] = 5;
      break;
    case 82:
      out[0] = 1;
      out[1] = 3;
      out[2] = 5;
      break;
    case 83:
      out[0] = 1;
      out[1] = 2;
      out[2] = 5;
      break;
    case 84:
      out[0] = 2;
      out[1] = 2;
      out[2] = 5;
      break;
    case 85:
      out[0] = 3;
      out[1] = 2;
      out[2] = 5;
      break;
    case 86:
      out[0] = 3;
      out[1] = 3;
      out[2] = 5;
      break;
    case 87:
      out[0] = 2;
      out[1] = 3;
      out[2] = 5;
      break;
    case 88:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 89:
      out[0] = 0;
      out[1] = 1;
      out[2] = 4;
      break;
    case 90:
      out[0] = 0;
      out[1] = 4;
      out[2] = 4;
      break;
    case 91:
      out[0] = 0;
      out[1] = 4;
      out[2] = 1;
      break;
    case 92:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 93:
      out[0] = 0;
      out[1] = 1;
      out[2] = 3;
      break;
    case 94:
      out[0] = 0;
      out[1] = 2;
      out[2] = 4;
      break;
    case 95:
      out[0] = 0;
      out[1] = 3;
      out[2] = 4;
      break;
    case 96:
      out[0] = 0;
      out[1] = 4;
      out[2] = 3;
      break;
    case 97:
      out[0] = 0;
      out[1] = 4;
      out[2] = 2;
      break;
    case 98:
      out[0] = 0;
      out[1] = 3;
      out[2] = 1;
      break;
    case 99:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 100:
      out[0] = 0;
      out[1] = 2;
      out[2] = 2;
      break;
    case 101:
      out[0] = 0;
      out[1] = 2;
      out[2] = 3;
      break;
    case 102:
      out[0] = 0;
      out[1] = 3;
      out[2] = 3;
      break;
    case 103:
      out[0] = 0;
      out[1] = 3;
      out[2] = 2;
      break;
    case 104:
      out[0] = 5;
      out[1] = 1;
      out[2] = 1;
      break;
    case 105:
      out[0] = 5;
      out[1] = 4;
      out[2] = 1;
      break;
    case 106:
      out[0] = 5;
      out[1] = 4;
      out[2] = 4;
      break;
    case 107:
      out[0] = 5;
      out[1] = 1;
      out[2] = 4;
      break;
    case 108:
      out[0] = 5;
      out[1] = 2;
      out[2] = 1;
      break;
    case 109:
      out[0] = 5;
      out[1] = 3;
      out[2] = 1;
      break;
    case 110:
      out[0] = 5;
      out[1] = 4;
      out[2] = 2;
      break;
    case 111:
      out[0] = 5;
      out[1] = 4;
      out[2] = 3;
      break;
    case 112:
      out[0] = 5;
      out[1] = 3;
      out[2] = 4;
      break;
    case 113:
      out[0] = 5;
      out[1] = 2;
      out[2] = 4;
      break;
    case 114:
      out[0] = 5;
      out[1] = 1;
      out[2] = 3;
      break;
    case 115:
      out[0] = 5;
      out[1] = 1;
      out[2] = 2;
      break;
    case 116:
      out[0] = 5;
      out[1] = 2;
      out[2] = 2;
      break;
    case 117:
      out[0] = 5;
      out[1] = 3;
      out[2] = 2;
      break;
    case 118:
      out[0] = 5;
      out[1] = 3;
      out[2] = 3;
      break;
    case 119:
      out[0] = 5;
      out[1] = 2;
      out[2] = 3;
      break;
    case 120:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 121:
      out[0] = 4;
      out[1] = 0;
      out[2] = 1;
      break;
    case 122:
      out[0] = 4;
      out[1] = 0;
      out[2] = 4;
      break;
    case 123:
      out[0] = 1;
      out[1] = 0;
      out[2] = 4;
      break;
    case 124:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 125:
      out[0] = 3;
      out[1] = 0;
      out[2] = 1;
      break;
    case 126:
      out[0] = 4;
      out[1] = 0;
      out[2] = 2;
      break;
    case 127:
      out[0] = 4;
      out[1] = 0;
      out[2] = 3;
      break;
    case 128:
      out[0] = 3;
      out[1] = 0;
      out[2] = 4;
      break;
    case 129:
      out[0] = 2;
      out[1] = 0;
      out[2] = 4;
      break;
    case 130:
      out[0] = 1;
      out[1] = 0;
      out[2] = 3;
      break;
    case 131:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 132:
      out[0] = 2;
      out[1] = 0;
      out[2] = 2;
      break;
    case 133:
      out[0] = 3;
      out[1] = 0;
      out[2] = 2;
      break;
    case 134:
      out[0] = 3;
      out[1] = 0;
      out[2] = 3;
      break;
    case 135:
      out[0] = 2;
      out[1] = 0;
      out[2] = 3;
      break;
    case 136:
      out[0] = 4;
      out[1] = 5;
      out[2] = 1;
      break;
    case 137:
      out[0] = 1;
      out[1] = 5;
      out[2] = 1;
      break;
    case 138:
      out[0] = 1;
      out[1] = 5;
      out[2] = 4;
      break;
    case 139:
      out[0] = 4;
      out[1] = 5;
      out[2] = 4;
      break;
    case 140:
      out[0] = 3;
      out[1] = 5;
      out[2] = 1;
      break;
    case 141:
      out[0] = 2;
      out[1] = 5;
      out[2] = 1;
      break;
    case 142:
      out[0] = 1;
      out[1] = 5;
      out[2] = 2;
      break;
    case 143:
      out[0] = 1;
      out[1] = 5;
      out[2] = 3;
      break;
    case 144:
      out[0] = 2;
      out[1] = 5;
      out[2] = 4;
      break;
    case 145:
      out[0] = 3;
      out[1] = 5;
      out[2] = 4;
      break;
    case 146:
      out[0] = 4;
      out[1] = 5;
      out[2] = 3;
      break;
    case 147:
      out[0] = 4;
      out[1] = 5;
      out[2] = 2;
      break;
    case 148:
      out[0] = 3;
      out[1] = 5;
      out[2] = 2;
      break;
    case 149:
      out[0] = 2;
      out[1] = 5;
      out[2] = 2;
      break;
    case 150:
      out[0] = 2;
      out[1] = 5;
      out[2] = 3;
      break;
    case 151:
      out[0] = 3;
      out[1] = 5;
      out[2] = 3;
      break;
    case 152:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    case 153:
      out[0] = 4;
      out[1] = 1;
      out[2] = 1;
      break;
    case 154:
      out[0] = 4;
      out[1] = 4;
      out[2] = 1;
      break;
    case 155:
      out[0] = 1;
      out[1] = 4;
      out[2] = 1;
      break;
    case 156:
      out[0] = 1;
      out[1] = 1;
      out[2] = 4;
      break;
    case 157:
      out[0] = 4;
      out[1] = 1;
      out[2] = 4;
      break;
    case 158:
      out[0] = 4;
      out[1] = 4;
      out[2] = 4;
      break;
    case 159:
      out[0] = 1;
      out[1] = 4;
      out[2] = 4;
      break;
    case 160:
      out[0] = 2;
      out[1] = 1;
      out[2] = 1;
      break;
    case 161:
      out[0] = 3;
      out[1] = 1;
      out[2] = 1;
      break;
    case 162:
      out[0] = 4;
      out[1] = 2;
      out[2] = 1;
      break;
    case 163:
      out[0] = 4;
      out[1] = 3;
      out[2] = 1;
      break;
    case 164:
      out[0] = 3;
      out[1] = 4;
      out[2] = 1;
      break;
    case 165:
      out[0] = 2;
      out[1] = 4;
      out[2] = 1;
      break;
    case 166:
      out[0] = 1;
      out[1] = 3;
      out[2] = 1;
      break;
    case 167:
      out[0] = 1;
      out[1] = 2;
      out[2] = 1;
      break;
    case 168:
      out[0] = 2;
      out[1] = 1;
      out[2] = 4;
      break;
    case 169:
      out[0] = 3;
      out[1] = 1;
      out[2] = 4;
      break;
    case 170:
      out[0] = 4;
      out[1] = 2;
      out[2] = 4;
      break;
    case 171:
      out[0] = 4;
      out[1] = 3;
      out[2] = 4;
      break;
    case 172:
      out[0] = 3;
      out[1] = 4;
      out[2] = 4;
      break;
    case 173:
      out[0] = 2;
      out[1] = 4;
      out[2] = 4;
      break;
    case 174:
      out[0] = 1;
      out[1] = 3;
      out[2] = 4;
      break;
    case 175:
      out[0] = 1;
      out[1] = 2;
      out[2] = 4;
      break;
    case 176:
      out[0] = 1;
      out[1] = 1;
      out[2] = 2;
      break;
    case 177:
      out[0] = 1;
      out[1] = 1;
      out[2] = 3;
      break;
    case 178:
      out[0] = 4;
      out[1] = 1;
      out[2] = 2;
      break;
    case 179:
      out[0] = 4;
      out[1] = 1;
      out[2] = 3;
      break;
    case 180:
      out[0] = 4;
      out[1] = 4;
      out[2] = 2;
      break;
    case 181:
      out[0] = 4;
      out[1] = 4;
      out[2] = 3;
      break;
    case 182:
      out[0] = 1;
      out[1] = 4;
      out[2] = 2;
      break;
    case 183:
      out[0] = 1;
      out[1] = 4;
      out[2] = 3;
      break;
    case 184:
      out[0] = 2;
      out[1] = 2;
      out[2] = 1;
      break;
    case 185:
      out[0] = 2;
      out[1] = 3;
      out[2] = 1;
      break;
    case 186:
      out[0] = 3;
      out[1] = 3;
      out[2] = 1;
      break;
    case 187:
      out[0] = 3;
      out[1] = 2;
      out[2] = 1;
      break;
    case 188:
      out[0] = 2;
      out[1] = 2;
      out[2] = 4;
      break;
    case 189:
      out[0] = 3;
      out[1] = 2;
      out[2] = 4;
      break;
    case 190:
      out[0] = 3;
      out[1] = 3;
      out[2] = 4;
      break;
    case 191:
      out[0] = 2;
      out[1] = 3;
      out[2] = 4;
      break;
    case 192:
      out[0] = 1;
      out[1] = 2;
      out[2] = 2;
      break;
    case 193:
      out[0] = 1;
      out[1] = 2;
      out[2] = 3;
      break;
    case 194:
      out[0] = 1;
      out[1] = 3;
      out[2] = 3;
      break;
    case 195:
      out[0] = 1;
      out[1] = 3;
      out[2] = 2;
      break;
    case 196:
      out[0] = 4;
      out[1] = 2;
      out[2] = 2;
      break;
    case 197:
      out[0] = 4;
      out[1] = 3;
      out[2] = 2;
      break;
    case 198:
      out[0] = 4;
      out[1] = 3;
      out[2] = 3;
      break;
    case 199:
      out[0] = 4;
      out[1] = 2;
      out[2] = 3;
      break;
    case 200:
      out[0] = 2;
      out[1] = 1;
      out[2] = 2;
      break;
    case 201:
      out[0] = 3;
      out[1] = 1;
      out[2] = 2;
      break;
    case 202:
      out[0] = 3;
      out[1] = 1;
      out[2] = 3;
      break;
    case 203:
      out[0] = 2;
      out[1] = 1;
      out[2] = 3;
      break;
    case 204:
      out[0] = 3;
      out[1] = 4;
      out[2] = 2;
      break;
    case 205:
      out[0] = 2;
      out[1] = 4;
      out[2] = 2;
      break;
    case 206:
      out[0] = 2;
      out[1] = 4;
      out[2] = 3;
      break;
    case 207:
      out[0] = 3;
      out[1] = 4;
      out[2] = 3;
      break;
    case 208:
      out[0] = 2;
      out[1] = 2;
      out[2] = 2;
      break;
    case 209:
      out[0] = 3;
      out[1] = 2;
      out[2] = 2;
      break;
    case 210:
      out[0] = 3;
      out[1] = 3;
      out[2] = 2;
      break;
    case 211:
      out[0] = 2;
      out[1] = 3;
      out[2] = 2;
      break;
    case 212:
      out[0] = 2;
      out[1] = 2;
      out[2] = 3;
      break;
    case 213:
      out[0] = 3;
      out[1] = 2;
      out[2] = 3;
      break;
    case 214:
      out[0] = 3;
      out[1] = 3;
      out[2] = 3;
      break;
    case 215:
      out[0] = 2;
      out[1] = 3;
      out[2] = 3;
      break;
    }
  }
}

} // namespace numeric::math

#endif
