#ifndef NUMERIC_MATH_BASIS_LAGRANGE_CUBE_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_CUBE_HPP_

#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math {

template <> struct BasisLagrange<mesh::RefElCube, 1> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 8;
  static constexpr dim_t num_interpolation_nodes = 8;

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
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x[2] - 1;
    const Scalar x3 = x1 * x2;
    const Scalar x4 = x2 * x[1];
    const Scalar x5 = x1 * x[2];
    const Scalar x6 = x[1] * x[2];
    out[0] = -x0 * x3;
    out[1] = x3 * x[0];
    out[2] = -x4 * x[0];
    out[3] = x0 * x4;
    out[4] = x0 * x5;
    out[5] = -x5 * x[0];
    out[6] = x6 * x[0];
    out[7] = -x0 * x6;
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

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[2] - 1;
    const Scalar x2 = x0 * x1;
    const Scalar x3 = x[0] - 1;
    const Scalar x4 = x1 * x3;
    const Scalar x5 = x0 * x3;
    const Scalar x6 = x1 * x[0];
    const Scalar x7 = x0 * x[0];
    const Scalar x8 = x1 * x[1];
    const Scalar x9 = x[0] * x[1];
    const Scalar x10 = x3 * x[1];
    const Scalar x11 = x0 * x[2];
    const Scalar x12 = x3 * x[2];
    const Scalar x13 = x[0] * x[2];
    const Scalar x14 = x[1] * x[2];
    out[0][0] = -x2;
    out[0][1] = -x4;
    out[0][2] = -x5;
    out[1][0] = x2;
    out[1][1] = x6;
    out[1][2] = x7;
    out[2][0] = -x8;
    out[2][1] = -x6;
    out[2][2] = -x9;
    out[3][0] = x8;
    out[3][1] = x4;
    out[3][2] = x10;
    out[4][0] = x11;
    out[4][1] = x12;
    out[4][2] = x5;
    out[5][0] = -x11;
    out[5][1] = -x13;
    out[5][2] = -x7;
    out[6][0] = x14;
    out[6][1] = x13;
    out[6][2] = x9;
    out[7][0] = -x14;
    out[7][1] = -x12;
    out[7][2] = -x10;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[3]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[0][2] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(1) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[1][2] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(1) / order;
    out[2][1] = static_cast<Scalar>(1) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(1) / order;
    out[3][2] = static_cast<Scalar>(0) / order;
    out[4][0] = static_cast<Scalar>(0) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[4][2] = static_cast<Scalar>(1) / order;
    out[5][0] = static_cast<Scalar>(1) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[5][2] = static_cast<Scalar>(1) / order;
    out[6][0] = static_cast<Scalar>(1) / order;
    out[6][1] = static_cast<Scalar>(1) / order;
    out[6][2] = static_cast<Scalar>(1) / order;
    out[7][0] = static_cast<Scalar>(0) / order;
    out[7][1] = static_cast<Scalar>(1) / order;
    out[7][2] = static_cast<Scalar>(1) / order;
  }
  template <typename Scalar>
  static constexpr void interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
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

  template <typename Element>
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs) {
    subelement_node_idxs(subelement, idxs, meta::type_tag<Element>{});
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElPoint>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      break;
    case 1:
      idxs[0] = 1;
      break;
    case 2:
      idxs[0] = 2;
      break;
    case 3:
      idxs[0] = 3;
      break;
    case 4:
      idxs[0] = 4;
      break;
    case 5:
      idxs[0] = 5;
      break;
    case 6:
      idxs[0] = 6;
      break;
    case 7:
      idxs[0] = 7;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      break;
    case 2:
      idxs[0] = 3;
      idxs[1] = 2;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      break;
    case 4:
      idxs[0] = 4;
      idxs[1] = 5;
      break;
    case 5:
      idxs[0] = 5;
      idxs[1] = 6;
      break;
    case 6:
      idxs[0] = 7;
      idxs[1] = 6;
      break;
    case 7:
      idxs[0] = 4;
      idxs[1] = 7;
      break;
    case 8:
      idxs[0] = 0;
      idxs[1] = 4;
      break;
    case 9:
      idxs[0] = 1;
      idxs[1] = 5;
      break;
    case 10:
      idxs[0] = 2;
      idxs[1] = 6;
      break;
    case 11:
      idxs[0] = 3;
      idxs[1] = 7;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 7;
      idxs[3] = 4;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 5;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 5;
      idxs[3] = 4;
      break;
    case 3:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 7;
      break;
    case 4:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 2;
      idxs[3] = 3;
      break;
    case 5:
      idxs[0] = 4;
      idxs[1] = 5;
      idxs[2] = 6;
      idxs[3] = 7;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTetra>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElCube>) {
    switch (subelement) {
    default:
      break;
    }
  }
};

template <> struct BasisLagrange<mesh::RefElCube, 2> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 27;
  static constexpr dim_t num_interpolation_nodes = 27;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x[2] - 1;
    const Scalar x3 = x0 * x1 * x2;
    const Scalar x4 = x[0] * x[1];
    const Scalar x5 = x4 * x[2];
    const Scalar x6 = 2 * x[0];
    const Scalar x7 = x6 * x[1];
    const Scalar x8 = x6 - 1;
    const Scalar x9 = x8 * x[2];
    const Scalar x10 = x1 * x2;
    const Scalar x11 = x10 * x9;
    const Scalar x12 = 2 * x[1];
    const Scalar x13 = x12 - 1;
    const Scalar x14 = x13 * x[2];
    const Scalar x15 = x0 * x7;
    const Scalar x16 = 2 * x[2] - 1;
    const Scalar x17 = x1 * x16;
    const Scalar x18 = (1.0 / 2.0) * x8;
    const Scalar x19 = x18 * x5;
    const Scalar x20 = (1.0 / 2.0) * x0;
    const Scalar x21 = x13 * x16;
    const Scalar x22 = x20 * x21;
    const Scalar x23 = x13 * x2;
    const Scalar x24 = x3 * x9;
    const Scalar x25 = x16 * x3;
    const Scalar x26 = x2 * x4;
    const Scalar x27 = x17 * x20;
    const Scalar x28 = x9 * x[1];
    const Scalar x29 = (1.0 / 2.0) * x13;
    const Scalar x30 = x29 * x[0];
    const Scalar x31 = (1.0 / 8.0) * x8;
    const Scalar x32 = x21 * x31;
    const Scalar x33 = x13 * x17 * x9;
    const Scalar x34 = (1.0 / 8.0) * x0;
    return 8 * coeffs[0] * x13 * x25 * x31 - 8 * coeffs[10] * x22 * x26 -
           8 * coeffs[11] * x18 * x25 * x[1] -
           8 * coeffs[12] * x14 * x27 * x[0] - 8 * coeffs[13] * x17 * x19 -
           8 * coeffs[14] * x22 * x5 - 8 * coeffs[15] * x27 * x28 -
           8 * coeffs[16] * x24 * x29 - 8 * coeffs[17] * x11 * x30 -
           8 * coeffs[18] * x19 * x23 - 8 * coeffs[19] * x20 * x23 * x28 +
           8 * coeffs[1] * x10 * x32 * x[0] + 8 * coeffs[20] * x12 * x24 +
           8 * coeffs[21] * x11 * x7 + 8 * coeffs[22] * x14 * x3 * x6 +
           8 * coeffs[23] * x14 * x15 * x2 + 8 * coeffs[24] * x25 * x7 +
           8 * coeffs[25] * x15 * x17 * x[2] - 64 * coeffs[26] * x3 * x5 +
           8 * coeffs[2] * x26 * x32 + 8 * coeffs[3] * x0 * x2 * x32 * x[1] +
           8 * coeffs[4] * x33 * x34 + coeffs[5] * x33 * x[0] +
           8 * coeffs[6] * x32 * x5 + 8 * coeffs[7] * x21 * x28 * x34 -
           8 * coeffs[8] * x25 * x30 - 8 * coeffs[9] * x10 * x16 * x18 * x4;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = 2 * x[0] - 1;
    const Scalar x1 = 2 * x[1] - 1;
    const Scalar x2 = 2 * x[2] - 1;
    const Scalar x3 = x0 * x1 * x2;
    const Scalar x4 = x[0] - 1;
    const Scalar x5 = x[1] - 1;
    const Scalar x6 = x[2] - 1;
    const Scalar x7 = x5 * x6;
    const Scalar x8 = x4 * x7;
    const Scalar x9 = x3 * x[0];
    const Scalar x10 = x6 * x[1];
    const Scalar x11 = x3 * x4;
    const Scalar x12 = x5 * x[2];
    const Scalar x13 = x[1] * x[2];
    const Scalar x14 = x1 * x[0];
    const Scalar x15 = 4 * x2;
    const Scalar x16 = x15 * x8;
    const Scalar x17 = x0 * x[1];
    const Scalar x18 = x15 * x17;
    const Scalar x19 = x7 * x[0];
    const Scalar x20 = x10 * x14;
    const Scalar x21 = x15 * x4;
    const Scalar x22 = x14 * x21;
    const Scalar x23 = x12 * x18;
    const Scalar x24 = x0 * x8;
    const Scalar x25 = 4 * x[2];
    const Scalar x26 = x1 * x25;
    const Scalar x27 = x0 * x25;
    const Scalar x28 = 16 * x13;
    const Scalar x29 = 16 * x[2];
    const Scalar x30 = x8 * x[0];
    const Scalar x31 = 16 * x2 * x[1];
    out[0] = x3 * x8;
    out[1] = x7 * x9;
    out[2] = x10 * x9;
    out[3] = x10 * x11;
    out[4] = x11 * x12;
    out[5] = x12 * x9;
    out[6] = x13 * x9;
    out[7] = x11 * x13;
    out[8] = -x14 * x16;
    out[9] = -x18 * x19;
    out[10] = -x20 * x21;
    out[11] = -x16 * x17;
    out[12] = -x12 * x22;
    out[13] = -x23 * x[0];
    out[14] = -x13 * x22;
    out[15] = -x23 * x4;
    out[16] = -x24 * x26;
    out[17] = -x14 * x27 * x7;
    out[18] = -x20 * x27;
    out[19] = -x0 * x10 * x26 * x4;
    out[20] = x24 * x28;
    out[21] = x0 * x19 * x28;
    out[22] = x14 * x29 * x8;
    out[23] = x20 * x29 * x4;
    out[24] = x30 * x31;
    out[25] = x12 * x31 * x4 * x[0];
    out[26] = -64 * x13 * x30;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 2 * x[1];
    const Scalar x1 = x0 * x[2];
    const Scalar x2 = 4 * x[0];
    const Scalar x3 = x2 - 3;
    const Scalar x4 = x[1] - 1;
    const Scalar x5 = x[2] - 1;
    const Scalar x6 = x4 * x5;
    const Scalar x7 = x3 * x6;
    const Scalar x8 = x2 - 1;
    const Scalar x9 = x6 * x8;
    const Scalar x10 = x[1] * x[2];
    const Scalar x11 = 2 * x[0];
    const Scalar x12 = x11 - 1;
    const Scalar x13 = x12 * x6;
    const Scalar x14 = 8 * coeffs[26];
    const Scalar x15 = (1.0 / 2.0) * x10;
    const Scalar x16 = 2 * x[2];
    const Scalar x17 = x16 - 1;
    const Scalar x18 = x17 * x4;
    const Scalar x19 = x18 * x8;
    const Scalar x20 = x15 * x3;
    const Scalar x21 = x0 - 1;
    const Scalar x22 = x21 * x5;
    const Scalar x23 = x1 * x12;
    const Scalar x24 = x17 * x[1];
    const Scalar x25 = (1.0 / 2.0) * x7;
    const Scalar x26 = (1.0 / 2.0) * x12;
    const Scalar x27 = x21 * x26;
    const Scalar x28 = x10 * x17;
    const Scalar x29 = x21 * x[2];
    const Scalar x30 = (1.0 / 2.0) * x9;
    const Scalar x31 = x13 * x21;
    const Scalar x32 = coeffs[24] * x17;
    const Scalar x33 = (1.0 / 8.0) * x21;
    const Scalar x34 = x28 * x33;
    const Scalar x35 = x22 * x24;
    const Scalar x36 = coeffs[12] * x[2];
    const Scalar x37 = x18 * x27;
    const Scalar x38 = (1.0 / 8.0) * x35;
    const Scalar x39 = (1.0 / 8.0) * x29;
    const Scalar x40 = x17 * x33;
    const Scalar x41 = (1.0 / 2.0) * x17;
    const Scalar x42 = coeffs[8] * x41;
    const Scalar x43 = 4 * x[1];
    const Scalar x44 = x43 - 3;
    const Scalar x45 = x11 * x[2];
    const Scalar x46 = x[0] - 1;
    const Scalar x47 = x46 * x5;
    const Scalar x48 = x45 * x47;
    const Scalar x49 = x43 - 1;
    const Scalar x50 = x22 * x46;
    const Scalar x51 = x[0] * x[2];
    const Scalar x52 = x41 * x[0];
    const Scalar x53 = x49 * x51;
    const Scalar x54 = coeffs[14] * x46;
    const Scalar x55 = x26 * x44;
    const Scalar x56 = coeffs[21] * x12;
    const Scalar x57 = x17 * x46;
    const Scalar x58 = coeffs[25] * x11;
    const Scalar x59 = x47 * x49;
    const Scalar x60 = x17 * x51;
    const Scalar x61 = coeffs[20] * x12;
    const Scalar x62 = (1.0 / 8.0) * x12;
    const Scalar x63 = x44 * x62;
    const Scalar x64 = x49 * x62;
    const Scalar x65 = x44 * x47;
    const Scalar x66 = x57 * x[2];
    const Scalar x67 = coeffs[1] * x[0];
    const Scalar x68 = x17 * x62;
    const Scalar x69 = x5 * x68;
    const Scalar x70 = coeffs[2] * x[0];
    const Scalar x71 = x17 * x26;
    const Scalar x72 = coeffs[9] * x[0];
    const Scalar x73 = x11 * x[1];
    const Scalar x74 = 4 * x[2];
    const Scalar x75 = x74 - 3;
    const Scalar x76 = x4 * x46;
    const Scalar x77 = x75 * x76;
    const Scalar x78 = x74 - 1;
    const Scalar x79 = x76 * x78;
    const Scalar x80 = x79 * x[1];
    const Scalar x81 = x18 * x46;
    const Scalar x82 = x[0] * x[1];
    const Scalar x83 = x46 * x75;
    const Scalar x84 = (1.0 / 2.0) * x21;
    const Scalar x85 = x82 * x84;
    const Scalar x86 = x4 * x78;
    const Scalar x87 = x11 * x21;
    const Scalar x88 = x24 * x46;
    const Scalar x89 = x26 * x[1];
    const Scalar x90 = x4 * x75;
    const Scalar x91 = x84 * x[0];
    const Scalar x92 = x12 * x33;
    const Scalar x93 = x92 * x[1];
    out[0] = 8 * coeffs[0] * x40 * x7 - 8 * coeffs[10] * x26 * x35 -
             8 * coeffs[11] * x24 * x25 - 8 * coeffs[13] * x15 * x19 -
             8 * coeffs[14] * x27 * x28 - 8 * coeffs[15] * x18 * x20 -
             8 * coeffs[16] * x25 * x29 - 8 * coeffs[17] * x29 * x30 -
             8 * coeffs[18] * x15 * x22 * x8 - 8 * coeffs[19] * x20 * x22 +
             8 * coeffs[1] * x40 * x9 + 8 * coeffs[20] * x1 * x7 +
             8 * coeffs[21] * x1 * x9 + 8 * coeffs[22] * x16 * x31 +
             8 * coeffs[23] * x22 * x23 + 8 * coeffs[25] * x18 * x23 +
             8 * coeffs[2] * x38 * x8 + 8 * coeffs[3] * x3 * x38 +
             8 * coeffs[4] * x18 * x3 * x39 + 8 * coeffs[5] * x19 * x39 +
             8 * coeffs[6] * x34 * x8 + 8 * coeffs[7] * x3 * x34 -
             8 * coeffs[9] * x24 * x30 + 8 * x0 * x13 * x32 -
             8 * x10 * x13 * x14 - 8 * x31 * x42 - 8 * x36 * x37;
    out[1] = 8 * coeffs[0] * x65 * x68 - 8 * coeffs[10] * x52 * x59 -
             8 * coeffs[11] * x50 * x71 - 8 * coeffs[13] * x27 * x60 -
             8 * coeffs[15] * x27 * x66 - 8 * coeffs[16] * x47 * x55 * x[2] -
             8 * coeffs[17] * x5 * x51 * x55 - 8 * coeffs[18] * x26 * x5 * x53 -
             8 * coeffs[19] * x26 * x59 * x[2] + 8 * coeffs[22] * x44 * x48 +
             8 * coeffs[23] * x48 * x49 + 8 * coeffs[3] * x59 * x68 +
             8 * coeffs[4] * x63 * x66 + 8 * coeffs[5] * x60 * x63 +
             8 * coeffs[6] * x60 * x64 + 8 * coeffs[7] * x64 * x66 +
             8 * x11 * x32 * x50 - 8 * x14 * x50 * x51 + 8 * x16 * x50 * x61 +
             8 * x22 * x45 * x56 - 8 * x22 * x71 * x72 + 8 * x29 * x57 * x58 -
             8 * x36 * x44 * x46 * x52 - 8 * x41 * x53 * x54 -
             8 * x42 * x65 * x[0] + 8 * x44 * x67 * x69 + 8 * x49 * x69 * x70;
    out[2] = 8 * coeffs[0] * x77 * x92 - 8 * coeffs[10] * x83 * x85 -
             8 * coeffs[11] * x77 * x89 - 8 * coeffs[12] * x79 * x91 -
             8 * coeffs[13] * x26 * x82 * x86 - 8 * coeffs[15] * x26 * x80 -
             8 * coeffs[16] * x37 * x46 - 8 * coeffs[17] * x37 * x[0] -
             8 * coeffs[18] * x24 * x27 * x[0] - 8 * coeffs[19] * x27 * x88 +
             8 * coeffs[22] * x81 * x87 + 8 * coeffs[23] * x87 * x88 +
             8 * coeffs[24] * x73 * x77 + 8 * coeffs[3] * x83 * x93 +
             8 * coeffs[4] * x79 * x92 + 8 * coeffs[5] * x86 * x92 * x[0] +
             8 * coeffs[6] * x78 * x82 * x92 + 8 * coeffs[7] * x46 * x78 * x93 -
             8 * coeffs[8] * x77 * x91 + 8 * x0 * x61 * x81 -
             8 * x14 * x81 * x82 + 8 * x18 * x56 * x73 - 8 * x54 * x78 * x85 +
             8 * x58 * x80 + 8 * x67 * x90 * x92 + 8 * x70 * x75 * x93 -
             8 * x72 * x89 * x90;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x0 - 3;
    const Scalar x2 = x[2] - 1;
    const Scalar x3 = 2 * x[2] - 1;
    const Scalar x4 = x2 * x3;
    const Scalar x5 = x[1] - 1;
    const Scalar x6 = 2 * x[1] - 1;
    const Scalar x7 = x5 * x6;
    const Scalar x8 = x4 * x7;
    const Scalar x9 = x[0] - 1;
    const Scalar x10 = 2 * x[0] - 1;
    const Scalar x11 = x10 * x9;
    const Scalar x12 = 4 * x[1];
    const Scalar x13 = x12 - 3;
    const Scalar x14 = x13 * x4;
    const Scalar x15 = 4 * x[2];
    const Scalar x16 = x15 - 3;
    const Scalar x17 = x16 * x7;
    const Scalar x18 = x0 - 1;
    const Scalar x19 = x10 * x[0];
    const Scalar x20 = x6 * x[1];
    const Scalar x21 = x18 * x4;
    const Scalar x22 = x12 - 1;
    const Scalar x23 = x22 * x4;
    const Scalar x24 = x16 * x20;
    const Scalar x25 = x1 * x20;
    const Scalar x26 = x3 * x[2];
    const Scalar x27 = x1 * x7;
    const Scalar x28 = x13 * x26;
    const Scalar x29 = x15 - 1;
    const Scalar x30 = x29 * x7;
    const Scalar x31 = x18 * x26;
    const Scalar x32 = x22 * x26;
    const Scalar x33 = x20 * x29;
    const Scalar x34 = x0 * x9;
    const Scalar x35 = x12 * x5;
    const Scalar x36 = x0 * x10;
    const Scalar x37 = x4 * x6;
    const Scalar x38 = x5 * x[1];
    const Scalar x39 = x36 * x38;
    const Scalar x40 = x10 * x12;
    const Scalar x41 = x1 * x35;
    const Scalar x42 = 4 * x11;
    const Scalar x43 = x11 * x35;
    const Scalar x44 = x15 * x3;
    const Scalar x45 = x10 * x7;
    const Scalar x46 = x26 * x6;
    const Scalar x47 = x11 * x6;
    const Scalar x48 = x15 * x2;
    const Scalar x49 = x11 * x48;
    const Scalar x50 = x3 * x7;
    const Scalar x51 = x2 * x[2];
    const Scalar x52 = x36 * x51;
    const Scalar x53 = x18 * x51;
    const Scalar x54 = x12 * x6;
    const Scalar x55 = x20 * x3;
    const Scalar x56 = x1 * x51;
    const Scalar x57 = 16 * x38;
    const Scalar x58 = 16 * x51;
    const Scalar x59 = x3 * x57;
    const Scalar x60 = x9 * x[0];
    const Scalar x61 = x58 * x60;
    const Scalar x62 = 16 * x60;
    const Scalar x63 = x10 * x57;
    const Scalar x64 = x57 * x60;
    const Scalar x65 = 64 * x51;
    out[0][0] = x1 * x8;
    out[0][1] = x11 * x14;
    out[0][2] = x11 * x17;
    out[1][0] = x18 * x8;
    out[1][1] = x14 * x19;
    out[1][2] = x17 * x19;
    out[2][0] = x20 * x21;
    out[2][1] = x19 * x23;
    out[2][2] = x19 * x24;
    out[3][0] = x25 * x4;
    out[3][1] = x11 * x23;
    out[3][2] = x11 * x24;
    out[4][0] = x26 * x27;
    out[4][1] = x11 * x28;
    out[4][2] = x11 * x30;
    out[5][0] = x31 * x7;
    out[5][1] = x19 * x28;
    out[5][2] = x19 * x30;
    out[6][0] = x20 * x31;
    out[6][1] = x19 * x32;
    out[6][2] = x19 * x33;
    out[7][0] = x25 * x26;
    out[7][1] = x11 * x32;
    out[7][2] = x11 * x33;
    out[8][0] = -4 * x10 * x8;
    out[8][1] = -x14 * x34;
    out[8][2] = -x17 * x34;
    out[9][0] = -x21 * x35;
    out[9][1] = -x36 * x37;
    out[9][2] = -x16 * x39;
    out[10][0] = -x37 * x40;
    out[10][1] = -x23 * x34;
    out[10][2] = -x24 * x34;
    out[11][0] = -x4 * x41;
    out[11][1] = -x37 * x42;
    out[11][2] = -x16 * x43;
    out[12][0] = -x44 * x45;
    out[12][1] = -x28 * x34;
    out[12][2] = -x30 * x34;
    out[13][0] = -x31 * x35;
    out[13][1] = -x36 * x46;
    out[13][2] = -x29 * x39;
    out[14][0] = -x40 * x46;
    out[14][1] = -x32 * x34;
    out[14][2] = -x33 * x34;
    out[15][0] = -x26 * x41;
    out[15][1] = -x44 * x47;
    out[15][2] = -x29 * x43;
    out[16][0] = -x27 * x48;
    out[16][1] = -x13 * x49;
    out[16][2] = -x42 * x50;
    out[17][0] = -x18 * x48 * x7;
    out[17][1] = -x13 * x52;
    out[17][2] = -x36 * x50;
    out[18][0] = -x53 * x54;
    out[18][1] = -x22 * x52;
    out[18][2] = -x36 * x55;
    out[19][0] = -x54 * x56;
    out[19][1] = -x22 * x49;
    out[19][2] = -x12 * x3 * x47;
    out[20][0] = x56 * x57;
    out[20][1] = x47 * x58;
    out[20][2] = x11 * x59;
    out[21][0] = x53 * x57;
    out[21][1] = x19 * x58 * x6;
    out[21][2] = x19 * x59;
    out[22][0] = x45 * x58;
    out[22][1] = x13 * x61;
    out[22][2] = x50 * x62;
    out[23][0] = x10 * x20 * x58;
    out[23][1] = x22 * x61;
    out[23][2] = x55 * x62;
    out[24][0] = x4 * x63;
    out[24][1] = x37 * x62;
    out[24][2] = x16 * x64;
    out[25][0] = x26 * x63;
    out[25][1] = x46 * x62;
    out[25][2] = x29 * x64;
    out[26][0] = -x10 * x38 * x65;
    out[26][1] = -x6 * x60 * x65;
    out[26][2] = -64 * x3 * x38 * x60;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[3]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[0][2] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(2) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[1][2] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(2) / order;
    out[2][1] = static_cast<Scalar>(2) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(2) / order;
    out[3][2] = static_cast<Scalar>(0) / order;
    out[4][0] = static_cast<Scalar>(0) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[4][2] = static_cast<Scalar>(2) / order;
    out[5][0] = static_cast<Scalar>(2) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[5][2] = static_cast<Scalar>(2) / order;
    out[6][0] = static_cast<Scalar>(2) / order;
    out[6][1] = static_cast<Scalar>(2) / order;
    out[6][2] = static_cast<Scalar>(2) / order;
    out[7][0] = static_cast<Scalar>(0) / order;
    out[7][1] = static_cast<Scalar>(2) / order;
    out[7][2] = static_cast<Scalar>(2) / order;
    out[8][0] = static_cast<Scalar>(1) / order;
    out[8][1] = static_cast<Scalar>(0) / order;
    out[8][2] = static_cast<Scalar>(0) / order;
    out[9][0] = static_cast<Scalar>(2) / order;
    out[9][1] = static_cast<Scalar>(1) / order;
    out[9][2] = static_cast<Scalar>(0) / order;
    out[10][0] = static_cast<Scalar>(1) / order;
    out[10][1] = static_cast<Scalar>(2) / order;
    out[10][2] = static_cast<Scalar>(0) / order;
    out[11][0] = static_cast<Scalar>(0) / order;
    out[11][1] = static_cast<Scalar>(1) / order;
    out[11][2] = static_cast<Scalar>(0) / order;
    out[12][0] = static_cast<Scalar>(1) / order;
    out[12][1] = static_cast<Scalar>(0) / order;
    out[12][2] = static_cast<Scalar>(2) / order;
    out[13][0] = static_cast<Scalar>(2) / order;
    out[13][1] = static_cast<Scalar>(1) / order;
    out[13][2] = static_cast<Scalar>(2) / order;
    out[14][0] = static_cast<Scalar>(1) / order;
    out[14][1] = static_cast<Scalar>(2) / order;
    out[14][2] = static_cast<Scalar>(2) / order;
    out[15][0] = static_cast<Scalar>(0) / order;
    out[15][1] = static_cast<Scalar>(1) / order;
    out[15][2] = static_cast<Scalar>(2) / order;
    out[16][0] = static_cast<Scalar>(0) / order;
    out[16][1] = static_cast<Scalar>(0) / order;
    out[16][2] = static_cast<Scalar>(1) / order;
    out[17][0] = static_cast<Scalar>(2) / order;
    out[17][1] = static_cast<Scalar>(0) / order;
    out[17][2] = static_cast<Scalar>(1) / order;
    out[18][0] = static_cast<Scalar>(2) / order;
    out[18][1] = static_cast<Scalar>(2) / order;
    out[18][2] = static_cast<Scalar>(1) / order;
    out[19][0] = static_cast<Scalar>(0) / order;
    out[19][1] = static_cast<Scalar>(2) / order;
    out[19][2] = static_cast<Scalar>(1) / order;
    out[20][0] = static_cast<Scalar>(0) / order;
    out[20][1] = static_cast<Scalar>(1) / order;
    out[20][2] = static_cast<Scalar>(1) / order;
    out[21][0] = static_cast<Scalar>(2) / order;
    out[21][1] = static_cast<Scalar>(1) / order;
    out[21][2] = static_cast<Scalar>(1) / order;
    out[22][0] = static_cast<Scalar>(1) / order;
    out[22][1] = static_cast<Scalar>(0) / order;
    out[22][2] = static_cast<Scalar>(1) / order;
    out[23][0] = static_cast<Scalar>(1) / order;
    out[23][1] = static_cast<Scalar>(2) / order;
    out[23][2] = static_cast<Scalar>(1) / order;
    out[24][0] = static_cast<Scalar>(1) / order;
    out[24][1] = static_cast<Scalar>(1) / order;
    out[24][2] = static_cast<Scalar>(0) / order;
    out[25][0] = static_cast<Scalar>(1) / order;
    out[25][1] = static_cast<Scalar>(1) / order;
    out[25][2] = static_cast<Scalar>(2) / order;
    out[26][0] = static_cast<Scalar>(1) / order;
    out[26][1] = static_cast<Scalar>(1) / order;
    out[26][2] = static_cast<Scalar>(1) / order;
  }
  template <typename Scalar>
  static constexpr void interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
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
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 21:
      out[0] = 2;
      out[1] = 1;
      out[2] = 1;
      break;
    case 22:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 23:
      out[0] = 1;
      out[1] = 2;
      out[2] = 1;
      break;
    case 24:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 25:
      out[0] = 1;
      out[1] = 1;
      out[2] = 2;
      break;
    case 26:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    }
  }

  template <typename Element>
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs) {
    subelement_node_idxs(subelement, idxs, meta::type_tag<Element>{});
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElPoint>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      break;
    case 1:
      idxs[0] = 1;
      break;
    case 2:
      idxs[0] = 2;
      break;
    case 3:
      idxs[0] = 3;
      break;
    case 4:
      idxs[0] = 4;
      break;
    case 5:
      idxs[0] = 5;
      break;
    case 6:
      idxs[0] = 6;
      break;
    case 7:
      idxs[0] = 7;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 8;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 9;
      break;
    case 2:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 10;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 11;
      break;
    case 4:
      idxs[0] = 4;
      idxs[1] = 5;
      idxs[2] = 12;
      break;
    case 5:
      idxs[0] = 5;
      idxs[1] = 6;
      idxs[2] = 13;
      break;
    case 6:
      idxs[0] = 7;
      idxs[1] = 6;
      idxs[2] = 14;
      break;
    case 7:
      idxs[0] = 4;
      idxs[1] = 7;
      idxs[2] = 15;
      break;
    case 8:
      idxs[0] = 0;
      idxs[1] = 4;
      idxs[2] = 16;
      break;
    case 9:
      idxs[0] = 1;
      idxs[1] = 5;
      idxs[2] = 17;
      break;
    case 10:
      idxs[0] = 2;
      idxs[1] = 6;
      idxs[2] = 18;
      break;
    case 11:
      idxs[0] = 3;
      idxs[1] = 7;
      idxs[2] = 19;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 7;
      idxs[3] = 4;
      idxs[4] = 11;
      idxs[5] = 19;
      idxs[6] = 15;
      idxs[7] = 16;
      idxs[8] = 20;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 5;
      idxs[4] = 9;
      idxs[5] = 18;
      idxs[6] = 13;
      idxs[7] = 17;
      idxs[8] = 21;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 5;
      idxs[3] = 4;
      idxs[4] = 8;
      idxs[5] = 17;
      idxs[6] = 12;
      idxs[7] = 16;
      idxs[8] = 22;
      break;
    case 3:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 7;
      idxs[4] = 10;
      idxs[5] = 18;
      idxs[6] = 14;
      idxs[7] = 19;
      idxs[8] = 23;
      break;
    case 4:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 2;
      idxs[3] = 3;
      idxs[4] = 8;
      idxs[5] = 9;
      idxs[6] = 10;
      idxs[7] = 11;
      idxs[8] = 24;
      break;
    case 5:
      idxs[0] = 4;
      idxs[1] = 5;
      idxs[2] = 6;
      idxs[3] = 7;
      idxs[4] = 12;
      idxs[5] = 13;
      idxs[6] = 14;
      idxs[7] = 15;
      idxs[8] = 25;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTetra>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElCube>) {
    switch (subelement) {
    default:
      break;
    }
  }
};

template <> struct BasisLagrange<mesh::RefElCube, 3> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 64;
  static constexpr dim_t num_interpolation_nodes = 64;

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
    const Scalar x18 = x13 * x8;
    const Scalar x19 = x18 * x[0];
    const Scalar x20 = x7 * x[2];
    const Scalar x21 = x20 * x5;
    const Scalar x22 = (1.0 / 9.0) * x21;
    const Scalar x23 = x22 * x[1];
    const Scalar x24 = x8 * x[0];
    const Scalar x25 = x6 * x[1];
    const Scalar x26 = x25 * x5;
    const Scalar x27 = (1.0 / 9.0) * x19 * x[2];
    const Scalar x28 = x22 * x25;
    const Scalar x29 = x15 * x3 * x[0];
    const Scalar x30 = (1.0 / 9.0) * x20;
    const Scalar x31 = x17 * x[0];
    const Scalar x32 = x12 * x31;
    const Scalar x33 = (1.0 / 81.0) * x21;
    const Scalar x34 = (1.0 / 81.0) * x[2];
    const Scalar x35 = x5 * x[1];
    const Scalar x36 = x1 * x12;
    const Scalar x37 = x36 * x6 * x8 * x[1];
    const Scalar x38 = x19 * x6;
    const Scalar x39 = x15 * x38;
    const Scalar x40 = (1.0 / 9.0) * x7;
    const Scalar x41 = x37 * x[0];
    const Scalar x42 = x17 * x26;
    const Scalar x43 = x14 * x36;
    const Scalar x44 = (1.0 / 81.0) * x7;
    const Scalar x45 = x24 * x35;
    const Scalar x46 = (1.0 / 81.0) * x6;
    const Scalar x47 = x31 * x36;
    const Scalar x48 = (1.0 / 729.0) * x36;
    const Scalar x49 = x17 * x48;
    return -729.0 / 8.0 * coeffs[0] * x49 * x5 * x6 * x7 * x8 -
           729.0 / 8.0 * coeffs[10] * x43 * x44 * x45 +
           (9.0 / 8.0) * coeffs[11] * x11 * x13 * x14 * x3 * x5 * x7 * x8 *
               x[0] * x[1] -
           9.0 / 8.0 * coeffs[12] * x17 * x41 +
           (9.0 / 8.0) * coeffs[13] * x1 * x13 * x14 * x3 * x5 * x6 * x8 *
               x[0] * x[1] +
           (9.0 / 8.0) * coeffs[14] * x1 * x11 * x14 * x3 * x5 * x6 * x7 * x8 *
               x[1] -
           729.0 / 8.0 * coeffs[15] * x12 * x42 * x44 * x8 -
           729.0 / 8.0 * coeffs[16] * x20 * x46 * x47 +
           (9.0 / 8.0) * coeffs[17] * x1 * x13 * x14 * x3 * x5 * x6 * x7 *
               x[0] * x[2] +
           (9.0 / 8.0) * coeffs[18] * x1 * x11 * x14 * x3 * x5 * x7 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[19] * x32 * x33 * x[1] +
           (1.0 / 8.0) * coeffs[1] * x1 * x11 * x13 * x14 * x3 * x5 * x7 * x8 *
               x[0] +
           (9.0 / 8.0) * coeffs[20] * x1 * x11 * x13 * x14 * x3 * x6 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[21] * x13 * x26 * x29 * x34 -
           729.0 / 8.0 * coeffs[22] * x25 * x33 * x43 +
           (9.0 / 8.0) * coeffs[23] * x11 * x13 * x14 * x3 * x5 * x6 * x7 *
               x[1] * x[2] +
           (9.0 / 8.0) * coeffs[24] * x1 * x11 * x13 * x3 * x5 * x6 * x7 * x8 *
               x[2] -
           729.0 / 8.0 * coeffs[25] * x16 * x18 * x21 * x46 -
           729.0 / 8.0 * coeffs[26] * x19 * x33 * x36 +
           (9.0 / 8.0) * coeffs[27] * x1 * x11 * x13 * x14 * x5 * x7 * x8 *
               x[0] * x[2] +
           (9.0 / 8.0) * coeffs[28] * x1 * x11 * x13 * x3 * x5 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[29] * x16 * x19 * x34 * x35 -
           729.0 / 8.0 * coeffs[2] * x45 * x49 -
           729.0 / 8.0 * coeffs[30] * x13 * x34 * x37 * x5 +
           (9.0 / 8.0) * coeffs[31] * x1 * x11 * x13 * x14 * x5 * x6 * x8 *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[32] * x22 * x37 +
           (81.0 / 8.0) * coeffs[33] * x11 * x13 * x3 * x5 * x6 * x7 * x8 *
               x[1] * x[2] +
           (81.0 / 8.0) * coeffs[34] * x1 * x11 * x14 * x5 * x6 * x7 * x8 *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[35] * x11 * x14 * x18 * x28 +
           (81.0 / 8.0) * coeffs[36] * x1 * x11 * x3 * x5 * x7 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[37] * x12 * x19 * x23 -
           729.0 / 8.0 * coeffs[38] * x16 * x23 * x24 +
           (81.0 / 8.0) * coeffs[39] * x11 * x13 * x14 * x5 * x7 * x8 * x[0] *
               x[1] * x[2] +
           (1.0 / 8.0) * coeffs[3] * x1 * x11 * x13 * x14 * x3 * x5 * x6 * x8 *
               x[1] -
           729.0 / 8.0 * coeffs[40] * x30 * x36 * x38 +
           (81.0 / 8.0) * coeffs[41] * x1 * x13 * x3 * x5 * x6 * x7 * x8 *
               x[0] * x[2] +
           (81.0 / 8.0) * coeffs[42] * x1 * x11 * x13 * x14 * x6 * x7 * x8 *
               x[0] * x[2] -
           729.0 / 8.0 * coeffs[43] * x22 * x39 +
           (81.0 / 8.0) * coeffs[44] * x1 * x11 * x13 * x3 * x6 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[45] * x1 * x26 * x27 * x3 -
           729.0 / 8.0 * coeffs[46] * x16 * x25 * x27 +
           (81.0 / 8.0) * coeffs[47] * x1 * x13 * x14 * x5 * x6 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[48] * x14 * x40 * x41 +
           (81.0 / 8.0) * coeffs[49] * x1 * x14 * x3 * x5 * x6 * x7 * x8 *
               x[0] * x[1] +
           (1.0 / 8.0) * coeffs[4] * x1 * x11 * x13 * x14 * x3 * x5 * x6 * x7 *
               x[2] +
           (81.0 / 8.0) * coeffs[50] * x11 * x13 * x14 * x3 * x6 * x7 * x8 *
               x[0] * x[1] -
           729.0 / 8.0 * coeffs[51] * x24 * x3 * x40 * x42 +
           (81.0 / 8.0) * coeffs[52] * x1 * x11 * x14 * x3 * x6 * x7 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[53] * x28 * x29 -
           729.0 / 8.0 * coeffs[54] * x25 * x30 * x32 +
           (81.0 / 8.0) * coeffs[55] * x13 * x14 * x3 * x5 * x6 * x7 * x[0] *
               x[1] * x[2] +
           (729.0 / 8.0) * coeffs[56] * x1 * x11 * x3 * x6 * x7 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[57] * x1 * x10 * x3 -
           729.0 / 8.0 * coeffs[58] * x12 * x13 * x9 +
           (729.0 / 8.0) * coeffs[59] * x13 * x3 * x5 * x6 * x7 * x8 * x[0] *
               x[1] * x[2] -
           1.0 / 8.0 * coeffs[5] * x21 * x47 -
           729.0 / 8.0 * coeffs[60] * x16 * x9 +
           (729.0 / 8.0) * coeffs[61] * x1 * x14 * x5 * x6 * x7 * x8 * x[0] *
               x[1] * x[2] +
           (729.0 / 8.0) * coeffs[62] * x11 * x13 * x14 * x6 * x7 * x8 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[63] * x10 * x17 +
           (1.0 / 8.0) * coeffs[6] * x1 * x11 * x13 * x14 * x3 * x5 * x[0] *
               x[1] * x[2] -
           729.0 / 8.0 * coeffs[7] * x42 * x48 * x[2] +
           (9.0 / 8.0) * coeffs[8] * x1 * x11 * x13 * x14 * x3 * x6 * x7 * x8 *
               x[0] -
           729.0 / 8.0 * coeffs[9] * x3 * x39 * x44 * x5;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[2] - 1;
    const Scalar x1 = x[0] - 1;
    const Scalar x2 = 3 * x[1];
    const Scalar x3 = x2 - 1;
    const Scalar x4 = x1 * x3;
    const Scalar x5 = x0 * x4;
    const Scalar x6 = x[1] - 1;
    const Scalar x7 = 3 * x[0];
    const Scalar x8 = x7 - 1;
    const Scalar x9 = 3 * x[2];
    const Scalar x10 = x9 - 1;
    const Scalar x11 = x7 - 2;
    const Scalar x12 = x2 - 2;
    const Scalar x13 = x9 - 2;
    const Scalar x14 = x10 * x11 * x12 * x13 * x6 * x8;
    const Scalar x15 = (1.0 / 8.0) * x14;
    const Scalar x16 = x3 * x[0];
    const Scalar x17 = x0 * x16;
    const Scalar x18 = x11 * x12 * x13;
    const Scalar x19 = x17 * x18;
    const Scalar x20 = x10 * x8;
    const Scalar x21 = x20 * x[1];
    const Scalar x22 = (1.0 / 8.0) * x21;
    const Scalar x23 = x18 * x22;
    const Scalar x24 = x15 * x[2];
    const Scalar x25 = x23 * x[2];
    const Scalar x26 = (9.0 / 8.0) * x[0];
    const Scalar x27 = x26 * x5;
    const Scalar x28 = x18 * x6;
    const Scalar x29 = x10 * x28;
    const Scalar x30 = x12 * x27;
    const Scalar x31 = x13 * x6;
    const Scalar x32 = x20 * x31;
    const Scalar x33 = x26 * x[1];
    const Scalar x34 = x0 * x14;
    const Scalar x35 = x21 * x31;
    const Scalar x36 = (9.0 / 8.0) * x11;
    const Scalar x37 = x17 * x36;
    const Scalar x38 = x10 * x18;
    const Scalar x39 = x13 * x21;
    const Scalar x40 = (9.0 / 8.0) * x[1];
    const Scalar x41 = x1 * x40;
    const Scalar x42 = x35 * x36;
    const Scalar x43 = x4 * x[2];
    const Scalar x44 = x26 * x43;
    const Scalar x45 = x12 * x44;
    const Scalar x46 = x14 * x[2];
    const Scalar x47 = (9.0 / 8.0) * x8;
    const Scalar x48 = x5 * x[2];
    const Scalar x49 = x28 * x48;
    const Scalar x50 = x12 * x6;
    const Scalar x51 = x20 * x50;
    const Scalar x52 = x36 * x48;
    const Scalar x53 = x19 * x[2];
    const Scalar x54 = x37 * x[2];
    const Scalar x55 = x40 * x8;
    const Scalar x56 = x12 * x21;
    const Scalar x57 = x18 * x48;
    const Scalar x58 = (81.0 / 8.0) * x[2];
    const Scalar x59 = x0 * x58;
    const Scalar x60 = x1 * x59;
    const Scalar x61 = x8 * x[1];
    const Scalar x62 = x28 * x61;
    const Scalar x63 = x31 * x61;
    const Scalar x64 = (81.0 / 8.0) * x48;
    const Scalar x65 = x11 * x64;
    const Scalar x66 = x11 * x21;
    const Scalar x67 = x50 * x66;
    const Scalar x68 = x21 * x6;
    const Scalar x69 = x59 * x[0];
    const Scalar x70 = x17 * x58;
    const Scalar x71 = (81.0 / 8.0) * x[0];
    const Scalar x72 = x64 * x[0];
    const Scalar x73 = x12 * x72;
    const Scalar x74 = x50 * x[0];
    const Scalar x75 = x10 * x65;
    const Scalar x76 = x71 * x[1];
    const Scalar x77 = x[0] * x[1];
    const Scalar x78 = x0 * x1;
    const Scalar x79 = x35 * x71;
    const Scalar x80 = x12 * x78;
    const Scalar x81 = x11 * x31;
    const Scalar x82 = x10 * x76 * x81;
    const Scalar x83 = x1 * x58;
    const Scalar x84 = (729.0 / 8.0) * x[2];
    const Scalar x85 = x78 * x84;
    const Scalar x86 = x63 * x[0];
    const Scalar x87 = (729.0 / 8.0) * x48;
    const Scalar x88 = x77 * x87;
    const Scalar x89 = x74 * x85;
    const Scalar x90 = x10 * x11;
    out[0] = -x15 * x5;
    out[1] = x15 * x17;
    out[2] = -x19 * x22;
    out[3] = x23 * x5;
    out[4] = x24 * x4;
    out[5] = -x16 * x24;
    out[6] = x16 * x25;
    out[7] = -x25 * x4;
    out[8] = x27 * x29;
    out[9] = -x30 * x32;
    out[10] = -x33 * x34;
    out[11] = x35 * x37;
    out[12] = -x27 * x38 * x[1];
    out[13] = x30 * x39;
    out[14] = x34 * x41;
    out[15] = -x42 * x5;
    out[16] = -x29 * x44;
    out[17] = x32 * x45;
    out[18] = x33 * x46;
    out[19] = -x16 * x42 * x[2];
    out[20] = x33 * x38 * x43;
    out[21] = -x39 * x45;
    out[22] = -x41 * x46;
    out[23] = x42 * x43;
    out[24] = x47 * x49;
    out[25] = -x51 * x52;
    out[26] = -x47 * x53 * x6;
    out[27] = x51 * x54;
    out[28] = x53 * x55;
    out[29] = -x54 * x56;
    out[30] = -x55 * x57;
    out[31] = x52 * x56;
    out[32] = -x60 * x62;
    out[33] = x63 * x65;
    out[34] = x60 * x67;
    out[35] = -x65 * x68;
    out[36] = x62 * x69;
    out[37] = -x11 * x63 * x70;
    out[38] = -x67 * x69;
    out[39] = x6 * x66 * x70;
    out[40] = -x49 * x71;
    out[41] = x31 * x73 * x8;
    out[42] = x74 * x75;
    out[43] = -x51 * x72;
    out[44] = x57 * x76;
    out[45] = -x13 * x61 * x73;
    out[46] = -x12 * x75 * x77;
    out[47] = x56 * x72;
    out[48] = -x29 * x76 * x78;
    out[49] = x79 * x80;
    out[50] = x5 * x82;
    out[51] = -x5 * x79;
    out[52] = x29 * x77 * x83;
    out[53] = -x12 * x35 * x83 * x[0];
    out[54] = -x43 * x82;
    out[55] = x43 * x79;
    out[56] = x28 * x77 * x85;
    out[57] = -x80 * x84 * x86;
    out[58] = -x81 * x88;
    out[59] = x86 * x87;
    out[60] = -x89 * x90 * x[1];
    out[61] = x21 * x89;
    out[62] = x6 * x88 * x90;
    out[63] = -x68 * x87 * x[0];
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
    const Scalar x29 = coeffs[63] * x23;
    const Scalar x30 = x16 * x6;
    const Scalar x31 = (1.0 / 9.0) * x9;
    const Scalar x32 = x11 * x22;
    const Scalar x33 = (1.0 / 9.0) * x27;
    const Scalar x34 = x10 * x26;
    const Scalar x35 = x31 * x34;
    const Scalar x36 = x22 * x26;
    const Scalar x37 = (1.0 / 81.0) * x36;
    const Scalar x38 = x37 * x9;
    const Scalar x39 = x10 * x15;
    const Scalar x40 = x32 * x39;
    const Scalar x41 = (1.0 / 9.0) * coeffs[48];
    const Scalar x42 = x11 * x8;
    const Scalar x43 = x34 * x42;
    const Scalar x44 = x23 * x43;
    const Scalar x45 = (1.0 / 9.0) * coeffs[51];
    const Scalar x46 = x18 * x3;
    const Scalar x47 = x1 * x18 + x1 * x3 + x46;
    const Scalar x48 = (1.0 / 9.0) * x47;
    const Scalar x49 = coeffs[37] * x23;
    const Scalar x50 = coeffs[38] * x16;
    const Scalar x51 = x37 * x42;
    const Scalar x52 = (1.0 / 81.0) * x47;
    const Scalar x53 = x34 * x9;
    const Scalar x54 = coeffs[19] * x23;
    const Scalar x55 = x16 * x52;
    const Scalar x56 = coeffs[29] * x27;
    const Scalar x57 = coeffs[26] * x9;
    const Scalar x58 = 3 * x20 + x46 + 3 * x5;
    const Scalar x59 = (1.0 / 9.0) * x58;
    const Scalar x60 = coeffs[32] * x16;
    const Scalar x61 = coeffs[35] * x23;
    const Scalar x62 = (1.0 / 81.0) * x58;
    const Scalar x63 = x16 * x62;
    const Scalar x64 = (1.0 / 729.0) * x36;
    const Scalar x65 = x42 * x64;
    const Scalar x66 = coeffs[2] * x16;
    const Scalar x67 = coeffs[30] * x9;
    const Scalar x68 = coeffs[5] * x39;
    const Scalar x69 = x64 * x9;
    const Scalar x70 = coeffs[25] * x27;
    const Scalar x71 = coeffs[7] * x16;
    const Scalar x72 = coeffs[0] * x39;
    const Scalar x73 = x10 * x14;
    const Scalar x74 = x16 + x39 + x73;
    const Scalar x75 = x0 * x11;
    const Scalar x76 = x75 * x9;
    const Scalar x77 = x4 * x76;
    const Scalar x78 = x10 * x22;
    const Scalar x79 = x23 + x73 + x78;
    const Scalar x80 = x19 * x79;
    const Scalar x81 = x19 * x74;
    const Scalar x82 = x27 * x75;
    const Scalar x83 = x4 * x82;
    const Scalar x84 = x11 * x3;
    const Scalar x85 = x31 * x80;
    const Scalar x86 = x0 * x26;
    const Scalar x87 = x26 * x3;
    const Scalar x88 = (1.0 / 81.0) * x87;
    const Scalar x89 = x88 * x9;
    const Scalar x90 = x20 * x84;
    const Scalar x91 = x42 * x86;
    const Scalar x92 = x4 * x91;
    const Scalar x93 = x15 * x22;
    const Scalar x94 = x14 * x15 + x14 * x22 + x93;
    const Scalar x95 = (1.0 / 9.0) * x94;
    const Scalar x96 = x42 * x88;
    const Scalar x97 = (1.0 / 81.0) * x94;
    const Scalar x98 = x86 * x9;
    const Scalar x99 = (1.0 / 81.0) * x19;
    const Scalar x100 = x94 * x99;
    const Scalar x101 = 3 * x39 + 3 * x78 + x93;
    const Scalar x102 = (1.0 / 9.0) * x101;
    const Scalar x103 = x101 * x99;
    const Scalar x104 = (1.0 / 729.0) * x87;
    const Scalar x105 = x104 * x42;
    const Scalar x106 = x104 * x9;
    const Scalar x107 = (1.0 / 81.0) * x101;
    const Scalar x108 = x11 * x7;
    const Scalar x109 = x108 + x42 + x9;
    const Scalar x110 = x0 * x10;
    const Scalar x111 = x110 * x4;
    const Scalar x112 = x109 * x19;
    const Scalar x113 = x110 * x23;
    const Scalar x114 = x110 * x16;
    const Scalar x115 = x11 * x26;
    const Scalar x116 = x108 + x115 + x27;
    const Scalar x117 = x116 * x19;
    const Scalar x118 = x10 * x3;
    const Scalar x119 = (1.0 / 9.0) * x118;
    const Scalar x120 = (1.0 / 9.0) * x109;
    const Scalar x121 = x0 * x22;
    const Scalar x122 = x121 * x16;
    const Scalar x123 = x122 * x4;
    const Scalar x124 = x22 * x3;
    const Scalar x125 = (1.0 / 81.0) * x124;
    const Scalar x126 = x125 * x16;
    const Scalar x127 = x118 * x20;
    const Scalar x128 = (1.0 / 9.0) * x116;
    const Scalar x129 = x121 * x39;
    const Scalar x130 = x129 * x4;
    const Scalar x131 = x26 * x8;
    const Scalar x132 = x131 + x26 * x7 + x7 * x8;
    const Scalar x133 = (1.0 / 9.0) * x132;
    const Scalar x134 = x125 * x39;
    const Scalar x135 = x132 * x99;
    const Scalar x136 = (1.0 / 81.0) * x127;
    const Scalar x137 = 3 * x115 + x131 + 3 * x42;
    const Scalar x138 = x137 * x19;
    const Scalar x139 = x137 * x23;
    const Scalar x140 = x137 * x99;
    const Scalar x141 = (1.0 / 729.0) * x124;
    const Scalar x142 = x141 * x20;
    out[0] =
        -729.0 / 8.0 * coeffs[10] * x43 * x55 +
        (9.0 / 8.0) * coeffs[11] * x10 * x11 * x22 * x26 * x47 * x8 * x[1] -
        729.0 / 8.0 * coeffs[12] * x25 * x51 +
        (9.0 / 8.0) * coeffs[13] * x11 * x15 * x22 * x26 * x6 * x8 * x[1] +
        (9.0 / 8.0) * coeffs[14] * x10 * x11 * x15 * x26 * x58 * x8 * x[1] -
        729.0 / 8.0 * coeffs[15] * x44 * x62 -
        729.0 / 8.0 * coeffs[16] * x21 * x38 * x39 +
        (9.0 / 8.0) * coeffs[17] * x10 * x15 * x22 * x26 * x6 * x8 * x[2] +
        (9.0 / 8.0) * coeffs[18] * x10 * x15 * x26 * x47 * x8 * x[1] * x[2] +
        (1.0 / 8.0) * coeffs[1] * x10 * x11 * x15 * x22 * x26 * x47 * x8 +
        (9.0 / 8.0) * coeffs[20] * x15 * x21 * x22 * x26 * x8 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[21] * x30 * x38 -
        729.0 / 8.0 * coeffs[22] * x53 * x63 +
        (9.0 / 8.0) * coeffs[23] * x10 * x22 * x26 * x58 * x8 * x[1] * x[2] +
        (9.0 / 8.0) * coeffs[24] * x10 * x11 * x15 * x22 * x58 * x8 * x[2] +
        (9.0 / 8.0) * coeffs[27] * x10 * x11 * x15 * x22 * x26 * x47 * x[2] +
        (9.0 / 8.0) * coeffs[28] * x11 * x15 * x22 * x47 * x8 * x[1] * x[2] +
        (9.0 / 8.0) * coeffs[31] * x11 * x15 * x22 * x26 * x58 * x[1] * x[2] +
        (81.0 / 8.0) * coeffs[33] * x10 * x11 * x22 * x58 * x8 * x[1] * x[2] +
        (81.0 / 8.0) * coeffs[34] * x10 * x11 * x15 * x26 * x58 * x[1] * x[2] +
        (81.0 / 8.0) * coeffs[36] * x10 * x11 * x15 * x47 * x8 * x[1] * x[2] +
        (81.0 / 8.0) * coeffs[39] * x10 * x11 * x22 * x26 * x47 * x[1] * x[2] +
        (1.0 / 8.0) * coeffs[3] * x11 * x15 * x22 * x26 * x58 * x8 * x[1] -
        729.0 / 8.0 * coeffs[40] * x21 * x31 * x40 +
        (81.0 / 8.0) * coeffs[41] * x10 * x11 * x15 * x22 * x6 * x8 * x[2] +
        (81.0 / 8.0) * coeffs[42] * x10 * x11 * x15 * x21 * x22 * x26 * x[2] -
        729.0 / 8.0 * coeffs[43] * x33 * x40 * x6 +
        (81.0 / 8.0) * coeffs[44] * x11 * x15 * x21 * x22 * x8 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[45] * x30 * x31 * x32 -
        729.0 / 8.0 * coeffs[46] * x25 * x32 * x33 +
        (81.0 / 8.0) * coeffs[47] * x11 * x15 * x22 * x26 * x6 * x[1] * x[2] +
        (81.0 / 8.0) * coeffs[49] * x10 * x11 * x15 * x26 * x6 * x8 * x[1] +
        (1.0 / 8.0) * coeffs[4] * x10 * x15 * x22 * x26 * x58 * x8 * x[2] +
        (81.0 / 8.0) * coeffs[50] * x10 * x11 * x21 * x22 * x26 * x8 * x[1] +
        (81.0 / 8.0) * coeffs[52] * x10 * x15 * x21 * x26 * x8 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[53] * x30 * x35 -
        729.0 / 8.0 * coeffs[54] * x24 * x35 +
        (81.0 / 8.0) * coeffs[55] * x10 * x22 * x26 * x6 * x8 * x[1] * x[2] +
        (729.0 / 8.0) * coeffs[56] * x10 * x11 * x15 * x21 * x8 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[58] * x13 * x24 +
        (729.0 / 8.0) * coeffs[59] * x10 * x11 * x22 * x6 * x8 * x[1] * x[2] -
        729.0 / 8.0 * coeffs[60] * x25 * x28 +
        (729.0 / 8.0) * coeffs[61] * x10 * x11 * x15 * x26 * x6 * x[1] * x[2] +
        (729.0 / 8.0) * coeffs[62] * x10 * x11 * x21 * x22 * x26 * x[1] * x[2] +
        (1.0 / 8.0) * coeffs[6] * x15 * x22 * x26 * x47 * x8 * x[1] * x[2] +
        (9.0 / 8.0) * coeffs[8] * x10 * x11 * x15 * x21 * x22 * x26 * x8 -
        729.0 / 8.0 * coeffs[9] * x39 * x51 * x6 -
        729.0 / 8.0 * x13 * x17 * x6 - 729.0 / 8.0 * x13 * x48 * x49 -
        729.0 / 8.0 * x13 * x59 * x60 - 729.0 / 8.0 * x25 * x41 * x43 -
        729.0 / 8.0 * x28 * x29 * x6 - 729.0 / 8.0 * x28 * x48 * x50 -
        729.0 / 8.0 * x28 * x59 * x61 - 729.0 / 8.0 * x32 * x55 * x56 -
        729.0 / 8.0 * x32 * x63 * x67 - 729.0 / 8.0 * x40 * x52 * x57 -
        729.0 / 8.0 * x40 * x62 * x70 - 729.0 / 8.0 * x44 * x45 * x6 -
        729.0 / 8.0 * x47 * x65 * x66 - 729.0 / 8.0 * x47 * x68 * x69 -
        729.0 / 8.0 * x52 * x53 * x54 - 729.0 / 8.0 * x58 * x65 * x72 -
        729.0 / 8.0 * x58 * x69 * x71;
    out[1] =
        -729.0 / 8.0 * coeffs[0] * x101 * x105 * x20 -
        729.0 / 8.0 * coeffs[10] * x81 * x96 +
        (9.0 / 8.0) * coeffs[11] * x11 * x18 * x26 * x3 * x79 * x8 * x[0] -
        729.0 / 8.0 * coeffs[12] * x100 * x91 +
        (9.0 / 8.0) * coeffs[13] * x0 * x11 * x26 * x3 * x8 * x94 * x[0] +
        (9.0 / 8.0) * coeffs[14] * x0 * x11 * x18 * x26 * x3 * x74 * x8 -
        729.0 / 8.0 * coeffs[15] * x20 * x79 * x96 -
        729.0 / 8.0 * coeffs[16] * x103 * x98 +
        (9.0 / 8.0) * coeffs[17] * x0 * x101 * x26 * x3 * x8 * x[0] * x[2] +
        (9.0 / 8.0) * coeffs[18] * x18 * x26 * x3 * x74 * x8 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[19] * x80 * x89 +
        (1.0 / 8.0) * coeffs[1] * x101 * x11 * x18 * x26 * x3 * x8 * x[0] +
        (9.0 / 8.0) * coeffs[20] * x0 * x18 * x26 * x8 * x94 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[21] * x4 * x97 * x98 -
        729.0 / 8.0 * coeffs[22] * x20 * x74 * x89 +
        (9.0 / 8.0) * coeffs[23] * x0 * x18 * x26 * x3 * x79 * x8 * x[2] +
        (9.0 / 8.0) * coeffs[24] * x0 * x101 * x11 * x18 * x3 * x8 * x[2] +
        (9.0 / 8.0) * coeffs[27] * x101 * x11 * x18 * x26 * x3 * x[0] * x[2] +
        (9.0 / 8.0) * coeffs[28] * x11 * x18 * x3 * x8 * x94 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[2] * x105 * x19 * x94 +
        (9.0 / 8.0) * coeffs[31] * x0 * x11 * x18 * x26 * x3 * x94 * x[2] -
        729.0 / 8.0 * coeffs[32] * x31 * x74 * x90 +
        (81.0 / 8.0) * coeffs[33] * x0 * x11 * x18 * x3 * x79 * x8 * x[2] +
        (81.0 / 8.0) * coeffs[34] * x0 * x11 * x18 * x26 * x3 * x74 * x[2] -
        729.0 / 8.0 * coeffs[35] * x33 * x79 * x90 +
        (81.0 / 8.0) * coeffs[36] * x11 * x18 * x3 * x74 * x8 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[37] * x84 * x85 -
        729.0 / 8.0 * coeffs[38] * x33 * x81 * x84 +
        (81.0 / 8.0) * coeffs[39] * x11 * x18 * x26 * x3 * x79 * x[0] * x[2] +
        (1.0 / 8.0) * coeffs[3] * x0 * x11 * x18 * x26 * x3 * x8 * x94 -
        729.0 / 8.0 * coeffs[40] * x102 * x19 * x76 +
        (81.0 / 8.0) * coeffs[41] * x0 * x101 * x11 * x3 * x8 * x[0] * x[2] +
        (81.0 / 8.0) * coeffs[42] * x0 * x101 * x11 * x18 * x26 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[43] * x102 * x83 +
        (81.0 / 8.0) * coeffs[44] * x0 * x11 * x18 * x8 * x94 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[45] * x77 * x95 -
        729.0 / 8.0 * coeffs[46] * x19 * x82 * x95 +
        (81.0 / 8.0) * coeffs[47] * x0 * x11 * x26 * x3 * x94 * x[0] * x[2] +
        (81.0 / 8.0) * coeffs[49] * x0 * x11 * x26 * x3 * x74 * x8 * x[0] +
        (1.0 / 8.0) * coeffs[4] * x0 * x101 * x18 * x26 * x3 * x8 * x[2] +
        (81.0 / 8.0) * coeffs[50] * x0 * x11 * x18 * x26 * x79 * x8 * x[0] +
        (81.0 / 8.0) * coeffs[52] * x0 * x18 * x26 * x74 * x8 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[53] * x31 * x4 * x74 * x86 -
        729.0 / 8.0 * coeffs[54] * x85 * x86 +
        (81.0 / 8.0) * coeffs[55] * x0 * x26 * x3 * x79 * x8 * x[0] * x[2] +
        (729.0 / 8.0) * coeffs[56] * x0 * x11 * x18 * x74 * x8 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[57] * x74 * x77 -
        729.0 / 8.0 * coeffs[58] * x76 * x80 +
        (729.0 / 8.0) * coeffs[59] * x0 * x11 * x3 * x79 * x8 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[5] * x101 * x106 * x19 -
        729.0 / 8.0 * coeffs[60] * x81 * x82 +
        (729.0 / 8.0) * coeffs[61] * x0 * x11 * x26 * x3 * x74 * x[0] * x[2] +
        (729.0 / 8.0) * coeffs[62] * x0 * x11 * x18 * x26 * x79 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[63] * x79 * x83 +
        (1.0 / 8.0) * coeffs[6] * x18 * x26 * x3 * x8 * x94 * x[0] * x[2] -
        729.0 / 8.0 * coeffs[7] * x106 * x20 * x94 +
        (9.0 / 8.0) * coeffs[8] * x0 * x101 * x11 * x18 * x26 * x8 * x[0] -
        729.0 / 8.0 * coeffs[9] * x107 * x92 - 729.0 / 8.0 * x100 * x56 * x84 -
        729.0 / 8.0 * x103 * x57 * x84 - 729.0 / 8.0 * x107 * x70 * x90 -
        729.0 / 8.0 * x41 * x81 * x91 - 729.0 / 8.0 * x45 * x79 * x92 -
        729.0 / 8.0 * x67 * x90 * x97;
    out[2] =
        -729.0 / 8.0 * coeffs[10] * x118 * x140 * x16 +
        (9.0 / 8.0) * coeffs[11] * x10 * x137 * x18 * x22 * x3 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[12] * x122 * x140 +
        (9.0 / 8.0) * coeffs[13] * x0 * x137 * x15 * x22 * x3 * x[0] * x[1] +
        (9.0 / 8.0) * coeffs[14] * x0 * x10 * x137 * x15 * x18 * x3 * x[1] -
        729.0 / 8.0 * coeffs[15] * x136 * x139 -
        729.0 / 8.0 * coeffs[16] * x129 * x135 +
        (9.0 / 8.0) * coeffs[17] * x0 * x10 * x132 * x15 * x22 * x3 * x[0] +
        (9.0 / 8.0) * coeffs[18] * x10 * x132 * x15 * x18 * x3 * x[0] * x[1] +
        (1.0 / 8.0) * coeffs[1] * x10 * x137 * x15 * x18 * x22 * x3 * x[0] +
        (9.0 / 8.0) * coeffs[20] * x0 * x132 * x15 * x18 * x22 * x[0] * x[1] -
        9.0 / 8.0 * coeffs[21] * x123 * x132 -
        729.0 / 8.0 * coeffs[22] * x132 * x136 * x16 +
        (9.0 / 8.0) * coeffs[23] * x0 * x10 * x132 * x18 * x22 * x3 * x[1] +
        (9.0 / 8.0) * coeffs[24] * x0 * x10 * x109 * x15 * x18 * x22 * x3 -
        729.0 / 8.0 * coeffs[25] * x116 * x134 * x20 -
        729.0 / 8.0 * coeffs[26] * x112 * x134 +
        (9.0 / 8.0) * coeffs[27] * x10 * x116 * x15 * x18 * x22 * x3 * x[0] +
        (9.0 / 8.0) * coeffs[28] * x109 * x15 * x18 * x22 * x3 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[29] * x117 * x126 -
        729.0 / 8.0 * coeffs[30] * x109 * x126 * x20 +
        (9.0 / 8.0) * coeffs[31] * x0 * x116 * x15 * x18 * x22 * x3 * x[1] +
        (81.0 / 8.0) * coeffs[33] * x0 * x10 * x109 * x18 * x22 * x3 * x[1] +
        (81.0 / 8.0) * coeffs[34] * x0 * x10 * x116 * x15 * x18 * x3 * x[1] +
        (81.0 / 8.0) * coeffs[36] * x10 * x109 * x15 * x18 * x3 * x[0] * x[1] +
        (81.0 / 8.0) * coeffs[39] * x10 * x116 * x18 * x22 * x3 * x[0] * x[1] +
        (1.0 / 8.0) * coeffs[3] * x0 * x137 * x15 * x18 * x22 * x3 * x[1] -
        81.0 / 8.0 * coeffs[40] * x112 * x129 +
        (81.0 / 8.0) * coeffs[41] * x0 * x10 * x109 * x15 * x22 * x3 * x[0] +
        (81.0 / 8.0) * coeffs[42] * x0 * x10 * x116 * x15 * x18 * x22 * x[0] -
        729.0 / 8.0 * coeffs[43] * x128 * x130 +
        (81.0 / 8.0) * coeffs[44] * x0 * x109 * x15 * x18 * x22 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[45] * x120 * x123 -
        81.0 / 8.0 * coeffs[46] * x117 * x122 +
        (81.0 / 8.0) * coeffs[47] * x0 * x116 * x15 * x22 * x3 * x[0] * x[1] +
        (81.0 / 8.0) * coeffs[49] * x0 * x10 * x137 * x15 * x3 * x[0] * x[1] +
        (1.0 / 8.0) * coeffs[4] * x0 * x10 * x132 * x15 * x18 * x22 * x3 +
        (81.0 / 8.0) * coeffs[50] * x0 * x10 * x137 * x18 * x22 * x[0] * x[1] +
        (81.0 / 8.0) * coeffs[52] * x0 * x10 * x132 * x15 * x18 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[53] * x114 * x133 * x4 -
        729.0 / 8.0 * coeffs[54] * x113 * x133 * x19 +
        (81.0 / 8.0) * coeffs[55] * x0 * x10 * x132 * x22 * x3 * x[0] * x[1] +
        (729.0 / 8.0) * coeffs[56] * x0 * x10 * x109 * x15 * x18 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[58] * x112 * x113 +
        (729.0 / 8.0) * coeffs[59] * x0 * x10 * x109 * x22 * x3 * x[0] * x[1] -
        729.0 / 8.0 * coeffs[60] * x114 * x117 +
        (729.0 / 8.0) * coeffs[61] * x0 * x10 * x116 * x15 * x3 * x[0] * x[1] +
        (729.0 / 8.0) * coeffs[62] * x0 * x10 * x116 * x18 * x22 * x[0] * x[1] +
        (1.0 / 8.0) * coeffs[6] * x132 * x15 * x18 * x22 * x3 * x[0] * x[1] +
        (9.0 / 8.0) * coeffs[8] * x0 * x10 * x137 * x15 * x18 * x22 * x[0] -
        9.0 / 8.0 * coeffs[9] * x130 * x137 - 729.0 / 8.0 * x109 * x111 * x17 -
        729.0 / 8.0 * x111 * x116 * x29 - 729.0 / 8.0 * x111 * x139 * x45 -
        729.0 / 8.0 * x112 * x119 * x49 - 729.0 / 8.0 * x114 * x138 * x41 -
        729.0 / 8.0 * x117 * x119 * x50 - 729.0 / 8.0 * x118 * x135 * x54 -
        729.0 / 8.0 * x120 * x127 * x60 - 729.0 / 8.0 * x127 * x128 * x61 -
        729.0 / 8.0 * x132 * x141 * x19 * x68 -
        729.0 / 8.0 * x132 * x142 * x71 - 729.0 / 8.0 * x137 * x142 * x72 -
        729.0 / 8.0 * x138 * x141 * x66;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = 3 * x[1];
    const Scalar x2 = x1 - 2;
    const Scalar x3 = x1 - 1;
    const Scalar x4 = x2 * x3;
    const Scalar x5 = (1.0 / 8.0) * x4;
    const Scalar x6 = x0 * x5;
    const Scalar x7 = x[0] - 1;
    const Scalar x8 = 3 * x[0];
    const Scalar x9 = x8 - 2;
    const Scalar x10 = x7 * x9;
    const Scalar x11 = x8 - 1;
    const Scalar x12 = x11 * x7;
    const Scalar x13 = x11 * x9;
    const Scalar x14 = 3 * x10 + 3 * x12 + x13;
    const Scalar x15 = x[2] - 1;
    const Scalar x16 = 3 * x[2];
    const Scalar x17 = x16 - 2;
    const Scalar x18 = x16 - 1;
    const Scalar x19 = x17 * x18;
    const Scalar x20 = x15 * x19;
    const Scalar x21 = x14 * x20;
    const Scalar x22 = (1.0 / 8.0) * x20;
    const Scalar x23 = x0 * x2;
    const Scalar x24 = x0 * x3;
    const Scalar x25 = 3 * x23 + 3 * x24 + x4;
    const Scalar x26 = x13 * x7;
    const Scalar x27 = x25 * x26;
    const Scalar x28 = x15 * x17;
    const Scalar x29 = x15 * x18;
    const Scalar x30 = x19 + 3 * x28 + 3 * x29;
    const Scalar x31 = x30 * x6;
    const Scalar x32 = x11 * x8 + x13 + x8 * x9;
    const Scalar x33 = x20 * x32;
    const Scalar x34 = x13 * x[0];
    const Scalar x35 = x22 * x34;
    const Scalar x36 = x5 * x[1];
    const Scalar x37 = x1 * x2 + x1 * x3 + x4;
    const Scalar x38 = x30 * x36;
    const Scalar x39 = x26 * x37;
    const Scalar x40 = x19 * x[2];
    const Scalar x41 = x40 * x6;
    const Scalar x42 = (1.0 / 8.0) * x40;
    const Scalar x43 = x16 * x17 + x16 * x18 + x19;
    const Scalar x44 = x43 * x6;
    const Scalar x45 = x34 * x42;
    const Scalar x46 = x36 * x40;
    const Scalar x47 = x36 * x43;
    const Scalar x48 = (9.0 / 8.0) * x20;
    const Scalar x49 = x7 * x8;
    const Scalar x50 = x9 * x[0];
    const Scalar x51 = x10 + x49 + x50;
    const Scalar x52 = x0 * x4;
    const Scalar x53 = x51 * x52;
    const Scalar x54 = x50 * x7;
    const Scalar x55 = x25 * x48;
    const Scalar x56 = (9.0 / 8.0) * x30;
    const Scalar x57 = x52 * x54;
    const Scalar x58 = x11 * x[0];
    const Scalar x59 = x12 + x49 + x58;
    const Scalar x60 = x48 * x59;
    const Scalar x61 = x58 * x7;
    const Scalar x62 = x56 * x61;
    const Scalar x63 = x2 * x[1];
    const Scalar x64 = x0 * x63;
    const Scalar x65 = (9.0 / 8.0) * x33;
    const Scalar x66 = x0 * x1;
    const Scalar x67 = x23 + x63 + x66;
    const Scalar x68 = x34 * x48;
    const Scalar x69 = x34 * x56;
    const Scalar x70 = x3 * x[1];
    const Scalar x71 = x0 * x70;
    const Scalar x72 = x24 + x66 + x70;
    const Scalar x73 = x4 * x[1];
    const Scalar x74 = x51 * x73;
    const Scalar x75 = x37 * x48;
    const Scalar x76 = x54 * x73;
    const Scalar x77 = (9.0 / 8.0) * x21;
    const Scalar x78 = x26 * x48;
    const Scalar x79 = x26 * x56;
    const Scalar x80 = (9.0 / 8.0) * x40;
    const Scalar x81 = x25 * x80;
    const Scalar x82 = (9.0 / 8.0) * x43;
    const Scalar x83 = x59 * x80;
    const Scalar x84 = x61 * x82;
    const Scalar x85 = x32 * x80;
    const Scalar x86 = x34 * x80;
    const Scalar x87 = x34 * x82;
    const Scalar x88 = x37 * x80;
    const Scalar x89 = x14 * x80;
    const Scalar x90 = x26 * x80;
    const Scalar x91 = x26 * x82;
    const Scalar x92 = (9.0 / 8.0) * x52;
    const Scalar x93 = x17 * x[2];
    const Scalar x94 = x15 * x93;
    const Scalar x95 = x14 * x94;
    const Scalar x96 = (9.0 / 8.0) * x27;
    const Scalar x97 = x15 * x16;
    const Scalar x98 = x28 + x93 + x97;
    const Scalar x99 = x26 * x92;
    const Scalar x100 = x18 * x[2];
    const Scalar x101 = x100 * x15;
    const Scalar x102 = x101 * x92;
    const Scalar x103 = x100 + x29 + x97;
    const Scalar x104 = x32 * x94;
    const Scalar x105 = (9.0 / 8.0) * x34;
    const Scalar x106 = x25 * x94;
    const Scalar x107 = x34 * x92;
    const Scalar x108 = x101 * x105;
    const Scalar x109 = (9.0 / 8.0) * x73;
    const Scalar x110 = x37 * x94;
    const Scalar x111 = x105 * x73;
    const Scalar x112 = x101 * x109;
    const Scalar x113 = (9.0 / 8.0) * x39;
    const Scalar x114 = x109 * x26;
    const Scalar x115 = (81.0 / 8.0) * x64;
    const Scalar x116 = (81.0 / 8.0) * x26;
    const Scalar x117 = x67 * x94;
    const Scalar x118 = x115 * x26;
    const Scalar x119 = (81.0 / 8.0) * x71;
    const Scalar x120 = x116 * x72;
    const Scalar x121 = x116 * x71;
    const Scalar x122 = x101 * x14;
    const Scalar x123 = x101 * x67;
    const Scalar x124 = (81.0 / 8.0) * x34;
    const Scalar x125 = x34 * x98;
    const Scalar x126 = x124 * x72;
    const Scalar x127 = x101 * x32;
    const Scalar x128 = x103 * x34;
    const Scalar x129 = (81.0 / 8.0) * x94;
    const Scalar x130 = (81.0 / 8.0) * x106;
    const Scalar x131 = (81.0 / 8.0) * x98;
    const Scalar x132 = x52 * x59;
    const Scalar x133 = x52 * x61;
    const Scalar x134 = (81.0 / 8.0) * x101;
    const Scalar x135 = x134 * x25;
    const Scalar x136 = (81.0 / 8.0) * x103;
    const Scalar x137 = (81.0 / 8.0) * x110;
    const Scalar x138 = x59 * x73;
    const Scalar x139 = x61 * x73;
    const Scalar x140 = x134 * x37;
    const Scalar x141 = x115 * x20;
    const Scalar x142 = (81.0 / 8.0) * x20;
    const Scalar x143 = x142 * x67;
    const Scalar x144 = x115 * x30;
    const Scalar x145 = x119 * x20;
    const Scalar x146 = x142 * x72;
    const Scalar x147 = x119 * x30;
    const Scalar x148 = x115 * x40;
    const Scalar x149 = (81.0 / 8.0) * x40;
    const Scalar x150 = x149 * x67;
    const Scalar x151 = x115 * x43;
    const Scalar x152 = x119 * x40;
    const Scalar x153 = x149 * x72;
    const Scalar x154 = x119 * x43;
    const Scalar x155 = (729.0 / 8.0) * x64;
    const Scalar x156 = x155 * x94;
    const Scalar x157 = (729.0 / 8.0) * x117;
    const Scalar x158 = x155 * x98;
    const Scalar x159 = (729.0 / 8.0) * x94;
    const Scalar x160 = x159 * x71;
    const Scalar x161 = x159 * x72;
    const Scalar x162 = (729.0 / 8.0) * x54;
    const Scalar x163 = x71 * x98;
    const Scalar x164 = (729.0 / 8.0) * x61;
    const Scalar x165 = x101 * x155;
    const Scalar x166 = x103 * x155;
    const Scalar x167 = (729.0 / 8.0) * x101 * x71;
    const Scalar x168 = x101 * x72;
    const Scalar x169 = x103 * x71;
    out[0][0] = -x21 * x6;
    out[0][1] = -x22 * x27;
    out[0][2] = -x26 * x31;
    out[1][0] = x33 * x6;
    out[1][1] = x25 * x35;
    out[1][2] = x31 * x34;
    out[2][0] = -x33 * x36;
    out[2][1] = -x35 * x37;
    out[2][2] = -x34 * x38;
    out[3][0] = x21 * x36;
    out[3][1] = x22 * x39;
    out[3][2] = x26 * x38;
    out[4][0] = x14 * x41;
    out[4][1] = x27 * x42;
    out[4][2] = x26 * x44;
    out[5][0] = -x32 * x41;
    out[5][1] = -x25 * x45;
    out[5][2] = -x34 * x44;
    out[6][0] = x32 * x46;
    out[6][1] = x37 * x45;
    out[6][2] = x34 * x47;
    out[7][0] = -x14 * x46;
    out[7][1] = -x39 * x42;
    out[7][2] = -x26 * x47;
    out[8][0] = x48 * x53;
    out[8][1] = x54 * x55;
    out[8][2] = x56 * x57;
    out[9][0] = -x52 * x60;
    out[9][1] = -x55 * x61;
    out[9][2] = -x52 * x62;
    out[10][0] = -x64 * x65;
    out[10][1] = -x67 * x68;
    out[10][2] = -x64 * x69;
    out[11][0] = x65 * x71;
    out[11][1] = x68 * x72;
    out[11][2] = x69 * x71;
    out[12][0] = -x48 * x74;
    out[12][1] = -x54 * x75;
    out[12][2] = -x56 * x76;
    out[13][0] = x60 * x73;
    out[13][1] = x61 * x75;
    out[13][2] = x62 * x73;
    out[14][0] = x64 * x77;
    out[14][1] = x67 * x78;
    out[14][2] = x64 * x79;
    out[15][0] = -x71 * x77;
    out[15][1] = -x72 * x78;
    out[15][2] = -x71 * x79;
    out[16][0] = -x53 * x80;
    out[16][1] = -x54 * x81;
    out[16][2] = -x57 * x82;
    out[17][0] = x52 * x83;
    out[17][1] = x61 * x81;
    out[17][2] = x52 * x84;
    out[18][0] = x64 * x85;
    out[18][1] = x67 * x86;
    out[18][2] = x64 * x87;
    out[19][0] = -x71 * x85;
    out[19][1] = -x72 * x86;
    out[19][2] = -x71 * x87;
    out[20][0] = x74 * x80;
    out[20][1] = x54 * x88;
    out[20][2] = x76 * x82;
    out[21][0] = -x73 * x83;
    out[21][1] = -x61 * x88;
    out[21][2] = -x73 * x84;
    out[22][0] = -x64 * x89;
    out[22][1] = -x67 * x90;
    out[22][2] = -x64 * x91;
    out[23][0] = x71 * x89;
    out[23][1] = x72 * x90;
    out[23][2] = x71 * x91;
    out[24][0] = x92 * x95;
    out[24][1] = x94 * x96;
    out[24][2] = x98 * x99;
    out[25][0] = -x102 * x14;
    out[25][1] = -x101 * x96;
    out[25][2] = -x103 * x99;
    out[26][0] = -x104 * x92;
    out[26][1] = -x105 * x106;
    out[26][2] = -x107 * x98;
    out[27][0] = x102 * x32;
    out[27][1] = x108 * x25;
    out[27][2] = x103 * x107;
    out[28][0] = x104 * x109;
    out[28][1] = x105 * x110;
    out[28][2] = x111 * x98;
    out[29][0] = -x112 * x32;
    out[29][1] = -x108 * x37;
    out[29][2] = -x103 * x111;
    out[30][0] = -x109 * x95;
    out[30][1] = -x113 * x94;
    out[30][2] = -x114 * x98;
    out[31][0] = x112 * x14;
    out[31][1] = x101 * x113;
    out[31][2] = x103 * x114;
    out[32][0] = -x115 * x95;
    out[32][1] = -x116 * x117;
    out[32][2] = -x118 * x98;
    out[33][0] = x119 * x95;
    out[33][1] = x120 * x94;
    out[33][2] = x121 * x98;
    out[34][0] = x115 * x122;
    out[34][1] = x116 * x123;
    out[34][2] = x103 * x118;
    out[35][0] = -x119 * x122;
    out[35][1] = -x101 * x120;
    out[35][2] = -x103 * x121;
    out[36][0] = x104 * x115;
    out[36][1] = x117 * x124;
    out[36][2] = x115 * x125;
    out[37][0] = -x104 * x119;
    out[37][1] = -x126 * x94;
    out[37][2] = -x119 * x125;
    out[38][0] = -x115 * x127;
    out[38][1] = -x123 * x124;
    out[38][2] = -x115 * x128;
    out[39][0] = x119 * x127;
    out[39][1] = x101 * x126;
    out[39][2] = x119 * x128;
    out[40][0] = -x129 * x53;
    out[40][1] = -x130 * x54;
    out[40][2] = -x131 * x57;
    out[41][0] = x129 * x132;
    out[41][1] = x130 * x61;
    out[41][2] = x131 * x133;
    out[42][0] = x134 * x53;
    out[42][1] = x135 * x54;
    out[42][2] = x136 * x57;
    out[43][0] = -x132 * x134;
    out[43][1] = -x135 * x61;
    out[43][2] = -x133 * x136;
    out[44][0] = x129 * x74;
    out[44][1] = x137 * x54;
    out[44][2] = x131 * x76;
    out[45][0] = -x129 * x138;
    out[45][1] = -x137 * x61;
    out[45][2] = -x131 * x139;
    out[46][0] = -x134 * x74;
    out[46][1] = -x140 * x54;
    out[46][2] = -x136 * x76;
    out[47][0] = x134 * x138;
    out[47][1] = x140 * x61;
    out[47][2] = x136 * x139;
    out[48][0] = -x141 * x51;
    out[48][1] = -x143 * x54;
    out[48][2] = -x144 * x54;
    out[49][0] = x141 * x59;
    out[49][1] = x143 * x61;
    out[49][2] = x144 * x61;
    out[50][0] = x145 * x51;
    out[50][1] = x146 * x54;
    out[50][2] = x147 * x54;
    out[51][0] = -x145 * x59;
    out[51][1] = -x146 * x61;
    out[51][2] = -x147 * x61;
    out[52][0] = x148 * x51;
    out[52][1] = x150 * x54;
    out[52][2] = x151 * x54;
    out[53][0] = -x148 * x59;
    out[53][1] = -x150 * x61;
    out[53][2] = -x151 * x61;
    out[54][0] = -x152 * x51;
    out[54][1] = -x153 * x54;
    out[54][2] = -x154 * x54;
    out[55][0] = x152 * x59;
    out[55][1] = x153 * x61;
    out[55][2] = x154 * x61;
    out[56][0] = x156 * x51;
    out[56][1] = x157 * x54;
    out[56][2] = x158 * x54;
    out[57][0] = -x156 * x59;
    out[57][1] = -x157 * x61;
    out[57][2] = -x158 * x61;
    out[58][0] = -x160 * x51;
    out[58][1] = -x161 * x54;
    out[58][2] = -x162 * x163;
    out[59][0] = x160 * x59;
    out[59][1] = x161 * x61;
    out[59][2] = x163 * x164;
    out[60][0] = -x165 * x51;
    out[60][1] = -x123 * x162;
    out[60][2] = -x166 * x54;
    out[61][0] = x165 * x59;
    out[61][1] = x123 * x164;
    out[61][2] = x166 * x61;
    out[62][0] = x167 * x51;
    out[62][1] = x162 * x168;
    out[62][2] = x162 * x169;
    out[63][0] = -x167 * x59;
    out[63][1] = -x164 * x168;
    out[63][2] = -x164 * x169;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[3]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[0][2] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(3) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[1][2] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(3) / order;
    out[2][1] = static_cast<Scalar>(3) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(3) / order;
    out[3][2] = static_cast<Scalar>(0) / order;
    out[4][0] = static_cast<Scalar>(0) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[4][2] = static_cast<Scalar>(3) / order;
    out[5][0] = static_cast<Scalar>(3) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[5][2] = static_cast<Scalar>(3) / order;
    out[6][0] = static_cast<Scalar>(3) / order;
    out[6][1] = static_cast<Scalar>(3) / order;
    out[6][2] = static_cast<Scalar>(3) / order;
    out[7][0] = static_cast<Scalar>(0) / order;
    out[7][1] = static_cast<Scalar>(3) / order;
    out[7][2] = static_cast<Scalar>(3) / order;
    out[8][0] = static_cast<Scalar>(1) / order;
    out[8][1] = static_cast<Scalar>(0) / order;
    out[8][2] = static_cast<Scalar>(0) / order;
    out[9][0] = static_cast<Scalar>(2) / order;
    out[9][1] = static_cast<Scalar>(0) / order;
    out[9][2] = static_cast<Scalar>(0) / order;
    out[10][0] = static_cast<Scalar>(3) / order;
    out[10][1] = static_cast<Scalar>(1) / order;
    out[10][2] = static_cast<Scalar>(0) / order;
    out[11][0] = static_cast<Scalar>(3) / order;
    out[11][1] = static_cast<Scalar>(2) / order;
    out[11][2] = static_cast<Scalar>(0) / order;
    out[12][0] = static_cast<Scalar>(1) / order;
    out[12][1] = static_cast<Scalar>(3) / order;
    out[12][2] = static_cast<Scalar>(0) / order;
    out[13][0] = static_cast<Scalar>(2) / order;
    out[13][1] = static_cast<Scalar>(3) / order;
    out[13][2] = static_cast<Scalar>(0) / order;
    out[14][0] = static_cast<Scalar>(0) / order;
    out[14][1] = static_cast<Scalar>(1) / order;
    out[14][2] = static_cast<Scalar>(0) / order;
    out[15][0] = static_cast<Scalar>(0) / order;
    out[15][1] = static_cast<Scalar>(2) / order;
    out[15][2] = static_cast<Scalar>(0) / order;
    out[16][0] = static_cast<Scalar>(1) / order;
    out[16][1] = static_cast<Scalar>(0) / order;
    out[16][2] = static_cast<Scalar>(3) / order;
    out[17][0] = static_cast<Scalar>(2) / order;
    out[17][1] = static_cast<Scalar>(0) / order;
    out[17][2] = static_cast<Scalar>(3) / order;
    out[18][0] = static_cast<Scalar>(3) / order;
    out[18][1] = static_cast<Scalar>(1) / order;
    out[18][2] = static_cast<Scalar>(3) / order;
    out[19][0] = static_cast<Scalar>(3) / order;
    out[19][1] = static_cast<Scalar>(2) / order;
    out[19][2] = static_cast<Scalar>(3) / order;
    out[20][0] = static_cast<Scalar>(1) / order;
    out[20][1] = static_cast<Scalar>(3) / order;
    out[20][2] = static_cast<Scalar>(3) / order;
    out[21][0] = static_cast<Scalar>(2) / order;
    out[21][1] = static_cast<Scalar>(3) / order;
    out[21][2] = static_cast<Scalar>(3) / order;
    out[22][0] = static_cast<Scalar>(0) / order;
    out[22][1] = static_cast<Scalar>(1) / order;
    out[22][2] = static_cast<Scalar>(3) / order;
    out[23][0] = static_cast<Scalar>(0) / order;
    out[23][1] = static_cast<Scalar>(2) / order;
    out[23][2] = static_cast<Scalar>(3) / order;
    out[24][0] = static_cast<Scalar>(0) / order;
    out[24][1] = static_cast<Scalar>(0) / order;
    out[24][2] = static_cast<Scalar>(1) / order;
    out[25][0] = static_cast<Scalar>(0) / order;
    out[25][1] = static_cast<Scalar>(0) / order;
    out[25][2] = static_cast<Scalar>(2) / order;
    out[26][0] = static_cast<Scalar>(3) / order;
    out[26][1] = static_cast<Scalar>(0) / order;
    out[26][2] = static_cast<Scalar>(1) / order;
    out[27][0] = static_cast<Scalar>(3) / order;
    out[27][1] = static_cast<Scalar>(0) / order;
    out[27][2] = static_cast<Scalar>(2) / order;
    out[28][0] = static_cast<Scalar>(3) / order;
    out[28][1] = static_cast<Scalar>(3) / order;
    out[28][2] = static_cast<Scalar>(1) / order;
    out[29][0] = static_cast<Scalar>(3) / order;
    out[29][1] = static_cast<Scalar>(3) / order;
    out[29][2] = static_cast<Scalar>(2) / order;
    out[30][0] = static_cast<Scalar>(0) / order;
    out[30][1] = static_cast<Scalar>(3) / order;
    out[30][2] = static_cast<Scalar>(1) / order;
    out[31][0] = static_cast<Scalar>(0) / order;
    out[31][1] = static_cast<Scalar>(3) / order;
    out[31][2] = static_cast<Scalar>(2) / order;
    out[32][0] = static_cast<Scalar>(0) / order;
    out[32][1] = static_cast<Scalar>(1) / order;
    out[32][2] = static_cast<Scalar>(1) / order;
    out[33][0] = static_cast<Scalar>(0) / order;
    out[33][1] = static_cast<Scalar>(2) / order;
    out[33][2] = static_cast<Scalar>(1) / order;
    out[34][0] = static_cast<Scalar>(0) / order;
    out[34][1] = static_cast<Scalar>(1) / order;
    out[34][2] = static_cast<Scalar>(2) / order;
    out[35][0] = static_cast<Scalar>(0) / order;
    out[35][1] = static_cast<Scalar>(2) / order;
    out[35][2] = static_cast<Scalar>(2) / order;
    out[36][0] = static_cast<Scalar>(3) / order;
    out[36][1] = static_cast<Scalar>(1) / order;
    out[36][2] = static_cast<Scalar>(1) / order;
    out[37][0] = static_cast<Scalar>(3) / order;
    out[37][1] = static_cast<Scalar>(2) / order;
    out[37][2] = static_cast<Scalar>(1) / order;
    out[38][0] = static_cast<Scalar>(3) / order;
    out[38][1] = static_cast<Scalar>(1) / order;
    out[38][2] = static_cast<Scalar>(2) / order;
    out[39][0] = static_cast<Scalar>(3) / order;
    out[39][1] = static_cast<Scalar>(2) / order;
    out[39][2] = static_cast<Scalar>(2) / order;
    out[40][0] = static_cast<Scalar>(1) / order;
    out[40][1] = static_cast<Scalar>(0) / order;
    out[40][2] = static_cast<Scalar>(1) / order;
    out[41][0] = static_cast<Scalar>(2) / order;
    out[41][1] = static_cast<Scalar>(0) / order;
    out[41][2] = static_cast<Scalar>(1) / order;
    out[42][0] = static_cast<Scalar>(1) / order;
    out[42][1] = static_cast<Scalar>(0) / order;
    out[42][2] = static_cast<Scalar>(2) / order;
    out[43][0] = static_cast<Scalar>(2) / order;
    out[43][1] = static_cast<Scalar>(0) / order;
    out[43][2] = static_cast<Scalar>(2) / order;
    out[44][0] = static_cast<Scalar>(1) / order;
    out[44][1] = static_cast<Scalar>(3) / order;
    out[44][2] = static_cast<Scalar>(1) / order;
    out[45][0] = static_cast<Scalar>(2) / order;
    out[45][1] = static_cast<Scalar>(3) / order;
    out[45][2] = static_cast<Scalar>(1) / order;
    out[46][0] = static_cast<Scalar>(1) / order;
    out[46][1] = static_cast<Scalar>(3) / order;
    out[46][2] = static_cast<Scalar>(2) / order;
    out[47][0] = static_cast<Scalar>(2) / order;
    out[47][1] = static_cast<Scalar>(3) / order;
    out[47][2] = static_cast<Scalar>(2) / order;
    out[48][0] = static_cast<Scalar>(1) / order;
    out[48][1] = static_cast<Scalar>(1) / order;
    out[48][2] = static_cast<Scalar>(0) / order;
    out[49][0] = static_cast<Scalar>(2) / order;
    out[49][1] = static_cast<Scalar>(1) / order;
    out[49][2] = static_cast<Scalar>(0) / order;
    out[50][0] = static_cast<Scalar>(1) / order;
    out[50][1] = static_cast<Scalar>(2) / order;
    out[50][2] = static_cast<Scalar>(0) / order;
    out[51][0] = static_cast<Scalar>(2) / order;
    out[51][1] = static_cast<Scalar>(2) / order;
    out[51][2] = static_cast<Scalar>(0) / order;
    out[52][0] = static_cast<Scalar>(1) / order;
    out[52][1] = static_cast<Scalar>(1) / order;
    out[52][2] = static_cast<Scalar>(3) / order;
    out[53][0] = static_cast<Scalar>(2) / order;
    out[53][1] = static_cast<Scalar>(1) / order;
    out[53][2] = static_cast<Scalar>(3) / order;
    out[54][0] = static_cast<Scalar>(1) / order;
    out[54][1] = static_cast<Scalar>(2) / order;
    out[54][2] = static_cast<Scalar>(3) / order;
    out[55][0] = static_cast<Scalar>(2) / order;
    out[55][1] = static_cast<Scalar>(2) / order;
    out[55][2] = static_cast<Scalar>(3) / order;
    out[56][0] = static_cast<Scalar>(1) / order;
    out[56][1] = static_cast<Scalar>(1) / order;
    out[56][2] = static_cast<Scalar>(1) / order;
    out[57][0] = static_cast<Scalar>(2) / order;
    out[57][1] = static_cast<Scalar>(1) / order;
    out[57][2] = static_cast<Scalar>(1) / order;
    out[58][0] = static_cast<Scalar>(1) / order;
    out[58][1] = static_cast<Scalar>(2) / order;
    out[58][2] = static_cast<Scalar>(1) / order;
    out[59][0] = static_cast<Scalar>(2) / order;
    out[59][1] = static_cast<Scalar>(2) / order;
    out[59][2] = static_cast<Scalar>(1) / order;
    out[60][0] = static_cast<Scalar>(1) / order;
    out[60][1] = static_cast<Scalar>(1) / order;
    out[60][2] = static_cast<Scalar>(2) / order;
    out[61][0] = static_cast<Scalar>(2) / order;
    out[61][1] = static_cast<Scalar>(1) / order;
    out[61][2] = static_cast<Scalar>(2) / order;
    out[62][0] = static_cast<Scalar>(1) / order;
    out[62][1] = static_cast<Scalar>(2) / order;
    out[62][2] = static_cast<Scalar>(2) / order;
    out[63][0] = static_cast<Scalar>(2) / order;
    out[63][1] = static_cast<Scalar>(2) / order;
    out[63][2] = static_cast<Scalar>(2) / order;
  }
  template <typename Scalar>
  static constexpr void interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
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
      out[0] = 1;
      out[1] = 3;
      out[2] = 0;
      break;
    case 13:
      out[0] = 2;
      out[1] = 3;
      out[2] = 0;
      break;
    case 14:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 15:
      out[0] = 0;
      out[1] = 2;
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
      out[0] = 1;
      out[1] = 3;
      out[2] = 3;
      break;
    case 21:
      out[0] = 2;
      out[1] = 3;
      out[2] = 3;
      break;
    case 22:
      out[0] = 0;
      out[1] = 1;
      out[2] = 3;
      break;
    case 23:
      out[0] = 0;
      out[1] = 2;
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
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 33:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 34:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 35:
      out[0] = 0;
      out[1] = 2;
      out[2] = 2;
      break;
    case 36:
      out[0] = 3;
      out[1] = 1;
      out[2] = 1;
      break;
    case 37:
      out[0] = 3;
      out[1] = 2;
      out[2] = 1;
      break;
    case 38:
      out[0] = 3;
      out[1] = 1;
      out[2] = 2;
      break;
    case 39:
      out[0] = 3;
      out[1] = 2;
      out[2] = 2;
      break;
    case 40:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 41:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 42:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 43:
      out[0] = 2;
      out[1] = 0;
      out[2] = 2;
      break;
    case 44:
      out[0] = 1;
      out[1] = 3;
      out[2] = 1;
      break;
    case 45:
      out[0] = 2;
      out[1] = 3;
      out[2] = 1;
      break;
    case 46:
      out[0] = 1;
      out[1] = 3;
      out[2] = 2;
      break;
    case 47:
      out[0] = 2;
      out[1] = 3;
      out[2] = 2;
      break;
    case 48:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 49:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 50:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 51:
      out[0] = 2;
      out[1] = 2;
      out[2] = 0;
      break;
    case 52:
      out[0] = 1;
      out[1] = 1;
      out[2] = 3;
      break;
    case 53:
      out[0] = 2;
      out[1] = 1;
      out[2] = 3;
      break;
    case 54:
      out[0] = 1;
      out[1] = 2;
      out[2] = 3;
      break;
    case 55:
      out[0] = 2;
      out[1] = 2;
      out[2] = 3;
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
      out[0] = 1;
      out[1] = 2;
      out[2] = 1;
      break;
    case 59:
      out[0] = 2;
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
      out[0] = 1;
      out[1] = 2;
      out[2] = 2;
      break;
    case 63:
      out[0] = 2;
      out[1] = 2;
      out[2] = 2;
      break;
    }
  }

  template <typename Element>
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs) {
    subelement_node_idxs(subelement, idxs, meta::type_tag<Element>{});
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElPoint>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      break;
    case 1:
      idxs[0] = 1;
      break;
    case 2:
      idxs[0] = 2;
      break;
    case 3:
      idxs[0] = 3;
      break;
    case 4:
      idxs[0] = 4;
      break;
    case 5:
      idxs[0] = 5;
      break;
    case 6:
      idxs[0] = 6;
      break;
    case 7:
      idxs[0] = 7;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 8;
      idxs[3] = 9;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 10;
      idxs[3] = 11;
      break;
    case 2:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 12;
      idxs[3] = 13;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 14;
      idxs[3] = 15;
      break;
    case 4:
      idxs[0] = 4;
      idxs[1] = 5;
      idxs[2] = 16;
      idxs[3] = 17;
      break;
    case 5:
      idxs[0] = 5;
      idxs[1] = 6;
      idxs[2] = 18;
      idxs[3] = 19;
      break;
    case 6:
      idxs[0] = 7;
      idxs[1] = 6;
      idxs[2] = 20;
      idxs[3] = 21;
      break;
    case 7:
      idxs[0] = 4;
      idxs[1] = 7;
      idxs[2] = 22;
      idxs[3] = 23;
      break;
    case 8:
      idxs[0] = 0;
      idxs[1] = 4;
      idxs[2] = 24;
      idxs[3] = 25;
      break;
    case 9:
      idxs[0] = 1;
      idxs[1] = 5;
      idxs[2] = 26;
      idxs[3] = 27;
      break;
    case 10:
      idxs[0] = 2;
      idxs[1] = 6;
      idxs[2] = 28;
      idxs[3] = 29;
      break;
    case 11:
      idxs[0] = 3;
      idxs[1] = 7;
      idxs[2] = 30;
      idxs[3] = 31;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 7;
      idxs[3] = 4;
      idxs[4] = 14;
      idxs[5] = 15;
      idxs[6] = 30;
      idxs[7] = 31;
      idxs[8] = 22;
      idxs[9] = 23;
      idxs[10] = 24;
      idxs[11] = 25;
      idxs[12] = 32;
      idxs[13] = 33;
      idxs[14] = 34;
      idxs[15] = 35;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 5;
      idxs[4] = 10;
      idxs[5] = 11;
      idxs[6] = 28;
      idxs[7] = 29;
      idxs[8] = 18;
      idxs[9] = 19;
      idxs[10] = 26;
      idxs[11] = 27;
      idxs[12] = 36;
      idxs[13] = 37;
      idxs[14] = 38;
      idxs[15] = 39;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 5;
      idxs[3] = 4;
      idxs[4] = 8;
      idxs[5] = 9;
      idxs[6] = 26;
      idxs[7] = 27;
      idxs[8] = 16;
      idxs[9] = 17;
      idxs[10] = 24;
      idxs[11] = 25;
      idxs[12] = 40;
      idxs[13] = 41;
      idxs[14] = 42;
      idxs[15] = 43;
      break;
    case 3:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 7;
      idxs[4] = 12;
      idxs[5] = 13;
      idxs[6] = 28;
      idxs[7] = 29;
      idxs[8] = 20;
      idxs[9] = 21;
      idxs[10] = 30;
      idxs[11] = 31;
      idxs[12] = 44;
      idxs[13] = 45;
      idxs[14] = 46;
      idxs[15] = 47;
      break;
    case 4:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 2;
      idxs[3] = 3;
      idxs[4] = 8;
      idxs[5] = 9;
      idxs[6] = 10;
      idxs[7] = 11;
      idxs[8] = 12;
      idxs[9] = 13;
      idxs[10] = 14;
      idxs[11] = 15;
      idxs[12] = 48;
      idxs[13] = 49;
      idxs[14] = 50;
      idxs[15] = 51;
      break;
    case 5:
      idxs[0] = 4;
      idxs[1] = 5;
      idxs[2] = 6;
      idxs[3] = 7;
      idxs[4] = 16;
      idxs[5] = 17;
      idxs[6] = 18;
      idxs[7] = 19;
      idxs[8] = 20;
      idxs[9] = 21;
      idxs[10] = 22;
      idxs[11] = 23;
      idxs[12] = 52;
      idxs[13] = 53;
      idxs[14] = 54;
      idxs[15] = 55;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTetra>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElCube>) {
    switch (subelement) {
    default:
      break;
    }
  }
};

template <> struct BasisLagrange<mesh::RefElCube, 4> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 4;
  static constexpr dim_t num_basis_functions = 125;
  static constexpr dim_t num_interpolation_nodes = 125;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 2 * x[0] - 1;
    const Scalar x1 = 2 * x[1] - 1;
    const Scalar x2 = 4 * x[0];
    const Scalar x3 = x2 - 1;
    const Scalar x4 = 4 * x[2];
    const Scalar x5 = x4 - 3;
    const Scalar x6 = x0 * x1 * x3 * x5 * x[2];
    const Scalar x7 = 2 * x[2] - 1;
    const Scalar x8 = 4 * x[1];
    const Scalar x9 = x8 - 3;
    const Scalar x10 = x7 * x9;
    const Scalar x11 = x[0] - 1;
    const Scalar x12 = x[1] - 1;
    const Scalar x13 = x[2] - 1;
    const Scalar x14 = x11 * x12 * x13;
    const Scalar x15 = x[0] * x[1];
    const Scalar x16 = x14 * x15;
    const Scalar x17 = x10 * x16;
    const Scalar x18 = 4096 * x17;
    const Scalar x19 = x8 - 1;
    const Scalar x20 = x2 - 3;
    const Scalar x21 = x19 * x20;
    const Scalar x22 = x21 * x[2];
    const Scalar x23 = x0 * x5;
    const Scalar x24 = x22 * x23;
    const Scalar x25 = 3072 * x17;
    const Scalar x26 = x3 * x5;
    const Scalar x27 = x17 * x26;
    const Scalar x28 = 2304 * x22;
    const Scalar x29 = x19 * x[2];
    const Scalar x30 = 3072 * x29;
    const Scalar x31 = 4096 * x16;
    const Scalar x32 = x21 * x7;
    const Scalar x33 = x0 * x1 * x[2];
    const Scalar x34 = x33 * x5;
    const Scalar x35 = x32 * x34;
    const Scalar x36 = x1 * x26;
    const Scalar x37 = x16 * x36;
    const Scalar x38 = 3072 * x[2];
    const Scalar x39 = x19 * x6;
    const Scalar x40 = x31 * x7;
    const Scalar x41 = x4 - 1;
    const Scalar x42 = x41 * x9;
    const Scalar x43 = x16 * x42;
    const Scalar x44 = 3072 * x43;
    const Scalar x45 = x20 * x[2];
    const Scalar x46 = x36 * x45;
    const Scalar x47 = 2304 * x43;
    const Scalar x48 = x22 * x26;
    const Scalar x49 = x0 * x26;
    const Scalar x50 = x29 * x49;
    const Scalar x51 = x21 * x41;
    const Scalar x52 = x34 * x51;
    const Scalar x53 = 3072 * x16;
    const Scalar x54 = x28 * x41;
    const Scalar x55 = x33 * x41;
    const Scalar x56 = x18 * x20;
    const Scalar x57 = x17 * x3;
    const Scalar x58 = x1 * x41;
    const Scalar x59 = 3072 * x45;
    const Scalar x60 = x3 * x55;
    const Scalar x61 = x0 * x41;
    const Scalar x62 = x16 * x32 * x58;
    const Scalar x63 = x19 * x60;
    const Scalar x64 = x1 * x27;
    const Scalar x65 = x12 * x[0];
    const Scalar x66 = x13 * x65;
    const Scalar x67 = x10 * x66;
    const Scalar x68 = x67 * x[1];
    const Scalar x69 = 256 * x20;
    const Scalar x70 = x22 * x49;
    const Scalar x71 = x10 * x70;
    const Scalar x72 = 192 * x[1];
    const Scalar x73 = x66 * x72;
    const Scalar x74 = x32 * x66;
    const Scalar x75 = 256 * x[1];
    const Scalar x76 = x74 * x75;
    const Scalar x77 = x6 * x72;
    const Scalar x78 = x20 * x77;
    const Scalar x79 = x42 * x66;
    const Scalar x80 = x70 * x[1];
    const Scalar x81 = x51 * x77;
    const Scalar x82 = x60 * x69;
    const Scalar x83 = x22 * x3;
    const Scalar x84 = x61 * x83;
    const Scalar x85 = x21 * x34;
    const Scalar x86 = x10 * x11;
    const Scalar x87 = x13 * x15;
    const Scalar x88 = x86 * x87;
    const Scalar x89 = 256 * x88;
    const Scalar x90 = 192 * x87;
    const Scalar x91 = x22 * x36;
    const Scalar x92 = x86 * x91;
    const Scalar x93 = x11 * x42;
    const Scalar x94 = x90 * x93;
    const Scalar x95 = x21 * x55;
    const Scalar x96 = x58 * x83;
    const Scalar x97 = x41 * x86;
    const Scalar x98 = x97 * x[1];
    const Scalar x99 = x65 * x98;
    const Scalar x100 = x65 * x72;
    const Scalar x101 = x100 * x97;
    const Scalar x102 = x6 * x65;
    const Scalar x103 = x11 * x41 * x65 * x75;
    const Scalar x104 = x32 * x41;
    const Scalar x105 = x104 * x11;
    const Scalar x106 = 16 * x[1];
    const Scalar x107 = x106 * x20;
    const Scalar x108 = 12 * x41;
    const Scalar x109 = x108 * x65;
    const Scalar x110 = x71 * x[1];
    const Scalar x111 = x52 * x86;
    const Scalar x112 = 16 * x15;
    const Scalar x113 = x39 * x97;
    const Scalar x114 = 16 * x21;
    const Scalar x115 = x114 * x6;
    const Scalar x116 = x10 * x87;
    const Scalar x117 = x21 * x6;
    const Scalar x118 = 12 * x117;
    const Scalar x119 = x114 * x60;
    const Scalar x120 = x10 * x14;
    const Scalar x121 = x120 * x6;
    const Scalar x122 = x120 * x72;
    const Scalar x123 = x14 * x32;
    const Scalar x124 = x123 * x75;
    const Scalar x125 = x14 * x42;
    const Scalar x126 = 144 * x125;
    const Scalar x127 = x120 * x[1];
    const Scalar x128 = x120 * x[0];
    const Scalar x129 = 256 * x128;
    const Scalar x130 = 192 * x[0];
    const Scalar x131 = x120 * x36;
    const Scalar x132 = x125 * x130;
    const Scalar x133 = x20 * x58;
    const Scalar x134 = 256 * x23;
    const Scalar x135 = 192 * x27;
    const Scalar x136 = x23 * x51;
    const Scalar x137 = x19 * x61;
    const Scalar x138 = x0 * x1 * x19 * x20 * x3 * x41 * x5 * x7 * x9;
    const Scalar x139 = x138 * x[2];
    const Scalar x140 = x107 * x61;
    const Scalar x141 = 12 * x51;
    const Scalar x142 = x141 * x49;
    const Scalar x143 = x106 * x36 * x61;
    const Scalar x144 = 16 * x1 * x136;
    const Scalar x145 = x36 * x88;
    const Scalar x146 = 16 * x137;
    const Scalar x147 = 16 * x65;
    const Scalar x148 = x12 * x6;
    const Scalar x149 = x117 * x13;
    const Scalar x150 = x106 * x86;
    const Scalar x151 = x13 * x138;
    const Scalar x152 = x151 * x[1];
    const Scalar x153 = x11 * x139;
    const Scalar x154 = x131 * x[0];
    return (1.0 / 27.0) * coeffs[0] * x138 * x14 -
           1.0 / 27.0 * coeffs[100] * x18 * x6 +
           (1.0 / 27.0) * coeffs[101] * x24 * x25 -
           1.0 / 27.0 * coeffs[102] * x27 * x28 +
           (1.0 / 27.0) * coeffs[103] * x0 * x27 * x30 -
           1.0 / 27.0 * coeffs[104] * x31 * x35 +
           (1.0 / 27.0) * coeffs[105] * x32 * x37 * x38 -
           1.0 / 27.0 * coeffs[106] * x39 * x40 +
           (1.0 / 27.0) * coeffs[107] * x20 * x34 * x44 -
           1.0 / 27.0 * coeffs[108] * x46 * x47 +
           (1.0 / 27.0) * coeffs[109] * x44 * x6 -
           1.0 / 27.0 * coeffs[10] * x146 * x154 -
           1.0 / 27.0 * coeffs[110] * x23 * x28 * x43 +
           64 * coeffs[111] * x43 * x48 - 1.0 / 27.0 * coeffs[112] * x47 * x50 +
           (1.0 / 27.0) * coeffs[113] * x52 * x53 -
           1.0 / 27.0 * coeffs[114] * x37 * x54 +
           (1.0 / 27.0) * coeffs[115] * x39 * x41 * x53 -
           1.0 / 27.0 * coeffs[116] * x55 * x56 +
           (1.0 / 27.0) * coeffs[117] * x57 * x58 * x59 -
           1.0 / 27.0 * coeffs[118] * x18 * x60 +
           (1.0 / 27.0) * coeffs[119] * x22 * x25 * x61 -
           1.0 / 27.0 * coeffs[11] * x140 * x36 * x67 -
           1.0 / 27.0 * coeffs[120] * x54 * x57 +
           (1.0 / 27.0) * coeffs[121] * x30 * x57 * x61 -
           1.0 / 27.0 * coeffs[122] * x31 * x32 * x55 +
           (1.0 / 27.0) * coeffs[123] * x3 * x38 * x62 -
           1.0 / 27.0 * coeffs[124] * x40 * x63 +
           (1.0 / 27.0) * coeffs[12] * x142 * x68 -
           1.0 / 27.0 * coeffs[13] * x143 * x74 -
           1.0 / 27.0 * coeffs[14] * x144 * x88 +
           (1.0 / 27.0) * coeffs[15] * x141 * x145 -
           1.0 / 27.0 * coeffs[16] * x145 * x146 -
           1.0 / 27.0 * coeffs[17] * x131 * x140 +
           (1.0 / 27.0) * coeffs[18] * x127 * x142 -
           1.0 / 27.0 * coeffs[19] * x123 * x143 +
           (1.0 / 27.0) * coeffs[1] * x151 * x65 -
           1.0 / 27.0 * coeffs[20] * x111 * x147 +
           (1.0 / 27.0) * coeffs[21] * x109 * x92 -
           1.0 / 27.0 * coeffs[22] * x113 * x147 -
           1.0 / 27.0 * coeffs[23] * x10 * x102 * x107 * x41 +
           (1.0 / 27.0) * coeffs[24] * x109 * x110 -
           1.0 / 27.0 * coeffs[25] * x102 * x104 * x106 -
           1.0 / 27.0 * coeffs[26] * x111 * x112 +
           (1.0 / 27.0) * coeffs[27] * x108 * x15 * x92 -
           1.0 / 27.0 * coeffs[28] * x112 * x113 -
           1.0 / 27.0 * coeffs[29] * x107 * x148 * x97 +
           (1.0 / 27.0) * coeffs[2] * x152 * x[0] +
           (1.0 / 27.0) * coeffs[30] * x108 * x11 * x110 * x12 -
           1.0 / 27.0 * coeffs[31] * x105 * x106 * x148 -
           1.0 / 27.0 * coeffs[32] * x114 * x121 +
           (1.0 / 27.0) * coeffs[33] * x118 * x125 -
           1.0 / 27.0 * coeffs[34] * x119 * x120 -
           1.0 / 27.0 * coeffs[35] * x115 * x67 +
           (1.0 / 27.0) * coeffs[36] * x118 * x79 -
           1.0 / 27.0 * coeffs[37] * x119 * x67 -
           1.0 / 27.0 * coeffs[38] * x115 * x116 +
           (1.0 / 27.0) * coeffs[39] * x118 * x42 * x87 +
           (1.0 / 27.0) * coeffs[3] * x11 * x152 -
           1.0 / 27.0 * coeffs[40] * x116 * x119 -
           1.0 / 27.0 * coeffs[41] * x149 * x150 +
           (4.0 / 9.0) * coeffs[42] * x149 * x93 * x[1] -
           1.0 / 27.0 * coeffs[43] * x13 * x150 * x21 * x60 +
           (1.0 / 27.0) * coeffs[44] * x121 * x20 * x75 -
           1.0 / 27.0 * coeffs[45] * x122 * x70 +
           (1.0 / 27.0) * coeffs[46] * x124 * x6 -
           1.0 / 27.0 * coeffs[47] * x125 * x78 +
           (1.0 / 27.0) * coeffs[48] * x126 * x80 -
           1.0 / 27.0 * coeffs[49] * x14 * x81 +
           (1.0 / 27.0) * coeffs[4] * x12 * x153 +
           (1.0 / 27.0) * coeffs[50] * x127 * x82 -
           1.0 / 27.0 * coeffs[51] * x122 * x84 +
           (1.0 / 27.0) * coeffs[52] * x124 * x60 +
           (1.0 / 27.0) * coeffs[53] * x6 * x68 * x69 -
           1.0 / 27.0 * coeffs[54] * x71 * x73 +
           (1.0 / 27.0) * coeffs[55] * x6 * x76 -
           1.0 / 27.0 * coeffs[56] * x78 * x79 +
           (16.0 / 3.0) * coeffs[57] * x79 * x80 -
           1.0 / 27.0 * coeffs[58] * x66 * x81 +
           (1.0 / 27.0) * coeffs[59] * x68 * x82 +
           (1.0 / 27.0) * coeffs[5] * x139 * x65 -
           1.0 / 27.0 * coeffs[60] * x10 * x73 * x84 +
           (1.0 / 27.0) * coeffs[61] * x60 * x76 +
           (1.0 / 27.0) * coeffs[62] * x129 * x85 -
           1.0 / 27.0 * coeffs[63] * x130 * x131 * x22 +
           (1.0 / 27.0) * coeffs[64] * x129 * x39 -
           1.0 / 27.0 * coeffs[65] * x132 * x85 +
           (1.0 / 27.0) * coeffs[66] * x126 * x91 * x[0] -
           1.0 / 27.0 * coeffs[67] * x132 * x39 +
           (1.0 / 27.0) * coeffs[68] * x129 * x95 -
           1.0 / 27.0 * coeffs[69] * x120 * x130 * x96 +
           (1.0 / 27.0) * coeffs[6] * x139 * x15 +
           (1.0 / 27.0) * coeffs[70] * x129 * x63 +
           (1.0 / 27.0) * coeffs[71] * x85 * x89 -
           1.0 / 27.0 * coeffs[72] * x90 * x92 +
           (1.0 / 27.0) * coeffs[73] * x39 * x89 -
           1.0 / 27.0 * coeffs[74] * x85 * x94 +
           (16.0 / 3.0) * coeffs[75] * x87 * x91 * x93 -
           1.0 / 27.0 * coeffs[76] * x39 * x94 +
           (1.0 / 27.0) * coeffs[77] * x89 * x95 -
           1.0 / 27.0 * coeffs[78] * x86 * x90 * x96 +
           (1.0 / 27.0) * coeffs[79] * x63 * x89 +
           (1.0 / 27.0) * coeffs[7] * x153 * x[1] +
           (1.0 / 27.0) * coeffs[80] * x133 * x134 * x17 -
           1.0 / 27.0 * coeffs[81] * x133 * x135 +
           (256.0 / 27.0) * coeffs[82] * x61 * x64 -
           64.0 / 9.0 * coeffs[83] * x136 * x17 +
           (16.0 / 3.0) * coeffs[84] * x27 * x51 -
           1.0 / 27.0 * coeffs[85] * x135 * x137 +
           (1.0 / 27.0) * coeffs[86] * x134 * x62 -
           64.0 / 9.0 * coeffs[87] * x104 * x37 +
           (256.0 / 27.0) * coeffs[88] * x137 * x37 * x7 +
           (1.0 / 27.0) * coeffs[89] * x34 * x69 * x99 -
           1.0 / 27.0 * coeffs[8] * x128 * x144 -
           1.0 / 27.0 * coeffs[90] * x101 * x46 +
           (256.0 / 27.0) * coeffs[91] * x102 * x98 -
           1.0 / 27.0 * coeffs[92] * x101 * x24 +
           (16.0 / 3.0) * coeffs[93] * x48 * x99 -
           1.0 / 27.0 * coeffs[94] * x101 * x50 +
           (1.0 / 27.0) * coeffs[95] * x103 * x35 -
           1.0 / 27.0 * coeffs[96] * x100 * x105 * x36 * x[2] +
           (1.0 / 27.0) * coeffs[97] * x103 * x39 * x7 -
           1.0 / 27.0 * coeffs[98] * x34 * x56 +
           (1.0 / 27.0) * coeffs[99] * x59 * x64 +
           (1.0 / 27.0) * coeffs[9] * x141 * x154;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[2] - 1;
    const Scalar x1 = x[0] - 1;
    const Scalar x2 = 4 * x[1];
    const Scalar x3 = x2 - 1;
    const Scalar x4 = x1 * x3;
    const Scalar x5 = x0 * x4;
    const Scalar x6 = x[1] - 1;
    const Scalar x7 = 2 * x[0] - 1;
    const Scalar x8 = 2 * x[1] - 1;
    const Scalar x9 = 2 * x[2] - 1;
    const Scalar x10 = 4 * x[0];
    const Scalar x11 = x10 - 1;
    const Scalar x12 = 4 * x[2];
    const Scalar x13 = x12 - 1;
    const Scalar x14 = x10 - 3;
    const Scalar x15 = x2 - 3;
    const Scalar x16 = x12 - 3;
    const Scalar x17 = x11 * x13 * x14 * x15 * x16 * x6 * x7 * x8 * x9;
    const Scalar x18 = (1.0 / 27.0) * x17;
    const Scalar x19 = x3 * x[0];
    const Scalar x20 = x0 * x19;
    const Scalar x21 = x14 * x15 * x16 * x7 * x8 * x9;
    const Scalar x22 = x20 * x21;
    const Scalar x23 = x11 * x13;
    const Scalar x24 = x23 * x[1];
    const Scalar x25 = (1.0 / 27.0) * x24;
    const Scalar x26 = x21 * x25;
    const Scalar x27 = x18 * x[2];
    const Scalar x28 = x26 * x[2];
    const Scalar x29 = (16.0 / 27.0) * x[0];
    const Scalar x30 = x5 * x6;
    const Scalar x31 = x29 * x30;
    const Scalar x32 = x13 * x21;
    const Scalar x33 = x30 * x[0];
    const Scalar x34 = x8 * x9;
    const Scalar x35 = x33 * x34;
    const Scalar x36 = x14 * x15 * x16;
    const Scalar x37 = (4.0 / 9.0) * x36;
    const Scalar x38 = x23 * x37;
    const Scalar x39 = x15 * x23;
    const Scalar x40 = x16 * x7;
    const Scalar x41 = x34 * x40;
    const Scalar x42 = x39 * x41;
    const Scalar x43 = x29 * x[1];
    const Scalar x44 = x0 * x17;
    const Scalar x45 = x20 * x6;
    const Scalar x46 = x24 * x37;
    const Scalar x47 = x7 * x9;
    const Scalar x48 = x46 * x47;
    const Scalar x49 = x24 * x41;
    const Scalar x50 = (16.0 / 27.0) * x14;
    const Scalar x51 = x45 * x50;
    const Scalar x52 = x32 * x43;
    const Scalar x53 = x34 * x[0];
    const Scalar x54 = x46 * x5;
    const Scalar x55 = x15 * x5;
    const Scalar x56 = x29 * x49;
    const Scalar x57 = (16.0 / 27.0) * x[1];
    const Scalar x58 = x1 * x57;
    const Scalar x59 = x49 * x50;
    const Scalar x60 = x6 * x[2];
    const Scalar x61 = x4 * x60;
    const Scalar x62 = x29 * x61;
    const Scalar x63 = x53 * x61;
    const Scalar x64 = x17 * x[2];
    const Scalar x65 = x19 * x60;
    const Scalar x66 = x4 * x[2];
    const Scalar x67 = x30 * x[2];
    const Scalar x68 = (16.0 / 27.0) * x11;
    const Scalar x69 = x67 * x7;
    const Scalar x70 = x38 * x8;
    const Scalar x71 = x34 * x39;
    const Scalar x72 = x7 * x[2];
    const Scalar x73 = x45 * x72;
    const Scalar x74 = x11 * x[2];
    const Scalar x75 = x57 * x74;
    const Scalar x76 = x20 * x72;
    const Scalar x77 = x15 * x24;
    const Scalar x78 = x34 * x50;
    const Scalar x79 = x21 * x5;
    const Scalar x80 = x72 * x8;
    const Scalar x81 = x24 * x55;
    const Scalar x82 = x72 * x81;
    const Scalar x83 = (256.0 / 27.0) * x21;
    const Scalar x84 = x11 * x[1];
    const Scalar x85 = x0 * x60;
    const Scalar x86 = x1 * x85;
    const Scalar x87 = x84 * x86;
    const Scalar x88 = x67 * x84;
    const Scalar x89 = (64.0 / 9.0) * x36;
    const Scalar x90 = x47 * x89;
    const Scalar x91 = (256.0 / 27.0) * x14;
    const Scalar x92 = x41 * x91;
    const Scalar x93 = x7 * x86;
    const Scalar x94 = x24 * x8;
    const Scalar x95 = x89 * x94;
    const Scalar x96 = x24 * x69;
    const Scalar x97 = (16.0 / 3.0) * x36;
    const Scalar x98 = x40 * x94;
    const Scalar x99 = (64.0 / 9.0) * x14;
    const Scalar x100 = x67 * x99;
    const Scalar x101 = x34 * x91;
    const Scalar x102 = x77 * x93;
    const Scalar x103 = x47 * x77;
    const Scalar x104 = x85 * x[0];
    const Scalar x105 = x74 * x[1];
    const Scalar x106 = x105 * x45;
    const Scalar x107 = x24 * x73;
    const Scalar x108 = x99 * x[2];
    const Scalar x109 = x108 * x45;
    const Scalar x110 = x53 * x91;
    const Scalar x111 = x33 * x[2];
    const Scalar x112 = x35 * x74;
    const Scalar x113 = (256.0 / 27.0) * x15;
    const Scalar x114 = x13 * x33;
    const Scalar x115 = x80 * x89;
    const Scalar x116 = x111 * x8;
    const Scalar x117 = (64.0 / 9.0) * x40;
    const Scalar x118 = x35 * x72;
    const Scalar x119 = x13 * x15;
    const Scalar x120 = (256.0 / 27.0) * x[0];
    const Scalar x121 = x[1] * x[2];
    const Scalar x122 = x53 * x89;
    const Scalar x123 = x5 * x[0];
    const Scalar x124 = x13 * x[1];
    const Scalar x125 = x117 * x[0];
    const Scalar x126 = x0 * x1 * x6;
    const Scalar x127 = x120 * x32 * x[1];
    const Scalar x128 = x122 * x24;
    const Scalar x129 = x113 * x49 * x[0];
    const Scalar x130 = x114 * x[1];
    const Scalar x131 = x33 * x9;
    const Scalar x132 = x24 * x97;
    const Scalar x133 = x35 * x40;
    const Scalar x134 = x16 * x35;
    const Scalar x135 = x24 * x99;
    const Scalar x136 = x1 * x60;
    const Scalar x137 = x61 * x[0];
    const Scalar x138 = x124 * x137;
    const Scalar x139 = x77 * x9;
    const Scalar x140 = (4096.0 / 27.0) * x[0];
    const Scalar x141 = (1024.0 / 9.0) * x36;
    const Scalar x142 = x111 * x47 * x[1];
    const Scalar x143 = x105 * x131;
    const Scalar x144 = (256.0 / 3.0) * x36;
    const Scalar x145 = (1024.0 / 9.0) * x40;
    const Scalar x146 = (4096.0 / 27.0) * x133;
    const Scalar x147 = (1024.0 / 9.0) * x14;
    const Scalar x148 = x8 * x[0];
    const Scalar x149 = x77 * x86;
    const Scalar x150 = (256.0 / 3.0) * x111;
    const Scalar x151 = x124 * x14;
    const Scalar x152 = x14 * x150;
    const Scalar x153 = (1024.0 / 9.0) * x111;
    const Scalar x154 = (4096.0 / 27.0) * x53;
    const Scalar x155 = (4096.0 / 27.0) * x118;
    out[0] = x18 * x5;
    out[1] = x18 * x20;
    out[2] = x22 * x25;
    out[3] = x26 * x5;
    out[4] = x27 * x4;
    out[5] = x19 * x27;
    out[6] = x19 * x28;
    out[7] = x28 * x4;
    out[8] = -x31 * x32;
    out[9] = x35 * x38;
    out[10] = -x31 * x42;
    out[11] = -x43 * x44;
    out[12] = x45 * x48;
    out[13] = -x49 * x51;
    out[14] = -x5 * x52;
    out[15] = x53 * x54;
    out[16] = -x55 * x56;
    out[17] = -x44 * x58;
    out[18] = x30 * x48;
    out[19] = -x30 * x59;
    out[20] = -x32 * x62;
    out[21] = x38 * x63;
    out[22] = -x42 * x62;
    out[23] = -x43 * x64;
    out[24] = x48 * x65;
    out[25] = -x59 * x65;
    out[26] = -x52 * x66;
    out[27] = x46 * x53 * x66;
    out[28] = -x15 * x56 * x66;
    out[29] = -x58 * x64;
    out[30] = x48 * x61;
    out[31] = -x59 * x61;
    out[32] = -x21 * x67 * x68;
    out[33] = x69 * x70;
    out[34] = -x50 * x69 * x71;
    out[35] = -x22 * x60 * x68;
    out[36] = x70 * x73;
    out[37] = -x51 * x71 * x72;
    out[38] = -x22 * x75;
    out[39] = x46 * x76 * x8;
    out[40] = -x76 * x77 * x78;
    out[41] = -x75 * x79;
    out[42] = x54 * x80;
    out[43] = -x78 * x82;
    out[44] = x83 * x87;
    out[45] = -x88 * x90;
    out[46] = x88 * x92;
    out[47] = -x93 * x95;
    out[48] = x96 * x97;
    out[49] = -x100 * x98;
    out[50] = x101 * x102;
    out[51] = -x100 * x103;
    out[52] = x101 * x96;
    out[53] = x104 * x83 * x84;
    out[54] = -x106 * x90;
    out[55] = x106 * x92;
    out[56] = -x104 * x7 * x95;
    out[57] = x107 * x97;
    out[58] = -x109 * x98;
    out[59] = x110 * x7 * x77 * x85;
    out[60] = -x103 * x109;
    out[61] = x101 * x107;
    out[62] = x111 * x83;
    out[63] = -x112 * x89;
    out[64] = x112 * x113 * x40;
    out[65] = -x114 * x115;
    out[66] = x116 * x23 * x97;
    out[67] = -x116 * x117 * x39;
    out[68] = x118 * x119 * x91;
    out[69] = -x108 * x35 * x39;
    out[70] = (256.0 / 27.0) * x118 * x39;
    out[71] = x120 * x121 * x79;
    out[72] = -x105 * x122 * x5;
    out[73] = x105 * x120 * x41 * x55;
    out[74] = -x115 * x123 * x124;
    out[75] = x123 * x94 * x97 * x[2];
    out[76] = -x125 * x8 * x81 * x[2];
    out[77] = x110 * x124 * x55 * x72;
    out[78] = -x108 * x53 * x81;
    out[79] = (256.0 / 27.0) * x53 * x82;
    out[80] = x126 * x127;
    out[81] = -x126 * x128;
    out[82] = x126 * x129;
    out[83] = -x130 * x90;
    out[84] = x131 * x132;
    out[85] = -x117 * x131 * x77;
    out[86] = x124 * x133 * x91;
    out[87] = -x134 * x135;
    out[88] = (256.0 / 27.0) * x133 * x24;
    out[89] = x127 * x136;
    out[90] = -x128 * x136;
    out[91] = x129 * x136;
    out[92] = -x138 * x90;
    out[93] = x132 * x137 * x9;
    out[94] = -x125 * x139 * x61;
    out[95] = x138 * x92;
    out[96] = -x135 * x16 * x63;
    out[97] = x120 * x49 * x61;
    out[98] = -x140 * x21 * x86 * x[1];
    out[99] = x141 * x53 * x87;
    out[100] = -x140 * x15 * x41 * x87;
    out[101] = x141 * x142;
    out[102] = -x143 * x144;
    out[103] = x143 * x145 * x15;
    out[104] = -x121 * x14 * x146;
    out[105] = x105 * x134 * x147;
    out[106] = -x105 * x146;
    out[107] = x124 * x141 * x148 * x93;
    out[108] = -x144 * x86 * x94 * x[0];
    out[109] = x145 * x148 * x149;
    out[110] = -x130 * x144 * x72;
    out[111] = 64 * x111 * x24 * x36;
    out[112] = -x150 * x40 * x77;
    out[113] = x116 * x145 * x151;
    out[114] = -x152 * x16 * x94;
    out[115] = x153 * x98;
    out[116] = -x119 * x14 * x154 * x93 * x[1];
    out[117] = x147 * x149 * x53;
    out[118] = -x102 * x154;
    out[119] = x119 * x142 * x147;
    out[120] = -x139 * x152;
    out[121] = x103 * x153;
    out[122] = -x151 * x155;
    out[123] = x147 * x24 * x35 * x[2];
    out[124] = -x155 * x24;
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
    const Scalar x12 = 2 * x[2];
    const Scalar x13 = x12 - 1;
    const Scalar x14 = 4 * x[2];
    const Scalar x15 = x14 - 3;
    const Scalar x16 = x13 * x15;
    const Scalar x17 = x16 * x[2];
    const Scalar x18 = x11 * x17;
    const Scalar x19 = 2 * x[1];
    const Scalar x20 = x19 - 1;
    const Scalar x21 = 4 * x[1];
    const Scalar x22 = x21 - 3;
    const Scalar x23 = x20 * x22;
    const Scalar x24 = x23 * x[1];
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
    const Scalar x37 = x22 * x36;
    const Scalar x38 = x37 * x[1];
    const Scalar x39 = x27 * x38;
    const Scalar x40 = 3072 * x39;
    const Scalar x41 = x30 * x6;
    const Scalar x42 = x41 * x[0];
    const Scalar x43 = x2 * x41;
    const Scalar x44 = x30 * x4 + x4 * x6 + x42 + x43;
    const Scalar x45 = x17 * x44;
    const Scalar x46 = 2304 * x39;
    const Scalar x47 = x20 * x36;
    const Scalar x48 = x47 * x[1];
    const Scalar x49 = x28 * x48;
    const Scalar x50 = x27 * x45;
    const Scalar x51 = 3072 * x50;
    const Scalar x52 = coeffs[105] * x48;
    const Scalar x53 = x14 - 1;
    const Scalar x54 = x15 * x53;
    const Scalar x55 = x54 * x[2];
    const Scalar x56 = x27 * x55;
    const Scalar x57 = x24 * x56;
    const Scalar x58 = 3072 * x57;
    const Scalar x59 = x24 * x44;
    const Scalar x60 = 2304 * coeffs[108];
    const Scalar x61 = x46 * x55;
    const Scalar x62 = x39 * x44;
    const Scalar x63 = 1728 * coeffs[111];
    const Scalar x64 = x55 * x63;
    const Scalar x65 = x48 * x56;
    const Scalar x66 = 3072 * x65;
    const Scalar x67 = 2304 * coeffs[114];
    const Scalar x68 = x13 * x53;
    const Scalar x69 = x68 * x[2];
    const Scalar x70 = x29 * x69;
    const Scalar x71 = x27 * x59;
    const Scalar x72 = 3072 * x69;
    const Scalar x73 = coeffs[117] * x72;
    const Scalar x74 = x40 * x69;
    const Scalar x75 = x44 * x69;
    const Scalar x76 = x49 * x69;
    const Scalar x77 = x27 * x48;
    const Scalar x78 = 3072 * coeffs[123];
    const Scalar x79 = coeffs[99] * x24;
    const Scalar x80 = x26 * x[1];
    const Scalar x81 = x23 * x36;
    const Scalar x82 = 256 * x81;
    const Scalar x83 = x80 * x82;
    const Scalar x84 = 192 * x81;
    const Scalar x85 = x80 * x84;
    const Scalar x86 = x55 * x85;
    const Scalar x87 = x44 * x81;
    const Scalar x88 = 144 * x55;
    const Scalar x89 = coeffs[75] * x88;
    const Scalar x90 = x69 * x83;
    const Scalar x91 = x75 * x84;
    const Scalar x92 = coeffs[89] * x24;
    const Scalar x93 = x25 * x[2];
    const Scalar x94 = x16 * x53;
    const Scalar x95 = 256 * x94;
    const Scalar x96 = x93 * x95;
    const Scalar x97 = x34 * x96;
    const Scalar x98 = 192 * x94;
    const Scalar x99 = x93 * x98;
    const Scalar x100 = x11 * x96;
    const Scalar x101 = coeffs[91] * x24;
    const Scalar x102 = x38 * x99;
    const Scalar x103 = 144 * x94;
    const Scalar x104 = coeffs[93] * x103;
    const Scalar x105 = coeffs[95] * x48;
    const Scalar x106 = coeffs[96] * x48;
    const Scalar x107 = coeffs[97] * x48;
    const Scalar x108 = x81 * x94;
    const Scalar x109 = x108 * x34;
    const Scalar x110 = x[1] * x[2];
    const Scalar x111 = 16 * x110;
    const Scalar x112 = x108 * x44;
    const Scalar x113 = 12 * x112;
    const Scalar x114 = x108 * x11;
    const Scalar x115 = x27 * x82;
    const Scalar x116 = coeffs[63] * x84;
    const Scalar x117 = x56 * x84;
    const Scalar x118 = 144 * coeffs[66];
    const Scalar x119 = x115 * x69;
    const Scalar x120 = x24 * x27;
    const Scalar x121 = 256 * x120;
    const Scalar x122 = x121 * x94;
    const Scalar x123 = coeffs[81] * x98;
    const Scalar x124 = x39 * x98;
    const Scalar x125 = coeffs[84] * x103;
    const Scalar x126 = 256 * x77;
    const Scalar x127 = x126 * x94;
    const Scalar x128 = coeffs[87] * x98;
    const Scalar x129 = 16 * x80;
    const Scalar x130 = 12 * x80;
    const Scalar x131 = 16 * x93;
    const Scalar x132 = 12 * x93;
    const Scalar x133 = x30 * x8;
    const Scalar x134 = x0 * x41 + x133 + x3 * x31 + x3 * x8;
    const Scalar x135 = x134 * x17;
    const Scalar x136 = 192 * x134;
    const Scalar x137 = x136 * x39;
    const Scalar x138 = coeffs[54] * x17;
    const Scalar x139 = x39 * x88;
    const Scalar x140 = x134 * x69;
    const Scalar x141 = coeffs[60] * x69;
    const Scalar x142 = 16 * x27;
    const Scalar x143 = x134 * x94;
    const Scalar x144 = x131 * x143;
    const Scalar x145 = coeffs[23] * x24;
    const Scalar x146 = x132 * x38;
    const Scalar x147 = coeffs[25] * x48;
    const Scalar x148 = x129 * x81;
    const Scalar x149 = x134 * x81;
    const Scalar x150 = x130 * x55;
    const Scalar x151 = x148 * x69;
    const Scalar x152 = x108 * x134;
    const Scalar x153 = coeffs[11] * x24;
    const Scalar x154 = 12 * x39;
    const Scalar x155 = 16 * x77;
    const Scalar x156 = x142 * x81;
    const Scalar x157 = 12 * coeffs[36];
    const Scalar x158 = x156 * x69;
    const Scalar x159 = 4 * x10 + x133 + 4 * x33 + 2 * x43;
    const Scalar x160 = x159 * x17;
    const Scalar x161 = 256 * x160;
    const Scalar x162 = 192 * x159;
    const Scalar x163 = x162 * x39;
    const Scalar x164 = coeffs[45] * x17;
    const Scalar x165 = x159 * x69;
    const Scalar x166 = coeffs[51] * x69;
    const Scalar x167 = x159 * x94;
    const Scalar x168 = x131 * x167;
    const Scalar x169 = coeffs[29] * x24;
    const Scalar x170 = coeffs[31] * x48;
    const Scalar x171 = x159 * x81;
    const Scalar x172 = x108 * x159;
    const Scalar x173 = coeffs[17] * x24;
    const Scalar x174 = 12 * coeffs[33];
    const Scalar x175 = x21 * x25;
    const Scalar x176 = x175 * x20;
    const Scalar x177 = x19 * x25;
    const Scalar x178 = x23 * x25;
    const Scalar x179 = x176 + x177 * x22 + x178 + x24;
    const Scalar x180 = x17 * x179;
    const Scalar x181 = x2 * x26;
    const Scalar x182 = 4096 * x181;
    const Scalar x183 = x182 * x9;
    const Scalar x184 = x17 * x32;
    const Scalar x185 = x25 * x37;
    const Scalar x186 = x175 * x22 + x175 * x36 + x185 + x38;
    const Scalar x187 = x181 * x186;
    const Scalar x188 = 3072 * x187;
    const Scalar x189 = 2304 * x186;
    const Scalar x190 = x181 * x42;
    const Scalar x191 = x17 * x190;
    const Scalar x192 = coeffs[103] * x9;
    const Scalar x193 = x25 * x47;
    const Scalar x194 = x176 + x177 * x36 + x193 + x48;
    const Scalar x195 = x17 * x194;
    const Scalar x196 = x182 * x32;
    const Scalar x197 = 3072 * x190;
    const Scalar x198 = x179 * x32;
    const Scalar x199 = x181 * x55;
    const Scalar x200 = 3072 * x199;
    const Scalar x201 = x190 * x55;
    const Scalar x202 = x200 * x9;
    const Scalar x203 = coeffs[110] * x32;
    const Scalar x204 = x189 * x199;
    const Scalar x205 = x186 * x190;
    const Scalar x206 = coeffs[112] * x9;
    const Scalar x207 = x194 * x32;
    const Scalar x208 = x179 * x69;
    const Scalar x209 = x179 * x190;
    const Scalar x210 = x187 * x72;
    const Scalar x211 = coeffs[119] * x32;
    const Scalar x212 = x190 * x69;
    const Scalar x213 = coeffs[121] * x9;
    const Scalar x214 = x194 * x69;
    const Scalar x215 = x190 * x194;
    const Scalar x216 = x26 * x[0];
    const Scalar x217 = 256 * x133;
    const Scalar x218 = x216 * x217;
    const Scalar x219 = 192 * x133;
    const Scalar x220 = x216 * x219;
    const Scalar x221 = x186 * x220;
    const Scalar x222 = x220 * x55;
    const Scalar x223 = x186 * x216;
    const Scalar x224 = x133 * x88;
    const Scalar x225 = x2 * x[2];
    const Scalar x226 = x225 * x95;
    const Scalar x227 = coeffs[90] * x42;
    const Scalar x228 = x225 * x98;
    const Scalar x229 = x226 * x9;
    const Scalar x230 = x186 * x228;
    const Scalar x231 = coeffs[92] * x32;
    const Scalar x232 = x225 * x42;
    const Scalar x233 = coeffs[94] * x9;
    const Scalar x234 = x133 * x94;
    const Scalar x235 = x179 * x234;
    const Scalar x236 = x[0] * x[2];
    const Scalar x237 = 16 * x236;
    const Scalar x238 = 12 * x234;
    const Scalar x239 = x186 * x238;
    const Scalar x240 = x194 * x234;
    const Scalar x241 = x181 * x217;
    const Scalar x242 = x187 * x219;
    const Scalar x243 = x199 * x219;
    const Scalar x244 = x181 * x95;
    const Scalar x245 = x244 * x9;
    const Scalar x246 = x187 * x98;
    const Scalar x247 = coeffs[83] * x32;
    const Scalar x248 = coeffs[85] * x9;
    const Scalar x249 = 16 * x216;
    const Scalar x250 = 16 * x225;
    const Scalar x251 = x19 * x37 + x21 * x23 + x21 * x47 + x81;
    const Scalar x252 = 256 * x181;
    const Scalar x253 = x251 * x252;
    const Scalar x254 = 192 * x251;
    const Scalar x255 = x253 * x9;
    const Scalar x256 = x199 * x254;
    const Scalar x257 = coeffs[74] * x32;
    const Scalar x258 = coeffs[76] * x9;
    const Scalar x259 = x32 * x69;
    const Scalar x260 = 16 * x181;
    const Scalar x261 = x251 * x94;
    const Scalar x262 = x250 * x261;
    const Scalar x263 = coeffs[26] * x32;
    const Scalar x264 = 12 * x261;
    const Scalar x265 = coeffs[28] * x9;
    const Scalar x266 = x133 * x251;
    const Scalar x267 = x249 * x266;
    const Scalar x268 = x216 * x55;
    const Scalar x269 = 12 * x266;
    const Scalar x270 = x234 * x251;
    const Scalar x271 = x270 * x[2];
    const Scalar x272 = coeffs[6] * x[0];
    const Scalar x273 = coeffs[14] * x32;
    const Scalar x274 = x260 * x261;
    const Scalar x275 = coeffs[16] * x9;
    const Scalar x276 = x260 * x266;
    const Scalar x277 = coeffs[7] * x2;
    const Scalar x278 = 4 * x178 + 2 * x185 + 4 * x193 + x81;
    const Scalar x279 = x252 * x278;
    const Scalar x280 = 192 * x278;
    const Scalar x281 = x279 * x9;
    const Scalar x282 = x199 * x280;
    const Scalar x283 = coeffs[65] * x32;
    const Scalar x284 = coeffs[67] * x9;
    const Scalar x285 = x278 * x94;
    const Scalar x286 = x250 * x285;
    const Scalar x287 = coeffs[20] * x32;
    const Scalar x288 = 12 * x285;
    const Scalar x289 = coeffs[22] * x9;
    const Scalar x290 = x133 * x278;
    const Scalar x291 = x249 * x290;
    const Scalar x292 = x234 * x278;
    const Scalar x293 = x292 * x[2];
    const Scalar x294 = coeffs[5] * x[0];
    const Scalar x295 = coeffs[10] * x9;
    const Scalar x296 = x260 * x285;
    const Scalar x297 = x260 * x290;
    const Scalar x298 = coeffs[8] * x32;
    const Scalar x299 = coeffs[4] * x2;
    const Scalar x300 = x14 * x26;
    const Scalar x301 = x13 * x300;
    const Scalar x302 = x12 * x26;
    const Scalar x303 = x16 * x26;
    const Scalar x304 = x15 * x302 + x17 + x301 + x303;
    const Scalar x305 = x24 * x304;
    const Scalar x306 = x2 * x25;
    const Scalar x307 = 4096 * x306;
    const Scalar x308 = x307 * x9;
    const Scalar x309 = x306 * x32;
    const Scalar x310 = x304 * x38;
    const Scalar x311 = 3072 * x310;
    const Scalar x312 = x306 * x42;
    const Scalar x313 = 2304 * x312;
    const Scalar x314 = x304 * x48;
    const Scalar x315 = x307 * x32;
    const Scalar x316 = x304 * x312;
    const Scalar x317 = 3072 * x316;
    const Scalar x318 = x26 * x54;
    const Scalar x319 = x15 * x300 + x300 * x53 + x318 + x55;
    const Scalar x320 = x306 * x319;
    const Scalar x321 = 3072 * x320;
    const Scalar x322 = x24 * x321;
    const Scalar x323 = x312 * x319;
    const Scalar x324 = x320 * x38;
    const Scalar x325 = 2304 * x324;
    const Scalar x326 = x321 * x48;
    const Scalar x327 = x26 * x68;
    const Scalar x328 = x301 + x302 * x53 + x327 + x69;
    const Scalar x329 = x24 * x328;
    const Scalar x330 = x328 * x38;
    const Scalar x331 = 3072 * x306 * x330;
    const Scalar x332 = x328 * x48;
    const Scalar x333 = x25 * x[0];
    const Scalar x334 = x217 * x333;
    const Scalar x335 = x219 * x333;
    const Scalar x336 = x319 * x335;
    const Scalar x337 = x333 * x38;
    const Scalar x338 = 144 * x133;
    const Scalar x339 = x32 * x82;
    const Scalar x340 = x2 * x[1];
    const Scalar x341 = x304 * x340;
    const Scalar x342 = x340 * x84;
    const Scalar x343 = x82 * x9;
    const Scalar x344 = x319 * x342;
    const Scalar x345 = x340 * x42;
    const Scalar x346 = x345 * x81;
    const Scalar x347 = x328 * x340;
    const Scalar x348 = x328 * x84;
    const Scalar x349 = x133 * x81;
    const Scalar x350 = 16 * x349;
    const Scalar x351 = x304 * x350;
    const Scalar x352 = x[0] * x[1];
    const Scalar x353 = x319 * x349;
    const Scalar x354 = 12 * x353;
    const Scalar x355 = x328 * x350;
    const Scalar x356 = x217 * x306;
    const Scalar x357 = x219 * x306;
    const Scalar x358 = x219 * x320;
    const Scalar x359 = x304 * x306;
    const Scalar x360 = x359 * x82;
    const Scalar x361 = x320 * x84;
    const Scalar x362 = x306 * x328;
    const Scalar x363 = x12 * x54 + x14 * x16 + x14 * x68 + x94;
    const Scalar x364 = 256 * x363;
    const Scalar x365 = x309 * x364;
    const Scalar x366 = 192 * x363;
    const Scalar x367 = x24 * x306;
    const Scalar x368 = x306 * x9;
    const Scalar x369 = x364 * x368;
    const Scalar x370 = x306 * x38;
    const Scalar x371 = x366 * x370;
    const Scalar x372 = 144 * x312 * x38;
    const Scalar x373 = 16 * x363;
    const Scalar x374 = x133 * x373;
    const Scalar x375 = x333 * x374;
    const Scalar x376 = 12 * x363;
    const Scalar x377 = x133 * x376;
    const Scalar x378 = x373 * x81;
    const Scalar x379 = x340 * x378;
    const Scalar x380 = x349 * x363;
    const Scalar x381 = x380 * x[1];
    const Scalar x382 = x306 * x378;
    const Scalar x383 = x312 * x81;
    const Scalar x384 = x306 * x374;
    const Scalar x385 = x25 * x380;
    const Scalar x386 = 4 * x303 + 2 * x318 + 4 * x327 + x94;
    const Scalar x387 = 256 * x386;
    const Scalar x388 = x367 * x387;
    const Scalar x389 = 192 * x386;
    const Scalar x390 = x312 * x389;
    const Scalar x391 = x370 * x389;
    const Scalar x392 = x387 * x48;
    const Scalar x393 = 16 * x386;
    const Scalar x394 = x133 * x393;
    const Scalar x395 = x333 * x394;
    const Scalar x396 = 12 * x386;
    const Scalar x397 = x133 * x396;
    const Scalar x398 = x393 * x81;
    const Scalar x399 = x340 * x398;
    const Scalar x400 = x349 * x386;
    const Scalar x401 = x400 * x[0];
    const Scalar x402 = x306 * x398;
    const Scalar x403 = x306 * x394;
    out[0] = (1.0 / 27.0) * coeffs[0] * x172 * x27 -
             1.0 / 27.0 * coeffs[100] * x18 * x29 +
             (1.0 / 27.0) * coeffs[101] * x35 * x40 -
             1.0 / 27.0 * coeffs[102] * x45 * x46 +
             (1.0 / 27.0) * coeffs[103] * x18 * x40 -
             1.0 / 27.0 * coeffs[104] * x35 * x49 -
             1.0 / 27.0 * coeffs[106] * x18 * x49 +
             (1.0 / 27.0) * coeffs[107] * x34 * x58 +
             (1.0 / 27.0) * coeffs[109] * x11 * x58 -
             1.0 / 27.0 * coeffs[10] * x114 * x142 -
             1.0 / 27.0 * coeffs[110] * x34 * x61 -
             1.0 / 27.0 * coeffs[112] * x11 * x61 +
             (1.0 / 27.0) * coeffs[113] * x34 * x66 +
             (1.0 / 27.0) * coeffs[115] * x11 * x66 -
             1.0 / 27.0 * coeffs[116] * x34 * x70 -
             1.0 / 27.0 * coeffs[118] * x11 * x70 +
             (1.0 / 27.0) * coeffs[119] * x34 * x74 -
             1.0 / 27.0 * coeffs[120] * x46 * x75 +
             (1.0 / 27.0) * coeffs[121] * x11 * x74 -
             1.0 / 27.0 * coeffs[122] * x34 * x76 -
             1.0 / 27.0 * coeffs[124] * x11 * x76 +
             (1.0 / 27.0) * coeffs[12] * x143 * x154 -
             1.0 / 27.0 * coeffs[13] * x143 * x155 -
             1.0 / 27.0 * coeffs[14] * x109 * x129 +
             (1.0 / 27.0) * coeffs[15] * x112 * x130 -
             1.0 / 27.0 * coeffs[16] * x114 * x129 +
             (1.0 / 27.0) * coeffs[18] * x154 * x167 -
             1.0 / 27.0 * coeffs[19] * x155 * x167 +
             (1.0 / 27.0) * coeffs[1] * x152 * x27 -
             1.0 / 27.0 * coeffs[20] * x109 * x131 +
             (1.0 / 27.0) * coeffs[21] * x112 * x132 -
             1.0 / 27.0 * coeffs[22] * x114 * x131 +
             (1.0 / 27.0) * coeffs[24] * x143 * x146 -
             1.0 / 27.0 * coeffs[26] * x109 * x111 +
             (1.0 / 27.0) * coeffs[27] * x110 * x113 -
             1.0 / 27.0 * coeffs[28] * x111 * x114 +
             (1.0 / 27.0) * coeffs[2] * x152 * x80 +
             (1.0 / 27.0) * coeffs[30] * x146 * x167 -
             1.0 / 27.0 * coeffs[32] * x156 * x160 -
             1.0 / 27.0 * coeffs[34] * x158 * x159 -
             1.0 / 27.0 * coeffs[35] * x135 * x156 -
             1.0 / 27.0 * coeffs[37] * x134 * x158 -
             1.0 / 27.0 * coeffs[38] * x135 * x148 +
             (1.0 / 27.0) * coeffs[39] * x149 * x150 +
             (1.0 / 27.0) * coeffs[3] * x172 * x80 -
             1.0 / 27.0 * coeffs[40] * x134 * x151 -
             1.0 / 27.0 * coeffs[41] * x148 * x160 +
             (1.0 / 27.0) * coeffs[42] * x150 * x171 -
             1.0 / 27.0 * coeffs[43] * x151 * x159 +
             (1.0 / 27.0) * coeffs[44] * x120 * x161 +
             (1.0 / 27.0) * coeffs[46] * x161 * x77 -
             1.0 / 27.0 * coeffs[47] * x162 * x57 +
             (1.0 / 27.0) * coeffs[48] * x139 * x159 -
             1.0 / 27.0 * coeffs[49] * x162 * x65 +
             (1.0 / 27.0) * coeffs[4] * x172 * x93 +
             (1.0 / 27.0) * coeffs[50] * x121 * x165 +
             (1.0 / 27.0) * coeffs[52] * x126 * x165 +
             (1.0 / 27.0) * coeffs[53] * x121 * x135 +
             (1.0 / 27.0) * coeffs[55] * x126 * x135 -
             1.0 / 27.0 * coeffs[56] * x136 * x57 +
             (1.0 / 27.0) * coeffs[57] * x134 * x139 -
             1.0 / 27.0 * coeffs[58] * x136 * x65 +
             (1.0 / 27.0) * coeffs[59] * x121 * x140 +
             (1.0 / 27.0) * coeffs[5] * x152 * x93 +
             (1.0 / 27.0) * coeffs[61] * x126 * x140 +
             (1.0 / 27.0) * coeffs[62] * x115 * x35 +
             (1.0 / 27.0) * coeffs[64] * x115 * x18 -
             1.0 / 27.0 * coeffs[65] * x117 * x34 -
             1.0 / 27.0 * coeffs[67] * x11 * x117 +
             (1.0 / 27.0) * coeffs[68] * x119 * x34 -
             1.0 / 27.0 * coeffs[69] * x27 * x91 +
             (1.0 / 27.0) * coeffs[6] * x110 * x152 +
             (1.0 / 27.0) * coeffs[70] * x11 * x119 +
             (1.0 / 27.0) * coeffs[71] * x35 * x83 -
             1.0 / 27.0 * coeffs[72] * x45 * x85 +
             (1.0 / 27.0) * coeffs[73] * x18 * x83 -
             1.0 / 27.0 * coeffs[74] * x34 * x86 -
             1.0 / 27.0 * coeffs[76] * x11 * x86 +
             (1.0 / 27.0) * coeffs[77] * x34 * x90 -
             1.0 / 27.0 * coeffs[78] * x80 * x91 +
             (1.0 / 27.0) * coeffs[79] * x11 * x90 +
             (1.0 / 27.0) * coeffs[7] * x110 * x172 +
             (1.0 / 27.0) * coeffs[80] * x122 * x34 +
             (1.0 / 27.0) * coeffs[82] * x11 * x122 -
             1.0 / 27.0 * coeffs[83] * x124 * x34 -
             1.0 / 27.0 * coeffs[85] * x11 * x124 +
             (1.0 / 27.0) * coeffs[86] * x127 * x34 +
             (1.0 / 27.0) * coeffs[88] * x11 * x127 -
             1.0 / 27.0 * coeffs[8] * x109 * x142 -
             1.0 / 27.0 * coeffs[90] * x59 * x99 -
             1.0 / 27.0 * coeffs[92] * x102 * x34 -
             1.0 / 27.0 * coeffs[94] * x102 * x11 -
             1.0 / 27.0 * coeffs[98] * x29 * x35 +
             (1.0 / 27.0) * coeffs[9] * x113 * x27 +
             (1.0 / 27.0) * x100 * x101 + (1.0 / 27.0) * x100 * x107 +
             (1.0 / 27.0) * x104 * x38 * x44 * x93 + (1.0 / 27.0) * x105 * x97 -
             1.0 / 27.0 * x106 * x44 * x99 - 1.0 / 27.0 * x116 * x50 +
             (1.0 / 27.0) * x118 * x56 * x87 - 1.0 / 27.0 * x123 * x71 +
             (1.0 / 27.0) * x125 * x62 - 1.0 / 27.0 * x128 * x44 * x77 -
             1.0 / 27.0 * x137 * x138 - 1.0 / 27.0 * x137 * x141 -
             1.0 / 27.0 * x142 * x143 * x153 - 1.0 / 27.0 * x142 * x167 * x173 -
             1.0 / 27.0 * x144 * x145 - 1.0 / 27.0 * x144 * x147 +
             (1.0 / 27.0) * x149 * x157 * x56 - 1.0 / 27.0 * x163 * x164 -
             1.0 / 27.0 * x163 * x166 - 1.0 / 27.0 * x168 * x169 -
             1.0 / 27.0 * x168 * x170 + (1.0 / 27.0) * x171 * x174 * x56 -
             1.0 / 27.0 * x44 * x65 * x67 + (1.0 / 27.0) * x51 * x52 +
             (1.0 / 27.0) * x51 * x79 - 1.0 / 27.0 * x56 * x59 * x60 +
             (1.0 / 27.0) * x62 * x64 + (1.0 / 27.0) * x71 * x73 +
             (1.0 / 27.0) * x75 * x77 * x78 + (1.0 / 27.0) * x80 * x87 * x89 +
             (1.0 / 27.0) * x92 * x97;
    out[1] = (1.0 / 27.0) * coeffs[0] * x181 * x292 -
             1.0 / 27.0 * coeffs[100] * x180 * x183 +
             (1.0 / 27.0) * coeffs[101] * x184 * x188 -
             1.0 / 27.0 * coeffs[102] * x189 * x191 -
             1.0 / 27.0 * coeffs[104] * x195 * x196 +
             (1.0 / 27.0) * coeffs[105] * x195 * x197 -
             1.0 / 27.0 * coeffs[106] * x183 * x195 +
             (1.0 / 27.0) * coeffs[107] * x198 * x200 +
             (1.0 / 27.0) * coeffs[109] * x179 * x202 +
             (1.0 / 27.0) * coeffs[113] * x200 * x207 +
             (1.0 / 27.0) * coeffs[115] * x194 * x202 -
             1.0 / 27.0 * coeffs[116] * x196 * x208 -
             1.0 / 27.0 * coeffs[118] * x183 * x208 -
             1.0 / 27.0 * coeffs[11] * x235 * x249 -
             1.0 / 27.0 * coeffs[120] * x189 * x212 -
             1.0 / 27.0 * coeffs[122] * x196 * x214 +
             (1.0 / 27.0) * coeffs[123] * x215 * x72 -
             1.0 / 27.0 * coeffs[124] * x183 * x214 +
             (1.0 / 27.0) * coeffs[12] * x223 * x238 -
             1.0 / 27.0 * coeffs[13] * x240 * x249 +
             (1.0 / 27.0) * coeffs[15] * x190 * x264 -
             1.0 / 27.0 * coeffs[17] * x235 * x260 +
             (1.0 / 27.0) * coeffs[18] * x187 * x238 -
             1.0 / 27.0 * coeffs[19] * x240 * x260 +
             (1.0 / 27.0) * coeffs[1] * x216 * x292 +
             (1.0 / 27.0) * coeffs[21] * x232 * x288 -
             1.0 / 27.0 * coeffs[23] * x235 * x237 +
             (1.0 / 27.0) * coeffs[24] * x236 * x239 -
             1.0 / 27.0 * coeffs[25] * x237 * x240 +
             (1.0 / 27.0) * coeffs[27] * x232 * x264 -
             1.0 / 27.0 * coeffs[29] * x235 * x250 +
             (1.0 / 27.0) * coeffs[2] * x216 * x270 +
             (1.0 / 27.0) * coeffs[30] * x225 * x239 -
             1.0 / 27.0 * coeffs[31] * x240 * x250 -
             1.0 / 27.0 * coeffs[32] * x17 * x297 -
             1.0 / 27.0 * coeffs[34] * x297 * x69 -
             1.0 / 27.0 * coeffs[35] * x17 * x291 -
             1.0 / 27.0 * coeffs[37] * x291 * x69 -
             1.0 / 27.0 * coeffs[38] * x17 * x267 +
             (1.0 / 27.0) * coeffs[39] * x268 * x269 +
             (1.0 / 27.0) * coeffs[3] * x181 * x270 -
             1.0 / 27.0 * coeffs[40] * x267 * x69 -
             1.0 / 27.0 * coeffs[41] * x17 * x276 +
             (1.0 / 27.0) * coeffs[42] * x199 * x269 -
             1.0 / 27.0 * coeffs[43] * x276 * x69 +
             (1.0 / 27.0) * coeffs[44] * x180 * x241 +
             (1.0 / 27.0) * coeffs[46] * x195 * x241 -
             1.0 / 27.0 * coeffs[47] * x179 * x243 +
             (1.0 / 27.0) * coeffs[48] * x187 * x224 -
             1.0 / 27.0 * coeffs[49] * x194 * x243 +
             (1.0 / 27.0) * coeffs[50] * x208 * x241 +
             (1.0 / 27.0) * coeffs[52] * x214 * x241 +
             (1.0 / 27.0) * coeffs[53] * x180 * x218 +
             (1.0 / 27.0) * coeffs[55] * x195 * x218 -
             1.0 / 27.0 * coeffs[56] * x179 * x222 +
             (1.0 / 27.0) * coeffs[57] * x223 * x224 -
             1.0 / 27.0 * coeffs[58] * x194 * x222 +
             (1.0 / 27.0) * coeffs[59] * x208 * x218 +
             (1.0 / 27.0) * coeffs[61] * x214 * x218 +
             (1.0 / 27.0) * coeffs[62] * x184 * x279 -
             1.0 / 27.0 * coeffs[63] * x191 * x280 +
             (1.0 / 27.0) * coeffs[64] * x17 * x281 +
             (1.0 / 27.0) * coeffs[66] * x190 * x278 * x88 +
             (1.0 / 27.0) * coeffs[68] * x259 * x279 -
             1.0 / 27.0 * coeffs[69] * x212 * x280 +
             (1.0 / 27.0) * coeffs[70] * x281 * x69 +
             (1.0 / 27.0) * coeffs[71] * x184 * x253 -
             1.0 / 27.0 * coeffs[72] * x191 * x254 +
             (1.0 / 27.0) * coeffs[73] * x17 * x255 +
             (1.0 / 27.0) * coeffs[77] * x253 * x259 -
             1.0 / 27.0 * coeffs[78] * x212 * x254 +
             (1.0 / 27.0) * coeffs[79] * x255 * x69 +
             (1.0 / 27.0) * coeffs[80] * x198 * x244 +
             (1.0 / 27.0) * coeffs[82] * x179 * x245 +
             (1.0 / 27.0) * coeffs[86] * x207 * x244 +
             (1.0 / 27.0) * coeffs[88] * x194 * x245 +
             (1.0 / 27.0) * coeffs[89] * x198 * x226 +
             (1.0 / 27.0) * coeffs[91] * x179 * x229 +
             (1.0 / 27.0) * coeffs[95] * x207 * x226 -
             1.0 / 27.0 * coeffs[96] * x194 * x228 * x42 +
             (1.0 / 27.0) * coeffs[97] * x194 * x229 -
             1.0 / 27.0 * coeffs[98] * x180 * x196 +
             (1.0 / 27.0) * coeffs[99] * x180 * x197 +
             (1.0 / 27.0) * coeffs[9] * x190 * x288 +
             (1.0 / 27.0) * x104 * x186 * x232 - 1.0 / 27.0 * x123 * x209 +
             (1.0 / 27.0) * x125 * x205 - 1.0 / 27.0 * x128 * x215 -
             1.0 / 27.0 * x138 * x221 - 1.0 / 27.0 * x141 * x221 +
             (1.0 / 27.0) * x157 * x268 * x290 - 1.0 / 27.0 * x164 * x242 -
             1.0 / 27.0 * x166 * x242 + (1.0 / 27.0) * x17 * x188 * x192 +
             (1.0 / 27.0) * x174 * x199 * x290 -
             1.0 / 27.0 * x179 * x201 * x60 - 1.0 / 27.0 * x179 * x227 * x228 +
             (1.0 / 27.0) * x190 * x251 * x89 - 1.0 / 27.0 * x194 * x201 * x67 -
             1.0 / 27.0 * x203 * x204 - 1.0 / 27.0 * x204 * x206 +
             (1.0 / 27.0) * x205 * x64 + (1.0 / 27.0) * x209 * x73 +
             (1.0 / 27.0) * x210 * x211 + (1.0 / 27.0) * x210 * x213 -
             1.0 / 27.0 * x230 * x231 - 1.0 / 27.0 * x230 * x233 -
             1.0 / 27.0 * x246 * x247 - 1.0 / 27.0 * x246 * x248 -
             1.0 / 27.0 * x256 * x257 - 1.0 / 27.0 * x256 * x258 -
             1.0 / 27.0 * x262 * x263 - 1.0 / 27.0 * x262 * x265 +
             (1.0 / 27.0) * x271 * x272 + (1.0 / 27.0) * x271 * x277 -
             1.0 / 27.0 * x273 * x274 - 1.0 / 27.0 * x274 * x275 -
             1.0 / 27.0 * x282 * x283 - 1.0 / 27.0 * x282 * x284 -
             1.0 / 27.0 * x286 * x287 - 1.0 / 27.0 * x286 * x289 +
             (1.0 / 27.0) * x293 * x294 + (1.0 / 27.0) * x293 * x299 -
             1.0 / 27.0 * x295 * x296 - 1.0 / 27.0 * x296 * x298;
    out[2] = (1.0 / 27.0) * coeffs[0] * x306 * x400 -
             1.0 / 27.0 * coeffs[100] * x305 * x308 +
             (1.0 / 27.0) * coeffs[101] * x309 * x311 -
             1.0 / 27.0 * coeffs[102] * x310 * x313 -
             1.0 / 27.0 * coeffs[104] * x314 * x315 -
             1.0 / 27.0 * coeffs[106] * x308 * x314 +
             (1.0 / 27.0) * coeffs[107] * x32 * x322 +
             (1.0 / 27.0) * coeffs[109] * x322 * x9 +
             (1.0 / 27.0) * coeffs[113] * x32 * x326 +
             (1.0 / 27.0) * coeffs[115] * x326 * x9 -
             1.0 / 27.0 * coeffs[116] * x315 * x329 +
             (1024.0 / 9.0) * coeffs[117] * x312 * x329 -
             1.0 / 27.0 * coeffs[118] * x308 * x329 -
             1.0 / 27.0 * coeffs[120] * x313 * x330 -
             1.0 / 27.0 * coeffs[122] * x315 * x332 -
             1.0 / 27.0 * coeffs[124] * x308 * x332 +
             (1.0 / 27.0) * coeffs[12] * x337 * x397 -
             1.0 / 27.0 * coeffs[13] * x395 * x48 +
             (1.0 / 27.0) * coeffs[15] * x346 * x396 +
             (1.0 / 27.0) * coeffs[18] * x370 * x397 -
             1.0 / 27.0 * coeffs[19] * x403 * x48 +
             (1.0 / 27.0) * coeffs[1] * x25 * x401 +
             (1.0 / 27.0) * coeffs[21] * x376 * x383 +
             (1.0 / 27.0) * coeffs[24] * x337 * x377 +
             (1.0 / 27.0) * coeffs[27] * x346 * x376 +
             (1.0 / 27.0) * coeffs[2] * x401 * x[1] +
             (1.0 / 27.0) * coeffs[30] * x370 * x377 -
             1.0 / 27.0 * coeffs[32] * x350 * x359 -
             1.0 / 27.0 * coeffs[34] * x306 * x355 -
             1.0 / 27.0 * coeffs[35] * x333 * x351 -
             1.0 / 27.0 * coeffs[37] * x333 * x355 -
             1.0 / 27.0 * coeffs[38] * x351 * x352 +
             (1.0 / 27.0) * coeffs[39] * x352 * x354 +
             (1.0 / 27.0) * coeffs[3] * x340 * x400 -
             1.0 / 27.0 * coeffs[40] * x352 * x355 -
             1.0 / 27.0 * coeffs[41] * x340 * x351 +
             (1.0 / 27.0) * coeffs[42] * x340 * x354 -
             1.0 / 27.0 * coeffs[43] * x340 * x355 +
             (1.0 / 27.0) * coeffs[44] * x305 * x356 -
             1.0 / 27.0 * coeffs[45] * x310 * x357 +
             (1.0 / 27.0) * coeffs[46] * x314 * x356 -
             1.0 / 27.0 * coeffs[47] * x24 * x358 +
             (1.0 / 27.0) * coeffs[48] * x324 * x338 -
             1.0 / 27.0 * coeffs[49] * x358 * x48 +
             (1.0 / 27.0) * coeffs[50] * x329 * x356 -
             1.0 / 27.0 * coeffs[51] * x330 * x357 +
             (1.0 / 27.0) * coeffs[52] * x332 * x356 +
             (1.0 / 27.0) * coeffs[53] * x305 * x334 -
             1.0 / 27.0 * coeffs[54] * x310 * x335 +
             (1.0 / 27.0) * coeffs[55] * x314 * x334 -
             1.0 / 27.0 * coeffs[56] * x24 * x336 +
             (1.0 / 27.0) * coeffs[57] * x319 * x337 * x338 -
             1.0 / 27.0 * coeffs[58] * x336 * x48 +
             (1.0 / 27.0) * coeffs[59] * x329 * x334 -
             1.0 / 27.0 * coeffs[60] * x330 * x335 +
             (1.0 / 27.0) * coeffs[61] * x332 * x334 +
             (1.0 / 27.0) * coeffs[62] * x32 * x360 +
             (1.0 / 27.0) * coeffs[64] * x360 * x9 +
             (1.0 / 27.0) * coeffs[68] * x339 * x362 -
             1.0 / 27.0 * coeffs[69] * x312 * x348 +
             (1.0 / 27.0) * coeffs[70] * x343 * x362 +
             (1.0 / 27.0) * coeffs[71] * x339 * x341 -
             1.0 / 27.0 * coeffs[72] * x304 * x342 * x42 +
             (1.0 / 27.0) * coeffs[73] * x341 * x343 +
             (16.0 / 3.0) * coeffs[75] * x319 * x346 +
             (1.0 / 27.0) * coeffs[77] * x339 * x347 -
             1.0 / 27.0 * coeffs[78] * x345 * x348 +
             (1.0 / 27.0) * coeffs[79] * x343 * x347 +
             (1.0 / 27.0) * coeffs[80] * x32 * x388 -
             1.0 / 27.0 * coeffs[81] * x24 * x390 +
             (1.0 / 27.0) * coeffs[82] * x388 * x9 +
             (1.0 / 27.0) * coeffs[84] * x372 * x386 +
             (1.0 / 27.0) * coeffs[86] * x309 * x392 -
             1.0 / 27.0 * coeffs[87] * x390 * x48 +
             (1.0 / 27.0) * coeffs[88] * x368 * x392 +
             (1.0 / 27.0) * coeffs[93] * x363 * x372 -
             1.0 / 27.0 * coeffs[98] * x305 * x315 +
             (1.0 / 27.0) * coeffs[9] * x383 * x396 +
             (1.0 / 27.0) * x101 * x369 + (1.0 / 27.0) * x105 * x365 -
             1.0 / 27.0 * x106 * x312 * x366 + (1.0 / 27.0) * x107 * x369 -
             1.0 / 27.0 * x116 * x316 + (1.0 / 27.0) * x118 * x323 * x81 -
             1.0 / 27.0 * x145 * x375 - 1.0 / 27.0 * x147 * x375 -
             1.0 / 27.0 * x153 * x395 + (1.0 / 27.0) * x157 * x333 * x353 -
             1.0 / 27.0 * x169 * x384 - 1.0 / 27.0 * x170 * x384 -
             1.0 / 27.0 * x173 * x403 + (1.0 / 27.0) * x174 * x320 * x349 +
             (1.0 / 27.0) * x192 * x306 * x311 - 1.0 / 27.0 * x203 * x325 -
             1.0 / 27.0 * x206 * x325 + (1.0 / 27.0) * x211 * x331 +
             (1.0 / 27.0) * x213 * x331 - 1.0 / 27.0 * x227 * x366 * x367 -
             1.0 / 27.0 * x231 * x371 - 1.0 / 27.0 * x233 * x371 -
             1.0 / 27.0 * x24 * x323 * x60 - 1.0 / 27.0 * x247 * x391 -
             1.0 / 27.0 * x248 * x391 - 1.0 / 27.0 * x257 * x344 -
             1.0 / 27.0 * x258 * x344 - 1.0 / 27.0 * x263 * x379 -
             1.0 / 27.0 * x265 * x379 + (1.0 / 27.0) * x272 * x381 -
             1.0 / 27.0 * x273 * x399 - 1.0 / 27.0 * x275 * x399 +
             (1.0 / 27.0) * x277 * x381 - 1.0 / 27.0 * x283 * x361 -
             1.0 / 27.0 * x284 * x361 - 1.0 / 27.0 * x287 * x382 -
             1.0 / 27.0 * x289 * x382 + (1.0 / 27.0) * x294 * x385 -
             1.0 / 27.0 * x295 * x402 - 1.0 / 27.0 * x298 * x402 +
             (1.0 / 27.0) * x299 * x385 + (1.0 / 27.0) * x312 * x332 * x78 +
             (1.0 / 27.0) * x317 * x52 + (1.0 / 27.0) * x317 * x79 +
             (1.0 / 27.0) * x323 * x38 * x63 - 1.0 / 27.0 * x323 * x48 * x67 +
             (1.0 / 27.0) * x365 * x92;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = 4 * x[1];
    const Scalar x2 = x1 - 1;
    const Scalar x3 = 2 * x[1];
    const Scalar x4 = x3 - 1;
    const Scalar x5 = x1 - 3;
    const Scalar x6 = x4 * x5;
    const Scalar x7 = x2 * x6;
    const Scalar x8 = (1.0 / 27.0) * x7;
    const Scalar x9 = x0 * x8;
    const Scalar x10 = x[0] - 1;
    const Scalar x11 = 2 * x[0];
    const Scalar x12 = x11 - 1;
    const Scalar x13 = 4 * x[0];
    const Scalar x14 = x13 - 3;
    const Scalar x15 = x12 * x14;
    const Scalar x16 = x10 * x15;
    const Scalar x17 = x13 - 1;
    const Scalar x18 = x12 * x17;
    const Scalar x19 = x10 * x18;
    const Scalar x20 = x14 * x17;
    const Scalar x21 = x10 * x20;
    const Scalar x22 = x15 * x17;
    const Scalar x23 = 4 * x16 + 4 * x19 + 2 * x21 + x22;
    const Scalar x24 = x[2] - 1;
    const Scalar x25 = 4 * x[2];
    const Scalar x26 = x25 - 1;
    const Scalar x27 = 2 * x[2];
    const Scalar x28 = x27 - 1;
    const Scalar x29 = x25 - 3;
    const Scalar x30 = x28 * x29;
    const Scalar x31 = x26 * x30;
    const Scalar x32 = x24 * x31;
    const Scalar x33 = x23 * x32;
    const Scalar x34 = (1.0 / 27.0) * x32;
    const Scalar x35 = x0 * x6;
    const Scalar x36 = x2 * x4;
    const Scalar x37 = x0 * x36;
    const Scalar x38 = x2 * x5;
    const Scalar x39 = x0 * x38;
    const Scalar x40 = 4 * x35 + 4 * x37 + 2 * x39 + x7;
    const Scalar x41 = x10 * x22;
    const Scalar x42 = x40 * x41;
    const Scalar x43 = x24 * x30;
    const Scalar x44 = x26 * x28;
    const Scalar x45 = x24 * x44;
    const Scalar x46 = x26 * x29;
    const Scalar x47 = x24 * x46;
    const Scalar x48 = x31 + 4 * x43 + 4 * x45 + 2 * x47;
    const Scalar x49 = x48 * x9;
    const Scalar x50 = x11 * x20 + x13 * x15 + x13 * x18 + x22;
    const Scalar x51 = x32 * x50;
    const Scalar x52 = x22 * x[0];
    const Scalar x53 = x34 * x52;
    const Scalar x54 = x8 * x[1];
    const Scalar x55 = x1 * x36 + x1 * x6 + x3 * x38 + x7;
    const Scalar x56 = x48 * x54;
    const Scalar x57 = x41 * x55;
    const Scalar x58 = x31 * x[2];
    const Scalar x59 = x58 * x9;
    const Scalar x60 = (1.0 / 27.0) * x58;
    const Scalar x61 = x25 * x30 + x25 * x44 + x27 * x46 + x31;
    const Scalar x62 = x61 * x9;
    const Scalar x63 = x52 * x60;
    const Scalar x64 = x54 * x58;
    const Scalar x65 = x54 * x61;
    const Scalar x66 = (16.0 / 27.0) * x32;
    const Scalar x67 = x10 * x13;
    const Scalar x68 = x12 * x67;
    const Scalar x69 = x10 * x14;
    const Scalar x70 = x15 * x[0];
    const Scalar x71 = x11 * x69 + x16 + x68 + x70;
    const Scalar x72 = x0 * x7;
    const Scalar x73 = x71 * x72;
    const Scalar x74 = x10 * x70;
    const Scalar x75 = x40 * x66;
    const Scalar x76 = x48 * x72;
    const Scalar x77 = (16.0 / 27.0) * x74;
    const Scalar x78 = (4.0 / 9.0) * x32;
    const Scalar x79 = x20 * x[0];
    const Scalar x80 = x13 * x69 + x17 * x67 + x21 + x79;
    const Scalar x81 = x72 * x80;
    const Scalar x82 = x10 * x79;
    const Scalar x83 = x78 * x82;
    const Scalar x84 = (4.0 / 9.0) * x82;
    const Scalar x85 = x18 * x[0];
    const Scalar x86 = x10 * x11 * x17 + x19 + x68 + x85;
    const Scalar x87 = x66 * x86;
    const Scalar x88 = x10 * x85;
    const Scalar x89 = (16.0 / 27.0) * x88;
    const Scalar x90 = x0 * x51;
    const Scalar x91 = x6 * x[1];
    const Scalar x92 = (16.0 / 27.0) * x91;
    const Scalar x93 = x0 * x1;
    const Scalar x94 = x4 * x93;
    const Scalar x95 = x0 * x5;
    const Scalar x96 = x3 * x95 + x35 + x91 + x94;
    const Scalar x97 = x52 * x66;
    const Scalar x98 = x0 * x92;
    const Scalar x99 = x48 * x52;
    const Scalar x100 = x38 * x[1];
    const Scalar x101 = (4.0 / 9.0) * x100;
    const Scalar x102 = x1 * x95 + x100 + x2 * x93 + x39;
    const Scalar x103 = x102 * x78;
    const Scalar x104 = x0 * x99;
    const Scalar x105 = x36 * x[1];
    const Scalar x106 = (16.0 / 27.0) * x105;
    const Scalar x107 = x0 * x2 * x3 + x105 + x37 + x94;
    const Scalar x108 = x7 * x[1];
    const Scalar x109 = x108 * x71;
    const Scalar x110 = x55 * x66;
    const Scalar x111 = x108 * x48;
    const Scalar x112 = x108 * x80;
    const Scalar x113 = x41 * x66;
    const Scalar x114 = x41 * x48;
    const Scalar x115 = x0 * x101;
    const Scalar x116 = x0 * x106;
    const Scalar x117 = (16.0 / 27.0) * x58;
    const Scalar x118 = x40 * x58;
    const Scalar x119 = x61 * x72;
    const Scalar x120 = (4.0 / 9.0) * x58;
    const Scalar x121 = x117 * x86;
    const Scalar x122 = x50 * x58;
    const Scalar x123 = x117 * x52;
    const Scalar x124 = x52 * x61;
    const Scalar x125 = x102 * x120;
    const Scalar x126 = x55 * x58;
    const Scalar x127 = x108 * x61;
    const Scalar x128 = x23 * x58;
    const Scalar x129 = x117 * x41;
    const Scalar x130 = x41 * x61;
    const Scalar x131 = (16.0 / 27.0) * x72;
    const Scalar x132 = x30 * x[2];
    const Scalar x133 = x23 * x24;
    const Scalar x134 = x132 * x133;
    const Scalar x135 = x24 * x42;
    const Scalar x136 = (16.0 / 27.0) * x135;
    const Scalar x137 = x24 * x25;
    const Scalar x138 = x137 * x28;
    const Scalar x139 = x24 * x29;
    const Scalar x140 = x132 + x138 + x139 * x27 + x43;
    const Scalar x141 = x131 * x41;
    const Scalar x142 = x46 * x[2];
    const Scalar x143 = (4.0 / 9.0) * x142;
    const Scalar x144 = x143 * x72;
    const Scalar x145 = x137 * x26 + x139 * x25 + x142 + x47;
    const Scalar x146 = (4.0 / 9.0) * x145;
    const Scalar x147 = x146 * x72;
    const Scalar x148 = x44 * x[2];
    const Scalar x149 = x131 * x148;
    const Scalar x150 = x138 + x148 + x24 * x26 * x27 + x45;
    const Scalar x151 = x132 * x24;
    const Scalar x152 = x151 * x50;
    const Scalar x153 = x40 * x52;
    const Scalar x154 = (16.0 / 27.0) * x151;
    const Scalar x155 = x131 * x52;
    const Scalar x156 = x24 * x50;
    const Scalar x157 = x153 * x24;
    const Scalar x158 = (16.0 / 27.0) * x148;
    const Scalar x159 = (16.0 / 27.0) * x108;
    const Scalar x160 = x52 * x55;
    const Scalar x161 = x159 * x52;
    const Scalar x162 = x108 * x156;
    const Scalar x163 = x160 * x24;
    const Scalar x164 = x108 * x146;
    const Scalar x165 = x159 * x41;
    const Scalar x166 = x108 * x133;
    const Scalar x167 = x24 * x57;
    const Scalar x168 = x0 * x134;
    const Scalar x169 = (256.0 / 27.0) * x91;
    const Scalar x170 = x151 * x41;
    const Scalar x171 = (256.0 / 27.0) * x96;
    const Scalar x172 = x0 * x41;
    const Scalar x173 = x140 * x172;
    const Scalar x174 = (64.0 / 9.0) * x100;
    const Scalar x175 = (64.0 / 9.0) * x102;
    const Scalar x176 = (256.0 / 27.0) * x105;
    const Scalar x177 = (256.0 / 27.0) * x107;
    const Scalar x178 = (64.0 / 9.0) * x142;
    const Scalar x179 = x0 * x133;
    const Scalar x180 = x178 * x179;
    const Scalar x181 = x24 * x41;
    const Scalar x182 = x178 * x181;
    const Scalar x183 = x145 * x172;
    const Scalar x184 = (64.0 / 9.0) * x91;
    const Scalar x185 = (16.0 / 3.0) * x142;
    const Scalar x186 = x100 * x185;
    const Scalar x187 = x102 * x185;
    const Scalar x188 = (16.0 / 3.0) * x100;
    const Scalar x189 = (64.0 / 9.0) * x105;
    const Scalar x190 = x148 * x179;
    const Scalar x191 = x148 * x181;
    const Scalar x192 = x150 * x172;
    const Scalar x193 = x0 * x169;
    const Scalar x194 = x151 * x52;
    const Scalar x195 = x140 * x52;
    const Scalar x196 = x0 * x174;
    const Scalar x197 = x0 * x176;
    const Scalar x198 = x0 * x156;
    const Scalar x199 = x178 * x198;
    const Scalar x200 = x24 * x52;
    const Scalar x201 = x178 * x200;
    const Scalar x202 = x0 * x184;
    const Scalar x203 = x145 * x52;
    const Scalar x204 = x0 * x203;
    const Scalar x205 = x148 * x156;
    const Scalar x206 = x148 * x200;
    const Scalar x207 = x150 * x52;
    const Scalar x208 = (256.0 / 27.0) * x151;
    const Scalar x209 = x208 * x40;
    const Scalar x210 = x140 * x72;
    const Scalar x211 = (256.0 / 27.0) * x74;
    const Scalar x212 = (64.0 / 9.0) * x151;
    const Scalar x213 = x40 * x82;
    const Scalar x214 = (64.0 / 9.0) * x82;
    const Scalar x215 = x72 * x86;
    const Scalar x216 = (256.0 / 27.0) * x88;
    const Scalar x217 = x178 * x24;
    const Scalar x218 = x217 * x40;
    const Scalar x219 = x145 * x72;
    const Scalar x220 = (64.0 / 9.0) * x219;
    const Scalar x221 = x185 * x24;
    const Scalar x222 = (16.0 / 3.0) * x82;
    const Scalar x223 = x148 * x24;
    const Scalar x224 = (256.0 / 27.0) * x223;
    const Scalar x225 = x223 * x40;
    const Scalar x226 = x150 * x72;
    const Scalar x227 = (64.0 / 9.0) * x223;
    const Scalar x228 = x208 * x55;
    const Scalar x229 = x108 * x140;
    const Scalar x230 = x55 * x82;
    const Scalar x231 = x108 * x86;
    const Scalar x232 = x217 * x55;
    const Scalar x233 = x108 * x145;
    const Scalar x234 = (64.0 / 9.0) * x233;
    const Scalar x235 = x223 * x55;
    const Scalar x236 = x108 * x150;
    const Scalar x237 = x193 * x32;
    const Scalar x238 = x171 * x32;
    const Scalar x239 = x193 * x48;
    const Scalar x240 = x32 * x80;
    const Scalar x241 = x214 * x32;
    const Scalar x242 = x48 * x82;
    const Scalar x243 = x196 * x32;
    const Scalar x244 = x175 * x32;
    const Scalar x245 = x196 * x48;
    const Scalar x246 = x0 * x188;
    const Scalar x247 = x102 * x222;
    const Scalar x248 = x197 * x32;
    const Scalar x249 = x177 * x32;
    const Scalar x250 = x197 * x48;
    const Scalar x251 = x0 * x189;
    const Scalar x252 = x193 * x58;
    const Scalar x253 = x171 * x58;
    const Scalar x254 = x193 * x61;
    const Scalar x255 = x58 * x80;
    const Scalar x256 = x214 * x58;
    const Scalar x257 = x61 * x82;
    const Scalar x258 = x196 * x58;
    const Scalar x259 = x175 * x58;
    const Scalar x260 = x196 * x61;
    const Scalar x261 = x197 * x58;
    const Scalar x262 = x177 * x58;
    const Scalar x263 = x197 * x61;
    const Scalar x264 = (4096.0 / 27.0) * x151;
    const Scalar x265 = x0 * x91;
    const Scalar x266 = x265 * x71;
    const Scalar x267 = x264 * x96;
    const Scalar x268 = x140 * x265;
    const Scalar x269 = (4096.0 / 27.0) * x74;
    const Scalar x270 = (1024.0 / 9.0) * x151;
    const Scalar x271 = x265 * x80;
    const Scalar x272 = x270 * x82;
    const Scalar x273 = (1024.0 / 9.0) * x82;
    const Scalar x274 = x264 * x86;
    const Scalar x275 = (4096.0 / 27.0) * x88;
    const Scalar x276 = x0 * x100;
    const Scalar x277 = x270 * x276;
    const Scalar x278 = x102 * x270;
    const Scalar x279 = x140 * x276;
    const Scalar x280 = (1024.0 / 9.0) * x74;
    const Scalar x281 = (256.0 / 3.0) * x151;
    const Scalar x282 = x276 * x80;
    const Scalar x283 = x102 * x82;
    const Scalar x284 = (256.0 / 3.0) * x82;
    const Scalar x285 = (1024.0 / 9.0) * x88;
    const Scalar x286 = x0 * x105;
    const Scalar x287 = x286 * x71;
    const Scalar x288 = x107 * x264;
    const Scalar x289 = x140 * x286;
    const Scalar x290 = x286 * x80;
    const Scalar x291 = x142 * x24;
    const Scalar x292 = (1024.0 / 9.0) * x291;
    const Scalar x293 = x291 * x96;
    const Scalar x294 = x145 * x265;
    const Scalar x295 = (256.0 / 3.0) * x291;
    const Scalar x296 = x292 * x86;
    const Scalar x297 = x276 * x295;
    const Scalar x298 = x102 * x295;
    const Scalar x299 = x145 * x276;
    const Scalar x300 = (256.0 / 3.0) * x299;
    const Scalar x301 = 64 * x291;
    const Scalar x302 = x107 * x291;
    const Scalar x303 = x145 * x286;
    const Scalar x304 = (4096.0 / 27.0) * x223;
    const Scalar x305 = x223 * x96;
    const Scalar x306 = x150 * x265;
    const Scalar x307 = (1024.0 / 9.0) * x223;
    const Scalar x308 = x304 * x86;
    const Scalar x309 = x276 * x307;
    const Scalar x310 = x102 * x223;
    const Scalar x311 = x150 * x276;
    const Scalar x312 = (256.0 / 3.0) * x223;
    const Scalar x313 = x107 * x223;
    const Scalar x314 = x150 * x286;
    out[0][0] = x33 * x9;
    out[0][1] = x34 * x42;
    out[0][2] = x41 * x49;
    out[1][0] = x51 * x9;
    out[1][1] = x40 * x53;
    out[1][2] = x49 * x52;
    out[2][0] = x51 * x54;
    out[2][1] = x53 * x55;
    out[2][2] = x52 * x56;
    out[3][0] = x33 * x54;
    out[3][1] = x34 * x57;
    out[3][2] = x41 * x56;
    out[4][0] = x23 * x59;
    out[4][1] = x42 * x60;
    out[4][2] = x41 * x62;
    out[5][0] = x50 * x59;
    out[5][1] = x40 * x63;
    out[5][2] = x52 * x62;
    out[6][0] = x50 * x64;
    out[6][1] = x55 * x63;
    out[6][2] = x52 * x65;
    out[7][0] = x23 * x64;
    out[7][1] = x57 * x60;
    out[7][2] = x41 * x65;
    out[8][0] = -x66 * x73;
    out[8][1] = -x74 * x75;
    out[8][2] = -x76 * x77;
    out[9][0] = x78 * x81;
    out[9][1] = x40 * x83;
    out[9][2] = x76 * x84;
    out[10][0] = -x72 * x87;
    out[10][1] = -x75 * x88;
    out[10][2] = -x76 * x89;
    out[11][0] = -x90 * x92;
    out[11][1] = -x96 * x97;
    out[11][2] = -x98 * x99;
    out[12][0] = x101 * x90;
    out[12][1] = x103 * x52;
    out[12][2] = x101 * x104;
    out[13][0] = -x106 * x90;
    out[13][1] = -x107 * x97;
    out[13][2] = -x104 * x106;
    out[14][0] = -x109 * x66;
    out[14][1] = -x110 * x74;
    out[14][2] = -x111 * x77;
    out[15][0] = x112 * x78;
    out[15][1] = x55 * x83;
    out[15][2] = x111 * x84;
    out[16][0] = -x108 * x87;
    out[16][1] = -x110 * x88;
    out[16][2] = -x111 * x89;
    out[17][0] = -x33 * x98;
    out[17][1] = -x113 * x96;
    out[17][2] = -x114 * x98;
    out[18][0] = x115 * x33;
    out[18][1] = x103 * x41;
    out[18][2] = x114 * x115;
    out[19][0] = -x116 * x33;
    out[19][1] = -x107 * x113;
    out[19][2] = -x114 * x116;
    out[20][0] = -x117 * x73;
    out[20][1] = -x118 * x77;
    out[20][2] = -x119 * x77;
    out[21][0] = x120 * x81;
    out[21][1] = x118 * x84;
    out[21][2] = x119 * x84;
    out[22][0] = -x121 * x72;
    out[22][1] = -x118 * x89;
    out[22][2] = -x119 * x89;
    out[23][0] = -x122 * x98;
    out[23][1] = -x123 * x96;
    out[23][2] = -x124 * x98;
    out[24][0] = x115 * x122;
    out[24][1] = x125 * x52;
    out[24][2] = x115 * x124;
    out[25][0] = -x116 * x122;
    out[25][1] = -x107 * x123;
    out[25][2] = -x116 * x124;
    out[26][0] = -x109 * x117;
    out[26][1] = -x126 * x77;
    out[26][2] = -x127 * x77;
    out[27][0] = x112 * x120;
    out[27][1] = x126 * x84;
    out[27][2] = x127 * x84;
    out[28][0] = -x108 * x121;
    out[28][1] = -x126 * x89;
    out[28][2] = -x127 * x89;
    out[29][0] = -x128 * x98;
    out[29][1] = -x129 * x96;
    out[29][2] = -x130 * x98;
    out[30][0] = x115 * x128;
    out[30][1] = x125 * x41;
    out[30][2] = x115 * x130;
    out[31][0] = -x116 * x128;
    out[31][1] = -x107 * x129;
    out[31][2] = -x116 * x130;
    out[32][0] = -x131 * x134;
    out[32][1] = -x132 * x136;
    out[32][2] = -x140 * x141;
    out[33][0] = x133 * x144;
    out[33][1] = x135 * x143;
    out[33][2] = x147 * x41;
    out[34][0] = -x133 * x149;
    out[34][1] = -x136 * x148;
    out[34][2] = -x141 * x150;
    out[35][0] = -x131 * x152;
    out[35][1] = -x153 * x154;
    out[35][2] = -x140 * x155;
    out[36][0] = x144 * x156;
    out[36][1] = x143 * x157;
    out[36][2] = x147 * x52;
    out[37][0] = -x149 * x156;
    out[37][1] = -x157 * x158;
    out[37][2] = -x150 * x155;
    out[38][0] = -x152 * x159;
    out[38][1] = -x154 * x160;
    out[38][2] = -x140 * x161;
    out[39][0] = x143 * x162;
    out[39][1] = x143 * x163;
    out[39][2] = x164 * x52;
    out[40][0] = -x158 * x162;
    out[40][1] = -x158 * x163;
    out[40][2] = -x150 * x161;
    out[41][0] = -x134 * x159;
    out[41][1] = -x154 * x57;
    out[41][2] = -x140 * x165;
    out[42][0] = x143 * x166;
    out[42][1] = x143 * x167;
    out[42][2] = x164 * x41;
    out[43][0] = -x158 * x166;
    out[43][1] = -x158 * x167;
    out[43][2] = -x150 * x165;
    out[44][0] = x168 * x169;
    out[44][1] = x170 * x171;
    out[44][2] = x169 * x173;
    out[45][0] = -x168 * x174;
    out[45][1] = -x170 * x175;
    out[45][2] = -x173 * x174;
    out[46][0] = x168 * x176;
    out[46][1] = x170 * x177;
    out[46][2] = x173 * x176;
    out[47][0] = -x180 * x91;
    out[47][1] = -x182 * x96;
    out[47][2] = -x183 * x184;
    out[48][0] = x179 * x186;
    out[48][1] = x181 * x187;
    out[48][2] = x183 * x188;
    out[49][0] = -x105 * x180;
    out[49][1] = -x107 * x182;
    out[49][2] = -x183 * x189;
    out[50][0] = x169 * x190;
    out[50][1] = x171 * x191;
    out[50][2] = x169 * x192;
    out[51][0] = -x174 * x190;
    out[51][1] = -x175 * x191;
    out[51][2] = -x174 * x192;
    out[52][0] = x176 * x190;
    out[52][1] = x177 * x191;
    out[52][2] = x176 * x192;
    out[53][0] = x152 * x193;
    out[53][1] = x171 * x194;
    out[53][2] = x193 * x195;
    out[54][0] = -x152 * x196;
    out[54][1] = -x175 * x194;
    out[54][2] = -x195 * x196;
    out[55][0] = x152 * x197;
    out[55][1] = x177 * x194;
    out[55][2] = x195 * x197;
    out[56][0] = -x199 * x91;
    out[56][1] = -x201 * x96;
    out[56][2] = -x202 * x203;
    out[57][0] = x186 * x198;
    out[57][1] = x187 * x200;
    out[57][2] = x188 * x204;
    out[58][0] = -x105 * x199;
    out[58][1] = -x107 * x201;
    out[58][2] = -x189 * x204;
    out[59][0] = x193 * x205;
    out[59][1] = x171 * x206;
    out[59][2] = x193 * x207;
    out[60][0] = -x196 * x205;
    out[60][1] = -x175 * x206;
    out[60][2] = -x196 * x207;
    out[61][0] = x197 * x205;
    out[61][1] = x177 * x206;
    out[61][2] = x197 * x207;
    out[62][0] = x208 * x73;
    out[62][1] = x209 * x74;
    out[62][2] = x210 * x211;
    out[63][0] = -x212 * x81;
    out[63][1] = -x212 * x213;
    out[63][2] = -x210 * x214;
    out[64][0] = x208 * x215;
    out[64][1] = x209 * x88;
    out[64][2] = x210 * x216;
    out[65][0] = -x217 * x73;
    out[65][1] = -x218 * x74;
    out[65][2] = -x220 * x74;
    out[66][0] = x221 * x81;
    out[66][1] = x213 * x221;
    out[66][2] = x219 * x222;
    out[67][0] = -x215 * x217;
    out[67][1] = -x218 * x88;
    out[67][2] = -x220 * x88;
    out[68][0] = x224 * x73;
    out[68][1] = x211 * x225;
    out[68][2] = x211 * x226;
    out[69][0] = -x227 * x81;
    out[69][1] = -x213 * x227;
    out[69][2] = -x214 * x226;
    out[70][0] = x215 * x224;
    out[70][1] = x216 * x225;
    out[70][2] = x216 * x226;
    out[71][0] = x109 * x208;
    out[71][1] = x228 * x74;
    out[71][2] = x211 * x229;
    out[72][0] = -x112 * x212;
    out[72][1] = -x212 * x230;
    out[72][2] = -x214 * x229;
    out[73][0] = x208 * x231;
    out[73][1] = x228 * x88;
    out[73][2] = x216 * x229;
    out[74][0] = -x109 * x217;
    out[74][1] = -x232 * x74;
    out[74][2] = -x234 * x74;
    out[75][0] = x112 * x221;
    out[75][1] = x221 * x230;
    out[75][2] = x222 * x233;
    out[76][0] = -x217 * x231;
    out[76][1] = -x232 * x88;
    out[76][2] = -x234 * x88;
    out[77][0] = x109 * x224;
    out[77][1] = x211 * x235;
    out[77][2] = x211 * x236;
    out[78][0] = -x112 * x227;
    out[78][1] = -x214 * x235;
    out[78][2] = -x214 * x236;
    out[79][0] = x224 * x231;
    out[79][1] = x216 * x235;
    out[79][2] = x216 * x236;
    out[80][0] = x237 * x71;
    out[80][1] = x238 * x74;
    out[80][2] = x239 * x74;
    out[81][0] = -x202 * x240;
    out[81][1] = -x241 * x96;
    out[81][2] = -x202 * x242;
    out[82][0] = x237 * x86;
    out[82][1] = x238 * x88;
    out[82][2] = x239 * x88;
    out[83][0] = -x243 * x71;
    out[83][1] = -x244 * x74;
    out[83][2] = -x245 * x74;
    out[84][0] = x240 * x246;
    out[84][1] = x247 * x32;
    out[84][2] = x242 * x246;
    out[85][0] = -x243 * x86;
    out[85][1] = -x244 * x88;
    out[85][2] = -x245 * x88;
    out[86][0] = x248 * x71;
    out[86][1] = x249 * x74;
    out[86][2] = x250 * x74;
    out[87][0] = -x240 * x251;
    out[87][1] = -x107 * x241;
    out[87][2] = -x242 * x251;
    out[88][0] = x248 * x86;
    out[88][1] = x249 * x88;
    out[88][2] = x250 * x88;
    out[89][0] = x252 * x71;
    out[89][1] = x253 * x74;
    out[89][2] = x254 * x74;
    out[90][0] = -x202 * x255;
    out[90][1] = -x256 * x96;
    out[90][2] = -x202 * x257;
    out[91][0] = x252 * x86;
    out[91][1] = x253 * x88;
    out[91][2] = x254 * x88;
    out[92][0] = -x258 * x71;
    out[92][1] = -x259 * x74;
    out[92][2] = -x260 * x74;
    out[93][0] = x246 * x255;
    out[93][1] = x247 * x58;
    out[93][2] = x246 * x257;
    out[94][0] = -x258 * x86;
    out[94][1] = -x259 * x88;
    out[94][2] = -x260 * x88;
    out[95][0] = x261 * x71;
    out[95][1] = x262 * x74;
    out[95][2] = x263 * x74;
    out[96][0] = -x251 * x255;
    out[96][1] = -x107 * x256;
    out[96][2] = -x251 * x257;
    out[97][0] = x261 * x86;
    out[97][1] = x262 * x88;
    out[97][2] = x263 * x88;
    out[98][0] = -x264 * x266;
    out[98][1] = -x267 * x74;
    out[98][2] = -x268 * x269;
    out[99][0] = x270 * x271;
    out[99][1] = x272 * x96;
    out[99][2] = x268 * x273;
    out[100][0] = -x265 * x274;
    out[100][1] = -x267 * x88;
    out[100][2] = -x268 * x275;
    out[101][0] = x277 * x71;
    out[101][1] = x278 * x74;
    out[101][2] = x279 * x280;
    out[102][0] = -x281 * x282;
    out[102][1] = -x281 * x283;
    out[102][2] = -x279 * x284;
    out[103][0] = x277 * x86;
    out[103][1] = x278 * x88;
    out[103][2] = x279 * x285;
    out[104][0] = -x264 * x287;
    out[104][1] = -x288 * x74;
    out[104][2] = -x269 * x289;
    out[105][0] = x270 * x290;
    out[105][1] = x107 * x272;
    out[105][2] = x273 * x289;
    out[106][0] = -x274 * x286;
    out[106][1] = -x288 * x88;
    out[106][2] = -x275 * x289;
    out[107][0] = x266 * x292;
    out[107][1] = x280 * x293;
    out[107][2] = x280 * x294;
    out[108][0] = -x271 * x295;
    out[108][1] = -x284 * x293;
    out[108][2] = -x284 * x294;
    out[109][0] = x265 * x296;
    out[109][1] = x285 * x293;
    out[109][2] = x285 * x294;
    out[110][0] = -x297 * x71;
    out[110][1] = -x298 * x74;
    out[110][2] = -x300 * x74;
    out[111][0] = x282 * x301;
    out[111][1] = x283 * x301;
    out[111][2] = 64 * x299 * x82;
    out[112][0] = -x297 * x86;
    out[112][1] = -x298 * x88;
    out[112][2] = -x300 * x88;
    out[113][0] = x287 * x292;
    out[113][1] = x280 * x302;
    out[113][2] = x280 * x303;
    out[114][0] = -x290 * x295;
    out[114][1] = -x284 * x302;
    out[114][2] = -x284 * x303;
    out[115][0] = x286 * x296;
    out[115][1] = x285 * x302;
    out[115][2] = x285 * x303;
    out[116][0] = -x266 * x304;
    out[116][1] = -x269 * x305;
    out[116][2] = -x269 * x306;
    out[117][0] = x271 * x307;
    out[117][1] = x273 * x305;
    out[117][2] = x273 * x306;
    out[118][0] = -x265 * x308;
    out[118][1] = -x275 * x305;
    out[118][2] = -x275 * x306;
    out[119][0] = x309 * x71;
    out[119][1] = x280 * x310;
    out[119][2] = x280 * x311;
    out[120][0] = -x282 * x312;
    out[120][1] = -x283 * x312;
    out[120][2] = -x284 * x311;
    out[121][0] = x309 * x86;
    out[121][1] = x285 * x310;
    out[121][2] = x285 * x311;
    out[122][0] = -x287 * x304;
    out[122][1] = -x269 * x313;
    out[122][2] = -x269 * x314;
    out[123][0] = x290 * x307;
    out[123][1] = x273 * x313;
    out[123][2] = x273 * x314;
    out[124][0] = -x286 * x308;
    out[124][1] = -x275 * x313;
    out[124][2] = -x275 * x314;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[3]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[0][2] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(4) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[1][2] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(4) / order;
    out[2][1] = static_cast<Scalar>(4) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(4) / order;
    out[3][2] = static_cast<Scalar>(0) / order;
    out[4][0] = static_cast<Scalar>(0) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[4][2] = static_cast<Scalar>(4) / order;
    out[5][0] = static_cast<Scalar>(4) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[5][2] = static_cast<Scalar>(4) / order;
    out[6][0] = static_cast<Scalar>(4) / order;
    out[6][1] = static_cast<Scalar>(4) / order;
    out[6][2] = static_cast<Scalar>(4) / order;
    out[7][0] = static_cast<Scalar>(0) / order;
    out[7][1] = static_cast<Scalar>(4) / order;
    out[7][2] = static_cast<Scalar>(4) / order;
    out[8][0] = static_cast<Scalar>(1) / order;
    out[8][1] = static_cast<Scalar>(0) / order;
    out[8][2] = static_cast<Scalar>(0) / order;
    out[9][0] = static_cast<Scalar>(2) / order;
    out[9][1] = static_cast<Scalar>(0) / order;
    out[9][2] = static_cast<Scalar>(0) / order;
    out[10][0] = static_cast<Scalar>(3) / order;
    out[10][1] = static_cast<Scalar>(0) / order;
    out[10][2] = static_cast<Scalar>(0) / order;
    out[11][0] = static_cast<Scalar>(4) / order;
    out[11][1] = static_cast<Scalar>(1) / order;
    out[11][2] = static_cast<Scalar>(0) / order;
    out[12][0] = static_cast<Scalar>(4) / order;
    out[12][1] = static_cast<Scalar>(2) / order;
    out[12][2] = static_cast<Scalar>(0) / order;
    out[13][0] = static_cast<Scalar>(4) / order;
    out[13][1] = static_cast<Scalar>(3) / order;
    out[13][2] = static_cast<Scalar>(0) / order;
    out[14][0] = static_cast<Scalar>(1) / order;
    out[14][1] = static_cast<Scalar>(4) / order;
    out[14][2] = static_cast<Scalar>(0) / order;
    out[15][0] = static_cast<Scalar>(2) / order;
    out[15][1] = static_cast<Scalar>(4) / order;
    out[15][2] = static_cast<Scalar>(0) / order;
    out[16][0] = static_cast<Scalar>(3) / order;
    out[16][1] = static_cast<Scalar>(4) / order;
    out[16][2] = static_cast<Scalar>(0) / order;
    out[17][0] = static_cast<Scalar>(0) / order;
    out[17][1] = static_cast<Scalar>(1) / order;
    out[17][2] = static_cast<Scalar>(0) / order;
    out[18][0] = static_cast<Scalar>(0) / order;
    out[18][1] = static_cast<Scalar>(2) / order;
    out[18][2] = static_cast<Scalar>(0) / order;
    out[19][0] = static_cast<Scalar>(0) / order;
    out[19][1] = static_cast<Scalar>(3) / order;
    out[19][2] = static_cast<Scalar>(0) / order;
    out[20][0] = static_cast<Scalar>(1) / order;
    out[20][1] = static_cast<Scalar>(0) / order;
    out[20][2] = static_cast<Scalar>(4) / order;
    out[21][0] = static_cast<Scalar>(2) / order;
    out[21][1] = static_cast<Scalar>(0) / order;
    out[21][2] = static_cast<Scalar>(4) / order;
    out[22][0] = static_cast<Scalar>(3) / order;
    out[22][1] = static_cast<Scalar>(0) / order;
    out[22][2] = static_cast<Scalar>(4) / order;
    out[23][0] = static_cast<Scalar>(4) / order;
    out[23][1] = static_cast<Scalar>(1) / order;
    out[23][2] = static_cast<Scalar>(4) / order;
    out[24][0] = static_cast<Scalar>(4) / order;
    out[24][1] = static_cast<Scalar>(2) / order;
    out[24][2] = static_cast<Scalar>(4) / order;
    out[25][0] = static_cast<Scalar>(4) / order;
    out[25][1] = static_cast<Scalar>(3) / order;
    out[25][2] = static_cast<Scalar>(4) / order;
    out[26][0] = static_cast<Scalar>(1) / order;
    out[26][1] = static_cast<Scalar>(4) / order;
    out[26][2] = static_cast<Scalar>(4) / order;
    out[27][0] = static_cast<Scalar>(2) / order;
    out[27][1] = static_cast<Scalar>(4) / order;
    out[27][2] = static_cast<Scalar>(4) / order;
    out[28][0] = static_cast<Scalar>(3) / order;
    out[28][1] = static_cast<Scalar>(4) / order;
    out[28][2] = static_cast<Scalar>(4) / order;
    out[29][0] = static_cast<Scalar>(0) / order;
    out[29][1] = static_cast<Scalar>(1) / order;
    out[29][2] = static_cast<Scalar>(4) / order;
    out[30][0] = static_cast<Scalar>(0) / order;
    out[30][1] = static_cast<Scalar>(2) / order;
    out[30][2] = static_cast<Scalar>(4) / order;
    out[31][0] = static_cast<Scalar>(0) / order;
    out[31][1] = static_cast<Scalar>(3) / order;
    out[31][2] = static_cast<Scalar>(4) / order;
    out[32][0] = static_cast<Scalar>(0) / order;
    out[32][1] = static_cast<Scalar>(0) / order;
    out[32][2] = static_cast<Scalar>(1) / order;
    out[33][0] = static_cast<Scalar>(0) / order;
    out[33][1] = static_cast<Scalar>(0) / order;
    out[33][2] = static_cast<Scalar>(2) / order;
    out[34][0] = static_cast<Scalar>(0) / order;
    out[34][1] = static_cast<Scalar>(0) / order;
    out[34][2] = static_cast<Scalar>(3) / order;
    out[35][0] = static_cast<Scalar>(4) / order;
    out[35][1] = static_cast<Scalar>(0) / order;
    out[35][2] = static_cast<Scalar>(1) / order;
    out[36][0] = static_cast<Scalar>(4) / order;
    out[36][1] = static_cast<Scalar>(0) / order;
    out[36][2] = static_cast<Scalar>(2) / order;
    out[37][0] = static_cast<Scalar>(4) / order;
    out[37][1] = static_cast<Scalar>(0) / order;
    out[37][2] = static_cast<Scalar>(3) / order;
    out[38][0] = static_cast<Scalar>(4) / order;
    out[38][1] = static_cast<Scalar>(4) / order;
    out[38][2] = static_cast<Scalar>(1) / order;
    out[39][0] = static_cast<Scalar>(4) / order;
    out[39][1] = static_cast<Scalar>(4) / order;
    out[39][2] = static_cast<Scalar>(2) / order;
    out[40][0] = static_cast<Scalar>(4) / order;
    out[40][1] = static_cast<Scalar>(4) / order;
    out[40][2] = static_cast<Scalar>(3) / order;
    out[41][0] = static_cast<Scalar>(0) / order;
    out[41][1] = static_cast<Scalar>(4) / order;
    out[41][2] = static_cast<Scalar>(1) / order;
    out[42][0] = static_cast<Scalar>(0) / order;
    out[42][1] = static_cast<Scalar>(4) / order;
    out[42][2] = static_cast<Scalar>(2) / order;
    out[43][0] = static_cast<Scalar>(0) / order;
    out[43][1] = static_cast<Scalar>(4) / order;
    out[43][2] = static_cast<Scalar>(3) / order;
    out[44][0] = static_cast<Scalar>(0) / order;
    out[44][1] = static_cast<Scalar>(1) / order;
    out[44][2] = static_cast<Scalar>(1) / order;
    out[45][0] = static_cast<Scalar>(0) / order;
    out[45][1] = static_cast<Scalar>(2) / order;
    out[45][2] = static_cast<Scalar>(1) / order;
    out[46][0] = static_cast<Scalar>(0) / order;
    out[46][1] = static_cast<Scalar>(3) / order;
    out[46][2] = static_cast<Scalar>(1) / order;
    out[47][0] = static_cast<Scalar>(0) / order;
    out[47][1] = static_cast<Scalar>(1) / order;
    out[47][2] = static_cast<Scalar>(2) / order;
    out[48][0] = static_cast<Scalar>(0) / order;
    out[48][1] = static_cast<Scalar>(2) / order;
    out[48][2] = static_cast<Scalar>(2) / order;
    out[49][0] = static_cast<Scalar>(0) / order;
    out[49][1] = static_cast<Scalar>(3) / order;
    out[49][2] = static_cast<Scalar>(2) / order;
    out[50][0] = static_cast<Scalar>(0) / order;
    out[50][1] = static_cast<Scalar>(1) / order;
    out[50][2] = static_cast<Scalar>(3) / order;
    out[51][0] = static_cast<Scalar>(0) / order;
    out[51][1] = static_cast<Scalar>(2) / order;
    out[51][2] = static_cast<Scalar>(3) / order;
    out[52][0] = static_cast<Scalar>(0) / order;
    out[52][1] = static_cast<Scalar>(3) / order;
    out[52][2] = static_cast<Scalar>(3) / order;
    out[53][0] = static_cast<Scalar>(4) / order;
    out[53][1] = static_cast<Scalar>(1) / order;
    out[53][2] = static_cast<Scalar>(1) / order;
    out[54][0] = static_cast<Scalar>(4) / order;
    out[54][1] = static_cast<Scalar>(2) / order;
    out[54][2] = static_cast<Scalar>(1) / order;
    out[55][0] = static_cast<Scalar>(4) / order;
    out[55][1] = static_cast<Scalar>(3) / order;
    out[55][2] = static_cast<Scalar>(1) / order;
    out[56][0] = static_cast<Scalar>(4) / order;
    out[56][1] = static_cast<Scalar>(1) / order;
    out[56][2] = static_cast<Scalar>(2) / order;
    out[57][0] = static_cast<Scalar>(4) / order;
    out[57][1] = static_cast<Scalar>(2) / order;
    out[57][2] = static_cast<Scalar>(2) / order;
    out[58][0] = static_cast<Scalar>(4) / order;
    out[58][1] = static_cast<Scalar>(3) / order;
    out[58][2] = static_cast<Scalar>(2) / order;
    out[59][0] = static_cast<Scalar>(4) / order;
    out[59][1] = static_cast<Scalar>(1) / order;
    out[59][2] = static_cast<Scalar>(3) / order;
    out[60][0] = static_cast<Scalar>(4) / order;
    out[60][1] = static_cast<Scalar>(2) / order;
    out[60][2] = static_cast<Scalar>(3) / order;
    out[61][0] = static_cast<Scalar>(4) / order;
    out[61][1] = static_cast<Scalar>(3) / order;
    out[61][2] = static_cast<Scalar>(3) / order;
    out[62][0] = static_cast<Scalar>(1) / order;
    out[62][1] = static_cast<Scalar>(0) / order;
    out[62][2] = static_cast<Scalar>(1) / order;
    out[63][0] = static_cast<Scalar>(2) / order;
    out[63][1] = static_cast<Scalar>(0) / order;
    out[63][2] = static_cast<Scalar>(1) / order;
    out[64][0] = static_cast<Scalar>(3) / order;
    out[64][1] = static_cast<Scalar>(0) / order;
    out[64][2] = static_cast<Scalar>(1) / order;
    out[65][0] = static_cast<Scalar>(1) / order;
    out[65][1] = static_cast<Scalar>(0) / order;
    out[65][2] = static_cast<Scalar>(2) / order;
    out[66][0] = static_cast<Scalar>(2) / order;
    out[66][1] = static_cast<Scalar>(0) / order;
    out[66][2] = static_cast<Scalar>(2) / order;
    out[67][0] = static_cast<Scalar>(3) / order;
    out[67][1] = static_cast<Scalar>(0) / order;
    out[67][2] = static_cast<Scalar>(2) / order;
    out[68][0] = static_cast<Scalar>(1) / order;
    out[68][1] = static_cast<Scalar>(0) / order;
    out[68][2] = static_cast<Scalar>(3) / order;
    out[69][0] = static_cast<Scalar>(2) / order;
    out[69][1] = static_cast<Scalar>(0) / order;
    out[69][2] = static_cast<Scalar>(3) / order;
    out[70][0] = static_cast<Scalar>(3) / order;
    out[70][1] = static_cast<Scalar>(0) / order;
    out[70][2] = static_cast<Scalar>(3) / order;
    out[71][0] = static_cast<Scalar>(1) / order;
    out[71][1] = static_cast<Scalar>(4) / order;
    out[71][2] = static_cast<Scalar>(1) / order;
    out[72][0] = static_cast<Scalar>(2) / order;
    out[72][1] = static_cast<Scalar>(4) / order;
    out[72][2] = static_cast<Scalar>(1) / order;
    out[73][0] = static_cast<Scalar>(3) / order;
    out[73][1] = static_cast<Scalar>(4) / order;
    out[73][2] = static_cast<Scalar>(1) / order;
    out[74][0] = static_cast<Scalar>(1) / order;
    out[74][1] = static_cast<Scalar>(4) / order;
    out[74][2] = static_cast<Scalar>(2) / order;
    out[75][0] = static_cast<Scalar>(2) / order;
    out[75][1] = static_cast<Scalar>(4) / order;
    out[75][2] = static_cast<Scalar>(2) / order;
    out[76][0] = static_cast<Scalar>(3) / order;
    out[76][1] = static_cast<Scalar>(4) / order;
    out[76][2] = static_cast<Scalar>(2) / order;
    out[77][0] = static_cast<Scalar>(1) / order;
    out[77][1] = static_cast<Scalar>(4) / order;
    out[77][2] = static_cast<Scalar>(3) / order;
    out[78][0] = static_cast<Scalar>(2) / order;
    out[78][1] = static_cast<Scalar>(4) / order;
    out[78][2] = static_cast<Scalar>(3) / order;
    out[79][0] = static_cast<Scalar>(3) / order;
    out[79][1] = static_cast<Scalar>(4) / order;
    out[79][2] = static_cast<Scalar>(3) / order;
    out[80][0] = static_cast<Scalar>(1) / order;
    out[80][1] = static_cast<Scalar>(1) / order;
    out[80][2] = static_cast<Scalar>(0) / order;
    out[81][0] = static_cast<Scalar>(2) / order;
    out[81][1] = static_cast<Scalar>(1) / order;
    out[81][2] = static_cast<Scalar>(0) / order;
    out[82][0] = static_cast<Scalar>(3) / order;
    out[82][1] = static_cast<Scalar>(1) / order;
    out[82][2] = static_cast<Scalar>(0) / order;
    out[83][0] = static_cast<Scalar>(1) / order;
    out[83][1] = static_cast<Scalar>(2) / order;
    out[83][2] = static_cast<Scalar>(0) / order;
    out[84][0] = static_cast<Scalar>(2) / order;
    out[84][1] = static_cast<Scalar>(2) / order;
    out[84][2] = static_cast<Scalar>(0) / order;
    out[85][0] = static_cast<Scalar>(3) / order;
    out[85][1] = static_cast<Scalar>(2) / order;
    out[85][2] = static_cast<Scalar>(0) / order;
    out[86][0] = static_cast<Scalar>(1) / order;
    out[86][1] = static_cast<Scalar>(3) / order;
    out[86][2] = static_cast<Scalar>(0) / order;
    out[87][0] = static_cast<Scalar>(2) / order;
    out[87][1] = static_cast<Scalar>(3) / order;
    out[87][2] = static_cast<Scalar>(0) / order;
    out[88][0] = static_cast<Scalar>(3) / order;
    out[88][1] = static_cast<Scalar>(3) / order;
    out[88][2] = static_cast<Scalar>(0) / order;
    out[89][0] = static_cast<Scalar>(1) / order;
    out[89][1] = static_cast<Scalar>(1) / order;
    out[89][2] = static_cast<Scalar>(4) / order;
    out[90][0] = static_cast<Scalar>(2) / order;
    out[90][1] = static_cast<Scalar>(1) / order;
    out[90][2] = static_cast<Scalar>(4) / order;
    out[91][0] = static_cast<Scalar>(3) / order;
    out[91][1] = static_cast<Scalar>(1) / order;
    out[91][2] = static_cast<Scalar>(4) / order;
    out[92][0] = static_cast<Scalar>(1) / order;
    out[92][1] = static_cast<Scalar>(2) / order;
    out[92][2] = static_cast<Scalar>(4) / order;
    out[93][0] = static_cast<Scalar>(2) / order;
    out[93][1] = static_cast<Scalar>(2) / order;
    out[93][2] = static_cast<Scalar>(4) / order;
    out[94][0] = static_cast<Scalar>(3) / order;
    out[94][1] = static_cast<Scalar>(2) / order;
    out[94][2] = static_cast<Scalar>(4) / order;
    out[95][0] = static_cast<Scalar>(1) / order;
    out[95][1] = static_cast<Scalar>(3) / order;
    out[95][2] = static_cast<Scalar>(4) / order;
    out[96][0] = static_cast<Scalar>(2) / order;
    out[96][1] = static_cast<Scalar>(3) / order;
    out[96][2] = static_cast<Scalar>(4) / order;
    out[97][0] = static_cast<Scalar>(3) / order;
    out[97][1] = static_cast<Scalar>(3) / order;
    out[97][2] = static_cast<Scalar>(4) / order;
    out[98][0] = static_cast<Scalar>(1) / order;
    out[98][1] = static_cast<Scalar>(1) / order;
    out[98][2] = static_cast<Scalar>(1) / order;
    out[99][0] = static_cast<Scalar>(2) / order;
    out[99][1] = static_cast<Scalar>(1) / order;
    out[99][2] = static_cast<Scalar>(1) / order;
    out[100][0] = static_cast<Scalar>(3) / order;
    out[100][1] = static_cast<Scalar>(1) / order;
    out[100][2] = static_cast<Scalar>(1) / order;
    out[101][0] = static_cast<Scalar>(1) / order;
    out[101][1] = static_cast<Scalar>(2) / order;
    out[101][2] = static_cast<Scalar>(1) / order;
    out[102][0] = static_cast<Scalar>(2) / order;
    out[102][1] = static_cast<Scalar>(2) / order;
    out[102][2] = static_cast<Scalar>(1) / order;
    out[103][0] = static_cast<Scalar>(3) / order;
    out[103][1] = static_cast<Scalar>(2) / order;
    out[103][2] = static_cast<Scalar>(1) / order;
    out[104][0] = static_cast<Scalar>(1) / order;
    out[104][1] = static_cast<Scalar>(3) / order;
    out[104][2] = static_cast<Scalar>(1) / order;
    out[105][0] = static_cast<Scalar>(2) / order;
    out[105][1] = static_cast<Scalar>(3) / order;
    out[105][2] = static_cast<Scalar>(1) / order;
    out[106][0] = static_cast<Scalar>(3) / order;
    out[106][1] = static_cast<Scalar>(3) / order;
    out[106][2] = static_cast<Scalar>(1) / order;
    out[107][0] = static_cast<Scalar>(1) / order;
    out[107][1] = static_cast<Scalar>(1) / order;
    out[107][2] = static_cast<Scalar>(2) / order;
    out[108][0] = static_cast<Scalar>(2) / order;
    out[108][1] = static_cast<Scalar>(1) / order;
    out[108][2] = static_cast<Scalar>(2) / order;
    out[109][0] = static_cast<Scalar>(3) / order;
    out[109][1] = static_cast<Scalar>(1) / order;
    out[109][2] = static_cast<Scalar>(2) / order;
    out[110][0] = static_cast<Scalar>(1) / order;
    out[110][1] = static_cast<Scalar>(2) / order;
    out[110][2] = static_cast<Scalar>(2) / order;
    out[111][0] = static_cast<Scalar>(2) / order;
    out[111][1] = static_cast<Scalar>(2) / order;
    out[111][2] = static_cast<Scalar>(2) / order;
    out[112][0] = static_cast<Scalar>(3) / order;
    out[112][1] = static_cast<Scalar>(2) / order;
    out[112][2] = static_cast<Scalar>(2) / order;
    out[113][0] = static_cast<Scalar>(1) / order;
    out[113][1] = static_cast<Scalar>(3) / order;
    out[113][2] = static_cast<Scalar>(2) / order;
    out[114][0] = static_cast<Scalar>(2) / order;
    out[114][1] = static_cast<Scalar>(3) / order;
    out[114][2] = static_cast<Scalar>(2) / order;
    out[115][0] = static_cast<Scalar>(3) / order;
    out[115][1] = static_cast<Scalar>(3) / order;
    out[115][2] = static_cast<Scalar>(2) / order;
    out[116][0] = static_cast<Scalar>(1) / order;
    out[116][1] = static_cast<Scalar>(1) / order;
    out[116][2] = static_cast<Scalar>(3) / order;
    out[117][0] = static_cast<Scalar>(2) / order;
    out[117][1] = static_cast<Scalar>(1) / order;
    out[117][2] = static_cast<Scalar>(3) / order;
    out[118][0] = static_cast<Scalar>(3) / order;
    out[118][1] = static_cast<Scalar>(1) / order;
    out[118][2] = static_cast<Scalar>(3) / order;
    out[119][0] = static_cast<Scalar>(1) / order;
    out[119][1] = static_cast<Scalar>(2) / order;
    out[119][2] = static_cast<Scalar>(3) / order;
    out[120][0] = static_cast<Scalar>(2) / order;
    out[120][1] = static_cast<Scalar>(2) / order;
    out[120][2] = static_cast<Scalar>(3) / order;
    out[121][0] = static_cast<Scalar>(3) / order;
    out[121][1] = static_cast<Scalar>(2) / order;
    out[121][2] = static_cast<Scalar>(3) / order;
    out[122][0] = static_cast<Scalar>(1) / order;
    out[122][1] = static_cast<Scalar>(3) / order;
    out[122][2] = static_cast<Scalar>(3) / order;
    out[123][0] = static_cast<Scalar>(2) / order;
    out[123][1] = static_cast<Scalar>(3) / order;
    out[123][2] = static_cast<Scalar>(3) / order;
    out[124][0] = static_cast<Scalar>(3) / order;
    out[124][1] = static_cast<Scalar>(3) / order;
    out[124][2] = static_cast<Scalar>(3) / order;
  }
  template <typename Scalar>
  static constexpr void interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
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
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 10:
      out[0] = 3;
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
      out[1] = 2;
      out[2] = 0;
      break;
    case 13:
      out[0] = 4;
      out[1] = 3;
      out[2] = 0;
      break;
    case 14:
      out[0] = 1;
      out[1] = 4;
      out[2] = 0;
      break;
    case 15:
      out[0] = 2;
      out[1] = 4;
      out[2] = 0;
      break;
    case 16:
      out[0] = 3;
      out[1] = 4;
      out[2] = 0;
      break;
    case 17:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 18:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 19:
      out[0] = 0;
      out[1] = 3;
      out[2] = 0;
      break;
    case 20:
      out[0] = 1;
      out[1] = 0;
      out[2] = 4;
      break;
    case 21:
      out[0] = 2;
      out[1] = 0;
      out[2] = 4;
      break;
    case 22:
      out[0] = 3;
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
      out[1] = 2;
      out[2] = 4;
      break;
    case 25:
      out[0] = 4;
      out[1] = 3;
      out[2] = 4;
      break;
    case 26:
      out[0] = 1;
      out[1] = 4;
      out[2] = 4;
      break;
    case 27:
      out[0] = 2;
      out[1] = 4;
      out[2] = 4;
      break;
    case 28:
      out[0] = 3;
      out[1] = 4;
      out[2] = 4;
      break;
    case 29:
      out[0] = 0;
      out[1] = 1;
      out[2] = 4;
      break;
    case 30:
      out[0] = 0;
      out[1] = 2;
      out[2] = 4;
      break;
    case 31:
      out[0] = 0;
      out[1] = 3;
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
      out[2] = 2;
      break;
    case 34:
      out[0] = 0;
      out[1] = 0;
      out[2] = 3;
      break;
    case 35:
      out[0] = 4;
      out[1] = 0;
      out[2] = 1;
      break;
    case 36:
      out[0] = 4;
      out[1] = 0;
      out[2] = 2;
      break;
    case 37:
      out[0] = 4;
      out[1] = 0;
      out[2] = 3;
      break;
    case 38:
      out[0] = 4;
      out[1] = 4;
      out[2] = 1;
      break;
    case 39:
      out[0] = 4;
      out[1] = 4;
      out[2] = 2;
      break;
    case 40:
      out[0] = 4;
      out[1] = 4;
      out[2] = 3;
      break;
    case 41:
      out[0] = 0;
      out[1] = 4;
      out[2] = 1;
      break;
    case 42:
      out[0] = 0;
      out[1] = 4;
      out[2] = 2;
      break;
    case 43:
      out[0] = 0;
      out[1] = 4;
      out[2] = 3;
      break;
    case 44:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 45:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 46:
      out[0] = 0;
      out[1] = 3;
      out[2] = 1;
      break;
    case 47:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 48:
      out[0] = 0;
      out[1] = 2;
      out[2] = 2;
      break;
    case 49:
      out[0] = 0;
      out[1] = 3;
      out[2] = 2;
      break;
    case 50:
      out[0] = 0;
      out[1] = 1;
      out[2] = 3;
      break;
    case 51:
      out[0] = 0;
      out[1] = 2;
      out[2] = 3;
      break;
    case 52:
      out[0] = 0;
      out[1] = 3;
      out[2] = 3;
      break;
    case 53:
      out[0] = 4;
      out[1] = 1;
      out[2] = 1;
      break;
    case 54:
      out[0] = 4;
      out[1] = 2;
      out[2] = 1;
      break;
    case 55:
      out[0] = 4;
      out[1] = 3;
      out[2] = 1;
      break;
    case 56:
      out[0] = 4;
      out[1] = 1;
      out[2] = 2;
      break;
    case 57:
      out[0] = 4;
      out[1] = 2;
      out[2] = 2;
      break;
    case 58:
      out[0] = 4;
      out[1] = 3;
      out[2] = 2;
      break;
    case 59:
      out[0] = 4;
      out[1] = 1;
      out[2] = 3;
      break;
    case 60:
      out[0] = 4;
      out[1] = 2;
      out[2] = 3;
      break;
    case 61:
      out[0] = 4;
      out[1] = 3;
      out[2] = 3;
      break;
    case 62:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 63:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 64:
      out[0] = 3;
      out[1] = 0;
      out[2] = 1;
      break;
    case 65:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 66:
      out[0] = 2;
      out[1] = 0;
      out[2] = 2;
      break;
    case 67:
      out[0] = 3;
      out[1] = 0;
      out[2] = 2;
      break;
    case 68:
      out[0] = 1;
      out[1] = 0;
      out[2] = 3;
      break;
    case 69:
      out[0] = 2;
      out[1] = 0;
      out[2] = 3;
      break;
    case 70:
      out[0] = 3;
      out[1] = 0;
      out[2] = 3;
      break;
    case 71:
      out[0] = 1;
      out[1] = 4;
      out[2] = 1;
      break;
    case 72:
      out[0] = 2;
      out[1] = 4;
      out[2] = 1;
      break;
    case 73:
      out[0] = 3;
      out[1] = 4;
      out[2] = 1;
      break;
    case 74:
      out[0] = 1;
      out[1] = 4;
      out[2] = 2;
      break;
    case 75:
      out[0] = 2;
      out[1] = 4;
      out[2] = 2;
      break;
    case 76:
      out[0] = 3;
      out[1] = 4;
      out[2] = 2;
      break;
    case 77:
      out[0] = 1;
      out[1] = 4;
      out[2] = 3;
      break;
    case 78:
      out[0] = 2;
      out[1] = 4;
      out[2] = 3;
      break;
    case 79:
      out[0] = 3;
      out[1] = 4;
      out[2] = 3;
      break;
    case 80:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 81:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 82:
      out[0] = 3;
      out[1] = 1;
      out[2] = 0;
      break;
    case 83:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 84:
      out[0] = 2;
      out[1] = 2;
      out[2] = 0;
      break;
    case 85:
      out[0] = 3;
      out[1] = 2;
      out[2] = 0;
      break;
    case 86:
      out[0] = 1;
      out[1] = 3;
      out[2] = 0;
      break;
    case 87:
      out[0] = 2;
      out[1] = 3;
      out[2] = 0;
      break;
    case 88:
      out[0] = 3;
      out[1] = 3;
      out[2] = 0;
      break;
    case 89:
      out[0] = 1;
      out[1] = 1;
      out[2] = 4;
      break;
    case 90:
      out[0] = 2;
      out[1] = 1;
      out[2] = 4;
      break;
    case 91:
      out[0] = 3;
      out[1] = 1;
      out[2] = 4;
      break;
    case 92:
      out[0] = 1;
      out[1] = 2;
      out[2] = 4;
      break;
    case 93:
      out[0] = 2;
      out[1] = 2;
      out[2] = 4;
      break;
    case 94:
      out[0] = 3;
      out[1] = 2;
      out[2] = 4;
      break;
    case 95:
      out[0] = 1;
      out[1] = 3;
      out[2] = 4;
      break;
    case 96:
      out[0] = 2;
      out[1] = 3;
      out[2] = 4;
      break;
    case 97:
      out[0] = 3;
      out[1] = 3;
      out[2] = 4;
      break;
    case 98:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    case 99:
      out[0] = 2;
      out[1] = 1;
      out[2] = 1;
      break;
    case 100:
      out[0] = 3;
      out[1] = 1;
      out[2] = 1;
      break;
    case 101:
      out[0] = 1;
      out[1] = 2;
      out[2] = 1;
      break;
    case 102:
      out[0] = 2;
      out[1] = 2;
      out[2] = 1;
      break;
    case 103:
      out[0] = 3;
      out[1] = 2;
      out[2] = 1;
      break;
    case 104:
      out[0] = 1;
      out[1] = 3;
      out[2] = 1;
      break;
    case 105:
      out[0] = 2;
      out[1] = 3;
      out[2] = 1;
      break;
    case 106:
      out[0] = 3;
      out[1] = 3;
      out[2] = 1;
      break;
    case 107:
      out[0] = 1;
      out[1] = 1;
      out[2] = 2;
      break;
    case 108:
      out[0] = 2;
      out[1] = 1;
      out[2] = 2;
      break;
    case 109:
      out[0] = 3;
      out[1] = 1;
      out[2] = 2;
      break;
    case 110:
      out[0] = 1;
      out[1] = 2;
      out[2] = 2;
      break;
    case 111:
      out[0] = 2;
      out[1] = 2;
      out[2] = 2;
      break;
    case 112:
      out[0] = 3;
      out[1] = 2;
      out[2] = 2;
      break;
    case 113:
      out[0] = 1;
      out[1] = 3;
      out[2] = 2;
      break;
    case 114:
      out[0] = 2;
      out[1] = 3;
      out[2] = 2;
      break;
    case 115:
      out[0] = 3;
      out[1] = 3;
      out[2] = 2;
      break;
    case 116:
      out[0] = 1;
      out[1] = 1;
      out[2] = 3;
      break;
    case 117:
      out[0] = 2;
      out[1] = 1;
      out[2] = 3;
      break;
    case 118:
      out[0] = 3;
      out[1] = 1;
      out[2] = 3;
      break;
    case 119:
      out[0] = 1;
      out[1] = 2;
      out[2] = 3;
      break;
    case 120:
      out[0] = 2;
      out[1] = 2;
      out[2] = 3;
      break;
    case 121:
      out[0] = 3;
      out[1] = 2;
      out[2] = 3;
      break;
    case 122:
      out[0] = 1;
      out[1] = 3;
      out[2] = 3;
      break;
    case 123:
      out[0] = 2;
      out[1] = 3;
      out[2] = 3;
      break;
    case 124:
      out[0] = 3;
      out[1] = 3;
      out[2] = 3;
      break;
    }
  }

  template <typename Element>
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs) {
    subelement_node_idxs(subelement, idxs, meta::type_tag<Element>{});
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElPoint>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      break;
    case 1:
      idxs[0] = 1;
      break;
    case 2:
      idxs[0] = 2;
      break;
    case 3:
      idxs[0] = 3;
      break;
    case 4:
      idxs[0] = 4;
      break;
    case 5:
      idxs[0] = 5;
      break;
    case 6:
      idxs[0] = 6;
      break;
    case 7:
      idxs[0] = 7;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 8;
      idxs[3] = 9;
      idxs[4] = 10;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 11;
      idxs[3] = 12;
      idxs[4] = 13;
      break;
    case 2:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 14;
      idxs[3] = 15;
      idxs[4] = 16;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 17;
      idxs[3] = 18;
      idxs[4] = 19;
      break;
    case 4:
      idxs[0] = 4;
      idxs[1] = 5;
      idxs[2] = 20;
      idxs[3] = 21;
      idxs[4] = 22;
      break;
    case 5:
      idxs[0] = 5;
      idxs[1] = 6;
      idxs[2] = 23;
      idxs[3] = 24;
      idxs[4] = 25;
      break;
    case 6:
      idxs[0] = 7;
      idxs[1] = 6;
      idxs[2] = 26;
      idxs[3] = 27;
      idxs[4] = 28;
      break;
    case 7:
      idxs[0] = 4;
      idxs[1] = 7;
      idxs[2] = 29;
      idxs[3] = 30;
      idxs[4] = 31;
      break;
    case 8:
      idxs[0] = 0;
      idxs[1] = 4;
      idxs[2] = 32;
      idxs[3] = 33;
      idxs[4] = 34;
      break;
    case 9:
      idxs[0] = 1;
      idxs[1] = 5;
      idxs[2] = 35;
      idxs[3] = 36;
      idxs[4] = 37;
      break;
    case 10:
      idxs[0] = 2;
      idxs[1] = 6;
      idxs[2] = 38;
      idxs[3] = 39;
      idxs[4] = 40;
      break;
    case 11:
      idxs[0] = 3;
      idxs[1] = 7;
      idxs[2] = 41;
      idxs[3] = 42;
      idxs[4] = 43;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 7;
      idxs[3] = 4;
      idxs[4] = 17;
      idxs[5] = 18;
      idxs[6] = 19;
      idxs[7] = 41;
      idxs[8] = 42;
      idxs[9] = 43;
      idxs[10] = 29;
      idxs[11] = 30;
      idxs[12] = 31;
      idxs[13] = 32;
      idxs[14] = 33;
      idxs[15] = 34;
      idxs[16] = 44;
      idxs[17] = 45;
      idxs[18] = 46;
      idxs[19] = 47;
      idxs[20] = 48;
      idxs[21] = 49;
      idxs[22] = 50;
      idxs[23] = 51;
      idxs[24] = 52;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 5;
      idxs[4] = 11;
      idxs[5] = 12;
      idxs[6] = 13;
      idxs[7] = 38;
      idxs[8] = 39;
      idxs[9] = 40;
      idxs[10] = 23;
      idxs[11] = 24;
      idxs[12] = 25;
      idxs[13] = 35;
      idxs[14] = 36;
      idxs[15] = 37;
      idxs[16] = 53;
      idxs[17] = 54;
      idxs[18] = 55;
      idxs[19] = 56;
      idxs[20] = 57;
      idxs[21] = 58;
      idxs[22] = 59;
      idxs[23] = 60;
      idxs[24] = 61;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 5;
      idxs[3] = 4;
      idxs[4] = 8;
      idxs[5] = 9;
      idxs[6] = 10;
      idxs[7] = 35;
      idxs[8] = 36;
      idxs[9] = 37;
      idxs[10] = 20;
      idxs[11] = 21;
      idxs[12] = 22;
      idxs[13] = 32;
      idxs[14] = 33;
      idxs[15] = 34;
      idxs[16] = 62;
      idxs[17] = 63;
      idxs[18] = 64;
      idxs[19] = 65;
      idxs[20] = 66;
      idxs[21] = 67;
      idxs[22] = 68;
      idxs[23] = 69;
      idxs[24] = 70;
      break;
    case 3:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 7;
      idxs[4] = 14;
      idxs[5] = 15;
      idxs[6] = 16;
      idxs[7] = 38;
      idxs[8] = 39;
      idxs[9] = 40;
      idxs[10] = 26;
      idxs[11] = 27;
      idxs[12] = 28;
      idxs[13] = 41;
      idxs[14] = 42;
      idxs[15] = 43;
      idxs[16] = 71;
      idxs[17] = 72;
      idxs[18] = 73;
      idxs[19] = 74;
      idxs[20] = 75;
      idxs[21] = 76;
      idxs[22] = 77;
      idxs[23] = 78;
      idxs[24] = 79;
      break;
    case 4:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 2;
      idxs[3] = 3;
      idxs[4] = 8;
      idxs[5] = 9;
      idxs[6] = 10;
      idxs[7] = 11;
      idxs[8] = 12;
      idxs[9] = 13;
      idxs[10] = 14;
      idxs[11] = 15;
      idxs[12] = 16;
      idxs[13] = 17;
      idxs[14] = 18;
      idxs[15] = 19;
      idxs[16] = 80;
      idxs[17] = 81;
      idxs[18] = 82;
      idxs[19] = 83;
      idxs[20] = 84;
      idxs[21] = 85;
      idxs[22] = 86;
      idxs[23] = 87;
      idxs[24] = 88;
      break;
    case 5:
      idxs[0] = 4;
      idxs[1] = 5;
      idxs[2] = 6;
      idxs[3] = 7;
      idxs[4] = 20;
      idxs[5] = 21;
      idxs[6] = 22;
      idxs[7] = 23;
      idxs[8] = 24;
      idxs[9] = 25;
      idxs[10] = 26;
      idxs[11] = 27;
      idxs[12] = 28;
      idxs[13] = 29;
      idxs[14] = 30;
      idxs[15] = 31;
      idxs[16] = 89;
      idxs[17] = 90;
      idxs[18] = 91;
      idxs[19] = 92;
      idxs[20] = 93;
      idxs[21] = 94;
      idxs[22] = 95;
      idxs[23] = 96;
      idxs[24] = 97;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTetra>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElCube>) {
    switch (subelement) {
    default:
      break;
    }
  }
};

template <> struct BasisLagrange<mesh::RefElCube, 5> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 5;
  static constexpr dim_t num_basis_functions = 216;
  static constexpr dim_t num_interpolation_nodes = 216;

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
    const Scalar x15 = x10 * x12;
    const Scalar x16 = x3 - 1;
    const Scalar x17 = x0 * x16 * x8;
    const Scalar x18 = x17 * x5;
    const Scalar x19 = x14 * x18;
    const Scalar x20 = x15 * x19;
    const Scalar x21 = x13 * x9;
    const Scalar x22 = x1 * x2 * x[0];
    const Scalar x23 = x[1] * x[2];
    const Scalar x24 = x22 * x23;
    const Scalar x25 = x21 * x24;
    const Scalar x26 = 31250 * x25;
    const Scalar x27 = 15625 * x25;
    const Scalar x28 = x12 * x6;
    const Scalar x29 = x14 * x28;
    const Scalar x30 = x17 * x29;
    const Scalar x31 = x30 * x5;
    const Scalar x32 = x10 * x31;
    const Scalar x33 = x0 * x5;
    const Scalar x34 = x33 * x8;
    const Scalar x35 = x26 * x34;
    const Scalar x36 = x7 - 1;
    const Scalar x37 = x36 * x4;
    const Scalar x38 = x29 * x37;
    const Scalar x39 = 62500 * x25;
    const Scalar x40 = x37 * x39;
    const Scalar x41 = x13 * x14;
    const Scalar x42 = x10 * x24;
    const Scalar x43 = x12 * x42;
    const Scalar x44 = 62500 * x37;
    const Scalar x45 = x43 * x44;
    const Scalar x46 = x19 * x36;
    const Scalar x47 = x13 * x6;
    const Scalar x48 = x43 * x47;
    const Scalar x49 = x10 * x38;
    const Scalar x50 = x0 * x16;
    const Scalar x51 = x11 - 1;
    const Scalar x52 = x10 * x51;
    const Scalar x53 = x4 * x52;
    const Scalar x54 = x28 * x53;
    const Scalar x55 = x28 * x39;
    const Scalar x56 = x17 * x53;
    const Scalar x57 = x0 * x12 * x16 * x36 * x4 * x5 * x51 * x8;
    const Scalar x58 = 125000 * x57;
    const Scalar x59 = x18 * x51;
    const Scalar x60 = x36 * x59;
    const Scalar x61 = x34 * x51;
    const Scalar x62 = x0 * x16 * x36 * x4 * x51 * x8;
    const Scalar x63 = 125000 * x62;
    const Scalar x64 = x12 * x52;
    const Scalar x65 = x16 * x33;
    const Scalar x66 = x64 * x65;
    const Scalar x67 = x36 * x65;
    const Scalar x68 = x28 * x52;
    const Scalar x69 = x14 * x9;
    const Scalar x70 = x59 * x69;
    const Scalar x71 = 62500 * x43;
    const Scalar x72 = 31250 * x43;
    const Scalar x73 = x24 * x69;
    const Scalar x74 = x28 * x73;
    const Scalar x75 = x14 * x42;
    const Scalar x76 = x14 * x6;
    const Scalar x77 = x60 * x76;
    const Scalar x78 = x6 * x69;
    const Scalar x79 = x51 * x78;
    const Scalar x80 = x37 * x79;
    const Scalar x81 = x27 * x76;
    const Scalar x82 = x34 * x53;
    const Scalar x83 = x14 * x62;
    const Scalar x84 = x5 * x83;
    const Scalar x85 = x47 * x75;
    const Scalar x86 = x37 * x85;
    const Scalar x87 = x52 * x65;
    const Scalar x88 = x14 * x37;
    const Scalar x89 = x52 * x67;
    const Scalar x90 = x20 * x37;
    const Scalar x91 = 1250 * x21;
    const Scalar x92 = x2 * x[1];
    const Scalar x93 = x[0] * x[2];
    const Scalar x94 = x92 * x93;
    const Scalar x95 = x91 * x94;
    const Scalar x96 = 625 * x21;
    const Scalar x97 = x94 * x96;
    const Scalar x98 = x37 * x68;
    const Scalar x99 = x34 * x98;
    const Scalar x100 = x10 * x92;
    const Scalar x101 = x100 * x93;
    const Scalar x102 = 2500 * x21;
    const Scalar x103 = x102 * x62;
    const Scalar x104 = 2500 * x57;
    const Scalar x105 = x18 * x36;
    const Scalar x106 = x105 * x68;
    const Scalar x107 = 1250 * x69;
    const Scalar x108 = x52 * x76;
    const Scalar x109 = x108 * x37;
    const Scalar x110 = x62 * x76;
    const Scalar x111 = x1 * x23;
    const Scalar x112 = x111 * x[0];
    const Scalar x113 = x112 * x91;
    const Scalar x114 = x112 * x52;
    const Scalar x115 = x114 * x96;
    const Scalar x116 = x113 * x38;
    const Scalar x117 = x29 * x[0];
    const Scalar x118 = 1250 * x64;
    const Scalar x119 = x16 * x5;
    const Scalar x120 = x119 * x8;
    const Scalar x121 = x120 * x25;
    const Scalar x122 = 1250 * x121;
    const Scalar x123 = x120 * x51;
    const Scalar x124 = 2500 * x123 * x37;
    const Scalar x125 = x22 * x[2];
    const Scalar x126 = x10 * x125;
    const Scalar x127 = x126 * x91;
    const Scalar x128 = x125 * x96;
    const Scalar x129 = x22 * x[1];
    const Scalar x130 = x129 * x29;
    const Scalar x131 = x129 * x91;
    const Scalar x132 = x102 * x57;
    const Scalar x133 = x31 * x36;
    const Scalar x134 = x129 * x47;
    const Scalar x135 = 50 * x21;
    const Scalar x136 = 25 * x21;
    const Scalar x137 = x136 * x38;
    const Scalar x138 = x23 * x[0];
    const Scalar x139 = x135 * x57;
    const Scalar x140 = x10 * x14;
    const Scalar x141 = x139 * x140;
    const Scalar x142 = x136 * x52;
    const Scalar x143 = x133 * x142;
    const Scalar x144 = 50 * x120;
    const Scalar x145 = x144 * x98;
    const Scalar x146 = x21 * x94;
    const Scalar x147 = x1 * x[2];
    const Scalar x148 = x147 * x92;
    const Scalar x149 = x148 * x96;
    const Scalar x150 = x148 * x91;
    const Scalar x151 = x147 * x76;
    const Scalar x152 = x136 * x53;
    const Scalar x153 = x142 * x34 * x38;
    const Scalar x154 = x117 * x135 * x62;
    const Scalar x155 = x10 * x147;
    const Scalar x156 = 50 * x57;
    const Scalar x157 =
        x10 * x12 * x13 * x14 * x16 * x36 * x4 * x5 * x51 * x6 * x8 * x9;
    const Scalar x158 = x157 * x[0];
    const Scalar x159 = x0 * x157;
    const Scalar x160 = x1 * x92;
    return -1.0 / 13824.0 * coeffs[0] * x1 * x159 * x2 +
           (625.0 / 13824.0) * coeffs[100] * x0 * x1 * x10 * x13 * x14 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[101] * x127 * x84 +
           (625.0 / 6912.0) * coeffs[102] * x0 * x1 * x10 * x13 * x14 * x16 *
               x2 * x36 * x4 * x51 * x6 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[103] * x105 * x108 * x128 +
           (625.0 / 13824.0) * coeffs[104] * x0 * x10 * x12 * x13 * x14 * x2 *
               x36 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[105] * x90 * x95 +
           (625.0 / 6912.0) * coeffs[106] * x0 * x10 * x12 * x13 * x14 * x16 *
               x2 * x36 * x4 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[107] * x32 * x36 * x97 -
           1.0 / 13824.0 * coeffs[108] * x95 * x99 +
           (625.0 / 3456.0) * coeffs[109] * x0 * x10 * x12 * x13 * x16 * x2 *
               x36 * x4 * x5 * x51 * x8 * x9 * x[0] * x[1] * x[2] +
           (25.0 / 6912.0) * coeffs[10] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x36 * x4 * x51 * x6 * x8 * x9 * x[0] -
           1.0 / 13824.0 * coeffs[110] * x101 * x103 * x28 +
           (625.0 / 6912.0) * coeffs[111] * x0 * x10 * x12 * x13 * x16 * x2 *
               x36 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (625.0 / 6912.0) * coeffs[112] * x0 * x10 * x12 * x14 * x2 * x36 *
               x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[113] * x101 * x104 * x69 +
           (625.0 / 3456.0) * coeffs[114] * x0 * x10 * x12 * x14 * x16 * x2 *
               x36 * x4 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[115] * x106 * x107 * x94 -
           1.0 / 13824.0 * coeffs[116] * x109 * x34 * x97 +
           (625.0 / 6912.0) * coeffs[117] * x0 * x10 * x13 * x14 * x16 * x2 *
               x36 * x4 * x5 * x51 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[118] * x101 * x110 * x91 +
           (625.0 / 13824.0) * coeffs[119] * x0 * x10 * x13 * x14 * x16 * x2 *
               x36 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[11] * x143 * x22 -
           1.0 / 13824.0 * coeffs[120] * x130 * x82 * x96 +
           (625.0 / 6912.0) * coeffs[121] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x4 * x5 * x51 * x8 * x9 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[122] * x131 * x30 * x53 +
           (625.0 / 13824.0) * coeffs[123] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] +
           (625.0 / 6912.0) * coeffs[124] * x0 * x1 * x12 * x13 * x14 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[125] * x129 * x132 * x14 +
           (625.0 / 3456.0) * coeffs[126] * x0 * x1 * x12 * x13 * x14 * x16 *
               x2 * x36 * x4 * x51 * x6 * x8 * x9 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[127] * x131 * x133 * x51 -
           1.0 / 13824.0 * coeffs[128] * x118 * x134 * x34 * x88 +
           (625.0 / 3456.0) * coeffs[129] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x36 * x4 * x5 * x51 * x8 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[12] * x120 * x130 * x152 -
           625.0 / 3456.0 * coeffs[130] * x134 * x15 * x83 +
           (625.0 / 6912.0) * coeffs[131] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x36 * x5 * x51 * x6 * x8 * x[0] * x[1] +
           (625.0 / 13824.0) * coeffs[132] * x0 * x1 * x10 * x12 * x13 * x14 *
               x2 * x36 * x4 * x5 * x51 * x6 * x9 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[133] * x131 * x66 * x88 +
           (625.0 / 6912.0) * coeffs[134] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x36 * x4 * x51 * x6 * x9 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[135] * x130 * x89 * x96 +
           (625.0 / 13824.0) * coeffs[136] * x0 * x1 * x10 * x12 * x13 * x14 *
               x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[137] * x113 * x12 * x19 * x53 +
           (625.0 / 6912.0) * coeffs[138] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x4 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[139] * x115 * x31 +
           (25.0 / 6912.0) * coeffs[13] * x1 * x12 * x13 * x14 * x16 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] -
           1.0 / 13824.0 * coeffs[140] * x116 * x61 +
           (625.0 / 3456.0) * coeffs[141] * x0 * x1 * x12 * x13 * x14 * x16 *
               x36 * x4 * x5 * x51 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[142] * x103 * x111 * x117 +
           (625.0 / 6912.0) * coeffs[143] * x0 * x1 * x12 * x13 * x14 * x16 *
               x36 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (625.0 / 6912.0) * coeffs[144] * x0 * x1 * x10 * x12 * x13 * x14 *
               x36 * x4 * x5 * x51 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[145] * x10 * x104 * x112 * x41 +
           (625.0 / 3456.0) * coeffs[146] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x36 * x4 * x51 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[147] * x112 * x118 * x46 * x47 -
           1.0 / 13824.0 * coeffs[148] * x115 * x33 * x38 +
           (625.0 / 6912.0) * coeffs[149] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x36 * x4 * x5 * x51 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[14] * x134 * x144 * x64 * x88 -
           1.0 / 13824.0 * coeffs[150] * x116 * x50 * x52 +
           (625.0 / 13824.0) * coeffs[151] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x36 * x5 * x51 * x6 * x9 * x[0] * x[1] * x[2] +
           (15625.0 / 13824.0) * coeffs[152] * x0 * x1 * x10 * x12 * x13 * x14 *
               x2 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[153] * x20 * x26 * x4 +
           (15625.0 / 6912.0) * coeffs[154] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x4 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[155] * x27 * x32 -
           1.0 / 13824.0 * coeffs[156] * x35 * x38 +
           (15625.0 / 3456.0) * coeffs[157] * x0 * x1 * x12 * x13 * x14 * x16 *
               x2 * x36 * x4 * x5 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[158] * x30 * x40 +
           (15625.0 / 6912.0) * coeffs[159] * x0 * x1 * x12 * x13 * x14 * x16 *
               x2 * x36 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (25.0 / 13824.0) * coeffs[15] * x1 * x10 * x12 * x13 * x14 * x16 *
               x2 * x36 * x4 * x5 * x51 * x6 * x9 * x[0] * x[1] +
           (15625.0 / 6912.0) * coeffs[160] * x0 * x1 * x10 * x12 * x13 * x14 *
               x2 * x36 * x4 * x5 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[161] * x18 * x41 * x45 +
           (15625.0 / 3456.0) * coeffs[162] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x36 * x4 * x6 * x8 * x[0] * x[1] * x[2] -
           15625.0 / 6912.0 * coeffs[163] * x46 * x48 -
           1.0 / 13824.0 * coeffs[164] * x27 * x33 * x49 +
           (15625.0 / 6912.0) * coeffs[165] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x36 * x4 * x5 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[166] * x26 * x49 * x50 +
           (15625.0 / 13824.0) * coeffs[167] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x36 * x5 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[168] * x35 * x54 +
           (15625.0 / 3456.0) * coeffs[169] * x0 * x1 * x10 * x12 * x13 * x16 *
               x2 * x4 * x5 * x51 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[16] * x153 * x92 * x[0] -
           1.0 / 13824.0 * coeffs[170] * x55 * x56 +
           (15625.0 / 6912.0) * coeffs[171] * x0 * x1 * x10 * x12 * x13 * x16 *
               x2 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (15625.0 / 3456.0) * coeffs[172] * x0 * x1 * x12 * x13 * x2 * x36 *
               x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[173] * x25 * x58 +
           (15625.0 / 1728.0) * coeffs[174] * x0 * x1 * x12 * x13 * x16 * x2 *
               x36 * x4 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[175] * x55 * x60 -
           1.0 / 13824.0 * coeffs[176] * x44 * x48 * x61 +
           (15625.0 / 1728.0) * coeffs[177] * x0 * x1 * x10 * x12 * x13 * x16 *
               x2 * x36 * x4 * x5 * x51 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[178] * x48 * x63 +
           (15625.0 / 3456.0) * coeffs[179] * x0 * x1 * x10 * x12 * x13 * x16 *
               x2 * x36 * x5 * x51 * x6 * x8 * x[0] * x[1] * x[2] +
           (25.0 / 6912.0) * coeffs[17] * x0 * x10 * x12 * x13 * x14 * x16 *
               x2 * x36 * x4 * x5 * x51 * x8 * x9 * x[0] * x[1] +
           (15625.0 / 6912.0) * coeffs[180] * x0 * x1 * x10 * x12 * x13 * x2 *
               x36 * x4 * x5 * x51 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[181] * x40 * x66 +
           (15625.0 / 3456.0) * coeffs[182] * x0 * x1 * x10 * x12 * x13 * x16 *
               x2 * x36 * x4 * x51 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[183] * x26 * x67 * x68 +
           (15625.0 / 6912.0) * coeffs[184] * x0 * x1 * x10 * x12 * x14 * x2 *
               x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[185] * x4 * x70 * x71 +
           (15625.0 / 3456.0) * coeffs[186] * x0 * x1 * x10 * x12 * x14 * x16 *
               x2 * x4 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[187] * x6 * x70 * x72 -
           1.0 / 13824.0 * coeffs[188] * x44 * x61 * x74 +
           (15625.0 / 1728.0) * coeffs[189] * x0 * x1 * x12 * x14 * x16 * x2 *
               x36 * x4 * x5 * x51 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[18] * x100 * x154 -
           1.0 / 13824.0 * coeffs[190] * x28 * x63 * x73 +
           (15625.0 / 3456.0) * coeffs[191] * x0 * x1 * x12 * x14 * x16 * x2 *
               x36 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (15625.0 / 3456.0) * coeffs[192] * x0 * x1 * x10 * x12 * x14 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[193] * x58 * x75 +
           (15625.0 / 1728.0) * coeffs[194] * x0 * x1 * x10 * x12 * x14 * x16 *
               x2 * x36 * x4 * x51 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[195] * x71 * x77 -
           1.0 / 13824.0 * coeffs[196] * x33 * x72 * x80 +
           (15625.0 / 3456.0) * coeffs[197] * x0 * x1 * x10 * x12 * x14 * x16 *
               x2 * x36 * x4 * x5 * x51 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[198] * x45 * x50 * x79 +
           (15625.0 / 6912.0) * coeffs[199] * x0 * x1 * x10 * x12 * x14 * x16 *
               x2 * x36 * x5 * x51 * x6 * x9 * x[0] * x[1] * x[2] +
           (25.0 / 13824.0) * coeffs[19] * x0 * x10 * x12 * x13 * x14 * x16 *
               x2 * x36 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] +
           (1.0 / 13824.0) * coeffs[1] * x1 * x10 * x12 * x13 * x14 * x16 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[0] -
           1.0 / 13824.0 * coeffs[200] * x81 * x82 +
           (15625.0 / 6912.0) * coeffs[201] * x0 * x1 * x10 * x13 * x14 * x16 *
               x2 * x4 * x5 * x51 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[202] * x26 * x56 * x76 +
           (15625.0 / 13824.0) * coeffs[203] * x0 * x1 * x10 * x13 * x14 * x16 *
               x2 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (15625.0 / 6912.0) * coeffs[204] * x0 * x1 * x13 * x14 * x2 * x36 *
               x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[205] * x39 * x84 +
           (15625.0 / 3456.0) * coeffs[206] * x0 * x1 * x13 * x14 * x16 * x2 *
               x36 * x4 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[207] * x26 * x77 -
           15625.0 / 6912.0 * coeffs[208] * x61 * x86 +
           (15625.0 / 3456.0) * coeffs[209] * x0 * x1 * x10 * x13 * x14 * x16 *
               x2 * x36 * x4 * x5 * x51 * x8 * x[0] * x[1] * x[2] +
           (25.0 / 13824.0) * coeffs[20] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x4 * x5 * x51 * x6 * x8 * x9 * x[1] -
           15625.0 / 3456.0 * coeffs[210] * x62 * x85 +
           (15625.0 / 6912.0) * coeffs[211] * x0 * x1 * x10 * x13 * x14 * x16 *
               x2 * x36 * x5 * x51 * x6 * x8 * x[0] * x[1] * x[2] +
           (15625.0 / 13824.0) * coeffs[212] * x0 * x1 * x10 * x13 * x14 * x2 *
               x36 * x4 * x5 * x51 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[213] * x26 * x87 * x88 +
           (15625.0 / 6912.0) * coeffs[214] * x0 * x1 * x10 * x13 * x14 * x16 *
               x2 * x36 * x4 * x51 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[215] * x81 * x89 -
           1.0 / 13824.0 * coeffs[21] * x139 * x160 * x76 +
           (25.0 / 6912.0) * coeffs[22] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x36 * x4 * x5 * x51 * x6 * x8 * x[1] -
           1.0 / 13824.0 * coeffs[23] * x137 * x160 * x87 -
           1.0 / 13824.0 * coeffs[24] * x147 * x153 * x[0] +
           (25.0 / 6912.0) * coeffs[25] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x36 * x4 * x5 * x51 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[26] * x154 * x155 +
           (25.0 / 13824.0) * coeffs[27] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x36 * x5 * x51 * x6 * x8 * x9 * x[0] * x[2] +
           (25.0 / 13824.0) * coeffs[28] * x1 * x10 * x12 * x13 * x14 * x16 *
               x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[29] * x112 * x123 * x135 * x38 -
           1.0 / 13824.0 * coeffs[2] * x158 * x92 +
           (25.0 / 6912.0) * coeffs[30] * x1 * x10 * x12 * x13 * x14 * x16 *
               x36 * x4 * x5 * x51 * x6 * x8 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[31] * x114 * x119 * x137 +
           (25.0 / 13824.0) * coeffs[32] * x0 * x10 * x12 * x13 * x14 * x36 *
               x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[33] * x138 * x141 +
           (25.0 / 6912.0) * coeffs[34] * x0 * x10 * x12 * x13 * x14 * x16 *
               x36 * x4 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[35] * x138 * x143 -
           1.0 / 13824.0 * coeffs[36] * x111 * x152 * x31 +
           (25.0 / 6912.0) * coeffs[37] * x0 * x1 * x12 * x13 * x14 * x16 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[38] * x111 * x140 * x156 * x47 +
           (25.0 / 13824.0) * coeffs[39] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x36 * x4 * x5 * x51 * x6 * x9 * x[1] * x[2] +
           (1.0 / 13824.0) * coeffs[3] * x0 * x10 * x12 * x13 * x14 * x16 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[1] +
           (25.0 / 13824.0) * coeffs[40] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x36 * x4 * x5 * x6 * x8 * x9 * x[2] -
           1.0 / 13824.0 * coeffs[41] * x139 * x155 * x2 * x6 +
           (25.0 / 6912.0) * coeffs[42] * x0 * x1 * x10 * x12 * x14 * x16 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[2] -
           1.0 / 13824.0 * coeffs[43] * x10 * x136 * x151 * x2 * x5 * x62 -
           1.0 / 13824.0 * coeffs[44] * x120 * x125 * x136 * x49 +
           (25.0 / 6912.0) * coeffs[45] * x1 * x10 * x12 * x13 * x16 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[46] * x125 * x145 * x69 +
           (25.0 / 13824.0) * coeffs[47] * x1 * x10 * x13 * x14 * x16 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[2] +
           (25.0 / 13824.0) * coeffs[48] * x10 * x12 * x13 * x14 * x16 * x2 *
               x36 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[49] * x145 * x146 +
           (1.0 / 13824.0) * coeffs[4] * x0 * x1 * x10 * x12 * x13 * x14 * x16 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[2] +
           (25.0 / 6912.0) * coeffs[50] * x10 * x12 * x14 * x16 * x2 * x36 *
               x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           25.0 / 13824.0 * coeffs[51] * x109 * x120 * x146 -
           1.0 / 13824.0 * coeffs[52] * x136 * x32 * x37 * x92 * x[2] +
           (25.0 / 6912.0) * coeffs[53] * x0 * x10 * x12 * x13 * x16 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[54] * x100 * x156 * x78 * x[2] +
           (25.0 / 13824.0) * coeffs[55] * x0 * x10 * x13 * x14 * x16 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[56] * x149 * x32 * x4 +
           (625.0 / 6912.0) * coeffs[57] * x0 * x1 * x12 * x13 * x14 * x16 *
               x2 * x36 * x4 * x5 * x6 * x8 * x9 * x[1] * x[2] -
           625.0 / 6912.0 * coeffs[58] * x148 * x47 * x90 +
           (625.0 / 13824.0) * coeffs[59] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x36 * x4 * x5 * x6 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[5] * x147 * x158 +
           (625.0 / 6912.0) * coeffs[60] * x0 * x1 * x10 * x12 * x13 * x16 *
               x2 * x4 * x5 * x51 * x6 * x8 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[61] * x132 * x148 * x6 +
           (625.0 / 3456.0) * coeffs[62] * x0 * x1 * x10 * x12 * x13 * x16 *
               x2 * x36 * x4 * x5 * x51 * x6 * x8 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[63] * x150 * x65 * x98 -
           1.0 / 13824.0 * coeffs[64] * x107 * x148 * x18 * x54 +
           (625.0 / 3456.0) * coeffs[65] * x0 * x1 * x12 * x14 * x16 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[66] * x100 * x104 * x151 +
           (625.0 / 6912.0) * coeffs[67] * x0 * x1 * x10 * x12 * x14 * x16 *
               x2 * x36 * x4 * x5 * x51 * x6 * x9 * x[1] * x[2] +
           (625.0 / 13824.0) * coeffs[68] * x0 * x1 * x10 * x13 * x14 * x16 *
               x2 * x4 * x5 * x51 * x6 * x8 * x9 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[69] * x110 * x150 * x5 +
           (1.0 / 13824.0) * coeffs[6] * x10 * x12 * x13 * x14 * x16 * x36 *
               x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] +
           (625.0 / 6912.0) * coeffs[70] * x0 * x1 * x10 * x13 * x14 * x16 *
               x2 * x36 * x4 * x5 * x51 * x6 * x8 * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[71] * x149 * x37 * x76 * x87 +
           (625.0 / 13824.0) * coeffs[72] * x1 * x10 * x12 * x13 * x14 * x16 *
               x2 * x4 * x5 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[73] * x122 * x38 +
           (625.0 / 6912.0) * coeffs[74] * x1 * x10 * x12 * x13 * x14 * x16 *
               x2 * x36 * x4 * x5 * x6 * x8 * x[0] * x[1] * x[2] -
           625.0 / 13824.0 * coeffs[75] * x119 * x25 * x49 -
           1.0 / 13824.0 * coeffs[76] * x122 * x54 +
           (625.0 / 3456.0) * coeffs[77] * x1 * x12 * x13 * x16 * x2 * x36 *
               x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[78] * x124 * x48 +
           (625.0 / 6912.0) * coeffs[79] * x1 * x10 * x12 * x13 * x16 * x2 *
               x36 * x4 * x5 * x51 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[7] * x159 * x23 +
           (625.0 / 6912.0) * coeffs[80] * x1 * x10 * x12 * x14 * x16 * x2 *
               x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[81] * x124 * x74 +
           (625.0 / 3456.0) * coeffs[82] * x1 * x10 * x12 * x14 * x16 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x[0] * x[1] * x[2] -
           625.0 / 6912.0 * coeffs[83] * x119 * x43 * x80 -
           625.0 / 13824.0 * coeffs[84] * x121 * x53 * x76 +
           (625.0 / 6912.0) * coeffs[85] * x1 * x13 * x14 * x16 * x2 * x36 *
               x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[1] * x[2] -
           625.0 / 6912.0 * coeffs[86] * x123 * x86 +
           (625.0 / 13824.0) * coeffs[87] * x1 * x10 * x13 * x14 * x16 * x2 *
               x36 * x4 * x5 * x51 * x6 * x9 * x[0] * x[1] * x[2] -
           1.0 / 13824.0 * coeffs[88] * x128 * x34 * x49 +
           (625.0 / 6912.0) * coeffs[89] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x36 * x4 * x5 * x8 * x9 * x[0] * x[2] +
           (25.0 / 13824.0) * coeffs[8] * x0 * x1 * x10 * x12 * x13 * x14 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[0] -
           1.0 / 13824.0 * coeffs[90] * x127 * x30 * x37 +
           (625.0 / 13824.0) * coeffs[91] * x0 * x1 * x10 * x12 * x13 * x14 *
               x16 * x2 * x36 * x5 * x6 * x8 * x9 * x[0] * x[2] +
           (625.0 / 6912.0) * coeffs[92] * x0 * x1 * x10 * x12 * x13 * x2 *
               x36 * x4 * x5 * x51 * x6 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[93] * x126 * x132 +
           (625.0 / 3456.0) * coeffs[94] * x0 * x1 * x10 * x12 * x13 * x16 *
               x2 * x36 * x4 * x51 * x6 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[95] * x106 * x125 * x91 -
           1.0 / 13824.0 * coeffs[96] * x107 * x125 * x99 +
           (625.0 / 3456.0) * coeffs[97] * x0 * x1 * x10 * x12 * x14 * x16 *
               x2 * x36 * x4 * x5 * x51 * x8 * x9 * x[0] * x[2] -
           625.0 / 3456.0 * coeffs[98] * x126 * x28 * x62 * x69 +
           (625.0 / 6912.0) * coeffs[99] * x0 * x1 * x10 * x12 * x14 * x16 *
               x2 * x36 * x5 * x51 * x6 * x8 * x9 * x[0] * x[2] -
           1.0 / 13824.0 * coeffs[9] * x141 * x22;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[2] - 1;
    const Scalar x1 = x[0] - 1;
    const Scalar x2 = 5 * x[1];
    const Scalar x3 = x2 - 1;
    const Scalar x4 = x1 * x3;
    const Scalar x5 = x0 * x4;
    const Scalar x6 = x[1] - 1;
    const Scalar x7 = 5 * x[0];
    const Scalar x8 = x7 - 1;
    const Scalar x9 = 5 * x[2];
    const Scalar x10 = x9 - 1;
    const Scalar x11 = x7 - 4;
    const Scalar x12 = x2 - 4;
    const Scalar x13 = x9 - 4;
    const Scalar x14 = x7 - 3;
    const Scalar x15 = x2 - 3;
    const Scalar x16 = x9 - 3;
    const Scalar x17 = x7 - 2;
    const Scalar x18 = x2 - 2;
    const Scalar x19 = x9 - 2;
    const Scalar x20 =
        x10 * x11 * x12 * x13 * x14 * x15 * x16 * x17 * x18 * x19 * x6 * x8;
    const Scalar x21 = (1.0 / 13824.0) * x20;
    const Scalar x22 = x3 * x[0];
    const Scalar x23 = x0 * x22;
    const Scalar x24 = x11 * x12 * x13 * x14 * x15 * x16 * x17 * x18 * x19;
    const Scalar x25 = x23 * x24;
    const Scalar x26 = x10 * x8;
    const Scalar x27 = x26 * x[1];
    const Scalar x28 = (1.0 / 13824.0) * x27;
    const Scalar x29 = x24 * x28;
    const Scalar x30 = x21 * x[2];
    const Scalar x31 = x29 * x[2];
    const Scalar x32 = (25.0 / 13824.0) * x[0];
    const Scalar x33 = x5 * x6;
    const Scalar x34 = x32 * x33;
    const Scalar x35 = x10 * x24;
    const Scalar x36 = x33 * x[0];
    const Scalar x37 = x18 * x19;
    const Scalar x38 = x36 * x37;
    const Scalar x39 = x11 * x12 * x13 * x14 * x15 * x16;
    const Scalar x40 = (25.0 / 6912.0) * x39;
    const Scalar x41 = x26 * x40;
    const Scalar x42 = x11 * x12 * x13 * x17;
    const Scalar x43 = (25.0 / 6912.0) * x42;
    const Scalar x44 = x15 * x16;
    const Scalar x45 = x38 * x44;
    const Scalar x46 = x26 * x45;
    const Scalar x47 = x26 * x44;
    const Scalar x48 = x12 * x17;
    const Scalar x49 = x14 * x37;
    const Scalar x50 = x13 * x49;
    const Scalar x51 = x48 * x50;
    const Scalar x52 = x47 * x51;
    const Scalar x53 = x32 * x[1];
    const Scalar x54 = x0 * x20;
    const Scalar x55 = x23 * x6;
    const Scalar x56 = x27 * x40;
    const Scalar x57 = x17 * x19;
    const Scalar x58 = x56 * x57;
    const Scalar x59 = x43 * x49;
    const Scalar x60 = x27 * x55;
    const Scalar x61 = x16 * x60;
    const Scalar x62 = (25.0 / 13824.0) * x11;
    const Scalar x63 = x44 * x62;
    const Scalar x64 = x17 * x60;
    const Scalar x65 = x50 * x64;
    const Scalar x66 = x35 * x53;
    const Scalar x67 = x37 * x[0];
    const Scalar x68 = x5 * x56;
    const Scalar x69 = x43 * x67;
    const Scalar x70 = x27 * x44;
    const Scalar x71 = x5 * x70;
    const Scalar x72 = x32 * x51;
    const Scalar x73 = (25.0 / 13824.0) * x[1];
    const Scalar x74 = x1 * x73;
    const Scalar x75 = x27 * x33;
    const Scalar x76 = x16 * x59;
    const Scalar x77 = x17 * x50;
    const Scalar x78 = x63 * x77;
    const Scalar x79 = x6 * x[2];
    const Scalar x80 = x4 * x79;
    const Scalar x81 = x32 * x80;
    const Scalar x82 = x67 * x80;
    const Scalar x83 = x20 * x[2];
    const Scalar x84 = x22 * x79;
    const Scalar x85 = x27 * x84;
    const Scalar x86 = x4 * x[2];
    const Scalar x87 = x70 * x86;
    const Scalar x88 = x27 * x80;
    const Scalar x89 = x33 * x[2];
    const Scalar x90 = (25.0 / 13824.0) * x8;
    const Scalar x91 = x17 * x18;
    const Scalar x92 = x41 * x91;
    const Scalar x93 = x15 * x59;
    const Scalar x94 = x26 * x93;
    const Scalar x95 = x48 * x49;
    const Scalar x96 = x47 * x62 * x95;
    const Scalar x97 = x55 * x[2];
    const Scalar x98 = x8 * x[2];
    const Scalar x99 = x73 * x98;
    const Scalar x100 = x23 * x[2];
    const Scalar x101 = x100 * x27;
    const Scalar x102 = x63 * x95;
    const Scalar x103 = x24 * x5;
    const Scalar x104 = x91 * x[2];
    const Scalar x105 = x5 * x[2];
    const Scalar x106 = x105 * x27;
    const Scalar x107 = (625.0 / 13824.0) * x24;
    const Scalar x108 = x8 * x[1];
    const Scalar x109 = x0 * x79;
    const Scalar x110 = x1 * x109;
    const Scalar x111 = x108 * x110;
    const Scalar x112 = x108 * x89;
    const Scalar x113 = (625.0 / 6912.0) * x39;
    const Scalar x114 = x113 * x57;
    const Scalar x115 = x16 * x42;
    const Scalar x116 = (625.0 / 6912.0) * x49;
    const Scalar x117 = x115 * x116;
    const Scalar x118 = x11 * x77;
    const Scalar x119 = (625.0 / 13824.0) * x44;
    const Scalar x120 = x118 * x119;
    const Scalar x121 = x110 * x27;
    const Scalar x122 = x113 * x91;
    const Scalar x123 = x75 * x[2];
    const Scalar x124 = (625.0 / 3456.0) * x39;
    const Scalar x125 = (625.0 / 3456.0) * x123;
    const Scalar x126 = x14 * x18;
    const Scalar x127 = x115 * x126;
    const Scalar x128 = x11 * x70;
    const Scalar x129 = x128 * x89;
    const Scalar x130 = x13 * x14;
    const Scalar x131 = (625.0 / 6912.0) * x130;
    const Scalar x132 = x15 * x42;
    const Scalar x133 = x116 * x132;
    const Scalar x134 = x14 * x19;
    const Scalar x135 = x132 * x134;
    const Scalar x136 = x42 * x49;
    const Scalar x137 = (625.0 / 6912.0) * x123;
    const Scalar x138 = (625.0 / 13824.0) * x95;
    const Scalar x139 = x110 * x128;
    const Scalar x140 = (625.0 / 6912.0) * x48;
    const Scalar x141 = x134 * x140;
    const Scalar x142 = x11 * x95;
    const Scalar x143 = (625.0 / 13824.0) * x17;
    const Scalar x144 = x109 * x[0];
    const Scalar x145 = x108 * x97;
    const Scalar x146 = x144 * x27;
    const Scalar x147 = x64 * x[2];
    const Scalar x148 = (625.0 / 3456.0) * x[2];
    const Scalar x149 = x148 * x42;
    const Scalar x150 = (625.0 / 6912.0) * x11;
    const Scalar x151 = x126 * x13;
    const Scalar x152 = x148 * x60;
    const Scalar x153 = x15 * x[2];
    const Scalar x154 = x11 * x[2];
    const Scalar x155 = x154 * x44;
    const Scalar x156 = (625.0 / 6912.0) * x[2];
    const Scalar x157 = x36 * x[2];
    const Scalar x158 = x113 * x98;
    const Scalar x159 = x45 * x98;
    const Scalar x160 = (625.0 / 6912.0) * x42;
    const Scalar x161 = (625.0 / 13824.0) * x48;
    const Scalar x162 = x130 * x159;
    const Scalar x163 = x10 * x36;
    const Scalar x164 = x104 * x113;
    const Scalar x165 = x124 * x18;
    const Scalar x166 = x149 * x18;
    const Scalar x167 = x140 * x151;
    const Scalar x168 = x132 * x38;
    const Scalar x169 = x10 * x14;
    const Scalar x170 = x148 * x26;
    const Scalar x171 = x11 * x130;
    const Scalar x172 = x12 * x38;
    const Scalar x173 = x171 * x172;
    const Scalar x174 = x131 * x48;
    const Scalar x175 = x153 * x38;
    const Scalar x176 = x154 * x45;
    const Scalar x177 = x169 * x176;
    const Scalar x178 = x14 * x[2];
    const Scalar x179 = x178 * x46;
    const Scalar x180 = x140 * x154;
    const Scalar x181 = x[0] * x[1];
    const Scalar x182 = (625.0 / 13824.0) * x181;
    const Scalar x183 = x5 * x67 * x[1];
    const Scalar x184 = x44 * x98;
    const Scalar x185 = x181 * x5;
    const Scalar x186 = x106 * x[0];
    const Scalar x187 = x71 * x[0];
    const Scalar x188 = x187 * x[2];
    const Scalar x189 = x10 * x181;
    const Scalar x190 = x105 * x189;
    const Scalar x191 = x15 * x186;
    const Scalar x192 = (625.0 / 3456.0) * x11 * x12 * x50;
    const Scalar x193 = x132 * x67;
    const Scalar x194 = (625.0 / 6912.0) * x51;
    const Scalar x195 = x12 * x154;
    const Scalar x196 = x0 * x1 * x6;
    const Scalar x197 = x182 * x35;
    const Scalar x198 = x196 * x67;
    const Scalar x199 = x113 * x27;
    const Scalar x200 = x160 * x70;
    const Scalar x201 = x70 * x[0];
    const Scalar x202 = (625.0 / 13824.0) * x201;
    const Scalar x203 = x202 * x51;
    const Scalar x204 = x163 * x[1];
    const Scalar x205 = x19 * x36;
    const Scalar x206 = x205 * x27;
    const Scalar x207 = x205 * x70;
    const Scalar x208 = (625.0 / 3456.0) * x42;
    const Scalar x209 = x115 * x38;
    const Scalar x210 = x209 * x[1];
    const Scalar x211 = (625.0 / 3456.0) * x27;
    const Scalar x212 = x16 * x173;
    const Scalar x213 = x16 * x38;
    const Scalar x214 = x213 * x27;
    const Scalar x215 = x143 * x45;
    const Scalar x216 = x10 * x[1];
    const Scalar x217 = x171 * x216;
    const Scalar x218 = x27 * x45;
    const Scalar x219 = x13 * x17;
    const Scalar x220 = x130 * x27;
    const Scalar x221 = x1 * x79;
    const Scalar x222 = x221 * x67;
    const Scalar x223 = x189 * x80;
    const Scalar x224 = x88 * x[0];
    const Scalar x225 = x19 * x201 * x80;
    const Scalar x226 = x16 * x224;
    const Scalar x227 = (625.0 / 6912.0) * x128;
    const Scalar x228 = x50 * x[0];
    const Scalar x229 = (15625.0 / 13824.0) * x110;
    const Scalar x230 = (15625.0 / 6912.0) * x39;
    const Scalar x231 = x111 * x67;
    const Scalar x232 = (15625.0 / 6912.0) * x44;
    const Scalar x233 = x51 * x[0];
    const Scalar x234 = x205 * x[1];
    const Scalar x235 = (15625.0 / 3456.0) * x98;
    const Scalar x236 = x184 * x234;
    const Scalar x237 = (15625.0 / 3456.0) * x42;
    const Scalar x238 = (15625.0 / 6912.0) * x48;
    const Scalar x239 = x130 * x238;
    const Scalar x240 = x213 * x[1];
    const Scalar x241 = x17 * x176;
    const Scalar x242 = (15625.0 / 13824.0) * x[1];
    const Scalar x243 = (15625.0 / 6912.0) * x11 * x[1];
    const Scalar x244 = x17 * x242;
    const Scalar x245 = x110 * x189;
    const Scalar x246 = (15625.0 / 3456.0) * x39;
    const Scalar x247 = x110 * x201;
    const Scalar x248 = x157 * x216;
    const Scalar x249 = (15625.0 / 1728.0) * x157;
    const Scalar x250 = x249 * x27;
    const Scalar x251 = x130 * x70;
    const Scalar x252 = (15625.0 / 3456.0) * x157;
    const Scalar x253 = x252 * x48;
    const Scalar x254 = (15625.0 / 3456.0) * x248;
    const Scalar x255 = (15625.0 / 1728.0) * x27;
    const Scalar x256 = x151 * x16;
    const Scalar x257 = x128 * x252;
    const Scalar x258 = x104 * x36;
    const Scalar x259 = (15625.0 / 6912.0) * x49;
    const Scalar x260 = (15625.0 / 3456.0) * x121;
    const Scalar x261 = x153 * x206;
    const Scalar x262 = (15625.0 / 3456.0) * x48;
    const Scalar x263 = x38 * x[2];
    const Scalar x264 = x154 * x172;
    const Scalar x265 = (15625.0 / 6912.0) * x17 * x175;
    const Scalar x266 = (15625.0 / 3456.0) * x27;
    const Scalar x267 = x175 * x266;
    const Scalar x268 = x134 * x238;
    const Scalar x269 = (15625.0 / 6912.0) * x27;
    out[0] = -x21 * x5;
    out[1] = x21 * x23;
    out[2] = -x25 * x28;
    out[3] = x29 * x5;
    out[4] = x30 * x4;
    out[5] = -x22 * x30;
    out[6] = x22 * x31;
    out[7] = -x31 * x4;
    out[8] = x34 * x35;
    out[9] = -x38 * x41;
    out[10] = x43 * x46;
    out[11] = -x34 * x52;
    out[12] = -x53 * x54;
    out[13] = x55 * x58;
    out[14] = -x59 * x61;
    out[15] = x63 * x65;
    out[16] = -x5 * x66;
    out[17] = x67 * x68;
    out[18] = -x69 * x71;
    out[19] = x71 * x72;
    out[20] = x54 * x74;
    out[21] = -x33 * x58;
    out[22] = x75 * x76;
    out[23] = -x75 * x78;
    out[24] = -x35 * x81;
    out[25] = x41 * x82;
    out[26] = -x47 * x69 * x80;
    out[27] = x52 * x81;
    out[28] = x53 * x83;
    out[29] = -x58 * x84;
    out[30] = x76 * x85;
    out[31] = -x78 * x85;
    out[32] = x66 * x86;
    out[33] = -x56 * x67 * x86;
    out[34] = x69 * x87;
    out[35] = -x72 * x87;
    out[36] = -x74 * x83;
    out[37] = x58 * x80;
    out[38] = -x76 * x88;
    out[39] = x78 * x88;
    out[40] = x24 * x89 * x90;
    out[41] = -x89 * x92;
    out[42] = x89 * x94;
    out[43] = -x89 * x96;
    out[44] = -x25 * x79 * x90;
    out[45] = x92 * x97;
    out[46] = -x94 * x97;
    out[47] = x96 * x97;
    out[48] = x25 * x99;
    out[49] = -x100 * x56 * x91;
    out[50] = x101 * x93;
    out[51] = -x101 * x102;
    out[52] = -x103 * x99;
    out[53] = x104 * x68;
    out[54] = -x106 * x93;
    out[55] = x102 * x106;
    out[56] = -x107 * x111;
    out[57] = x112 * x114;
    out[58] = -x112 * x117;
    out[59] = x112 * x120;
    out[60] = x121 * x122;
    out[61] = -x123 * x124 * x17;
    out[62] = x125 * x127;
    out[63] = -x129 * x131 * x91;
    out[64] = -x121 * x133;
    out[65] = x125 * x135;
    out[66] = -x125 * x136;
    out[67] = x118 * x137 * x15;
    out[68] = x138 * x139;
    out[69] = -x129 * x141;
    out[70] = x137 * x142 * x16;
    out[71] = -x129 * x143 * x49;
    out[72] = x107 * x108 * x144;
    out[73] = -x114 * x145;
    out[74] = x117 * x145;
    out[75] = -x120 * x145;
    out[76] = -x122 * x146;
    out[77] = x124 * x147;
    out[78] = -x126 * x149 * x61;
    out[79] = x147 * x150 * x151 * x44;
    out[80] = x133 * x146;
    out[81] = -x135 * x152;
    out[82] = x136 * x152;
    out[83] = -x150 * x153 * x65;
    out[84] = -x128 * x138 * x144;
    out[85] = x141 * x155 * x60;
    out[86] = -x142 * x156 * x61;
    out[87] = x11 * x119 * x147 * x49;
    out[88] = -x107 * x157;
    out[89] = x158 * x38;
    out[90] = -x159 * x160;
    out[91] = x161 * x162;
    out[92] = x163 * x164;
    out[93] = -x157 * x165 * x26;
    out[94] = x166 * x36 * x47;
    out[95] = -x157 * x167 * x47;
    out[96] = -x156 * x168 * x169;
    out[97] = x15 * x170 * x173;
    out[98] = -x168 * x170;
    out[99] = x174 * x175 * x26;
    out[100] = x161 * x177;
    out[101] = -x12 * x150 * x179;
    out[102] = x180 * x46;
    out[103] = -x161 * x179;
    out[104] = x103 * x182 * x[2];
    out[105] = -x158 * x183;
    out[106] = x160 * x183 * x184;
    out[107] = -x119 * x185 * x51 * x98;
    out[108] = -x10 * x164 * x185;
    out[109] = x165 * x186;
    out[110] = -x166 * x187;
    out[111] = x167 * x188;
    out[112] = x133 * x190;
    out[113] = -x191 * x192;
    out[114] = (625.0 / 3456.0) * x106 * x193;
    out[115] = -x191 * x194;
    out[116] = -x119 * x142 * x190;
    out[117] = x116 * x187 * x195;
    out[118] = -x180 * x67 * x71;
    out[119] = x138 * x188;
    out[120] = -x196 * x197;
    out[121] = x198 * x199;
    out[122] = -x198 * x200;
    out[123] = x196 * x203;
    out[124] = x114 * x204;
    out[125] = -x124 * x206;
    out[126] = x207 * x208;
    out[127] = -x174 * x207;
    out[128] = -625.0 / 6912.0 * x169 * x210;
    out[129] = x211 * x212;
    out[130] = -x209 * x211;
    out[131] = x174 * x214;
    out[132] = x215 * x217;
    out[133] = -x11 * x131 * x218;
    out[134] = x150 * x218 * x219;
    out[135] = -x215 * x220;
    out[136] = x197 * x221;
    out[137] = -x199 * x222;
    out[138] = x200 * x222;
    out[139] = -x203 * x221;
    out[140] = -x114 * x223;
    out[141] = x124 * x19 * x224;
    out[142] = -x208 * x225;
    out[143] = x174 * x225;
    out[144] = x117 * x223;
    out[145] = -x192 * x226;
    out[146] = x115 * x211 * x82;
    out[147] = -x194 * x226;
    out[148] = -x120 * x223;
    out[149] = x227 * x228 * x80;
    out[150] = -x219 * x227 * x82;
    out[151] = x202 * x77 * x80;
    out[152] = x181 * x229 * x24;
    out[153] = -x230 * x231;
    out[154] = x231 * x232 * x42;
    out[155] = -15625.0 / 13824.0 * x111 * x233 * x44;
    out[156] = -x157 * x230 * x57 * x[1];
    out[157] = x234 * x235 * x39;
    out[158] = -x236 * x237;
    out[159] = x236 * x239;
    out[160] = (15625.0 / 6912.0) * x178 * x210;
    out[161] = -x212 * x235 * x[1];
    out[162] = x210 * x235;
    out[163] = -x239 * x240 * x98;
    out[164] = -x130 * x241 * x242;
    out[165] = x162 * x243;
    out[166] = -x159 * x219 * x243;
    out[167] = x162 * x244;
    out[168] = -x230 * x245 * x91;
    out[169] = x121 * x18 * x246 * x[0];
    out[170] = -x18 * x237 * x247;
    out[171] = x151 * x238 * x247;
    out[172] = x17 * x246 * x248;
    out[173] = -x250 * x39;
    out[174] = x249 * x42 * x70;
    out[175] = -x251 * x253;
    out[176] = -x127 * x254;
    out[177] = x195 * x255 * x256 * x36;
    out[178] = -x115 * x18 * x250;
    out[179] = x253 * x256 * x27;
    out[180] = x104 * x171 * x204 * x232;
    out[181] = -x151 * x257;
    out[182] = (15625.0 / 3456.0) * x128 * x13 * x258;
    out[183] = -15625.0 / 6912.0 * x251 * x258;
    out[184] = x132 * x245 * x259;
    out[185] = -x11 * x12 * x15 * x228 * x260;
    out[186] = x193 * x260;
    out[187] = -15625.0 / 6912.0 * x121 * x15 * x233;
    out[188] = -x135 * x254;
    out[189] = (15625.0 / 1728.0) * x12 * x171 * x261;
    out[190] = -x132 * x19 * x250;
    out[191] = x130 * x261 * x262;
    out[192] = x169 * x237 * x263 * x[1];
    out[193] = -15625.0 / 1728.0 * x220 * x264;
    out[194] = x255 * x263 * x42;
    out[195] = -x220 * x262 * x263;
    out[196] = -x217 * x265;
    out[197] = x171 * x267;
    out[198] = -x11 * x219 * x267;
    out[199] = x220 * x265;
    out[200] = -x142 * x189 * x229 * x44;
    out[201] = x12 * x139 * x259 * x[0];
    out[202] = -x139 * x238 * x67;
    out[203] = x201 * x229 * x95;
    out[204] = x155 * x204 * x268;
    out[205] = -x12 * x134 * x257;
    out[206] = x128 * x19 * x253;
    out[207] = -x157 * x268 * x70;
    out[208] = -x154 * x169 * x238 * x240;
    out[209] = x14 * x16 * x264 * x266;
    out[210] = -x154 * x214 * x262;
    out[211] = x178 * x214 * x238;
    out[212] = x177 * x244;
    out[213] = -x14 * x176 * x269;
    out[214] = x241 * x269;
    out[215] = -15625.0 / 13824.0 * x17 * x178 * x218;
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
    const Scalar x27 = x11 * x26;
    const Scalar x28 = x14 * x27;
    const Scalar x29 = x15 * x26;
    const Scalar x30 = x12 * x27;
    const Scalar x31 = x30 * x[0];
    const Scalar x32 = x13 * x30;
    const Scalar x33 = x16 + x28 + x29 + x31 + x32;
    const Scalar x34 = x4 * x5;
    const Scalar x35 = x3 * x34;
    const Scalar x36 = x35 * x[1];
    const Scalar x37 = x0 * x1;
    const Scalar x38 = x36 * x37;
    const Scalar x39 = x8 * x9;
    const Scalar x40 = x39 * x7;
    const Scalar x41 = x40 * x[2];
    const Scalar x42 = 31250 * x41;
    const Scalar x43 = coeffs[153] * x42;
    const Scalar x44 = x18 * x26;
    const Scalar x45 = x17 * x27;
    const Scalar x46 = x45 * x[0];
    const Scalar x47 = x13 * x45;
    const Scalar x48 = x19 + x28 + x44 + x46 + x47;
    const Scalar x49 = x20 * x26;
    const Scalar x50 = x49 * x[0];
    const Scalar x51 = x13 * x49;
    const Scalar x52 = x21 + x29 + x44 + x50 + x51;
    const Scalar x53 = x38 * x52;
    const Scalar x54 = 15625 * x41;
    const Scalar x55 = coeffs[155] * x54;
    const Scalar x56 = x2 - 1;
    const Scalar x57 = x3 * x56;
    const Scalar x58 = x4 * x57;
    const Scalar x59 = x58 * x[1];
    const Scalar x60 = x37 * x59;
    const Scalar x61 = x25 * x60;
    const Scalar x62 = coeffs[156] * x42;
    const Scalar x63 = 62500 * x41;
    const Scalar x64 = coeffs[158] * x63;
    const Scalar x65 = x5 * x57;
    const Scalar x66 = x65 * x[1];
    const Scalar x67 = x37 * x66;
    const Scalar x68 = coeffs[161] * x63;
    const Scalar x69 = x52 * x67;
    const Scalar x70 = coeffs[163] * x42;
    const Scalar x71 = x34 * x56;
    const Scalar x72 = x71 * x[1];
    const Scalar x73 = x37 * x72;
    const Scalar x74 = x25 * x73;
    const Scalar x75 = coeffs[164] * x54;
    const Scalar x76 = x25 * x38;
    const Scalar x77 = x6 - 1;
    const Scalar x78 = x7 * x77;
    const Scalar x79 = x78 * x8;
    const Scalar x80 = x79 * x[2];
    const Scalar x81 = 31250 * x80;
    const Scalar x82 = coeffs[168] * x81;
    const Scalar x83 = x48 * x80;
    const Scalar x84 = 62500 * x37;
    const Scalar x85 = x36 * x84;
    const Scalar x86 = x33 * x80;
    const Scalar x87 = 125000 * x60;
    const Scalar x88 = x52 * x60;
    const Scalar x89 = 62500 * x80;
    const Scalar x90 = coeffs[175] * x89;
    const Scalar x91 = x25 * x67;
    const Scalar x92 = coeffs[176] * x89;
    const Scalar x93 = 125000 * x67;
    const Scalar x94 = x72 * x84;
    const Scalar x95 = x52 * x73;
    const Scalar x96 = coeffs[183] * x81;
    const Scalar x97 = x78 * x9;
    const Scalar x98 = x97 * x[2];
    const Scalar x99 = x33 * x98;
    const Scalar x100 = 31250 * x98;
    const Scalar x101 = coeffs[187] * x100;
    const Scalar x102 = 62500 * x98;
    const Scalar x103 = coeffs[188] * x102;
    const Scalar x104 = x48 * x98;
    const Scalar x105 = coeffs[195] * x102;
    const Scalar x106 = coeffs[196] * x100;
    const Scalar x107 = x39 * x77;
    const Scalar x108 = x107 * x[2];
    const Scalar x109 = 15625 * x108;
    const Scalar x110 = coeffs[200] * x109;
    const Scalar x111 = x38 * x48;
    const Scalar x112 = 31250 * x108;
    const Scalar x113 = coeffs[202] * x112;
    const Scalar x114 = x33 * x60;
    const Scalar x115 = 62500 * x108;
    const Scalar x116 = coeffs[205] * x115;
    const Scalar x117 = coeffs[207] * x112;
    const Scalar x118 = coeffs[208] * x112;
    const Scalar x119 = x48 * x67;
    const Scalar x120 = coeffs[210] * x115;
    const Scalar x121 = x33 * x73;
    const Scalar x122 = coeffs[213] * x112;
    const Scalar x123 = coeffs[215] * x109;
    const Scalar x124 = x1 * x[1];
    const Scalar x125 = x34 * x57;
    const Scalar x126 = 1250 * x125;
    const Scalar x127 = x124 * x126;
    const Scalar x128 = coeffs[105] * x41;
    const Scalar x129 = x124 * x125;
    const Scalar x130 = 625 * x52;
    const Scalar x131 = 2500 * x125;
    const Scalar x132 = x124 * x131;
    const Scalar x133 = 625 * x25;
    const Scalar x134 = x108 * x129;
    const Scalar x135 = x108 * x126;
    const Scalar x136 = x124 * x48;
    const Scalar x137 = coeffs[137] * x36;
    const Scalar x138 = x0 * x[2];
    const Scalar x139 = x39 * x78;
    const Scalar x140 = 1250 * x139;
    const Scalar x141 = x138 * x140;
    const Scalar x142 = coeffs[139] * x36;
    const Scalar x143 = x138 * x139;
    const Scalar x144 = coeffs[142] * x59;
    const Scalar x145 = 2500 * x139;
    const Scalar x146 = x138 * x145;
    const Scalar x147 = coeffs[145] * x66;
    const Scalar x148 = x143 * x72;
    const Scalar x149 = coeffs[150] * x72;
    const Scalar x150 = x33 * x37;
    const Scalar x151 = x125 * x37;
    const Scalar x152 = x108 * x151;
    const Scalar x153 = 625 * x139;
    const Scalar x154 = coeffs[120] * x153;
    const Scalar x155 = coeffs[122] * x140;
    const Scalar x156 = coeffs[125] * x145;
    const Scalar x157 = coeffs[127] * x140;
    const Scalar x158 = coeffs[128] * x140;
    const Scalar x159 = coeffs[130] * x145;
    const Scalar x160 = coeffs[133] * x140;
    const Scalar x161 = coeffs[135] * x153;
    const Scalar x162 = x[1] * x[2];
    const Scalar x163 = x125 * x139;
    const Scalar x164 = 50 * x163;
    const Scalar x165 = 25 * x163;
    const Scalar x166 = x165 * x52;
    const Scalar x167 = x151 * x41;
    const Scalar x168 = x126 * x37;
    const Scalar x169 = x131 * x37;
    const Scalar x170 = x165 * x25;
    const Scalar x171 = x20 * x27;
    const Scalar x172 = x10 * x22 + x10 * x30 + x10 * x45 + x10 * x49 + x171;
    const Scalar x173 = 1250 * x172;
    const Scalar x174 = 625 * x172;
    const Scalar x175 = 2500 * x172;
    const Scalar x176 = 50 * x143;
    const Scalar x177 = coeffs[29] * x59;
    const Scalar x178 = 25 * x172;
    const Scalar x179 = 50 * x129;
    const Scalar x180 = coeffs[49] * x80;
    const Scalar x181 = 25 * x139;
    const Scalar x182 = 50 * x139;
    const Scalar x183 = 50 * x151;
    const Scalar x184 = coeffs[46] * x98;
    const Scalar x185 = x163 * x172;
    const Scalar x186 = x171 + 5 * x24 + 5 * x32 + 5 * x47 + 5 * x51;
    const Scalar x187 = 625 * x186;
    const Scalar x188 = 1250 * x186;
    const Scalar x189 = 2500 * x186;
    const Scalar x190 = 25 * x186;
    const Scalar x191 = coeffs[38] * x66;
    const Scalar x192 = coeffs[54] * x98;
    const Scalar x193 = x163 * x186;
    const Scalar x194 = coeffs[41] * x80;
    const Scalar x195 = x0 * x2;
    const Scalar x196 = x195 * x4;
    const Scalar x197 = x196 * x3;
    const Scalar x198 = x195 * x5;
    const Scalar x199 = x198 * x3;
    const Scalar x200 = x195 * x34;
    const Scalar x201 = x0 * x35;
    const Scalar x202 = x197 + x199 + x200 + x201 + x36;
    const Scalar x203 = x1 * x13;
    const Scalar x204 = x202 * x31;
    const Scalar x205 = x202 * x203;
    const Scalar x206 = x205 * x50;
    const Scalar x207 = x195 * x57;
    const Scalar x208 = x196 * x56;
    const Scalar x209 = x0 * x58;
    const Scalar x210 = x197 + x207 + x208 + x209 + x59;
    const Scalar x211 = x203 * x210;
    const Scalar x212 = x211 * x23;
    const Scalar x213 = x198 * x56;
    const Scalar x214 = x0 * x65;
    const Scalar x215 = x199 + x207 + x213 + x214 + x66;
    const Scalar x216 = x203 * x215;
    const Scalar x217 = x216 * x50;
    const Scalar x218 = x0 * x71;
    const Scalar x219 = x200 + x208 + x213 + x218 + x72;
    const Scalar x220 = x203 * x219;
    const Scalar x221 = x220 * x23;
    const Scalar x222 = x219 * x46;
    const Scalar x223 = x205 * x23;
    const Scalar x224 = x203 * x89;
    const Scalar x225 = x202 * x46;
    const Scalar x226 = x31 * x80;
    const Scalar x227 = 125000 * x211;
    const Scalar x228 = x211 * x50;
    const Scalar x229 = x216 * x23;
    const Scalar x230 = x46 * x80;
    const Scalar x231 = 125000 * x216;
    const Scalar x232 = x219 * x31;
    const Scalar x233 = x220 * x50;
    const Scalar x234 = x102 * x203;
    const Scalar x235 = x46 * x98;
    const Scalar x236 = x31 * x98;
    const Scalar x237 = x203 * x225;
    const Scalar x238 = x211 * x31;
    const Scalar x239 = x216 * x46;
    const Scalar x240 = x203 * x232;
    const Scalar x241 = x13 * x140 * x[2];
    const Scalar x242 = x13 * x50;
    const Scalar x243 = x153 * x[2];
    const Scalar x244 = x210 * x[2];
    const Scalar x245 = x13 * x23;
    const Scalar x246 = coeffs[140] * x245;
    const Scalar x247 = x13 * x46;
    const Scalar x248 = x13 * x31;
    const Scalar x249 = x215 * x[2];
    const Scalar x250 = coeffs[147] * x242;
    const Scalar x251 = coeffs[148] * x245;
    const Scalar x252 = x1 * x[0];
    const Scalar x253 = x210 * x252;
    const Scalar x254 = 1250 * x171;
    const Scalar x255 = x254 * x41;
    const Scalar x256 = x219 * x252;
    const Scalar x257 = 625 * x41;
    const Scalar x258 = x254 * x80;
    const Scalar x259 = x202 * x252;
    const Scalar x260 = 2500 * x171;
    const Scalar x261 = x260 * x80;
    const Scalar x262 = x215 * x252;
    const Scalar x263 = x260 * x98;
    const Scalar x264 = x254 * x98;
    const Scalar x265 = 625 * x108;
    const Scalar x266 = x171 * x265;
    const Scalar x267 = x108 * x254;
    const Scalar x268 = x139 * x171;
    const Scalar x269 = x268 * x[0];
    const Scalar x270 = 25 * x[2];
    const Scalar x271 = 25 * x1;
    const Scalar x272 = 50 * x1;
    const Scalar x273 = x13 * x268;
    const Scalar x274 = x125 + x2 * x35 + x2 * x58 + x2 * x65 + x2 * x71;
    const Scalar x275 = x1 * x274;
    const Scalar x276 = 1250 * x248;
    const Scalar x277 = x275 * x41;
    const Scalar x278 = 625 * x242;
    const Scalar x279 = coeffs[107] * x278;
    const Scalar x280 = coeffs[108] * x245;
    const Scalar x281 = 1250 * x275;
    const Scalar x282 = 2500 * x13;
    const Scalar x283 = x275 * x282;
    const Scalar x284 = coeffs[115] * x242;
    const Scalar x285 = x245 * x275;
    const Scalar x286 = x274 * x[2];
    const Scalar x287 = x182 * x248;
    const Scalar x288 = x181 * x242;
    const Scalar x289 = 50 * x171;
    const Scalar x290 = x275 * x289;
    const Scalar x291 = 25 * x171;
    const Scalar x292 = x291 * x[0];
    const Scalar x293 = x182 * x247;
    const Scalar x294 = x13 * x291;
    const Scalar x295 = x125 + 5 * x201 + 5 * x209 + 5 * x214 + 5 * x218;
    const Scalar x296 = x1 * x295;
    const Scalar x297 = x108 * x296;
    const Scalar x298 = coeffs[103] * x278;
    const Scalar x299 = 1250 * x296;
    const Scalar x300 = coeffs[90] * x247;
    const Scalar x301 = x282 * x296;
    const Scalar x302 = coeffs[95] * x242;
    const Scalar x303 = coeffs[96] * x245;
    const Scalar x304 = x295 * x[2];
    const Scalar x305 = x289 * x296;
    const Scalar x306 = x1 * x6;
    const Scalar x307 = x306 * x8;
    const Scalar x308 = x307 * x7;
    const Scalar x309 = x306 * x9;
    const Scalar x310 = x309 * x7;
    const Scalar x311 = x306 * x39;
    const Scalar x312 = x1 * x40;
    const Scalar x313 = x308 + x310 + x311 + x312 + x41;
    const Scalar x314 = 31250 * x313;
    const Scalar x315 = x0 * x36;
    const Scalar x316 = x242 * x315;
    const Scalar x317 = 15625 * x313;
    const Scalar x318 = x0 * x59;
    const Scalar x319 = x245 * x318;
    const Scalar x320 = 62500 * x313;
    const Scalar x321 = x0 * x66;
    const Scalar x322 = x242 * x321;
    const Scalar x323 = x0 * x72;
    const Scalar x324 = x245 * x323;
    const Scalar x325 = x306 * x78;
    const Scalar x326 = x307 * x77;
    const Scalar x327 = x1 * x79;
    const Scalar x328 = x308 + x325 + x326 + x327 + x80;
    const Scalar x329 = 31250 * x328;
    const Scalar x330 = x245 * x315;
    const Scalar x331 = x247 * x328;
    const Scalar x332 = 62500 * x0;
    const Scalar x333 = x332 * x36;
    const Scalar x334 = x248 * x328;
    const Scalar x335 = 125000 * x318;
    const Scalar x336 = 62500 * x328;
    const Scalar x337 = x242 * x318;
    const Scalar x338 = x245 * x321;
    const Scalar x339 = 125000 * x321;
    const Scalar x340 = x332 * x72;
    const Scalar x341 = x242 * x323;
    const Scalar x342 = x309 * x77;
    const Scalar x343 = x1 * x97;
    const Scalar x344 = x310 + x325 + x342 + x343 + x98;
    const Scalar x345 = x248 * x344;
    const Scalar x346 = 31250 * x344;
    const Scalar x347 = 62500 * x344;
    const Scalar x348 = x247 * x344;
    const Scalar x349 = x1 * x107;
    const Scalar x350 = x108 + x311 + x326 + x342 + x349;
    const Scalar x351 = 15625 * x350;
    const Scalar x352 = 31250 * x350;
    const Scalar x353 = 62500 * x350;
    const Scalar x354 = x126 * x248;
    const Scalar x355 = x313 * x[1];
    const Scalar x356 = x126 * x[1];
    const Scalar x357 = x131 * x[1];
    const Scalar x358 = 625 * x245;
    const Scalar x359 = x125 * x358;
    const Scalar x360 = x350 * x[1];
    const Scalar x361 = x254 * x[0];
    const Scalar x362 = 625 * x171;
    const Scalar x363 = x362 * x[0];
    const Scalar x364 = x260 * x[0];
    const Scalar x365 = x0 * x350;
    const Scalar x366 = x125 * x171;
    const Scalar x367 = x366 * x[0];
    const Scalar x368 = 50 * x[1];
    const Scalar x369 = 25 * x367;
    const Scalar x370 = x13 * x362;
    const Scalar x371 = x13 * x254;
    const Scalar x372 = x171 * x282;
    const Scalar x373 = x0 * x313;
    const Scalar x374 = x0 * x126;
    const Scalar x375 = x0 * x131;
    const Scalar x376 = 50 * x0;
    const Scalar x377 = x13 * x366;
    const Scalar x378 = 25 * x377;
    const Scalar x379 = x107 * x6 + x139 + x40 * x6 + x6 * x79 + x6 * x97;
    const Scalar x380 = x0 * x379;
    const Scalar x381 = 1250 * x380;
    const Scalar x382 = x282 * x380;
    const Scalar x383 = x380 * x72;
    const Scalar x384 = 1250 * x247;
    const Scalar x385 = x289 * x380;
    const Scalar x386 = x379 * x[1];
    const Scalar x387 = 50 * x125;
    const Scalar x388 = x248 * x387;
    const Scalar x389 = 25 * x125;
    const Scalar x390 = x242 * x389;
    const Scalar x391 = x245 * x389;
    const Scalar x392 = x247 * x387;
    const Scalar x393 = x139 + 5 * x312 + 5 * x327 + 5 * x343 + 5 * x349;
    const Scalar x394 = x0 * x393;
    const Scalar x395 = x36 * x394;
    const Scalar x396 = x282 * x394;
    const Scalar x397 = 1250 * x394;
    const Scalar x398 = x394 * x72;
    const Scalar x399 = x289 * x394;
    const Scalar x400 = x393 * x[1];
    out[0] =
        -1.0 / 13824.0 * coeffs[0] * x193 * x37 +
        (625.0 / 13824.0) * coeffs[100] * x0 * x1 * x25 * x3 * x4 * x5 * x56 *
            x77 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[101] * x135 * x150 +
        (625.0 / 6912.0) * coeffs[102] * x0 * x1 * x3 * x4 * x48 * x5 * x56 *
            x77 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[103] * x130 * x152 +
        (625.0 / 13824.0) * coeffs[104] * x1 * x25 * x3 * x4 * x5 * x56 * x7 *
            x8 * x9 * x[1] * x[2] +
        (625.0 / 6912.0) * coeffs[106] * x1 * x3 * x4 * x48 * x5 * x56 * x7 *
            x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[107] * x129 * x130 * x41 -
        1.0 / 13824.0 * coeffs[108] * x127 * x25 * x80 +
        (625.0 / 3456.0) * coeffs[109] * x1 * x3 * x33 * x4 * x5 * x56 * x7 *
            x77 * x8 * x[1] * x[2] +
        (25.0 / 6912.0) * coeffs[10] * x0 * x1 * x3 * x4 * x48 * x5 * x56 * x7 *
            x77 * x8 * x9 -
        1.0 / 13824.0 * coeffs[110] * x132 * x83 +
        (625.0 / 6912.0) * coeffs[111] * x1 * x3 * x4 * x5 * x52 * x56 * x7 *
            x77 * x8 * x[1] * x[2] +
        (625.0 / 6912.0) * coeffs[112] * x1 * x25 * x3 * x4 * x5 * x56 * x7 *
            x77 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[113] * x132 * x99 +
        (625.0 / 3456.0) * coeffs[114] * x1 * x3 * x4 * x48 * x5 * x56 * x7 *
            x77 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[115] * x127 * x52 * x98 -
        1.0 / 13824.0 * coeffs[116] * x133 * x134 +
        (625.0 / 6912.0) * coeffs[117] * x1 * x3 * x33 * x4 * x5 * x56 * x77 *
            x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[118] * x135 * x136 +
        (625.0 / 13824.0) * coeffs[119] * x1 * x3 * x4 * x5 * x52 * x56 * x77 *
            x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[11] * x166 * x37 +
        (625.0 / 6912.0) * coeffs[121] * x0 * x1 * x3 * x33 * x4 * x5 * x7 *
            x77 * x8 * x9 * x[1] +
        (625.0 / 13824.0) * coeffs[123] * x0 * x1 * x3 * x4 * x5 * x52 * x7 *
            x77 * x8 * x9 * x[1] +
        (625.0 / 6912.0) * coeffs[124] * x0 * x1 * x25 * x3 * x4 * x56 * x7 *
            x77 * x8 * x9 * x[1] +
        (625.0 / 3456.0) * coeffs[126] * x0 * x1 * x3 * x4 * x48 * x56 * x7 *
            x77 * x8 * x9 * x[1] +
        (625.0 / 3456.0) * coeffs[129] * x0 * x1 * x3 * x33 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[1] -
        1.0 / 13824.0 * coeffs[12] * x172 * x181 * x38 +
        (625.0 / 6912.0) * coeffs[131] * x0 * x1 * x3 * x5 * x52 * x56 * x7 *
            x77 * x8 * x9 * x[1] +
        (625.0 / 13824.0) * coeffs[132] * x0 * x1 * x25 * x4 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[1] +
        (625.0 / 6912.0) * coeffs[134] * x0 * x1 * x4 * x48 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[1] +
        (625.0 / 13824.0) * coeffs[136] * x0 * x25 * x3 * x4 * x5 * x7 * x77 *
            x8 * x9 * x[1] * x[2] +
        (625.0 / 6912.0) * coeffs[138] * x0 * x3 * x4 * x48 * x5 * x7 * x77 *
            x8 * x9 * x[1] * x[2] +
        (25.0 / 6912.0) * coeffs[13] * x0 * x1 * x172 * x3 * x4 * x56 * x7 *
            x77 * x8 * x9 * x[1] -
        1.0 / 13824.0 * coeffs[140] * x141 * x25 * x59 +
        (625.0 / 3456.0) * coeffs[141] * x0 * x3 * x33 * x4 * x56 * x7 * x77 *
            x8 * x9 * x[1] * x[2] +
        (625.0 / 6912.0) * coeffs[143] * x0 * x3 * x4 * x52 * x56 * x7 * x77 *
            x8 * x9 * x[1] * x[2] +
        (625.0 / 6912.0) * coeffs[144] * x0 * x25 * x3 * x5 * x56 * x7 * x77 *
            x8 * x9 * x[1] * x[2] +
        (625.0 / 3456.0) * coeffs[146] * x0 * x3 * x48 * x5 * x56 * x7 * x77 *
            x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[147] * x141 * x52 * x66 -
        1.0 / 13824.0 * coeffs[148] * x133 * x148 +
        (625.0 / 6912.0) * coeffs[149] * x0 * x33 * x4 * x5 * x56 * x7 * x77 *
            x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[14] * x172 * x182 * x67 +
        (625.0 / 13824.0) * coeffs[151] * x0 * x4 * x5 * x52 * x56 * x7 * x77 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 13824.0) * coeffs[152] * x0 * x1 * x25 * x3 * x4 * x5 * x7 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[154] * x0 * x1 * x3 * x4 * x48 * x5 * x7 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[157] * x0 * x1 * x3 * x33 * x4 * x56 * x7 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[159] * x0 * x1 * x3 * x4 * x52 * x56 * x7 *
            x8 * x9 * x[1] * x[2] +
        (25.0 / 13824.0) * coeffs[15] * x0 * x1 * x172 * x4 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[1] +
        (15625.0 / 6912.0) * coeffs[160] * x0 * x1 * x25 * x3 * x5 * x56 * x7 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[162] * x0 * x1 * x3 * x48 * x5 * x56 * x7 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[165] * x0 * x1 * x33 * x4 * x5 * x56 * x7 *
            x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[166] * x42 * x48 * x73 +
        (15625.0 / 13824.0) * coeffs[167] * x0 * x1 * x4 * x5 * x52 * x56 * x7 *
            x8 * x9 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[169] * x0 * x1 * x3 * x33 * x4 * x5 * x7 *
            x77 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[16] * x124 * x170 -
        1.0 / 13824.0 * coeffs[170] * x83 * x85 +
        (15625.0 / 6912.0) * coeffs[171] * x0 * x1 * x3 * x4 * x5 * x52 * x7 *
            x77 * x8 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[172] * x0 * x1 * x25 * x3 * x4 * x56 * x7 *
            x77 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[173] * x86 * x87 +
        (15625.0 / 1728.0) * coeffs[174] * x0 * x1 * x3 * x4 * x48 * x56 * x7 *
            x77 * x8 * x[1] * x[2] +
        (15625.0 / 1728.0) * coeffs[177] * x0 * x1 * x3 * x33 * x5 * x56 * x7 *
            x77 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[178] * x83 * x93 +
        (15625.0 / 3456.0) * coeffs[179] * x0 * x1 * x3 * x5 * x52 * x56 * x7 *
            x77 * x8 * x[1] * x[2] +
        (25.0 / 6912.0) * coeffs[17] * x1 * x3 * x33 * x4 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[1] +
        (15625.0 / 6912.0) * coeffs[180] * x0 * x1 * x25 * x4 * x5 * x56 * x7 *
            x77 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[181] * x86 * x94 +
        (15625.0 / 3456.0) * coeffs[182] * x0 * x1 * x4 * x48 * x5 * x56 * x7 *
            x77 * x8 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[184] * x0 * x1 * x25 * x3 * x4 * x5 * x7 *
            x77 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[185] * x85 * x99 +
        (15625.0 / 3456.0) * coeffs[186] * x0 * x1 * x3 * x4 * x48 * x5 * x7 *
            x77 * x9 * x[1] * x[2] +
        (15625.0 / 1728.0) * coeffs[189] * x0 * x1 * x3 * x33 * x4 * x56 * x7 *
            x77 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[18] * x136 * x164 -
        1.0 / 13824.0 * coeffs[190] * x104 * x87 +
        (15625.0 / 3456.0) * coeffs[191] * x0 * x1 * x3 * x4 * x52 * x56 * x7 *
            x77 * x9 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[192] * x0 * x1 * x25 * x3 * x5 * x56 * x7 *
            x77 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[193] * x93 * x99 +
        (15625.0 / 1728.0) * coeffs[194] * x0 * x1 * x3 * x48 * x5 * x56 * x7 *
            x77 * x9 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[197] * x0 * x1 * x33 * x4 * x5 * x56 * x7 *
            x77 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[198] * x104 * x94 +
        (15625.0 / 6912.0) * coeffs[199] * x0 * x1 * x4 * x5 * x52 * x56 * x7 *
            x77 * x9 * x[1] * x[2] +
        (25.0 / 13824.0) * coeffs[19] * x1 * x3 * x4 * x5 * x52 * x56 * x7 *
            x77 * x8 * x9 * x[1] +
        (1.0 / 13824.0) * coeffs[1] * x0 * x1 * x172 * x3 * x4 * x5 * x56 * x7 *
            x77 * x8 * x9 +
        (15625.0 / 6912.0) * coeffs[201] * x0 * x1 * x3 * x33 * x4 * x5 *
            x77 * x8 * x9 * x[1] * x[2] +
        (15625.0 / 13824.0) * coeffs[203] * x0 * x1 * x3 * x4 * x5 * x52 *
            x77 * x8 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[204] * x0 * x1 * x25 * x3 * x4 * x56 *
            x77 * x8 * x9 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[206] * x0 * x1 * x3 * x4 * x48 * x56 *
            x77 * x8 * x9 * x[1] * x[2] +
        (15625.0 / 3456.0) * coeffs[209] * x0 * x1 * x3 * x33 * x5 * x56 *
            x77 * x8 * x9 * x[1] * x[2] +
        (25.0 / 13824.0) * coeffs[20] * x0 * x1 * x186 * x3 * x4 * x5 * x7 *
            x77 * x8 * x9 * x[1] +
        (15625.0 / 6912.0) * coeffs[211] * x0 * x1 * x3 * x5 * x52 * x56 *
            x77 * x8 * x9 * x[1] * x[2] +
        (15625.0 / 13824.0) * coeffs[212] * x0 * x1 * x25 * x4 * x5 * x56 *
            x77 * x8 * x9 * x[1] * x[2] +
        (15625.0 / 6912.0) * coeffs[214] * x0 * x1 * x4 * x48 * x5 * x56 *
            x77 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[21] * x182 * x186 * x60 +
        (25.0 / 6912.0) * coeffs[22] * x0 * x1 * x186 * x3 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[1] -
        1.0 / 13824.0 * coeffs[23] * x181 * x186 * x73 -
        1.0 / 13824.0 * coeffs[24] * x138 * x170 +
        (25.0 / 6912.0) * coeffs[25] * x0 * x3 * x33 * x4 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[26] * x138 * x164 * x48 +
        (25.0 / 13824.0) * coeffs[27] * x0 * x3 * x4 * x5 * x52 * x56 * x7 *
            x77 * x8 * x9 * x[2] +
        (25.0 / 13824.0) * coeffs[28] * x0 * x172 * x3 * x4 * x5 * x7 *
            x77 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[2] * x124 * x185 +
        (25.0 / 6912.0) * coeffs[30] * x0 * x172 * x3 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[31] * x148 * x178 +
        (25.0 / 13824.0) * coeffs[32] * x25 * x3 * x4 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[33] * x162 * x164 * x33 +
        (25.0 / 6912.0) * coeffs[34] * x3 * x4 * x48 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[35] * x162 * x166 -
        1.0 / 13824.0 * coeffs[36] * x143 * x190 * x36 +
        (25.0 / 6912.0) * coeffs[37] * x0 * x186 * x3 * x4 * x56 * x7 *
            x77 * x8 * x9 * x[1] * x[2] +
        (25.0 / 13824.0) * coeffs[39] * x0 * x186 * x4 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[1] * x[2] +
        (1.0 / 13824.0) * coeffs[3] * x1 * x186 * x3 * x4 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[1] +
        (25.0 / 13824.0) * coeffs[40] * x0 * x1 * x186 * x3 * x4 * x5 *
            x56 * x7 * x8 * x9 * x[2] +
        (25.0 / 6912.0) * coeffs[42] * x0 * x1 * x186 * x3 * x4 * x5 *
            x56 * x7 * x77 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[43] * x152 * x190 -
        1.0 / 13824.0 * coeffs[44] * x167 * x178 +
        (25.0 / 6912.0) * coeffs[45] * x0 * x1 * x172 * x3 * x4 * x5 *
            x56 * x7 * x77 * x8 * x[2] +
        (25.0 / 13824.0) * coeffs[47] * x0 * x1 * x172 * x3 * x4 * x5 * x56 *
            x77 * x8 * x9 * x[2] +
        (25.0 / 13824.0) * coeffs[48] * x1 * x172 * x3 * x4 * x5 *
            x56 * x7 * x8 * x9 * x[1] * x[2] +
        (1.0 / 13824.0) * coeffs[4] * x0 * x186 * x3 * x4 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[2] +
        (25.0 / 6912.0) * coeffs[50] * x1 * x172 * x3 * x4 * x5 * x56 * x7 *
            x77 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[51] * x134 * x178 -
        1.0 / 13824.0 * coeffs[52] * x129 * x190 * x41 +
        (25.0 / 6912.0) * coeffs[53] * x1 * x186 * x3 * x4 * x5 * x56 * x7 *
            x77 * x8 * x[1] * x[2] +
        (25.0 / 13824.0) * coeffs[55] * x1 * x186 * x3 * x4 * x5 * x56 *
            x77 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[56] * x187 * x38 * x41 +
        (625.0 / 6912.0) * coeffs[57] * x0 * x1 * x186 * x3 * x4 *
            x56 * x7 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[58] * x188 * x41 * x67 +
        (625.0 / 13824.0) * coeffs[59] * x0 * x1 * x186 * x4 * x5 *
            x56 * x7 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[5] * x138 * x185 +
        (625.0 / 6912.0) * coeffs[60] * x0 * x1 * x186 * x3 * x4 * x5 * x7 *
            x77 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[61] * x189 * x60 * x80 +
        (625.0 / 3456.0) * coeffs[62] * x0 * x1 * x186 * x3 * x5 * x56 * x7 *
            x77 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[63] * x188 * x73 * x80 -
        1.0 / 13824.0 * coeffs[64] * x188 * x38 * x98 +
        (625.0 / 3456.0) * coeffs[65] * x0 * x1 * x186 * x3 * x4 * x56 * x7 *
            x77 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[66] * x189 * x67 * x98 +
        (625.0 / 6912.0) * coeffs[67] * x0 * x1 * x186 * x4 * x5 * x56 * x7 *
            x77 * x9 * x[1] * x[2] +
        (625.0 / 13824.0) * coeffs[68] * x0 * x1 * x186 * x3 * x4 * x5 *
            x77 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[69] * x108 * x188 * x60 +
        (1.0 / 13824.0) * coeffs[6] * x172 * x3 * x4 * x5 * x56 * x7 *
            x77 * x8 * x9 * x[1] * x[2] +
        (625.0 / 6912.0) * coeffs[70] * x0 * x1 * x186 * x3 * x5 * x56 *
            x77 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[71] * x108 * x187 * x73 +
        (625.0 / 13824.0) * coeffs[72] * x0 * x1 *
            x172 * x3 * x4 * x5 * x7 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[73] * x173 * x41 * x60 +
        (625.0 / 6912.0) * coeffs[74] * x0 * x1 * x172 * x3 * x5 *
            x56 * x7 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[75] * x174 * x41 * x73 -
        1.0 / 13824.0 * coeffs[76] * x173 * x38 * x80 +
        (625.0 / 3456.0) * coeffs[77] * x0 * x1 * x172 * x3 * x4 * x56 * x7 *
            x77 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[78] * x175 * x67 * x80 +
        (625.0 / 6912.0) * coeffs[79] * x0 * x1 * x172 * x4 * x5 * x56 * x7 *
            x77 * x8 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[7] * x162 * x193 +
        (625.0 / 6912.0) * coeffs[80] * x0 * x1 * x172 * x3 * x4 * x5 * x7 *
            x77 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[81] * x175 * x60 * x98 +
        (625.0 / 3456.0) * coeffs[82] * x0 * x1 * x172 * x3 * x5 * x56 * x7 *
            x77 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[83] * x173 * x73 * x98 -
        1.0 / 13824.0 * coeffs[84] * x108 * x174 * x38 +
        (625.0 / 6912.0) * coeffs[85] * x0 * x1 * x172 * x3 * x4 * x56 *
            x77 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[86] * x108 * x173 * x67 +
        (625.0 / 13824.0) * coeffs[87] * x0 * x1 * x172 * x4 * x5 * x56 *
            x77 * x8 * x9 * x[1] * x[2] -
        1.0 / 13824.0 * coeffs[88] * x133 * x167 +
        (625.0 / 6912.0) * coeffs[89] * x0 * x1 * x3 * x33 * x4 * x5 *
            x56 * x7 * x8 * x9 * x[2] +
        (25.0 / 13824.0) * coeffs[8] * x0 * x1 * x25 * x3 * x4 * x5 * x56 * x7 *
            x77 * x8 * x9 -
        1.0 / 13824.0 * coeffs[90] * x168 * x41 * x48 +
        (625.0 / 13824.0) * coeffs[91] * x0 * x1 * x3 * x4 * x5 * x52 *
            x56 * x7 * x8 * x9 * x[2] +
        (625.0 / 6912.0) * coeffs[92] * x0 * x1 * x25 * x3 * x4 * x5 *
            x56 * x7 * x77 * x8 * x[2] -
        1.0 / 13824.0 * coeffs[93] * x169 * x86 +
        (625.0 / 3456.0) * coeffs[94] * x0 * x1 * x3 * x4 * x48 * x5 *
            x56 * x7 * x77 * x8 * x[2] -
        1.0 / 13824.0 * coeffs[95] * x168 * x52 * x80 -
        1.0 / 13824.0 * coeffs[96] * x168 * x25 * x98 +
        (625.0 / 3456.0) * coeffs[97] * x0 * x1 * x3 * x33 * x4 * x5 *
            x56 * x7 * x77 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[98] * x104 * x169 +
        (625.0 / 6912.0) * coeffs[99] * x0 * x1 * x3 * x4 * x5 * x52 *
            x56 * x7 * x77 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[9] * x150 * x164 - 1.0 / 13824.0 * x101 * x53 -
        1.0 / 13824.0 * x103 * x61 - 1.0 / 13824.0 * x105 * x69 -
        1.0 / 13824.0 * x106 * x74 - 1.0 / 13824.0 * x110 * x76 -
        1.0 / 13824.0 * x111 * x113 - 1.0 / 13824.0 * x111 * x155 -
        1.0 / 13824.0 * x114 * x116 - 1.0 / 13824.0 * x114 * x156 -
        1.0 / 13824.0 * x117 * x88 - 1.0 / 13824.0 * x118 * x91 -
        1.0 / 13824.0 * x119 * x120 - 1.0 / 13824.0 * x119 * x159 -
        1.0 / 13824.0 * x121 * x122 - 1.0 / 13824.0 * x121 * x160 -
        1.0 / 13824.0 * x123 * x95 - 1.0 / 13824.0 * x127 * x128 * x33 -
        1.0 / 13824.0 * x130 * x142 * x143 - 1.0 / 13824.0 * x137 * x141 * x33 -
        1.0 / 13824.0 * x141 * x149 * x48 - 1.0 / 13824.0 * x144 * x146 * x48 -
        1.0 / 13824.0 * x146 * x147 * x33 - 1.0 / 13824.0 * x154 * x76 -
        1.0 / 13824.0 * x157 * x88 - 1.0 / 13824.0 * x158 * x91 -
        1.0 / 13824.0 * x161 * x95 - 1.0 / 13824.0 * x172 * x176 * x177 -
        1.0 / 13824.0 * x172 * x179 * x180 -
        1.0 / 13824.0 * x172 * x183 * x184 -
        1.0 / 13824.0 * x176 * x186 * x191 -
        1.0 / 13824.0 * x179 * x186 * x192 -
        1.0 / 13824.0 * x183 * x186 * x194 - 1.0 / 13824.0 * x33 * x38 * x43 -
        1.0 / 13824.0 * x33 * x67 * x68 - 1.0 / 13824.0 * x48 * x60 * x64 -
        1.0 / 13824.0 * x53 * x55 - 1.0 / 13824.0 * x61 * x62 -
        1.0 / 13824.0 * x69 * x70 - 1.0 / 13824.0 * x74 * x75 -
        1.0 / 13824.0 * x76 * x82 - 1.0 / 13824.0 * x88 * x90 -
        1.0 / 13824.0 * x91 * x92 - 1.0 / 13824.0 * x95 * x96;
    out[1] =
        -1.0 / 13824.0 * coeffs[0] * x273 * x296 +
        (625.0 / 13824.0) * coeffs[100] * x1 * x11 * x12 * x13 * x17 * x295 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[101] * x276 * x297 +
        (625.0 / 6912.0) * coeffs[102] * x1 * x11 * x13 * x17 * x26 * x295 *
            x77 * x8 * x9 * x[0] * x[2] +
        (625.0 / 13824.0) * coeffs[104] * x1 * x11 * x12 * x13 * x17 * x274 *
            x7 * x8 * x9 * x[0] * x[2] +
        (625.0 / 6912.0) * coeffs[106] * x1 * x11 * x13 * x17 * x26 * x274 *
            x7 * x8 * x9 * x[0] * x[2] +
        (625.0 / 3456.0) * coeffs[109] * x1 * x11 * x12 * x13 * x26 * x274 *
            x7 * x77 * x8 * x[0] * x[2] +
        (25.0 / 6912.0) * coeffs[10] * x1 * x11 * x13 * x17 * x26 * x295 * x7 *
            x77 * x8 * x9 * x[0] -
        1.0 / 13824.0 * coeffs[110] * x230 * x283 +
        (625.0 / 6912.0) * coeffs[111] * x1 * x12 * x13 * x17 * x26 * x274 *
            x7 * x77 * x8 * x[0] * x[2] +
        (625.0 / 6912.0) * coeffs[112] * x1 * x11 * x12 * x13 * x17 * x274 *
            x7 * x77 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[113] * x236 * x283 +
        (625.0 / 3456.0) * coeffs[114] * x1 * x11 * x13 * x17 * x26 * x274 *
            x7 * x77 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[116] * x265 * x285 +
        (625.0 / 6912.0) * coeffs[117] * x1 * x11 * x12 * x13 * x26 * x274 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[118] * x108 * x247 * x281 +
        (625.0 / 13824.0) * coeffs[119] * x1 * x12 * x13 * x17 * x26 * x274 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[11] * x288 * x296 +
        (625.0 / 6912.0) * coeffs[121] * x1 * x11 * x12 * x13 * x202 * x26 *
            x7 * x77 * x8 * x9 * x[0] +
        (625.0 / 13824.0) * coeffs[123] * x1 * x12 * x13 * x17 * x202 * x26 *
            x7 * x77 * x8 * x9 * x[0] +
        (625.0 / 6912.0) * coeffs[124] * x1 * x11 * x12 * x13 * x17 * x210 *
            x7 * x77 * x8 * x9 * x[0] +
        (625.0 / 3456.0) * coeffs[126] * x1 * x11 * x13 * x17 * x210 * x26 *
            x7 * x77 * x8 * x9 * x[0] +
        (625.0 / 3456.0) * coeffs[129] * x1 * x11 * x12 * x13 * x215 * x26 *
            x7 * x77 * x8 * x9 * x[0] -
        1.0 / 13824.0 * coeffs[12] * x202 * x269 * x271 +
        (625.0 / 6912.0) * coeffs[131] * x1 * x12 * x13 * x17 * x215 * x26 *
            x7 * x77 * x8 * x9 * x[0] +
        (625.0 / 13824.0) * coeffs[132] * x1 * x11 * x12 * x13 * x17 * x219 *
            x7 * x77 * x8 * x9 * x[0] +
        (625.0 / 6912.0) * coeffs[134] * x1 * x11 * x13 * x17 * x219 * x26 *
            x7 * x77 * x8 * x9 * x[0] +
        (625.0 / 13824.0) * coeffs[136] * x11 * x12 * x13 * x17 * x202 * x7 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[137] * x204 * x241 +
        (625.0 / 6912.0) * coeffs[138] * x11 * x13 * x17 * x202 * x26 * x7 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[139] * x202 * x242 * x243 +
        (25.0 / 6912.0) * coeffs[13] * x1 * x11 * x12 * x17 * x210 * x26 * x7 *
            x77 * x8 * x9 * x[0] +
        (625.0 / 3456.0) * coeffs[141] * x11 * x12 * x13 * x210 * x26 * x7 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[142] * x145 * x244 * x247 +
        (625.0 / 6912.0) * coeffs[143] * x12 * x13 * x17 * x210 * x26 * x7 *
            x77 * x8 * x9 * x[0] * x[2] +
        (625.0 / 6912.0) * coeffs[144] * x11 * x12 * x13 * x17 * x215 * x7 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[145] * x145 * x248 * x249 +
        (625.0 / 3456.0) * coeffs[146] * x11 * x13 * x17 * x215 * x26 * x7 *
            x77 * x8 * x9 * x[0] * x[2] +
        (625.0 / 6912.0) * coeffs[149] * x11 * x12 * x13 * x219 * x26 * x7 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[14] * x215 * x269 * x272 -
        1.0 / 13824.0 * coeffs[150] * x222 * x241 +
        (625.0 / 13824.0) * coeffs[151] * x12 * x13 * x17 * x219 * x26 * x7 *
            x77 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 13824.0) * coeffs[152] * x1 * x11 * x12 * x13 * x17 * x202 *
            x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[154] * x1 * x11 * x13 * x17 * x202 * x26 *
            x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[157] * x1 * x11 * x12 * x13 * x210 * x26 *
            x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[159] * x1 * x12 * x13 * x17 * x210 * x26 *
            x7 * x8 * x9 * x[0] * x[2] +
        (25.0 / 13824.0) * coeffs[15] * x1 * x11 * x12 * x17 * x219 * x26 * x7 *
            x77 * x8 * x9 * x[0] +
        (15625.0 / 6912.0) * coeffs[160] * x1 * x11 * x12 * x13 * x17 * x215 *
            x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[162] * x1 * x11 * x13 * x17 * x215 * x26 *
            x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[165] * x1 * x11 * x12 * x13 * x219 * x26 *
            x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[166] * x203 * x222 * x42 +
        (15625.0 / 13824.0) * coeffs[167] * x1 * x12 * x13 * x17 * x219 * x26 *
            x7 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[169] * x1 * x11 * x12 * x13 * x202 * x26 *
            x7 * x77 * x8 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[16] * x181 * x285 -
        1.0 / 13824.0 * coeffs[170] * x224 * x225 +
        (15625.0 / 6912.0) * coeffs[171] * x1 * x12 * x13 * x17 * x202 *
            x26 * x7 * x77 * x8 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[172] * x1 * x11 * x12 * x13 * x17 *
            x210 * x7 * x77 * x8 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[173] * x226 * x227 +
        (15625.0 / 1728.0) * coeffs[174] * x1 * x11 * x13 * x17 * x210 *
            x26 * x7 * x77 * x8 * x[0] * x[2] +
        (15625.0 / 1728.0) * coeffs[177] * x1 * x11 * x12 * x13 * x215 *
            x26 * x7 * x77 * x8 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[178] * x230 * x231 +
        (15625.0 / 3456.0) * coeffs[179] * x1 * x12 * x13 * x17 * x215 *
            x26 * x7 * x77 * x8 * x[0] * x[2] +
        (25.0 / 6912.0) * coeffs[17] * x1 * x11 * x12 * x13 * x26 * x274 * x7 *
            x77 * x8 * x9 * x[0] +
        (15625.0 / 6912.0) * coeffs[180] * x1 * x11 * x12 * x13 * x17 *
            x219 * x7 * x77 * x8 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[181] * x224 * x232 +
        (15625.0 / 3456.0) * coeffs[182] * x1 * x11 * x13 * x17 * x219 *
            x26 * x7 * x77 * x8 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[184] * x1 * x11 * x12 * x13 * x17 *
            x202 * x7 * x77 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[185] * x204 * x234 +
        (15625.0 / 3456.0) * coeffs[186] * x1 * x11 * x13 * x17 * x202 *
            x26 * x7 * x77 * x9 * x[0] * x[2] +
        (15625.0 / 1728.0) * coeffs[189] * x1 * x11 * x12 * x13 * x210 *
            x26 * x7 * x77 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[18] * x275 * x293 -
        1.0 / 13824.0 * coeffs[190] * x227 * x235 +
        (15625.0 / 3456.0) * coeffs[191] * x1 * x12 * x13 * x17 * x210 *
            x26 * x7 * x77 * x9 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[192] * x1 * x11 * x12 * x13 * x17 *
            x215 * x7 * x77 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[193] * x231 * x236 +
        (15625.0 / 1728.0) * coeffs[194] * x1 * x11 * x13 * x17 * x215 *
            x26 * x7 * x77 * x9 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[197] * x1 * x11 * x12 * x13 * x219 *
            x26 * x7 * x77 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[198] * x222 * x234 +
        (15625.0 / 6912.0) * coeffs[199] * x1 * x12 * x13 * x17 * x219 *
            x26 * x7 * x77 * x9 * x[0] * x[2] +
        (25.0 / 13824.0) * coeffs[19] * x1 * x12 * x13 * x17 * x26 * x274 * x7 *
            x77 * x8 * x9 * x[0] +
        (1.0 / 13824.0) * coeffs[1] * x1 * x11 * x12 * x17 * x26 * x295 * x7 *
            x77 * x8 * x9 * x[0] +
        (15625.0 / 6912.0) * coeffs[201] * x1 * x11 * x12 * x13 * x202 * x26 *
            x77 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 13824.0) * coeffs[203] * x1 * x12 * x13 * x17 * x202 * x26 *
            x77 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[204] * x1 * x11 * x12 * x13 * x17 * x210 *
            x77 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[206] * x1 * x11 * x13 * x17 * x210 * x26 *
            x77 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 3456.0) * coeffs[209] * x1 * x11 * x12 * x13 * x215 * x26 *
            x77 * x8 * x9 * x[0] * x[2] +
        (25.0 / 13824.0) * coeffs[20] * x1 * x11 * x12 * x13 * x17 * x202 *
            x26 * x7 * x77 * x8 * x9 +
        (15625.0 / 6912.0) * coeffs[211] * x1 * x12 * x13 * x17 * x215 * x26 *
            x77 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 13824.0) * coeffs[212] * x1 * x11 * x12 * x13 * x17 * x219 *
            x77 * x8 * x9 * x[0] * x[2] +
        (15625.0 / 6912.0) * coeffs[214] * x1 * x11 * x13 * x17 * x219 * x26 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[21] * x210 * x272 * x273 +
        (25.0 / 6912.0) * coeffs[22] * x1 * x11 * x12 * x13 * x17 * x215 *
            x26 * x7 * x77 * x8 * x9 -
        1.0 / 13824.0 * coeffs[23] * x219 * x271 * x273 -
        1.0 / 13824.0 * coeffs[24] * x181 * x245 * x304 +
        (25.0 / 6912.0) * coeffs[25] * x11 * x12 * x13 * x26 * x295 * x7 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[26] * x293 * x304 +
        (25.0 / 13824.0) * coeffs[27] * x12 * x13 * x17 * x26 * x295 * x7 *
            x77 * x8 * x9 * x[0] * x[2] +
        (25.0 / 13824.0) * coeffs[28] * x11 * x12 * x17 * x202 * x26 * x7 *
            x77 * x8 * x9 * x[0] * x[2] -
        25.0 / 6912.0 * coeffs[29] * x244 * x269 -
        1.0 / 13824.0 * coeffs[2] * x269 * x275 +
        (25.0 / 6912.0) * coeffs[30] * x11 * x12 * x17 * x215 * x26 * x7 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[31] * x219 * x269 * x270 +
        (25.0 / 13824.0) * coeffs[32] * x11 * x12 * x13 * x17 * x274 * x7 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[33] * x286 * x287 +
        (25.0 / 6912.0) * coeffs[34] * x11 * x13 * x17 * x26 * x274 * x7 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[35] * x286 * x288 -
        1.0 / 13824.0 * coeffs[36] * x202 * x270 * x273 +
        (25.0 / 6912.0) * coeffs[37] * x11 * x12 * x13 * x17 * x210 * x26 * x7 *
            x77 * x8 * x9 * x[2] -
        25.0 / 6912.0 * coeffs[38] * x249 * x273 +
        (25.0 / 13824.0) * coeffs[39] * x11 * x12 * x13 * x17 * x219 *
            x26 * x7 * x77 * x8 * x9 * x[2] +
        (1.0 / 13824.0) * coeffs[3] * x1 * x11 * x12 * x13 * x17 * x26 *
            x274 * x7 * x77 * x8 * x9 +
        (25.0 / 13824.0) * coeffs[40] * x1 * x11 * x12 * x13 * x17 * x26 *
            x295 * x7 * x8 * x9 * x[2] +
        (25.0 / 6912.0) * coeffs[42] * x1 * x11 * x12 * x13 * x17 * x26 *
            x295 * x7 * x77 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[43] * x294 * x297 -
        1.0 / 13824.0 * coeffs[44] * x292 * x296 * x41 +
        (25.0 / 6912.0) * coeffs[45] * x1 * x11 * x12 * x17 * x26 * x295 * x7 *
            x77 * x8 * x[0] * x[2] +
        (25.0 / 13824.0) * coeffs[47] * x1 * x11 * x12 * x17 * x26 * x295 *
            x77 * x8 * x9 * x[0] * x[2] +
        (25.0 / 13824.0) * coeffs[48] * x1 * x11 * x12 * x17 * x26 *
            x274 * x7 * x8 * x9 * x[0] * x[2] +
        (1.0 / 13824.0) * coeffs[4] * x11 * x12 * x13 * x17 * x26 * x295 * x7 *
            x77 * x8 * x9 * x[2] +
        (25.0 / 6912.0) * coeffs[50] * x1 * x11 * x12 * x17 * x26 * x274 * x7 *
            x77 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[51] * x108 * x275 * x292 -
        1.0 / 13824.0 * coeffs[52] * x277 * x294 +
        (25.0 / 6912.0) * coeffs[53] * x1 * x11 * x12 * x13 * x17 * x26 *
            x274 * x7 * x77 * x8 * x[2] +
        (25.0 / 13824.0) * coeffs[55] * x1 * x11 * x12 * x13 * x17 * x26 *
            x274 * x77 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[56] * x171 * x205 * x257 +
        (625.0 / 6912.0) * coeffs[57] * x1 * x11 * x12 * x13 * x17 * x210 *
            x26 * x7 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[58] * x216 * x255 +
        (625.0 / 13824.0) * coeffs[59] * x1 * x11 * x12 * x13 * x17 * x219 *
            x26 * x7 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[5] * x269 * x304 +
        (625.0 / 6912.0) * coeffs[60] * x1 * x11 * x12 * x13 * x17 * x202 *
            x26 * x7 * x77 * x8 * x[2] -
        1.0 / 13824.0 * coeffs[61] * x211 * x261 +
        (625.0 / 3456.0) * coeffs[62] * x1 * x11 * x12 * x13 * x17 * x215 *
            x26 * x7 * x77 * x8 * x[2] -
        1.0 / 13824.0 * coeffs[63] * x220 * x258 -
        1.0 / 13824.0 * coeffs[64] * x205 * x264 +
        (625.0 / 3456.0) * coeffs[65] * x1 * x11 * x12 * x13 * x17 * x210 *
            x26 * x7 * x77 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[66] * x216 * x263 +
        (625.0 / 6912.0) * coeffs[67] * x1 * x11 * x12 * x13 * x17 * x219 *
            x26 * x7 * x77 * x9 * x[2] +
        (625.0 / 13824.0) * coeffs[68] * x1 * x11 * x12 * x13 * x17 * x202 *
            x26 * x77 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[69] * x211 * x267 +
        (1.0 / 13824.0) * coeffs[6] * x11 * x12 * x17 * x26 * x274 * x7 *
            x77 * x8 * x9 * x[0] * x[2] +
        (625.0 / 6912.0) * coeffs[70] * x1 * x11 * x12 * x13 * x17 * x215 *
            x26 * x77 * x8 * x9 * x[2] -
        1.0 / 13824.0 * coeffs[71] * x220 * x266 +
        (625.0 / 13824.0) * coeffs[72] * x1 * x11 * x12 * x17 * x202 *
            x26 * x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[73] * x253 * x255 +
        (625.0 / 6912.0) * coeffs[74] * x1 * x11 * x12 * x17 * x215 *
            x26 * x7 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[75] * x171 * x256 * x257 -
        1.0 / 13824.0 * coeffs[76] * x258 * x259 +
        (625.0 / 3456.0) * coeffs[77] * x1 * x11 * x12 * x17 * x210 * x26 * x7 *
            x77 * x8 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[78] * x261 * x262 +
        (625.0 / 6912.0) * coeffs[79] * x1 * x11 * x12 * x17 * x219 * x26 * x7 *
            x77 * x8 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[7] * x273 * x286 +
        (625.0 / 6912.0) * coeffs[80] * x1 * x11 * x12 * x17 * x202 * x26 * x7 *
            x77 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[81] * x253 * x263 +
        (625.0 / 3456.0) * coeffs[82] * x1 * x11 * x12 * x17 * x215 * x26 * x7 *
            x77 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[83] * x256 * x264 -
        1.0 / 13824.0 * coeffs[84] * x259 * x266 +
        (625.0 / 6912.0) * coeffs[85] * x1 * x11 * x12 * x17 * x210 * x26 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[86] * x262 * x267 +
        (625.0 / 13824.0) * coeffs[87] * x1 * x11 * x12 * x17 * x219 * x26 *
            x77 * x8 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[88] * x245 * x257 * x296 +
        (625.0 / 6912.0) * coeffs[89] * x1 * x11 * x12 * x13 * x26 *
            x295 * x7 * x8 * x9 * x[0] * x[2] +
        (25.0 / 13824.0) * coeffs[8] * x1 * x11 * x12 * x13 * x17 * x295 * x7 *
            x77 * x8 * x9 * x[0] +
        (625.0 / 13824.0) * coeffs[91] * x1 * x12 * x13 * x17 * x26 *
            x295 * x7 * x8 * x9 * x[0] * x[2] +
        (625.0 / 6912.0) * coeffs[92] * x1 * x11 * x12 * x13 * x17 * x295 * x7 *
            x77 * x8 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[93] * x226 * x301 +
        (625.0 / 3456.0) * coeffs[94] * x1 * x11 * x13 * x17 * x26 * x295 * x7 *
            x77 * x8 * x[0] * x[2] +
        (625.0 / 3456.0) * coeffs[97] * x1 * x11 * x12 * x13 * x26 * x295 * x7 *
            x77 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[98] * x235 * x301 +
        (625.0 / 6912.0) * coeffs[99] * x1 * x12 * x13 * x17 * x26 * x295 * x7 *
            x77 * x9 * x[0] * x[2] -
        1.0 / 13824.0 * coeffs[9] * x287 * x296 - 1.0 / 13824.0 * x101 * x206 -
        1.0 / 13824.0 * x103 * x212 - 1.0 / 13824.0 * x105 * x217 -
        1.0 / 13824.0 * x106 * x221 - 1.0 / 13824.0 * x110 * x223 -
        1.0 / 13824.0 * x113 * x237 - 1.0 / 13824.0 * x116 * x238 -
        1.0 / 13824.0 * x117 * x228 - 1.0 / 13824.0 * x118 * x229 -
        1.0 / 13824.0 * x120 * x239 - 1.0 / 13824.0 * x122 * x240 -
        1.0 / 13824.0 * x123 * x233 - 1.0 / 13824.0 * x128 * x275 * x276 -
        1.0 / 13824.0 * x13 * x192 * x290 - 1.0 / 13824.0 * x13 * x194 * x305 -
        1.0 / 13824.0 * x140 * x244 * x246 -
        1.0 / 13824.0 * x140 * x249 * x250 - 1.0 / 13824.0 * x154 * x223 -
        1.0 / 13824.0 * x155 * x237 - 1.0 / 13824.0 * x156 * x238 -
        1.0 / 13824.0 * x157 * x228 - 1.0 / 13824.0 * x158 * x229 -
        1.0 / 13824.0 * x159 * x239 - 1.0 / 13824.0 * x160 * x240 -
        1.0 / 13824.0 * x161 * x233 - 1.0 / 13824.0 * x180 * x290 * x[0] -
        1.0 / 13824.0 * x184 * x305 * x[0] - 1.0 / 13824.0 * x203 * x204 * x43 -
        1.0 / 13824.0 * x206 * x55 - 1.0 / 13824.0 * x211 * x46 * x64 -
        1.0 / 13824.0 * x212 * x62 - 1.0 / 13824.0 * x216 * x31 * x68 -
        1.0 / 13824.0 * x217 * x70 - 1.0 / 13824.0 * x219 * x243 * x251 -
        1.0 / 13824.0 * x221 * x75 - 1.0 / 13824.0 * x223 * x82 -
        1.0 / 13824.0 * x228 * x90 - 1.0 / 13824.0 * x229 * x92 -
        1.0 / 13824.0 * x233 * x96 - 1.0 / 13824.0 * x277 * x279 -
        1.0 / 13824.0 * x280 * x281 * x80 - 1.0 / 13824.0 * x281 * x284 * x98 -
        1.0 / 13824.0 * x297 * x298 - 1.0 / 13824.0 * x299 * x300 * x41 -
        1.0 / 13824.0 * x299 * x302 * x80 - 1.0 / 13824.0 * x299 * x303 * x98;
    out[2] =
        -1.0 / 13824.0 * coeffs[0] * x377 * x394 +
        (625.0 / 13824.0) * coeffs[100] * x0 * x11 * x12 * x13 * x17 * x3 *
            x350 * x4 * x5 * x56 * x[0] -
        1.0 / 13824.0 * coeffs[101] * x354 * x365 +
        (625.0 / 6912.0) * coeffs[102] * x0 * x11 * x13 * x17 * x26 * x3 *
            x350 * x4 * x5 * x56 * x[0] +
        (625.0 / 13824.0) * coeffs[104] * x11 * x12 * x13 * x17 * x3 * x313 *
            x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[105] * x354 * x355 +
        (625.0 / 6912.0) * coeffs[106] * x11 * x13 * x17 * x26 * x3 * x313 *
            x4 * x5 * x56 * x[0] * x[1] +
        (625.0 / 3456.0) * coeffs[109] * x11 * x12 * x13 * x26 * x3 * x328 *
            x4 * x5 * x56 * x[0] * x[1] +
        (25.0 / 6912.0) * coeffs[10] * x0 * x11 * x13 * x17 * x26 * x3 * x393 *
            x4 * x5 * x56 * x[0] -
        1.0 / 13824.0 * coeffs[110] * x331 * x357 +
        (625.0 / 6912.0) * coeffs[111] * x12 * x13 * x17 * x26 * x3 * x328 *
            x4 * x5 * x56 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[112] * x11 * x12 * x13 * x17 * x3 * x344 *
            x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[113] * x345 * x357 +
        (625.0 / 3456.0) * coeffs[114] * x11 * x13 * x17 * x26 * x3 * x344 *
            x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[116] * x359 * x360 +
        (625.0 / 6912.0) * coeffs[117] * x11 * x12 * x13 * x26 * x3 * x350 *
            x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[118] * x247 * x350 * x356 +
        (625.0 / 13824.0) * coeffs[119] * x12 * x13 * x17 * x26 * x3 * x350 *
            x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[11] * x390 * x394 -
        1.0 / 13824.0 * coeffs[120] * x358 * x395 +
        (625.0 / 6912.0) * coeffs[121] * x0 * x11 * x12 * x13 * x26 * x3 *
            x393 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[122] * x384 * x395 +
        (625.0 / 13824.0) * coeffs[123] * x0 * x12 * x13 * x17 * x26 * x3 *
            x393 * x4 * x5 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[124] * x0 * x11 * x12 * x13 * x17 * x3 *
            x393 * x4 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[125] * x31 * x396 * x59 +
        (625.0 / 3456.0) * coeffs[126] * x0 * x11 * x13 * x17 * x26 * x3 *
            x393 * x4 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[127] * x242 * x397 * x59 -
        1.0 / 13824.0 * coeffs[128] * x245 * x397 * x66 +
        (625.0 / 3456.0) * coeffs[129] * x0 * x11 * x12 * x13 * x26 * x3 *
            x393 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[12] * x292 * x395 -
        1.0 / 13824.0 * coeffs[130] * x396 * x46 * x66 +
        (625.0 / 6912.0) * coeffs[131] * x0 * x12 * x13 * x17 * x26 * x3 *
            x393 * x5 * x56 * x[0] * x[1] +
        (625.0 / 13824.0) * coeffs[132] * x0 * x11 * x12 * x13 * x17 * x393 *
            x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[133] * x276 * x398 +
        (625.0 / 6912.0) * coeffs[134] * x0 * x11 * x13 * x17 * x26 * x393 *
            x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[135] * x278 * x398 +
        (625.0 / 13824.0) * coeffs[136] * x0 * x11 * x12 * x13 * x17 * x3 *
            x379 * x4 * x5 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[138] * x0 * x11 * x13 * x17 * x26 * x3 *
            x379 * x4 * x5 * x[0] * x[1] +
        (25.0 / 6912.0) * coeffs[13] * x0 * x11 * x12 * x17 * x26 * x3 * x393 *
            x4 * x56 * x[0] * x[1] +
        (625.0 / 3456.0) * coeffs[141] * x0 * x11 * x12 * x13 * x26 * x3 *
            x379 * x4 * x56 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[143] * x0 * x12 * x13 * x17 * x26 * x3 *
            x379 * x4 * x56 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[144] * x0 * x11 * x12 * x13 * x17 * x3 *
            x379 * x5 * x56 * x[0] * x[1] +
        (625.0 / 3456.0) * coeffs[146] * x0 * x11 * x13 * x17 * x26 * x3 *
            x379 * x5 * x56 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[149] * x0 * x11 * x12 * x13 * x26 * x379 *
            x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[14] * x399 * x66 * x[0] +
        (625.0 / 13824.0) * coeffs[151] * x0 * x12 * x13 * x17 * x26 * x379 *
            x4 * x5 * x56 * x[0] * x[1] +
        (15625.0 / 13824.0) * coeffs[152] * x0 * x11 * x12 * x13 * x17 * x3 *
            x313 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[153] * x248 * x314 * x315 +
        (15625.0 / 6912.0) * coeffs[154] * x0 * x11 * x13 * x17 * x26 * x3 *
            x313 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[155] * x316 * x317 -
        1.0 / 13824.0 * coeffs[156] * x314 * x319 +
        (15625.0 / 3456.0) * coeffs[157] * x0 * x11 * x12 * x13 * x26 * x3 *
            x313 * x4 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[158] * x247 * x318 * x320 +
        (15625.0 / 6912.0) * coeffs[159] * x0 * x12 * x13 * x17 * x26 * x3 *
            x313 * x4 * x56 * x[0] * x[1] +
        (25.0 / 13824.0) * coeffs[15] * x0 * x11 * x12 * x17 * x26 * x393 * x4 *
            x5 * x56 * x[0] * x[1] +
        (15625.0 / 6912.0) * coeffs[160] * x0 * x11 * x12 * x13 * x17 * x3 *
            x313 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[161] * x248 * x320 * x321 +
        (15625.0 / 3456.0) * coeffs[162] * x0 * x11 * x13 * x17 * x26 * x3 *
            x313 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[163] * x314 * x322 -
        1.0 / 13824.0 * coeffs[164] * x317 * x324 +
        (15625.0 / 6912.0) * coeffs[165] * x0 * x11 * x12 * x13 * x26 * x313 *
            x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[166] * x247 * x314 * x323 +
        (15625.0 / 13824.0) * coeffs[167] * x0 * x12 * x13 * x17 * x26 *
            x313 * x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[168] * x329 * x330 +
        (15625.0 / 3456.0) * coeffs[169] * x0 * x11 * x12 * x13 * x26 * x3 *
            x328 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[16] * x391 * x400 -
        1.0 / 13824.0 * coeffs[170] * x331 * x333 +
        (15625.0 / 6912.0) * coeffs[171] * x0 * x12 * x13 * x17 * x26 * x3 *
            x328 * x4 * x5 * x[0] * x[1] +
        (15625.0 / 3456.0) * coeffs[172] * x0 * x11 * x12 * x13 * x17 * x3 *
            x328 * x4 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[173] * x334 * x335 +
        (15625.0 / 1728.0) * coeffs[174] * x0 * x11 * x13 * x17 * x26 * x3 *
            x328 * x4 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[175] * x336 * x337 -
        1.0 / 13824.0 * coeffs[176] * x336 * x338 +
        (15625.0 / 1728.0) * coeffs[177] * x0 * x11 * x12 * x13 * x26 * x3 *
            x328 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[178] * x331 * x339 +
        (15625.0 / 3456.0) * coeffs[179] * x0 * x12 * x13 * x17 * x26 * x3 *
            x328 * x5 * x56 * x[0] * x[1] +
        (25.0 / 6912.0) * coeffs[17] * x11 * x12 * x13 * x26 * x3 *
            x393 * x4 * x5 * x56 * x[0] * x[1] +
        (15625.0 / 6912.0) * coeffs[180] * x0 * x11 * x12 * x13 * x17 *
            x328 * x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[181] * x334 * x340 +
        (15625.0 / 3456.0) * coeffs[182] * x0 * x11 * x13 * x17 * x26 *
            x328 * x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[183] * x329 * x341 +
        (15625.0 / 6912.0) * coeffs[184] * x0 * x11 * x12 * x13 * x17 * x3 *
            x344 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[185] * x333 * x345 +
        (15625.0 / 3456.0) * coeffs[186] * x0 * x11 * x13 * x17 * x26 * x3 *
            x344 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[187] * x316 * x346 -
        1.0 / 13824.0 * coeffs[188] * x319 * x347 +
        (15625.0 / 1728.0) * coeffs[189] * x0 * x11 * x12 * x13 * x26 * x3 *
            x344 * x4 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[18] * x392 * x400 -
        1.0 / 13824.0 * coeffs[190] * x335 * x348 +
        (15625.0 / 3456.0) * coeffs[191] * x0 * x12 * x13 * x17 * x26 * x3 *
            x344 * x4 * x56 * x[0] * x[1] +
        (15625.0 / 3456.0) * coeffs[192] * x0 * x11 * x12 * x13 * x17 * x3 *
            x344 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[193] * x339 * x345 +
        (15625.0 / 1728.0) * coeffs[194] * x0 * x11 * x13 * x17 * x26 * x3 *
            x344 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[195] * x322 * x347 -
        1.0 / 13824.0 * coeffs[196] * x324 * x346 +
        (15625.0 / 3456.0) * coeffs[197] * x0 * x11 * x12 * x13 * x26 *
            x344 * x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[198] * x340 * x348 +
        (15625.0 / 6912.0) * coeffs[199] * x0 * x12 * x13 * x17 * x26 *
            x344 * x4 * x5 * x56 * x[0] * x[1] +
        (25.0 / 13824.0) * coeffs[19] * x12 * x13 * x17 * x26 * x3 *
            x393 * x4 * x5 * x56 * x[0] * x[1] +
        (1.0 / 13824.0) * coeffs[1] * x0 * x11 * x12 * x17 * x26 * x3 *
            x393 * x4 * x5 * x56 * x[0] -
        1.0 / 13824.0 * coeffs[200] * x330 * x351 +
        (15625.0 / 6912.0) * coeffs[201] * x0 * x11 * x12 * x13 * x26 * x3 *
            x350 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[202] * x247 * x315 * x352 +
        (15625.0 / 13824.0) * coeffs[203] * x0 * x12 * x13 * x17 * x26 * x3 *
            x350 * x4 * x5 * x[0] * x[1] +
        (15625.0 / 6912.0) * coeffs[204] * x0 * x11 * x12 * x13 * x17 * x3 *
            x350 * x4 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[205] * x248 * x318 * x353 +
        (15625.0 / 3456.0) * coeffs[206] * x0 * x11 * x13 * x17 * x26 * x3 *
            x350 * x4 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[207] * x337 * x352 -
        1.0 / 13824.0 * coeffs[208] * x338 * x352 +
        (15625.0 / 3456.0) * coeffs[209] * x0 * x11 * x12 * x13 * x26 * x3 *
            x350 * x5 * x56 * x[0] * x[1] +
        (25.0 / 13824.0) * coeffs[20] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x393 * x4 * x5 * x[1] -
        1.0 / 13824.0 * coeffs[210] * x247 * x321 * x353 +
        (15625.0 / 6912.0) * coeffs[211] * x0 * x12 * x13 * x17 * x26 * x3 *
            x350 * x5 * x56 * x[0] * x[1] +
        (15625.0 / 13824.0) * coeffs[212] * x0 * x11 * x12 * x13 * x17 *
            x350 * x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[213] * x248 * x323 * x352 +
        (15625.0 / 6912.0) * coeffs[214] * x0 * x11 * x13 * x17 * x26 *
            x350 * x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[215] * x341 * x351 -
        1.0 / 13824.0 * coeffs[21] * x13 * x399 * x59 +
        (25.0 / 6912.0) * coeffs[22] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x393 * x5 * x56 * x[1] -
        1.0 / 13824.0 * coeffs[23] * x294 * x398 -
        1.0 / 13824.0 * coeffs[24] * x380 * x391 +
        (25.0 / 6912.0) * coeffs[25] * x0 * x11 * x12 * x13 * x26 * x3 *
            x379 * x4 * x5 * x56 * x[0] -
        1.0 / 13824.0 * coeffs[26] * x380 * x392 +
        (25.0 / 13824.0) * coeffs[27] * x0 * x12 * x13 * x17 * x26 * x3 *
            x379 * x4 * x5 * x56 * x[0] +
        (25.0 / 13824.0) * coeffs[28] * x0 * x11 * x12 * x17 * x26 * x3 *
            x379 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[2] * x367 * x400 +
        (25.0 / 6912.0) * coeffs[30] * x0 * x11 * x12 * x17 * x26 * x3 *
            x379 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[31] * x292 * x383 +
        (25.0 / 13824.0) * coeffs[32] * x11 * x12 * x13 * x17 * x3 *
            x379 * x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[33] * x386 * x388 +
        (25.0 / 6912.0) * coeffs[34] * x11 * x13 * x17 * x26 * x3 *
            x379 * x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[35] * x386 * x390 -
        1.0 / 13824.0 * coeffs[36] * x294 * x36 * x380 +
        (25.0 / 6912.0) * coeffs[37] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x379 * x4 * x56 * x[1] +
        (25.0 / 13824.0) * coeffs[39] * x0 * x11 * x12 * x13 * x17 * x26 *
            x379 * x4 * x5 * x56 * x[1] +
        (1.0 / 13824.0) * coeffs[3] * x11 * x12 * x13 * x17 * x26 * x3 *
            x393 * x4 * x5 * x56 * x[1] +
        (25.0 / 13824.0) * coeffs[40] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x313 * x4 * x5 * x56 -
        1.0 / 13824.0 * coeffs[41] * x328 * x376 * x377 +
        (25.0 / 6912.0) * coeffs[42] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x344 * x4 * x5 * x56 -
        1.0 / 13824.0 * coeffs[43] * x365 * x378 -
        1.0 / 13824.0 * coeffs[44] * x369 * x373 +
        (25.0 / 6912.0) * coeffs[45] * x0 * x11 * x12 * x17 * x26 * x3 *
            x328 * x4 * x5 * x56 * x[0] -
        1.0 / 13824.0 * coeffs[46] * x344 * x367 * x376 +
        (25.0 / 13824.0) * coeffs[47] * x0 * x11 * x12 * x17 * x26 * x3 *
            x350 * x4 * x5 * x56 * x[0] +
        (25.0 / 13824.0) * coeffs[48] * x11 * x12 * x17 * x26 * x3 *
            x313 * x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[49] * x328 * x367 * x368 +
        (1.0 / 13824.0) * coeffs[4] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x379 * x4 * x5 * x56 +
        (25.0 / 6912.0) * coeffs[50] * x11 * x12 * x17 * x26 * x3 *
            x344 * x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[51] * x360 * x369 -
        1.0 / 13824.0 * coeffs[52] * x355 * x378 +
        (25.0 / 6912.0) * coeffs[53] * x11 * x12 * x13 * x17 * x26 * x3 *
            x328 * x4 * x5 * x56 * x[1] -
        1.0 / 13824.0 * coeffs[54] * x344 * x368 * x377 +
        (25.0 / 13824.0) * coeffs[55] * x11 * x12 * x13 * x17 * x26 * x3 *
            x350 * x4 * x5 * x56 * x[1] -
        1.0 / 13824.0 * coeffs[56] * x313 * x315 * x370 +
        (625.0 / 6912.0) * coeffs[57] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x313 * x4 * x56 * x[1] -
        1.0 / 13824.0 * coeffs[58] * x313 * x321 * x371 +
        (625.0 / 13824.0) * coeffs[59] * x0 * x11 * x12 * x13 * x17 * x26 *
            x313 * x4 * x5 * x56 * x[1] -
        1.0 / 13824.0 * coeffs[5] * x367 * x380 +
        (625.0 / 6912.0) * coeffs[60] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x328 * x4 * x5 * x[1] -
        1.0 / 13824.0 * coeffs[61] * x318 * x328 * x372 +
        (625.0 / 3456.0) * coeffs[62] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x328 * x5 * x56 * x[1] -
        1.0 / 13824.0 * coeffs[63] * x323 * x328 * x371 -
        1.0 / 13824.0 * coeffs[64] * x315 * x344 * x371 +
        (625.0 / 3456.0) * coeffs[65] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x344 * x4 * x56 * x[1] -
        1.0 / 13824.0 * coeffs[66] * x321 * x344 * x372 +
        (625.0 / 6912.0) * coeffs[67] * x0 * x11 * x12 * x13 * x17 * x26 *
            x344 * x4 * x5 * x56 * x[1] +
        (625.0 / 13824.0) * coeffs[68] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x350 * x4 * x5 * x[1] -
        1.0 / 13824.0 * coeffs[69] * x318 * x350 * x371 +
        (1.0 / 13824.0) * coeffs[6] * x11 * x12 * x17 * x26 * x3 *
            x379 * x4 * x5 * x56 * x[0] * x[1] +
        (625.0 / 6912.0) * coeffs[70] * x0 * x11 * x12 * x13 * x17 * x26 * x3 *
            x350 * x5 * x56 * x[1] -
        1.0 / 13824.0 * coeffs[71] * x323 * x350 * x370 +
        (625.0 / 13824.0) * coeffs[72] * x0 * x11 * x12 * x17 * x26 * x3 *
            x313 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[73] * x313 * x318 * x361 +
        (625.0 / 6912.0) * coeffs[74] * x0 * x11 * x12 * x17 * x26 * x3 *
            x313 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[75] * x313 * x323 * x363 -
        1.0 / 13824.0 * coeffs[76] * x315 * x328 * x361 +
        (625.0 / 3456.0) * coeffs[77] * x0 * x11 * x12 * x17 * x26 * x3 *
            x328 * x4 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[78] * x321 * x328 * x364 +
        (625.0 / 6912.0) * coeffs[79] * x0 * x11 * x12 * x17 * x26 *
            x328 * x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[7] * x377 * x386 +
        (625.0 / 6912.0) * coeffs[80] * x0 * x11 * x12 * x17 * x26 * x3 *
            x344 * x4 * x5 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[81] * x318 * x344 * x364 +
        (625.0 / 3456.0) * coeffs[82] * x0 * x11 * x12 * x17 * x26 * x3 *
            x344 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[83] * x323 * x344 * x361 -
        1.0 / 13824.0 * coeffs[84] * x315 * x350 * x363 +
        (625.0 / 6912.0) * coeffs[85] * x0 * x11 * x12 * x17 * x26 * x3 *
            x350 * x4 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[86] * x321 * x350 * x361 +
        (625.0 / 13824.0) * coeffs[87] * x0 * x11 * x12 * x17 * x26 *
            x350 * x4 * x5 * x56 * x[0] * x[1] -
        1.0 / 13824.0 * coeffs[88] * x359 * x373 +
        (625.0 / 6912.0) * coeffs[89] * x0 * x11 * x12 * x13 * x26 * x3 *
            x313 * x4 * x5 * x56 * x[0] +
        (25.0 / 13824.0) * coeffs[8] * x0 * x11 * x12 * x13 * x17 * x3 *
            x393 * x4 * x5 * x56 * x[0] +
        (625.0 / 13824.0) * coeffs[91] * x0 * x12 * x13 * x17 * x26 * x3 *
            x313 * x4 * x5 * x56 * x[0] +
        (625.0 / 6912.0) * coeffs[92] * x0 * x11 * x12 * x13 * x17 * x3 *
            x328 * x4 * x5 * x56 * x[0] -
        1.0 / 13824.0 * coeffs[93] * x334 * x375 +
        (625.0 / 3456.0) * coeffs[94] * x0 * x11 * x13 * x17 * x26 * x3 *
            x328 * x4 * x5 * x56 * x[0] +
        (625.0 / 3456.0) * coeffs[97] * x0 * x11 * x12 * x13 * x26 * x3 *
            x344 * x4 * x5 * x56 * x[0] -
        1.0 / 13824.0 * coeffs[98] * x348 * x375 +
        (625.0 / 6912.0) * coeffs[99] * x0 * x12 * x13 * x17 * x26 * x3 *
            x344 * x4 * x5 * x56 * x[0] -
        1.0 / 13824.0 * coeffs[9] * x388 * x394 -
        1.0 / 13824.0 * x125 * x279 * x355 -
        1.0 / 13824.0 * x125 * x298 * x365 - 1.0 / 13824.0 * x13 * x191 * x385 -
        1.0 / 13824.0 * x137 * x276 * x380 -
        1.0 / 13824.0 * x142 * x278 * x380 - 1.0 / 13824.0 * x144 * x382 * x46 -
        1.0 / 13824.0 * x147 * x31 * x382 - 1.0 / 13824.0 * x149 * x380 * x384 -
        1.0 / 13824.0 * x177 * x385 * x[0] - 1.0 / 13824.0 * x246 * x381 * x59 -
        1.0 / 13824.0 * x250 * x381 * x66 - 625.0 / 13824.0 * x251 * x383 -
        1.0 / 13824.0 * x280 * x328 * x356 -
        1.0 / 13824.0 * x284 * x344 * x356 -
        1.0 / 13824.0 * x300 * x313 * x374 -
        1.0 / 13824.0 * x302 * x328 * x374 - 1.0 / 13824.0 * x303 * x344 * x374;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = 5 * x[1];
    const Scalar x2 = x1 - 4;
    const Scalar x3 = x1 - 3;
    const Scalar x4 = x2 * x3;
    const Scalar x5 = x1 - 1;
    const Scalar x6 = x1 - 2;
    const Scalar x7 = x5 * x6;
    const Scalar x8 = x4 * x7;
    const Scalar x9 = (1.0 / 13824.0) * x8;
    const Scalar x10 = x0 * x9;
    const Scalar x11 = x[0] - 1;
    const Scalar x12 = 5 * x[0];
    const Scalar x13 = x12 - 2;
    const Scalar x14 = x12 - 4;
    const Scalar x15 = x12 - 3;
    const Scalar x16 = x14 * x15;
    const Scalar x17 = x13 * x16;
    const Scalar x18 = x11 * x17;
    const Scalar x19 = x12 - 1;
    const Scalar x20 = x16 * x19;
    const Scalar x21 = x11 * x20;
    const Scalar x22 = x13 * x19;
    const Scalar x23 = x14 * x22;
    const Scalar x24 = x11 * x23;
    const Scalar x25 = x15 * x22;
    const Scalar x26 = x11 * x25;
    const Scalar x27 = x16 * x22;
    const Scalar x28 = 5 * x18 + 5 * x21 + 5 * x24 + 5 * x26 + x27;
    const Scalar x29 = x[2] - 1;
    const Scalar x30 = 5 * x[2];
    const Scalar x31 = x30 - 4;
    const Scalar x32 = x30 - 3;
    const Scalar x33 = x31 * x32;
    const Scalar x34 = x30 - 1;
    const Scalar x35 = x30 - 2;
    const Scalar x36 = x34 * x35;
    const Scalar x37 = x33 * x36;
    const Scalar x38 = x29 * x37;
    const Scalar x39 = x28 * x38;
    const Scalar x40 = (1.0 / 13824.0) * x38;
    const Scalar x41 = x4 * x6;
    const Scalar x42 = x0 * x41;
    const Scalar x43 = x4 * x5;
    const Scalar x44 = x0 * x43;
    const Scalar x45 = x2 * x7;
    const Scalar x46 = x0 * x45;
    const Scalar x47 = x3 * x7;
    const Scalar x48 = x0 * x47;
    const Scalar x49 = 5 * x42 + 5 * x44 + 5 * x46 + 5 * x48 + x8;
    const Scalar x50 = x11 * x27;
    const Scalar x51 = x49 * x50;
    const Scalar x52 = x33 * x35;
    const Scalar x53 = x29 * x52;
    const Scalar x54 = x33 * x34;
    const Scalar x55 = x29 * x54;
    const Scalar x56 = x31 * x36;
    const Scalar x57 = x29 * x56;
    const Scalar x58 = x32 * x36;
    const Scalar x59 = x29 * x58;
    const Scalar x60 = x37 + 5 * x53 + 5 * x55 + 5 * x57 + 5 * x59;
    const Scalar x61 = x10 * x60;
    const Scalar x62 = x12 * x17 + x12 * x20 + x12 * x23 + x12 * x25 + x27;
    const Scalar x63 = x38 * x62;
    const Scalar x64 = x27 * x[0];
    const Scalar x65 = x40 * x64;
    const Scalar x66 = x9 * x[1];
    const Scalar x67 = x1 * x41 + x1 * x43 + x1 * x45 + x1 * x47 + x8;
    const Scalar x68 = x60 * x66;
    const Scalar x69 = x50 * x67;
    const Scalar x70 = x37 * x[2];
    const Scalar x71 = x10 * x70;
    const Scalar x72 = (1.0 / 13824.0) * x70;
    const Scalar x73 = x30 * x52 + x30 * x54 + x30 * x56 + x30 * x58 + x37;
    const Scalar x74 = x10 * x73;
    const Scalar x75 = x64 * x72;
    const Scalar x76 = x66 * x70;
    const Scalar x77 = x66 * x73;
    const Scalar x78 = (25.0 / 13824.0) * x38;
    const Scalar x79 = x11 * x12;
    const Scalar x80 = x16 * x79;
    const Scalar x81 = x13 * x79;
    const Scalar x82 = x14 * x81;
    const Scalar x83 = x15 * x81;
    const Scalar x84 = x17 * x[0];
    const Scalar x85 = x18 + x80 + x82 + x83 + x84;
    const Scalar x86 = x0 * x8;
    const Scalar x87 = x85 * x86;
    const Scalar x88 = x11 * x84;
    const Scalar x89 = x49 * x78;
    const Scalar x90 = x60 * x86;
    const Scalar x91 = (25.0 / 13824.0) * x88;
    const Scalar x92 = (25.0 / 6912.0) * x38;
    const Scalar x93 = x19 * x79;
    const Scalar x94 = x14 * x93;
    const Scalar x95 = x15 * x93;
    const Scalar x96 = x20 * x[0];
    const Scalar x97 = x21 + x80 + x94 + x95 + x96;
    const Scalar x98 = x86 * x97;
    const Scalar x99 = x11 * x96;
    const Scalar x100 = x49 * x92;
    const Scalar x101 = (25.0 / 6912.0) * x90;
    const Scalar x102 = x22 * x79;
    const Scalar x103 = x23 * x[0];
    const Scalar x104 = x102 + x103 + x24 + x82 + x94;
    const Scalar x105 = x104 * x92;
    const Scalar x106 = x103 * x11;
    const Scalar x107 = x25 * x[0];
    const Scalar x108 = x102 + x107 + x26 + x83 + x95;
    const Scalar x109 = x108 * x78;
    const Scalar x110 = x107 * x11;
    const Scalar x111 = (25.0 / 13824.0) * x110;
    const Scalar x112 = x0 * x63;
    const Scalar x113 = x41 * x[1];
    const Scalar x114 = (25.0 / 13824.0) * x113;
    const Scalar x115 = x0 * x1;
    const Scalar x116 = x115 * x4;
    const Scalar x117 = x115 * x6;
    const Scalar x118 = x117 * x2;
    const Scalar x119 = x117 * x3;
    const Scalar x120 = x113 + x116 + x118 + x119 + x42;
    const Scalar x121 = x64 * x78;
    const Scalar x122 = x0 * x114;
    const Scalar x123 = x60 * x64;
    const Scalar x124 = x43 * x[1];
    const Scalar x125 = (25.0 / 6912.0) * x124;
    const Scalar x126 = x115 * x5;
    const Scalar x127 = x126 * x2;
    const Scalar x128 = x126 * x3;
    const Scalar x129 = x116 + x124 + x127 + x128 + x44;
    const Scalar x130 = x64 * x92;
    const Scalar x131 = x0 * x123;
    const Scalar x132 = x45 * x[1];
    const Scalar x133 = (25.0 / 6912.0) * x132;
    const Scalar x134 = x115 * x7;
    const Scalar x135 = x118 + x127 + x132 + x134 + x46;
    const Scalar x136 = x47 * x[1];
    const Scalar x137 = (25.0 / 13824.0) * x136;
    const Scalar x138 = x119 + x128 + x134 + x136 + x48;
    const Scalar x139 = x8 * x[1];
    const Scalar x140 = x139 * x85;
    const Scalar x141 = x67 * x78;
    const Scalar x142 = x139 * x60;
    const Scalar x143 = x139 * x97;
    const Scalar x144 = x67 * x92;
    const Scalar x145 = (25.0 / 6912.0) * x142;
    const Scalar x146 = x50 * x78;
    const Scalar x147 = x50 * x60;
    const Scalar x148 = x0 * x125;
    const Scalar x149 = x50 * x92;
    const Scalar x150 = x0 * x133;
    const Scalar x151 = x0 * x137;
    const Scalar x152 = (25.0 / 13824.0) * x70;
    const Scalar x153 = x49 * x70;
    const Scalar x154 = x73 * x86;
    const Scalar x155 = (25.0 / 6912.0) * x70;
    const Scalar x156 = (25.0 / 6912.0) * x99;
    const Scalar x157 = x104 * x155;
    const Scalar x158 = (25.0 / 6912.0) * x106;
    const Scalar x159 = x108 * x152;
    const Scalar x160 = x62 * x70;
    const Scalar x161 = x152 * x64;
    const Scalar x162 = x64 * x73;
    const Scalar x163 = x155 * x64;
    const Scalar x164 = x67 * x70;
    const Scalar x165 = x139 * x73;
    const Scalar x166 = x155 * x67;
    const Scalar x167 = x28 * x70;
    const Scalar x168 = x152 * x50;
    const Scalar x169 = x50 * x73;
    const Scalar x170 = x155 * x50;
    const Scalar x171 = (25.0 / 13824.0) * x86;
    const Scalar x172 = x52 * x[2];
    const Scalar x173 = x28 * x29;
    const Scalar x174 = x172 * x173;
    const Scalar x175 = x29 * x51;
    const Scalar x176 = (25.0 / 13824.0) * x175;
    const Scalar x177 = x29 * x30;
    const Scalar x178 = x177 * x33;
    const Scalar x179 = x177 * x35;
    const Scalar x180 = x179 * x31;
    const Scalar x181 = x179 * x32;
    const Scalar x182 = x172 + x178 + x180 + x181 + x53;
    const Scalar x183 = x171 * x50;
    const Scalar x184 = x54 * x[2];
    const Scalar x185 = (25.0 / 6912.0) * x184;
    const Scalar x186 = x173 * x86;
    const Scalar x187 = (25.0 / 6912.0) * x86;
    const Scalar x188 = x177 * x34;
    const Scalar x189 = x188 * x31;
    const Scalar x190 = x188 * x32;
    const Scalar x191 = x178 + x184 + x189 + x190 + x55;
    const Scalar x192 = x191 * x50;
    const Scalar x193 = x56 * x[2];
    const Scalar x194 = (25.0 / 6912.0) * x193;
    const Scalar x195 = x177 * x36;
    const Scalar x196 = x180 + x189 + x193 + x195 + x57;
    const Scalar x197 = x187 * x196;
    const Scalar x198 = x58 * x[2];
    const Scalar x199 = x171 * x198;
    const Scalar x200 = x181 + x190 + x195 + x198 + x59;
    const Scalar x201 = x172 * x29;
    const Scalar x202 = x201 * x62;
    const Scalar x203 = x49 * x64;
    const Scalar x204 = (25.0 / 13824.0) * x201;
    const Scalar x205 = x171 * x64;
    const Scalar x206 = x185 * x29;
    const Scalar x207 = x206 * x62;
    const Scalar x208 = x191 * x64;
    const Scalar x209 = x29 * x62;
    const Scalar x210 = x193 * x209;
    const Scalar x211 = x203 * x29;
    const Scalar x212 = (25.0 / 13824.0) * x198;
    const Scalar x213 = (25.0 / 13824.0) * x139;
    const Scalar x214 = x64 * x67;
    const Scalar x215 = x213 * x64;
    const Scalar x216 = (25.0 / 6912.0) * x139;
    const Scalar x217 = x139 * x209;
    const Scalar x218 = x214 * x29;
    const Scalar x219 = x196 * x216;
    const Scalar x220 = x213 * x50;
    const Scalar x221 = x139 * x173;
    const Scalar x222 = x29 * x69;
    const Scalar x223 = x0 * x174;
    const Scalar x224 = (625.0 / 13824.0) * x113;
    const Scalar x225 = x201 * x50;
    const Scalar x226 = (625.0 / 13824.0) * x120;
    const Scalar x227 = x0 * x50;
    const Scalar x228 = x182 * x227;
    const Scalar x229 = (625.0 / 6912.0) * x124;
    const Scalar x230 = (625.0 / 6912.0) * x225;
    const Scalar x231 = (625.0 / 6912.0) * x132;
    const Scalar x232 = (625.0 / 13824.0) * x136;
    const Scalar x233 = (625.0 / 13824.0) * x138;
    const Scalar x234 = (625.0 / 6912.0) * x184;
    const Scalar x235 = x0 * x173;
    const Scalar x236 = x234 * x235;
    const Scalar x237 = x29 * x50;
    const Scalar x238 = x234 * x237;
    const Scalar x239 = x0 * x192;
    const Scalar x240 = (625.0 / 6912.0) * x113;
    const Scalar x241 = (625.0 / 3456.0) * x184;
    const Scalar x242 = x235 * x241;
    const Scalar x243 = x237 * x241;
    const Scalar x244 = (625.0 / 3456.0) * x239;
    const Scalar x245 = (625.0 / 6912.0) * x136;
    const Scalar x246 = x193 * x235;
    const Scalar x247 = x193 * x237;
    const Scalar x248 = (625.0 / 6912.0) * x120;
    const Scalar x249 = x196 * x227;
    const Scalar x250 = (625.0 / 3456.0) * x124;
    const Scalar x251 = (625.0 / 3456.0) * x247;
    const Scalar x252 = (625.0 / 3456.0) * x132;
    const Scalar x253 = (625.0 / 6912.0) * x138;
    const Scalar x254 = x198 * x235;
    const Scalar x255 = x198 * x237;
    const Scalar x256 = x200 * x227;
    const Scalar x257 = (625.0 / 6912.0) * x255;
    const Scalar x258 = x0 * x224;
    const Scalar x259 = x201 * x64;
    const Scalar x260 = x182 * x64;
    const Scalar x261 = x0 * x229;
    const Scalar x262 = (625.0 / 6912.0) * x259;
    const Scalar x263 = x0 * x231;
    const Scalar x264 = x0 * x232;
    const Scalar x265 = x0 * x209;
    const Scalar x266 = x234 * x265;
    const Scalar x267 = x29 * x64;
    const Scalar x268 = x234 * x267;
    const Scalar x269 = x0 * x208;
    const Scalar x270 = x241 * x265;
    const Scalar x271 = x241 * x267;
    const Scalar x272 = x0 * x240;
    const Scalar x273 = x193 * x267;
    const Scalar x274 = x196 * x64;
    const Scalar x275 = x0 * x250;
    const Scalar x276 = (625.0 / 3456.0) * x273;
    const Scalar x277 = x0 * x252;
    const Scalar x278 = x0 * x245;
    const Scalar x279 = x198 * x209;
    const Scalar x280 = x198 * x267;
    const Scalar x281 = x200 * x64;
    const Scalar x282 = (625.0 / 6912.0) * x280;
    const Scalar x283 = (625.0 / 13824.0) * x201;
    const Scalar x284 = x283 * x49;
    const Scalar x285 = x182 * x86;
    const Scalar x286 = (625.0 / 13824.0) * x88;
    const Scalar x287 = (625.0 / 6912.0) * x201;
    const Scalar x288 = x287 * x49;
    const Scalar x289 = (625.0 / 6912.0) * x285;
    const Scalar x290 = x104 * x86;
    const Scalar x291 = x108 * x86;
    const Scalar x292 = (625.0 / 13824.0) * x110;
    const Scalar x293 = x234 * x29;
    const Scalar x294 = x293 * x49;
    const Scalar x295 = x191 * x86;
    const Scalar x296 = (625.0 / 6912.0) * x88;
    const Scalar x297 = x241 * x29;
    const Scalar x298 = x297 * x49;
    const Scalar x299 = (625.0 / 3456.0) * x295;
    const Scalar x300 = (625.0 / 6912.0) * x110;
    const Scalar x301 = x193 * x29;
    const Scalar x302 = (625.0 / 6912.0) * x301;
    const Scalar x303 = x301 * x49;
    const Scalar x304 = x196 * x86;
    const Scalar x305 = (625.0 / 3456.0) * x301;
    const Scalar x306 = (625.0 / 3456.0) * x99;
    const Scalar x307 = (625.0 / 3456.0) * x106;
    const Scalar x308 = x198 * x29;
    const Scalar x309 = (625.0 / 13824.0) * x308;
    const Scalar x310 = x308 * x49;
    const Scalar x311 = x200 * x86;
    const Scalar x312 = (625.0 / 6912.0) * x308;
    const Scalar x313 = (625.0 / 6912.0) * x99;
    const Scalar x314 = (625.0 / 6912.0) * x106;
    const Scalar x315 = x283 * x67;
    const Scalar x316 = x139 * x182;
    const Scalar x317 = x287 * x67;
    const Scalar x318 = x104 * x139;
    const Scalar x319 = x108 * x139;
    const Scalar x320 = x293 * x67;
    const Scalar x321 = x139 * x191;
    const Scalar x322 = x297 * x67;
    const Scalar x323 = x301 * x67;
    const Scalar x324 = x139 * x196;
    const Scalar x325 = x305 * x67;
    const Scalar x326 = x308 * x67;
    const Scalar x327 = x139 * x200;
    const Scalar x328 = x312 * x67;
    const Scalar x329 = x258 * x38;
    const Scalar x330 = x226 * x38;
    const Scalar x331 = x258 * x60;
    const Scalar x332 = x272 * x38;
    const Scalar x333 = x248 * x38;
    const Scalar x334 = x272 * x60;
    const Scalar x335 = x261 * x38;
    const Scalar x336 = x129 * x38;
    const Scalar x337 = x261 * x60;
    const Scalar x338 = x275 * x38;
    const Scalar x339 = x275 * x60;
    const Scalar x340 = x263 * x38;
    const Scalar x341 = x135 * x38;
    const Scalar x342 = x263 * x60;
    const Scalar x343 = x277 * x38;
    const Scalar x344 = x277 * x60;
    const Scalar x345 = x264 * x38;
    const Scalar x346 = x233 * x38;
    const Scalar x347 = x264 * x60;
    const Scalar x348 = x278 * x38;
    const Scalar x349 = x253 * x38;
    const Scalar x350 = x278 * x60;
    const Scalar x351 = x258 * x70;
    const Scalar x352 = x226 * x70;
    const Scalar x353 = x258 * x73;
    const Scalar x354 = x272 * x70;
    const Scalar x355 = x248 * x70;
    const Scalar x356 = x272 * x73;
    const Scalar x357 = x261 * x70;
    const Scalar x358 = x129 * x70;
    const Scalar x359 = x261 * x73;
    const Scalar x360 = x275 * x70;
    const Scalar x361 = x275 * x73;
    const Scalar x362 = x263 * x70;
    const Scalar x363 = x135 * x70;
    const Scalar x364 = x263 * x73;
    const Scalar x365 = x277 * x70;
    const Scalar x366 = x277 * x73;
    const Scalar x367 = x264 * x70;
    const Scalar x368 = x233 * x70;
    const Scalar x369 = x264 * x73;
    const Scalar x370 = x278 * x70;
    const Scalar x371 = x253 * x70;
    const Scalar x372 = x278 * x73;
    const Scalar x373 = (15625.0 / 13824.0) * x201;
    const Scalar x374 = x0 * x113;
    const Scalar x375 = x374 * x85;
    const Scalar x376 = x120 * x373;
    const Scalar x377 = x182 * x374;
    const Scalar x378 = (15625.0 / 13824.0) * x88;
    const Scalar x379 = (15625.0 / 6912.0) * x201;
    const Scalar x380 = x374 * x97;
    const Scalar x381 = x120 * x379;
    const Scalar x382 = (15625.0 / 6912.0) * x377;
    const Scalar x383 = x104 * x379;
    const Scalar x384 = x108 * x373;
    const Scalar x385 = (15625.0 / 13824.0) * x110;
    const Scalar x386 = x0 * x124;
    const Scalar x387 = x379 * x386;
    const Scalar x388 = x129 * x379;
    const Scalar x389 = x182 * x386;
    const Scalar x390 = (15625.0 / 6912.0) * x88;
    const Scalar x391 = (15625.0 / 3456.0) * x201;
    const Scalar x392 = x386 * x97;
    const Scalar x393 = x129 * x391;
    const Scalar x394 = (15625.0 / 3456.0) * x389;
    const Scalar x395 = x104 * x391;
    const Scalar x396 = (15625.0 / 6912.0) * x110;
    const Scalar x397 = x0 * x132;
    const Scalar x398 = x379 * x397;
    const Scalar x399 = x135 * x379;
    const Scalar x400 = x182 * x397;
    const Scalar x401 = x397 * x97;
    const Scalar x402 = x135 * x391;
    const Scalar x403 = (15625.0 / 3456.0) * x400;
    const Scalar x404 = x0 * x136;
    const Scalar x405 = x404 * x85;
    const Scalar x406 = x138 * x373;
    const Scalar x407 = x182 * x404;
    const Scalar x408 = x404 * x97;
    const Scalar x409 = x138 * x379;
    const Scalar x410 = (15625.0 / 6912.0) * x407;
    const Scalar x411 = x184 * x29;
    const Scalar x412 = (15625.0 / 6912.0) * x375;
    const Scalar x413 = x120 * x411;
    const Scalar x414 = x191 * x374;
    const Scalar x415 = (15625.0 / 3456.0) * x411;
    const Scalar x416 = (15625.0 / 3456.0) * x99;
    const Scalar x417 = x104 * x415;
    const Scalar x418 = (15625.0 / 3456.0) * x106;
    const Scalar x419 = (15625.0 / 6912.0) * x411;
    const Scalar x420 = x108 * x374;
    const Scalar x421 = x386 * x415;
    const Scalar x422 = x129 * x415;
    const Scalar x423 = x191 * x386;
    const Scalar x424 = (15625.0 / 3456.0) * x88;
    const Scalar x425 = (15625.0 / 1728.0) * x411;
    const Scalar x426 = x129 * x425;
    const Scalar x427 = (15625.0 / 1728.0) * x423;
    const Scalar x428 = x104 * x425;
    const Scalar x429 = (15625.0 / 3456.0) * x110;
    const Scalar x430 = x397 * x415;
    const Scalar x431 = x135 * x415;
    const Scalar x432 = x191 * x397;
    const Scalar x433 = x135 * x425;
    const Scalar x434 = (15625.0 / 1728.0) * x432;
    const Scalar x435 = x138 * x411;
    const Scalar x436 = x191 * x404;
    const Scalar x437 = x138 * x415;
    const Scalar x438 = x108 * x404;
    const Scalar x439 = x120 * x301;
    const Scalar x440 = x196 * x374;
    const Scalar x441 = (15625.0 / 3456.0) * x301;
    const Scalar x442 = x104 * x441;
    const Scalar x443 = (15625.0 / 6912.0) * x301;
    const Scalar x444 = x386 * x441;
    const Scalar x445 = x129 * x301;
    const Scalar x446 = x196 * x386;
    const Scalar x447 = (15625.0 / 1728.0) * x301;
    const Scalar x448 = (15625.0 / 1728.0) * x99;
    const Scalar x449 = x104 * x447;
    const Scalar x450 = (15625.0 / 1728.0) * x106;
    const Scalar x451 = x397 * x441;
    const Scalar x452 = x135 * x301;
    const Scalar x453 = x196 * x397;
    const Scalar x454 = x135 * x447;
    const Scalar x455 = x138 * x301;
    const Scalar x456 = x196 * x404;
    const Scalar x457 = (15625.0 / 13824.0) * x308;
    const Scalar x458 = x120 * x308;
    const Scalar x459 = x200 * x374;
    const Scalar x460 = (15625.0 / 6912.0) * x308;
    const Scalar x461 = (15625.0 / 6912.0) * x99;
    const Scalar x462 = x104 * x460;
    const Scalar x463 = (15625.0 / 6912.0) * x106;
    const Scalar x464 = x386 * x460;
    const Scalar x465 = x129 * x308;
    const Scalar x466 = x200 * x386;
    const Scalar x467 = (15625.0 / 3456.0) * x308;
    const Scalar x468 = x104 * x467;
    const Scalar x469 = x397 * x460;
    const Scalar x470 = x135 * x308;
    const Scalar x471 = x200 * x397;
    const Scalar x472 = x138 * x308;
    const Scalar x473 = x200 * x404;
    const Scalar x474 = x138 * x460;
    out[0][0] = -x10 * x39;
    out[0][1] = -x40 * x51;
    out[0][2] = -x50 * x61;
    out[1][0] = x10 * x63;
    out[1][1] = x49 * x65;
    out[1][2] = x61 * x64;
    out[2][0] = -x63 * x66;
    out[2][1] = -x65 * x67;
    out[2][2] = -x64 * x68;
    out[3][0] = x39 * x66;
    out[3][1] = x40 * x69;
    out[3][2] = x50 * x68;
    out[4][0] = x28 * x71;
    out[4][1] = x51 * x72;
    out[4][2] = x50 * x74;
    out[5][0] = -x62 * x71;
    out[5][1] = -x49 * x75;
    out[5][2] = -x64 * x74;
    out[6][0] = x62 * x76;
    out[6][1] = x67 * x75;
    out[6][2] = x64 * x77;
    out[7][0] = -x28 * x76;
    out[7][1] = -x69 * x72;
    out[7][2] = -x50 * x77;
    out[8][0] = x78 * x87;
    out[8][1] = x88 * x89;
    out[8][2] = x90 * x91;
    out[9][0] = -x92 * x98;
    out[9][1] = -x100 * x99;
    out[9][2] = -x101 * x99;
    out[10][0] = x105 * x86;
    out[10][1] = x100 * x106;
    out[10][2] = x101 * x106;
    out[11][0] = -x109 * x86;
    out[11][1] = -x110 * x89;
    out[11][2] = -x111 * x90;
    out[12][0] = -x112 * x114;
    out[12][1] = -x120 * x121;
    out[12][2] = -x122 * x123;
    out[13][0] = x112 * x125;
    out[13][1] = x129 * x130;
    out[13][2] = x125 * x131;
    out[14][0] = -x112 * x133;
    out[14][1] = -x130 * x135;
    out[14][2] = -x131 * x133;
    out[15][0] = x112 * x137;
    out[15][1] = x121 * x138;
    out[15][2] = x131 * x137;
    out[16][0] = -x140 * x78;
    out[16][1] = -x141 * x88;
    out[16][2] = -x142 * x91;
    out[17][0] = x143 * x92;
    out[17][1] = x144 * x99;
    out[17][2] = x145 * x99;
    out[18][0] = -x105 * x139;
    out[18][1] = -x106 * x144;
    out[18][2] = -x106 * x145;
    out[19][0] = x109 * x139;
    out[19][1] = x110 * x141;
    out[19][2] = x111 * x142;
    out[20][0] = x122 * x39;
    out[20][1] = x120 * x146;
    out[20][2] = x122 * x147;
    out[21][0] = -x148 * x39;
    out[21][1] = -x129 * x149;
    out[21][2] = -x147 * x148;
    out[22][0] = x150 * x39;
    out[22][1] = x135 * x149;
    out[22][2] = x147 * x150;
    out[23][0] = -x151 * x39;
    out[23][1] = -x138 * x146;
    out[23][2] = -x147 * x151;
    out[24][0] = -x152 * x87;
    out[24][1] = -x153 * x91;
    out[24][2] = -x154 * x91;
    out[25][0] = x155 * x98;
    out[25][1] = x153 * x156;
    out[25][2] = x154 * x156;
    out[26][0] = -x157 * x86;
    out[26][1] = -x153 * x158;
    out[26][2] = -x154 * x158;
    out[27][0] = x159 * x86;
    out[27][1] = x111 * x153;
    out[27][2] = x111 * x154;
    out[28][0] = x122 * x160;
    out[28][1] = x120 * x161;
    out[28][2] = x122 * x162;
    out[29][0] = -x148 * x160;
    out[29][1] = -x129 * x163;
    out[29][2] = -x148 * x162;
    out[30][0] = x150 * x160;
    out[30][1] = x135 * x163;
    out[30][2] = x150 * x162;
    out[31][0] = -x151 * x160;
    out[31][1] = -x138 * x161;
    out[31][2] = -x151 * x162;
    out[32][0] = x140 * x152;
    out[32][1] = x164 * x91;
    out[32][2] = x165 * x91;
    out[33][0] = -x143 * x155;
    out[33][1] = -x166 * x99;
    out[33][2] = -x156 * x165;
    out[34][0] = x139 * x157;
    out[34][1] = x106 * x166;
    out[34][2] = x158 * x165;
    out[35][0] = -x139 * x159;
    out[35][1] = -x111 * x164;
    out[35][2] = -x111 * x165;
    out[36][0] = -x122 * x167;
    out[36][1] = -x120 * x168;
    out[36][2] = -x122 * x169;
    out[37][0] = x148 * x167;
    out[37][1] = x129 * x170;
    out[37][2] = x148 * x169;
    out[38][0] = -x150 * x167;
    out[38][1] = -x135 * x170;
    out[38][2] = -x150 * x169;
    out[39][0] = x151 * x167;
    out[39][1] = x138 * x168;
    out[39][2] = x151 * x169;
    out[40][0] = x171 * x174;
    out[40][1] = x172 * x176;
    out[40][2] = x182 * x183;
    out[41][0] = -x185 * x186;
    out[41][1] = -x175 * x185;
    out[41][2] = -x187 * x192;
    out[42][0] = x186 * x194;
    out[42][1] = x175 * x194;
    out[42][2] = x197 * x50;
    out[43][0] = -x173 * x199;
    out[43][1] = -x176 * x198;
    out[43][2] = -x183 * x200;
    out[44][0] = -x171 * x202;
    out[44][1] = -x203 * x204;
    out[44][2] = -x182 * x205;
    out[45][0] = x207 * x86;
    out[45][1] = x203 * x206;
    out[45][2] = x187 * x208;
    out[46][0] = -x187 * x210;
    out[46][1] = -x194 * x211;
    out[46][2] = -x197 * x64;
    out[47][0] = x199 * x209;
    out[47][1] = x211 * x212;
    out[47][2] = x200 * x205;
    out[48][0] = x202 * x213;
    out[48][1] = x204 * x214;
    out[48][2] = x182 * x215;
    out[49][0] = -x139 * x207;
    out[49][1] = -x206 * x214;
    out[49][2] = -x208 * x216;
    out[50][0] = x194 * x217;
    out[50][1] = x194 * x218;
    out[50][2] = x219 * x64;
    out[51][0] = -x212 * x217;
    out[51][1] = -x212 * x218;
    out[51][2] = -x200 * x215;
    out[52][0] = -x174 * x213;
    out[52][1] = -x204 * x69;
    out[52][2] = -x182 * x220;
    out[53][0] = x185 * x221;
    out[53][1] = x206 * x69;
    out[53][2] = x192 * x216;
    out[54][0] = -x194 * x221;
    out[54][1] = -x194 * x222;
    out[54][2] = -x219 * x50;
    out[55][0] = x212 * x221;
    out[55][1] = x212 * x222;
    out[55][2] = x200 * x220;
    out[56][0] = -x223 * x224;
    out[56][1] = -x225 * x226;
    out[56][2] = -x224 * x228;
    out[57][0] = x223 * x229;
    out[57][1] = x129 * x230;
    out[57][2] = x228 * x229;
    out[58][0] = -x223 * x231;
    out[58][1] = -x135 * x230;
    out[58][2] = -x228 * x231;
    out[59][0] = x223 * x232;
    out[59][1] = x225 * x233;
    out[59][2] = x228 * x232;
    out[60][0] = x113 * x236;
    out[60][1] = x120 * x238;
    out[60][2] = x239 * x240;
    out[61][0] = -x124 * x242;
    out[61][1] = -x129 * x243;
    out[61][2] = -x124 * x244;
    out[62][0] = x132 * x242;
    out[62][1] = x135 * x243;
    out[62][2] = x132 * x244;
    out[63][0] = -x136 * x236;
    out[63][1] = -x138 * x238;
    out[63][2] = -x239 * x245;
    out[64][0] = -x240 * x246;
    out[64][1] = -x247 * x248;
    out[64][2] = -x240 * x249;
    out[65][0] = x246 * x250;
    out[65][1] = x129 * x251;
    out[65][2] = x249 * x250;
    out[66][0] = -x246 * x252;
    out[66][1] = -x135 * x251;
    out[66][2] = -x249 * x252;
    out[67][0] = x245 * x246;
    out[67][1] = x247 * x253;
    out[67][2] = x245 * x249;
    out[68][0] = x224 * x254;
    out[68][1] = x226 * x255;
    out[68][2] = x224 * x256;
    out[69][0] = -x229 * x254;
    out[69][1] = -x129 * x257;
    out[69][2] = -x229 * x256;
    out[70][0] = x231 * x254;
    out[70][1] = x135 * x257;
    out[70][2] = x231 * x256;
    out[71][0] = -x232 * x254;
    out[71][1] = -x233 * x255;
    out[71][2] = -x232 * x256;
    out[72][0] = x202 * x258;
    out[72][1] = x226 * x259;
    out[72][2] = x258 * x260;
    out[73][0] = -x202 * x261;
    out[73][1] = -x129 * x262;
    out[73][2] = -x260 * x261;
    out[74][0] = x202 * x263;
    out[74][1] = x135 * x262;
    out[74][2] = x260 * x263;
    out[75][0] = -x202 * x264;
    out[75][1] = -x233 * x259;
    out[75][2] = -x260 * x264;
    out[76][0] = -x113 * x266;
    out[76][1] = -x120 * x268;
    out[76][2] = -x240 * x269;
    out[77][0] = x124 * x270;
    out[77][1] = x129 * x271;
    out[77][2] = x250 * x269;
    out[78][0] = -x132 * x270;
    out[78][1] = -x135 * x271;
    out[78][2] = -x252 * x269;
    out[79][0] = x136 * x266;
    out[79][1] = x138 * x268;
    out[79][2] = x245 * x269;
    out[80][0] = x210 * x272;
    out[80][1] = x248 * x273;
    out[80][2] = x272 * x274;
    out[81][0] = -x210 * x275;
    out[81][1] = -x129 * x276;
    out[81][2] = -x274 * x275;
    out[82][0] = x210 * x277;
    out[82][1] = x135 * x276;
    out[82][2] = x274 * x277;
    out[83][0] = -x210 * x278;
    out[83][1] = -x253 * x273;
    out[83][2] = -x274 * x278;
    out[84][0] = -x258 * x279;
    out[84][1] = -x226 * x280;
    out[84][2] = -x258 * x281;
    out[85][0] = x261 * x279;
    out[85][1] = x129 * x282;
    out[85][2] = x261 * x281;
    out[86][0] = -x263 * x279;
    out[86][1] = -x135 * x282;
    out[86][2] = -x263 * x281;
    out[87][0] = x264 * x279;
    out[87][1] = x233 * x280;
    out[87][2] = x264 * x281;
    out[88][0] = -x283 * x87;
    out[88][1] = -x284 * x88;
    out[88][2] = -x285 * x286;
    out[89][0] = x287 * x98;
    out[89][1] = x288 * x99;
    out[89][2] = x289 * x99;
    out[90][0] = -x287 * x290;
    out[90][1] = -x106 * x288;
    out[90][2] = -x106 * x289;
    out[91][0] = x283 * x291;
    out[91][1] = x110 * x284;
    out[91][2] = x285 * x292;
    out[92][0] = x293 * x87;
    out[92][1] = x294 * x88;
    out[92][2] = x295 * x296;
    out[93][0] = -x297 * x98;
    out[93][1] = -x298 * x99;
    out[93][2] = -x299 * x99;
    out[94][0] = x290 * x297;
    out[94][1] = x106 * x298;
    out[94][2] = x106 * x299;
    out[95][0] = -x291 * x293;
    out[95][1] = -x110 * x294;
    out[95][2] = -x295 * x300;
    out[96][0] = -x302 * x87;
    out[96][1] = -x296 * x303;
    out[96][2] = -x296 * x304;
    out[97][0] = x305 * x98;
    out[97][1] = x303 * x306;
    out[97][2] = x304 * x306;
    out[98][0] = -x290 * x305;
    out[98][1] = -x303 * x307;
    out[98][2] = -x304 * x307;
    out[99][0] = x291 * x302;
    out[99][1] = x300 * x303;
    out[99][2] = x300 * x304;
    out[100][0] = x309 * x87;
    out[100][1] = x286 * x310;
    out[100][2] = x286 * x311;
    out[101][0] = -x312 * x98;
    out[101][1] = -x310 * x313;
    out[101][2] = -x311 * x313;
    out[102][0] = x290 * x312;
    out[102][1] = x310 * x314;
    out[102][2] = x311 * x314;
    out[103][0] = -x291 * x309;
    out[103][1] = -x292 * x310;
    out[103][2] = -x292 * x311;
    out[104][0] = x140 * x283;
    out[104][1] = x315 * x88;
    out[104][2] = x286 * x316;
    out[105][0] = -x143 * x287;
    out[105][1] = -x317 * x99;
    out[105][2] = -x313 * x316;
    out[106][0] = x287 * x318;
    out[106][1] = x106 * x317;
    out[106][2] = x314 * x316;
    out[107][0] = -x283 * x319;
    out[107][1] = -x110 * x315;
    out[107][2] = -x292 * x316;
    out[108][0] = -x140 * x293;
    out[108][1] = -x320 * x88;
    out[108][2] = -x296 * x321;
    out[109][0] = x143 * x297;
    out[109][1] = x322 * x99;
    out[109][2] = x306 * x321;
    out[110][0] = -x297 * x318;
    out[110][1] = -x106 * x322;
    out[110][2] = -x307 * x321;
    out[111][0] = x293 * x319;
    out[111][1] = x110 * x320;
    out[111][2] = x300 * x321;
    out[112][0] = x140 * x302;
    out[112][1] = x296 * x323;
    out[112][2] = x296 * x324;
    out[113][0] = -x143 * x305;
    out[113][1] = -x325 * x99;
    out[113][2] = -x306 * x324;
    out[114][0] = x305 * x318;
    out[114][1] = x106 * x325;
    out[114][2] = x307 * x324;
    out[115][0] = -x302 * x319;
    out[115][1] = -x300 * x323;
    out[115][2] = -x300 * x324;
    out[116][0] = -x140 * x309;
    out[116][1] = -x286 * x326;
    out[116][2] = -x286 * x327;
    out[117][0] = x143 * x312;
    out[117][1] = x328 * x99;
    out[117][2] = x313 * x327;
    out[118][0] = -x312 * x318;
    out[118][1] = -x106 * x328;
    out[118][2] = -x314 * x327;
    out[119][0] = x309 * x319;
    out[119][1] = x292 * x326;
    out[119][2] = x292 * x327;
    out[120][0] = -x329 * x85;
    out[120][1] = -x330 * x88;
    out[120][2] = -x331 * x88;
    out[121][0] = x332 * x97;
    out[121][1] = x333 * x99;
    out[121][2] = x334 * x99;
    out[122][0] = -x104 * x332;
    out[122][1] = -x106 * x333;
    out[122][2] = -x106 * x334;
    out[123][0] = x108 * x329;
    out[123][1] = x110 * x330;
    out[123][2] = x110 * x331;
    out[124][0] = x335 * x85;
    out[124][1] = x296 * x336;
    out[124][2] = x337 * x88;
    out[125][0] = -x338 * x97;
    out[125][1] = -x306 * x336;
    out[125][2] = -x339 * x99;
    out[126][0] = x104 * x338;
    out[126][1] = x307 * x336;
    out[126][2] = x106 * x339;
    out[127][0] = -x108 * x335;
    out[127][1] = -x300 * x336;
    out[127][2] = -x110 * x337;
    out[128][0] = -x340 * x85;
    out[128][1] = -x296 * x341;
    out[128][2] = -x342 * x88;
    out[129][0] = x343 * x97;
    out[129][1] = x306 * x341;
    out[129][2] = x344 * x99;
    out[130][0] = -x104 * x343;
    out[130][1] = -x307 * x341;
    out[130][2] = -x106 * x344;
    out[131][0] = x108 * x340;
    out[131][1] = x300 * x341;
    out[131][2] = x110 * x342;
    out[132][0] = x345 * x85;
    out[132][1] = x346 * x88;
    out[132][2] = x347 * x88;
    out[133][0] = -x348 * x97;
    out[133][1] = -x349 * x99;
    out[133][2] = -x350 * x99;
    out[134][0] = x104 * x348;
    out[134][1] = x106 * x349;
    out[134][2] = x106 * x350;
    out[135][0] = -x108 * x345;
    out[135][1] = -x110 * x346;
    out[135][2] = -x110 * x347;
    out[136][0] = x351 * x85;
    out[136][1] = x352 * x88;
    out[136][2] = x353 * x88;
    out[137][0] = -x354 * x97;
    out[137][1] = -x355 * x99;
    out[137][2] = -x356 * x99;
    out[138][0] = x104 * x354;
    out[138][1] = x106 * x355;
    out[138][2] = x106 * x356;
    out[139][0] = -x108 * x351;
    out[139][1] = -x110 * x352;
    out[139][2] = -x110 * x353;
    out[140][0] = -x357 * x85;
    out[140][1] = -x296 * x358;
    out[140][2] = -x359 * x88;
    out[141][0] = x360 * x97;
    out[141][1] = x306 * x358;
    out[141][2] = x361 * x99;
    out[142][0] = -x104 * x360;
    out[142][1] = -x307 * x358;
    out[142][2] = -x106 * x361;
    out[143][0] = x108 * x357;
    out[143][1] = x300 * x358;
    out[143][2] = x110 * x359;
    out[144][0] = x362 * x85;
    out[144][1] = x296 * x363;
    out[144][2] = x364 * x88;
    out[145][0] = -x365 * x97;
    out[145][1] = -x306 * x363;
    out[145][2] = -x366 * x99;
    out[146][0] = x104 * x365;
    out[146][1] = x307 * x363;
    out[146][2] = x106 * x366;
    out[147][0] = -x108 * x362;
    out[147][1] = -x300 * x363;
    out[147][2] = -x110 * x364;
    out[148][0] = -x367 * x85;
    out[148][1] = -x368 * x88;
    out[148][2] = -x369 * x88;
    out[149][0] = x370 * x97;
    out[149][1] = x371 * x99;
    out[149][2] = x372 * x99;
    out[150][0] = -x104 * x370;
    out[150][1] = -x106 * x371;
    out[150][2] = -x106 * x372;
    out[151][0] = x108 * x367;
    out[151][1] = x110 * x368;
    out[151][2] = x110 * x369;
    out[152][0] = x373 * x375;
    out[152][1] = x376 * x88;
    out[152][2] = x377 * x378;
    out[153][0] = -x379 * x380;
    out[153][1] = -x381 * x99;
    out[153][2] = -x382 * x99;
    out[154][0] = x374 * x383;
    out[154][1] = x106 * x381;
    out[154][2] = x106 * x382;
    out[155][0] = -x374 * x384;
    out[155][1] = -x110 * x376;
    out[155][2] = -x377 * x385;
    out[156][0] = -x387 * x85;
    out[156][1] = -x388 * x88;
    out[156][2] = -x389 * x390;
    out[157][0] = x391 * x392;
    out[157][1] = x393 * x99;
    out[157][2] = x394 * x99;
    out[158][0] = -x386 * x395;
    out[158][1] = -x106 * x393;
    out[158][2] = -x106 * x394;
    out[159][0] = x108 * x387;
    out[159][1] = x110 * x388;
    out[159][2] = x389 * x396;
    out[160][0] = x398 * x85;
    out[160][1] = x399 * x88;
    out[160][2] = x390 * x400;
    out[161][0] = -x391 * x401;
    out[161][1] = -x402 * x99;
    out[161][2] = -x403 * x99;
    out[162][0] = x395 * x397;
    out[162][1] = x106 * x402;
    out[162][2] = x106 * x403;
    out[163][0] = -x108 * x398;
    out[163][1] = -x110 * x399;
    out[163][2] = -x396 * x400;
    out[164][0] = -x373 * x405;
    out[164][1] = -x406 * x88;
    out[164][2] = -x378 * x407;
    out[165][0] = x379 * x408;
    out[165][1] = x409 * x99;
    out[165][2] = x410 * x99;
    out[166][0] = -x383 * x404;
    out[166][1] = -x106 * x409;
    out[166][2] = -x106 * x410;
    out[167][0] = x384 * x404;
    out[167][1] = x110 * x406;
    out[167][2] = x385 * x407;
    out[168][0] = -x411 * x412;
    out[168][1] = -x390 * x413;
    out[168][2] = -x390 * x414;
    out[169][0] = x380 * x415;
    out[169][1] = x413 * x416;
    out[169][2] = x414 * x416;
    out[170][0] = -x374 * x417;
    out[170][1] = -x413 * x418;
    out[170][2] = -x414 * x418;
    out[171][0] = x419 * x420;
    out[171][1] = x396 * x413;
    out[171][2] = x396 * x414;
    out[172][0] = x421 * x85;
    out[172][1] = x422 * x88;
    out[172][2] = x423 * x424;
    out[173][0] = -x392 * x425;
    out[173][1] = -x426 * x99;
    out[173][2] = -x427 * x99;
    out[174][0] = x386 * x428;
    out[174][1] = x106 * x426;
    out[174][2] = x106 * x427;
    out[175][0] = -x108 * x421;
    out[175][1] = -x110 * x422;
    out[175][2] = -x423 * x429;
    out[176][0] = -x430 * x85;
    out[176][1] = -x431 * x88;
    out[176][2] = -x424 * x432;
    out[177][0] = x401 * x425;
    out[177][1] = x433 * x99;
    out[177][2] = x434 * x99;
    out[178][0] = -x397 * x428;
    out[178][1] = -x106 * x433;
    out[178][2] = -x106 * x434;
    out[179][0] = x108 * x430;
    out[179][1] = x110 * x431;
    out[179][2] = x429 * x432;
    out[180][0] = x405 * x419;
    out[180][1] = x390 * x435;
    out[180][2] = x390 * x436;
    out[181][0] = -x408 * x415;
    out[181][1] = -x437 * x99;
    out[181][2] = -x416 * x436;
    out[182][0] = x404 * x417;
    out[182][1] = x106 * x437;
    out[182][2] = x418 * x436;
    out[183][0] = -x419 * x438;
    out[183][1] = -x396 * x435;
    out[183][2] = -x396 * x436;
    out[184][0] = x301 * x412;
    out[184][1] = x390 * x439;
    out[184][2] = x390 * x440;
    out[185][0] = -x380 * x441;
    out[185][1] = -x416 * x439;
    out[185][2] = -x416 * x440;
    out[186][0] = x374 * x442;
    out[186][1] = x418 * x439;
    out[186][2] = x418 * x440;
    out[187][0] = -x420 * x443;
    out[187][1] = -x396 * x439;
    out[187][2] = -x396 * x440;
    out[188][0] = -x444 * x85;
    out[188][1] = -x424 * x445;
    out[188][2] = -x424 * x446;
    out[189][0] = x392 * x447;
    out[189][1] = x445 * x448;
    out[189][2] = x446 * x448;
    out[190][0] = -x386 * x449;
    out[190][1] = -x445 * x450;
    out[190][2] = -x446 * x450;
    out[191][0] = x108 * x444;
    out[191][1] = x429 * x445;
    out[191][2] = x429 * x446;
    out[192][0] = x451 * x85;
    out[192][1] = x424 * x452;
    out[192][2] = x424 * x453;
    out[193][0] = -x401 * x447;
    out[193][1] = -x454 * x99;
    out[193][2] = -x448 * x453;
    out[194][0] = x397 * x449;
    out[194][1] = x106 * x454;
    out[194][2] = x450 * x453;
    out[195][0] = -x108 * x451;
    out[195][1] = -x429 * x452;
    out[195][2] = -x429 * x453;
    out[196][0] = -x405 * x443;
    out[196][1] = -x390 * x455;
    out[196][2] = -x390 * x456;
    out[197][0] = x408 * x441;
    out[197][1] = x416 * x455;
    out[197][2] = x416 * x456;
    out[198][0] = -x404 * x442;
    out[198][1] = -x418 * x455;
    out[198][2] = -x418 * x456;
    out[199][0] = x438 * x443;
    out[199][1] = x396 * x455;
    out[199][2] = x396 * x456;
    out[200][0] = -x375 * x457;
    out[200][1] = -x378 * x458;
    out[200][2] = -x378 * x459;
    out[201][0] = x380 * x460;
    out[201][1] = x458 * x461;
    out[201][2] = x459 * x461;
    out[202][0] = -x374 * x462;
    out[202][1] = -x458 * x463;
    out[202][2] = -x459 * x463;
    out[203][0] = x420 * x457;
    out[203][1] = x385 * x458;
    out[203][2] = x385 * x459;
    out[204][0] = x464 * x85;
    out[204][1] = x390 * x465;
    out[204][2] = x390 * x466;
    out[205][0] = -x392 * x467;
    out[205][1] = -x416 * x465;
    out[205][2] = -x416 * x466;
    out[206][0] = x386 * x468;
    out[206][1] = x418 * x465;
    out[206][2] = x418 * x466;
    out[207][0] = -x108 * x464;
    out[207][1] = -x396 * x465;
    out[207][2] = -x396 * x466;
    out[208][0] = -x469 * x85;
    out[208][1] = -x390 * x470;
    out[208][2] = -x390 * x471;
    out[209][0] = x401 * x467;
    out[209][1] = x416 * x470;
    out[209][2] = x416 * x471;
    out[210][0] = -x397 * x468;
    out[210][1] = -x418 * x470;
    out[210][2] = -x418 * x471;
    out[211][0] = x108 * x469;
    out[211][1] = x396 * x470;
    out[211][2] = x396 * x471;
    out[212][0] = x405 * x457;
    out[212][1] = x378 * x472;
    out[212][2] = x378 * x473;
    out[213][0] = -x408 * x460;
    out[213][1] = -x474 * x99;
    out[213][2] = -x461 * x473;
    out[214][0] = x404 * x462;
    out[214][1] = x106 * x474;
    out[214][2] = x463 * x473;
    out[215][0] = -x438 * x457;
    out[215][1] = -x385 * x472;
    out[215][2] = -x385 * x473;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[3]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[0][2] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(5) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[1][2] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(5) / order;
    out[2][1] = static_cast<Scalar>(5) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(5) / order;
    out[3][2] = static_cast<Scalar>(0) / order;
    out[4][0] = static_cast<Scalar>(0) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[4][2] = static_cast<Scalar>(5) / order;
    out[5][0] = static_cast<Scalar>(5) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[5][2] = static_cast<Scalar>(5) / order;
    out[6][0] = static_cast<Scalar>(5) / order;
    out[6][1] = static_cast<Scalar>(5) / order;
    out[6][2] = static_cast<Scalar>(5) / order;
    out[7][0] = static_cast<Scalar>(0) / order;
    out[7][1] = static_cast<Scalar>(5) / order;
    out[7][2] = static_cast<Scalar>(5) / order;
    out[8][0] = static_cast<Scalar>(1) / order;
    out[8][1] = static_cast<Scalar>(0) / order;
    out[8][2] = static_cast<Scalar>(0) / order;
    out[9][0] = static_cast<Scalar>(2) / order;
    out[9][1] = static_cast<Scalar>(0) / order;
    out[9][2] = static_cast<Scalar>(0) / order;
    out[10][0] = static_cast<Scalar>(3) / order;
    out[10][1] = static_cast<Scalar>(0) / order;
    out[10][2] = static_cast<Scalar>(0) / order;
    out[11][0] = static_cast<Scalar>(4) / order;
    out[11][1] = static_cast<Scalar>(0) / order;
    out[11][2] = static_cast<Scalar>(0) / order;
    out[12][0] = static_cast<Scalar>(5) / order;
    out[12][1] = static_cast<Scalar>(1) / order;
    out[12][2] = static_cast<Scalar>(0) / order;
    out[13][0] = static_cast<Scalar>(5) / order;
    out[13][1] = static_cast<Scalar>(2) / order;
    out[13][2] = static_cast<Scalar>(0) / order;
    out[14][0] = static_cast<Scalar>(5) / order;
    out[14][1] = static_cast<Scalar>(3) / order;
    out[14][2] = static_cast<Scalar>(0) / order;
    out[15][0] = static_cast<Scalar>(5) / order;
    out[15][1] = static_cast<Scalar>(4) / order;
    out[15][2] = static_cast<Scalar>(0) / order;
    out[16][0] = static_cast<Scalar>(1) / order;
    out[16][1] = static_cast<Scalar>(5) / order;
    out[16][2] = static_cast<Scalar>(0) / order;
    out[17][0] = static_cast<Scalar>(2) / order;
    out[17][1] = static_cast<Scalar>(5) / order;
    out[17][2] = static_cast<Scalar>(0) / order;
    out[18][0] = static_cast<Scalar>(3) / order;
    out[18][1] = static_cast<Scalar>(5) / order;
    out[18][2] = static_cast<Scalar>(0) / order;
    out[19][0] = static_cast<Scalar>(4) / order;
    out[19][1] = static_cast<Scalar>(5) / order;
    out[19][2] = static_cast<Scalar>(0) / order;
    out[20][0] = static_cast<Scalar>(0) / order;
    out[20][1] = static_cast<Scalar>(1) / order;
    out[20][2] = static_cast<Scalar>(0) / order;
    out[21][0] = static_cast<Scalar>(0) / order;
    out[21][1] = static_cast<Scalar>(2) / order;
    out[21][2] = static_cast<Scalar>(0) / order;
    out[22][0] = static_cast<Scalar>(0) / order;
    out[22][1] = static_cast<Scalar>(3) / order;
    out[22][2] = static_cast<Scalar>(0) / order;
    out[23][0] = static_cast<Scalar>(0) / order;
    out[23][1] = static_cast<Scalar>(4) / order;
    out[23][2] = static_cast<Scalar>(0) / order;
    out[24][0] = static_cast<Scalar>(1) / order;
    out[24][1] = static_cast<Scalar>(0) / order;
    out[24][2] = static_cast<Scalar>(5) / order;
    out[25][0] = static_cast<Scalar>(2) / order;
    out[25][1] = static_cast<Scalar>(0) / order;
    out[25][2] = static_cast<Scalar>(5) / order;
    out[26][0] = static_cast<Scalar>(3) / order;
    out[26][1] = static_cast<Scalar>(0) / order;
    out[26][2] = static_cast<Scalar>(5) / order;
    out[27][0] = static_cast<Scalar>(4) / order;
    out[27][1] = static_cast<Scalar>(0) / order;
    out[27][2] = static_cast<Scalar>(5) / order;
    out[28][0] = static_cast<Scalar>(5) / order;
    out[28][1] = static_cast<Scalar>(1) / order;
    out[28][2] = static_cast<Scalar>(5) / order;
    out[29][0] = static_cast<Scalar>(5) / order;
    out[29][1] = static_cast<Scalar>(2) / order;
    out[29][2] = static_cast<Scalar>(5) / order;
    out[30][0] = static_cast<Scalar>(5) / order;
    out[30][1] = static_cast<Scalar>(3) / order;
    out[30][2] = static_cast<Scalar>(5) / order;
    out[31][0] = static_cast<Scalar>(5) / order;
    out[31][1] = static_cast<Scalar>(4) / order;
    out[31][2] = static_cast<Scalar>(5) / order;
    out[32][0] = static_cast<Scalar>(1) / order;
    out[32][1] = static_cast<Scalar>(5) / order;
    out[32][2] = static_cast<Scalar>(5) / order;
    out[33][0] = static_cast<Scalar>(2) / order;
    out[33][1] = static_cast<Scalar>(5) / order;
    out[33][2] = static_cast<Scalar>(5) / order;
    out[34][0] = static_cast<Scalar>(3) / order;
    out[34][1] = static_cast<Scalar>(5) / order;
    out[34][2] = static_cast<Scalar>(5) / order;
    out[35][0] = static_cast<Scalar>(4) / order;
    out[35][1] = static_cast<Scalar>(5) / order;
    out[35][2] = static_cast<Scalar>(5) / order;
    out[36][0] = static_cast<Scalar>(0) / order;
    out[36][1] = static_cast<Scalar>(1) / order;
    out[36][2] = static_cast<Scalar>(5) / order;
    out[37][0] = static_cast<Scalar>(0) / order;
    out[37][1] = static_cast<Scalar>(2) / order;
    out[37][2] = static_cast<Scalar>(5) / order;
    out[38][0] = static_cast<Scalar>(0) / order;
    out[38][1] = static_cast<Scalar>(3) / order;
    out[38][2] = static_cast<Scalar>(5) / order;
    out[39][0] = static_cast<Scalar>(0) / order;
    out[39][1] = static_cast<Scalar>(4) / order;
    out[39][2] = static_cast<Scalar>(5) / order;
    out[40][0] = static_cast<Scalar>(0) / order;
    out[40][1] = static_cast<Scalar>(0) / order;
    out[40][2] = static_cast<Scalar>(1) / order;
    out[41][0] = static_cast<Scalar>(0) / order;
    out[41][1] = static_cast<Scalar>(0) / order;
    out[41][2] = static_cast<Scalar>(2) / order;
    out[42][0] = static_cast<Scalar>(0) / order;
    out[42][1] = static_cast<Scalar>(0) / order;
    out[42][2] = static_cast<Scalar>(3) / order;
    out[43][0] = static_cast<Scalar>(0) / order;
    out[43][1] = static_cast<Scalar>(0) / order;
    out[43][2] = static_cast<Scalar>(4) / order;
    out[44][0] = static_cast<Scalar>(5) / order;
    out[44][1] = static_cast<Scalar>(0) / order;
    out[44][2] = static_cast<Scalar>(1) / order;
    out[45][0] = static_cast<Scalar>(5) / order;
    out[45][1] = static_cast<Scalar>(0) / order;
    out[45][2] = static_cast<Scalar>(2) / order;
    out[46][0] = static_cast<Scalar>(5) / order;
    out[46][1] = static_cast<Scalar>(0) / order;
    out[46][2] = static_cast<Scalar>(3) / order;
    out[47][0] = static_cast<Scalar>(5) / order;
    out[47][1] = static_cast<Scalar>(0) / order;
    out[47][2] = static_cast<Scalar>(4) / order;
    out[48][0] = static_cast<Scalar>(5) / order;
    out[48][1] = static_cast<Scalar>(5) / order;
    out[48][2] = static_cast<Scalar>(1) / order;
    out[49][0] = static_cast<Scalar>(5) / order;
    out[49][1] = static_cast<Scalar>(5) / order;
    out[49][2] = static_cast<Scalar>(2) / order;
    out[50][0] = static_cast<Scalar>(5) / order;
    out[50][1] = static_cast<Scalar>(5) / order;
    out[50][2] = static_cast<Scalar>(3) / order;
    out[51][0] = static_cast<Scalar>(5) / order;
    out[51][1] = static_cast<Scalar>(5) / order;
    out[51][2] = static_cast<Scalar>(4) / order;
    out[52][0] = static_cast<Scalar>(0) / order;
    out[52][1] = static_cast<Scalar>(5) / order;
    out[52][2] = static_cast<Scalar>(1) / order;
    out[53][0] = static_cast<Scalar>(0) / order;
    out[53][1] = static_cast<Scalar>(5) / order;
    out[53][2] = static_cast<Scalar>(2) / order;
    out[54][0] = static_cast<Scalar>(0) / order;
    out[54][1] = static_cast<Scalar>(5) / order;
    out[54][2] = static_cast<Scalar>(3) / order;
    out[55][0] = static_cast<Scalar>(0) / order;
    out[55][1] = static_cast<Scalar>(5) / order;
    out[55][2] = static_cast<Scalar>(4) / order;
    out[56][0] = static_cast<Scalar>(0) / order;
    out[56][1] = static_cast<Scalar>(1) / order;
    out[56][2] = static_cast<Scalar>(1) / order;
    out[57][0] = static_cast<Scalar>(0) / order;
    out[57][1] = static_cast<Scalar>(2) / order;
    out[57][2] = static_cast<Scalar>(1) / order;
    out[58][0] = static_cast<Scalar>(0) / order;
    out[58][1] = static_cast<Scalar>(3) / order;
    out[58][2] = static_cast<Scalar>(1) / order;
    out[59][0] = static_cast<Scalar>(0) / order;
    out[59][1] = static_cast<Scalar>(4) / order;
    out[59][2] = static_cast<Scalar>(1) / order;
    out[60][0] = static_cast<Scalar>(0) / order;
    out[60][1] = static_cast<Scalar>(1) / order;
    out[60][2] = static_cast<Scalar>(2) / order;
    out[61][0] = static_cast<Scalar>(0) / order;
    out[61][1] = static_cast<Scalar>(2) / order;
    out[61][2] = static_cast<Scalar>(2) / order;
    out[62][0] = static_cast<Scalar>(0) / order;
    out[62][1] = static_cast<Scalar>(3) / order;
    out[62][2] = static_cast<Scalar>(2) / order;
    out[63][0] = static_cast<Scalar>(0) / order;
    out[63][1] = static_cast<Scalar>(4) / order;
    out[63][2] = static_cast<Scalar>(2) / order;
    out[64][0] = static_cast<Scalar>(0) / order;
    out[64][1] = static_cast<Scalar>(1) / order;
    out[64][2] = static_cast<Scalar>(3) / order;
    out[65][0] = static_cast<Scalar>(0) / order;
    out[65][1] = static_cast<Scalar>(2) / order;
    out[65][2] = static_cast<Scalar>(3) / order;
    out[66][0] = static_cast<Scalar>(0) / order;
    out[66][1] = static_cast<Scalar>(3) / order;
    out[66][2] = static_cast<Scalar>(3) / order;
    out[67][0] = static_cast<Scalar>(0) / order;
    out[67][1] = static_cast<Scalar>(4) / order;
    out[67][2] = static_cast<Scalar>(3) / order;
    out[68][0] = static_cast<Scalar>(0) / order;
    out[68][1] = static_cast<Scalar>(1) / order;
    out[68][2] = static_cast<Scalar>(4) / order;
    out[69][0] = static_cast<Scalar>(0) / order;
    out[69][1] = static_cast<Scalar>(2) / order;
    out[69][2] = static_cast<Scalar>(4) / order;
    out[70][0] = static_cast<Scalar>(0) / order;
    out[70][1] = static_cast<Scalar>(3) / order;
    out[70][2] = static_cast<Scalar>(4) / order;
    out[71][0] = static_cast<Scalar>(0) / order;
    out[71][1] = static_cast<Scalar>(4) / order;
    out[71][2] = static_cast<Scalar>(4) / order;
    out[72][0] = static_cast<Scalar>(5) / order;
    out[72][1] = static_cast<Scalar>(1) / order;
    out[72][2] = static_cast<Scalar>(1) / order;
    out[73][0] = static_cast<Scalar>(5) / order;
    out[73][1] = static_cast<Scalar>(2) / order;
    out[73][2] = static_cast<Scalar>(1) / order;
    out[74][0] = static_cast<Scalar>(5) / order;
    out[74][1] = static_cast<Scalar>(3) / order;
    out[74][2] = static_cast<Scalar>(1) / order;
    out[75][0] = static_cast<Scalar>(5) / order;
    out[75][1] = static_cast<Scalar>(4) / order;
    out[75][2] = static_cast<Scalar>(1) / order;
    out[76][0] = static_cast<Scalar>(5) / order;
    out[76][1] = static_cast<Scalar>(1) / order;
    out[76][2] = static_cast<Scalar>(2) / order;
    out[77][0] = static_cast<Scalar>(5) / order;
    out[77][1] = static_cast<Scalar>(2) / order;
    out[77][2] = static_cast<Scalar>(2) / order;
    out[78][0] = static_cast<Scalar>(5) / order;
    out[78][1] = static_cast<Scalar>(3) / order;
    out[78][2] = static_cast<Scalar>(2) / order;
    out[79][0] = static_cast<Scalar>(5) / order;
    out[79][1] = static_cast<Scalar>(4) / order;
    out[79][2] = static_cast<Scalar>(2) / order;
    out[80][0] = static_cast<Scalar>(5) / order;
    out[80][1] = static_cast<Scalar>(1) / order;
    out[80][2] = static_cast<Scalar>(3) / order;
    out[81][0] = static_cast<Scalar>(5) / order;
    out[81][1] = static_cast<Scalar>(2) / order;
    out[81][2] = static_cast<Scalar>(3) / order;
    out[82][0] = static_cast<Scalar>(5) / order;
    out[82][1] = static_cast<Scalar>(3) / order;
    out[82][2] = static_cast<Scalar>(3) / order;
    out[83][0] = static_cast<Scalar>(5) / order;
    out[83][1] = static_cast<Scalar>(4) / order;
    out[83][2] = static_cast<Scalar>(3) / order;
    out[84][0] = static_cast<Scalar>(5) / order;
    out[84][1] = static_cast<Scalar>(1) / order;
    out[84][2] = static_cast<Scalar>(4) / order;
    out[85][0] = static_cast<Scalar>(5) / order;
    out[85][1] = static_cast<Scalar>(2) / order;
    out[85][2] = static_cast<Scalar>(4) / order;
    out[86][0] = static_cast<Scalar>(5) / order;
    out[86][1] = static_cast<Scalar>(3) / order;
    out[86][2] = static_cast<Scalar>(4) / order;
    out[87][0] = static_cast<Scalar>(5) / order;
    out[87][1] = static_cast<Scalar>(4) / order;
    out[87][2] = static_cast<Scalar>(4) / order;
    out[88][0] = static_cast<Scalar>(1) / order;
    out[88][1] = static_cast<Scalar>(0) / order;
    out[88][2] = static_cast<Scalar>(1) / order;
    out[89][0] = static_cast<Scalar>(2) / order;
    out[89][1] = static_cast<Scalar>(0) / order;
    out[89][2] = static_cast<Scalar>(1) / order;
    out[90][0] = static_cast<Scalar>(3) / order;
    out[90][1] = static_cast<Scalar>(0) / order;
    out[90][2] = static_cast<Scalar>(1) / order;
    out[91][0] = static_cast<Scalar>(4) / order;
    out[91][1] = static_cast<Scalar>(0) / order;
    out[91][2] = static_cast<Scalar>(1) / order;
    out[92][0] = static_cast<Scalar>(1) / order;
    out[92][1] = static_cast<Scalar>(0) / order;
    out[92][2] = static_cast<Scalar>(2) / order;
    out[93][0] = static_cast<Scalar>(2) / order;
    out[93][1] = static_cast<Scalar>(0) / order;
    out[93][2] = static_cast<Scalar>(2) / order;
    out[94][0] = static_cast<Scalar>(3) / order;
    out[94][1] = static_cast<Scalar>(0) / order;
    out[94][2] = static_cast<Scalar>(2) / order;
    out[95][0] = static_cast<Scalar>(4) / order;
    out[95][1] = static_cast<Scalar>(0) / order;
    out[95][2] = static_cast<Scalar>(2) / order;
    out[96][0] = static_cast<Scalar>(1) / order;
    out[96][1] = static_cast<Scalar>(0) / order;
    out[96][2] = static_cast<Scalar>(3) / order;
    out[97][0] = static_cast<Scalar>(2) / order;
    out[97][1] = static_cast<Scalar>(0) / order;
    out[97][2] = static_cast<Scalar>(3) / order;
    out[98][0] = static_cast<Scalar>(3) / order;
    out[98][1] = static_cast<Scalar>(0) / order;
    out[98][2] = static_cast<Scalar>(3) / order;
    out[99][0] = static_cast<Scalar>(4) / order;
    out[99][1] = static_cast<Scalar>(0) / order;
    out[99][2] = static_cast<Scalar>(3) / order;
    out[100][0] = static_cast<Scalar>(1) / order;
    out[100][1] = static_cast<Scalar>(0) / order;
    out[100][2] = static_cast<Scalar>(4) / order;
    out[101][0] = static_cast<Scalar>(2) / order;
    out[101][1] = static_cast<Scalar>(0) / order;
    out[101][2] = static_cast<Scalar>(4) / order;
    out[102][0] = static_cast<Scalar>(3) / order;
    out[102][1] = static_cast<Scalar>(0) / order;
    out[102][2] = static_cast<Scalar>(4) / order;
    out[103][0] = static_cast<Scalar>(4) / order;
    out[103][1] = static_cast<Scalar>(0) / order;
    out[103][2] = static_cast<Scalar>(4) / order;
    out[104][0] = static_cast<Scalar>(1) / order;
    out[104][1] = static_cast<Scalar>(5) / order;
    out[104][2] = static_cast<Scalar>(1) / order;
    out[105][0] = static_cast<Scalar>(2) / order;
    out[105][1] = static_cast<Scalar>(5) / order;
    out[105][2] = static_cast<Scalar>(1) / order;
    out[106][0] = static_cast<Scalar>(3) / order;
    out[106][1] = static_cast<Scalar>(5) / order;
    out[106][2] = static_cast<Scalar>(1) / order;
    out[107][0] = static_cast<Scalar>(4) / order;
    out[107][1] = static_cast<Scalar>(5) / order;
    out[107][2] = static_cast<Scalar>(1) / order;
    out[108][0] = static_cast<Scalar>(1) / order;
    out[108][1] = static_cast<Scalar>(5) / order;
    out[108][2] = static_cast<Scalar>(2) / order;
    out[109][0] = static_cast<Scalar>(2) / order;
    out[109][1] = static_cast<Scalar>(5) / order;
    out[109][2] = static_cast<Scalar>(2) / order;
    out[110][0] = static_cast<Scalar>(3) / order;
    out[110][1] = static_cast<Scalar>(5) / order;
    out[110][2] = static_cast<Scalar>(2) / order;
    out[111][0] = static_cast<Scalar>(4) / order;
    out[111][1] = static_cast<Scalar>(5) / order;
    out[111][2] = static_cast<Scalar>(2) / order;
    out[112][0] = static_cast<Scalar>(1) / order;
    out[112][1] = static_cast<Scalar>(5) / order;
    out[112][2] = static_cast<Scalar>(3) / order;
    out[113][0] = static_cast<Scalar>(2) / order;
    out[113][1] = static_cast<Scalar>(5) / order;
    out[113][2] = static_cast<Scalar>(3) / order;
    out[114][0] = static_cast<Scalar>(3) / order;
    out[114][1] = static_cast<Scalar>(5) / order;
    out[114][2] = static_cast<Scalar>(3) / order;
    out[115][0] = static_cast<Scalar>(4) / order;
    out[115][1] = static_cast<Scalar>(5) / order;
    out[115][2] = static_cast<Scalar>(3) / order;
    out[116][0] = static_cast<Scalar>(1) / order;
    out[116][1] = static_cast<Scalar>(5) / order;
    out[116][2] = static_cast<Scalar>(4) / order;
    out[117][0] = static_cast<Scalar>(2) / order;
    out[117][1] = static_cast<Scalar>(5) / order;
    out[117][2] = static_cast<Scalar>(4) / order;
    out[118][0] = static_cast<Scalar>(3) / order;
    out[118][1] = static_cast<Scalar>(5) / order;
    out[118][2] = static_cast<Scalar>(4) / order;
    out[119][0] = static_cast<Scalar>(4) / order;
    out[119][1] = static_cast<Scalar>(5) / order;
    out[119][2] = static_cast<Scalar>(4) / order;
    out[120][0] = static_cast<Scalar>(1) / order;
    out[120][1] = static_cast<Scalar>(1) / order;
    out[120][2] = static_cast<Scalar>(0) / order;
    out[121][0] = static_cast<Scalar>(2) / order;
    out[121][1] = static_cast<Scalar>(1) / order;
    out[121][2] = static_cast<Scalar>(0) / order;
    out[122][0] = static_cast<Scalar>(3) / order;
    out[122][1] = static_cast<Scalar>(1) / order;
    out[122][2] = static_cast<Scalar>(0) / order;
    out[123][0] = static_cast<Scalar>(4) / order;
    out[123][1] = static_cast<Scalar>(1) / order;
    out[123][2] = static_cast<Scalar>(0) / order;
    out[124][0] = static_cast<Scalar>(1) / order;
    out[124][1] = static_cast<Scalar>(2) / order;
    out[124][2] = static_cast<Scalar>(0) / order;
    out[125][0] = static_cast<Scalar>(2) / order;
    out[125][1] = static_cast<Scalar>(2) / order;
    out[125][2] = static_cast<Scalar>(0) / order;
    out[126][0] = static_cast<Scalar>(3) / order;
    out[126][1] = static_cast<Scalar>(2) / order;
    out[126][2] = static_cast<Scalar>(0) / order;
    out[127][0] = static_cast<Scalar>(4) / order;
    out[127][1] = static_cast<Scalar>(2) / order;
    out[127][2] = static_cast<Scalar>(0) / order;
    out[128][0] = static_cast<Scalar>(1) / order;
    out[128][1] = static_cast<Scalar>(3) / order;
    out[128][2] = static_cast<Scalar>(0) / order;
    out[129][0] = static_cast<Scalar>(2) / order;
    out[129][1] = static_cast<Scalar>(3) / order;
    out[129][2] = static_cast<Scalar>(0) / order;
    out[130][0] = static_cast<Scalar>(3) / order;
    out[130][1] = static_cast<Scalar>(3) / order;
    out[130][2] = static_cast<Scalar>(0) / order;
    out[131][0] = static_cast<Scalar>(4) / order;
    out[131][1] = static_cast<Scalar>(3) / order;
    out[131][2] = static_cast<Scalar>(0) / order;
    out[132][0] = static_cast<Scalar>(1) / order;
    out[132][1] = static_cast<Scalar>(4) / order;
    out[132][2] = static_cast<Scalar>(0) / order;
    out[133][0] = static_cast<Scalar>(2) / order;
    out[133][1] = static_cast<Scalar>(4) / order;
    out[133][2] = static_cast<Scalar>(0) / order;
    out[134][0] = static_cast<Scalar>(3) / order;
    out[134][1] = static_cast<Scalar>(4) / order;
    out[134][2] = static_cast<Scalar>(0) / order;
    out[135][0] = static_cast<Scalar>(4) / order;
    out[135][1] = static_cast<Scalar>(4) / order;
    out[135][2] = static_cast<Scalar>(0) / order;
    out[136][0] = static_cast<Scalar>(1) / order;
    out[136][1] = static_cast<Scalar>(1) / order;
    out[136][2] = static_cast<Scalar>(5) / order;
    out[137][0] = static_cast<Scalar>(2) / order;
    out[137][1] = static_cast<Scalar>(1) / order;
    out[137][2] = static_cast<Scalar>(5) / order;
    out[138][0] = static_cast<Scalar>(3) / order;
    out[138][1] = static_cast<Scalar>(1) / order;
    out[138][2] = static_cast<Scalar>(5) / order;
    out[139][0] = static_cast<Scalar>(4) / order;
    out[139][1] = static_cast<Scalar>(1) / order;
    out[139][2] = static_cast<Scalar>(5) / order;
    out[140][0] = static_cast<Scalar>(1) / order;
    out[140][1] = static_cast<Scalar>(2) / order;
    out[140][2] = static_cast<Scalar>(5) / order;
    out[141][0] = static_cast<Scalar>(2) / order;
    out[141][1] = static_cast<Scalar>(2) / order;
    out[141][2] = static_cast<Scalar>(5) / order;
    out[142][0] = static_cast<Scalar>(3) / order;
    out[142][1] = static_cast<Scalar>(2) / order;
    out[142][2] = static_cast<Scalar>(5) / order;
    out[143][0] = static_cast<Scalar>(4) / order;
    out[143][1] = static_cast<Scalar>(2) / order;
    out[143][2] = static_cast<Scalar>(5) / order;
    out[144][0] = static_cast<Scalar>(1) / order;
    out[144][1] = static_cast<Scalar>(3) / order;
    out[144][2] = static_cast<Scalar>(5) / order;
    out[145][0] = static_cast<Scalar>(2) / order;
    out[145][1] = static_cast<Scalar>(3) / order;
    out[145][2] = static_cast<Scalar>(5) / order;
    out[146][0] = static_cast<Scalar>(3) / order;
    out[146][1] = static_cast<Scalar>(3) / order;
    out[146][2] = static_cast<Scalar>(5) / order;
    out[147][0] = static_cast<Scalar>(4) / order;
    out[147][1] = static_cast<Scalar>(3) / order;
    out[147][2] = static_cast<Scalar>(5) / order;
    out[148][0] = static_cast<Scalar>(1) / order;
    out[148][1] = static_cast<Scalar>(4) / order;
    out[148][2] = static_cast<Scalar>(5) / order;
    out[149][0] = static_cast<Scalar>(2) / order;
    out[149][1] = static_cast<Scalar>(4) / order;
    out[149][2] = static_cast<Scalar>(5) / order;
    out[150][0] = static_cast<Scalar>(3) / order;
    out[150][1] = static_cast<Scalar>(4) / order;
    out[150][2] = static_cast<Scalar>(5) / order;
    out[151][0] = static_cast<Scalar>(4) / order;
    out[151][1] = static_cast<Scalar>(4) / order;
    out[151][2] = static_cast<Scalar>(5) / order;
    out[152][0] = static_cast<Scalar>(1) / order;
    out[152][1] = static_cast<Scalar>(1) / order;
    out[152][2] = static_cast<Scalar>(1) / order;
    out[153][0] = static_cast<Scalar>(2) / order;
    out[153][1] = static_cast<Scalar>(1) / order;
    out[153][2] = static_cast<Scalar>(1) / order;
    out[154][0] = static_cast<Scalar>(3) / order;
    out[154][1] = static_cast<Scalar>(1) / order;
    out[154][2] = static_cast<Scalar>(1) / order;
    out[155][0] = static_cast<Scalar>(4) / order;
    out[155][1] = static_cast<Scalar>(1) / order;
    out[155][2] = static_cast<Scalar>(1) / order;
    out[156][0] = static_cast<Scalar>(1) / order;
    out[156][1] = static_cast<Scalar>(2) / order;
    out[156][2] = static_cast<Scalar>(1) / order;
    out[157][0] = static_cast<Scalar>(2) / order;
    out[157][1] = static_cast<Scalar>(2) / order;
    out[157][2] = static_cast<Scalar>(1) / order;
    out[158][0] = static_cast<Scalar>(3) / order;
    out[158][1] = static_cast<Scalar>(2) / order;
    out[158][2] = static_cast<Scalar>(1) / order;
    out[159][0] = static_cast<Scalar>(4) / order;
    out[159][1] = static_cast<Scalar>(2) / order;
    out[159][2] = static_cast<Scalar>(1) / order;
    out[160][0] = static_cast<Scalar>(1) / order;
    out[160][1] = static_cast<Scalar>(3) / order;
    out[160][2] = static_cast<Scalar>(1) / order;
    out[161][0] = static_cast<Scalar>(2) / order;
    out[161][1] = static_cast<Scalar>(3) / order;
    out[161][2] = static_cast<Scalar>(1) / order;
    out[162][0] = static_cast<Scalar>(3) / order;
    out[162][1] = static_cast<Scalar>(3) / order;
    out[162][2] = static_cast<Scalar>(1) / order;
    out[163][0] = static_cast<Scalar>(4) / order;
    out[163][1] = static_cast<Scalar>(3) / order;
    out[163][2] = static_cast<Scalar>(1) / order;
    out[164][0] = static_cast<Scalar>(1) / order;
    out[164][1] = static_cast<Scalar>(4) / order;
    out[164][2] = static_cast<Scalar>(1) / order;
    out[165][0] = static_cast<Scalar>(2) / order;
    out[165][1] = static_cast<Scalar>(4) / order;
    out[165][2] = static_cast<Scalar>(1) / order;
    out[166][0] = static_cast<Scalar>(3) / order;
    out[166][1] = static_cast<Scalar>(4) / order;
    out[166][2] = static_cast<Scalar>(1) / order;
    out[167][0] = static_cast<Scalar>(4) / order;
    out[167][1] = static_cast<Scalar>(4) / order;
    out[167][2] = static_cast<Scalar>(1) / order;
    out[168][0] = static_cast<Scalar>(1) / order;
    out[168][1] = static_cast<Scalar>(1) / order;
    out[168][2] = static_cast<Scalar>(2) / order;
    out[169][0] = static_cast<Scalar>(2) / order;
    out[169][1] = static_cast<Scalar>(1) / order;
    out[169][2] = static_cast<Scalar>(2) / order;
    out[170][0] = static_cast<Scalar>(3) / order;
    out[170][1] = static_cast<Scalar>(1) / order;
    out[170][2] = static_cast<Scalar>(2) / order;
    out[171][0] = static_cast<Scalar>(4) / order;
    out[171][1] = static_cast<Scalar>(1) / order;
    out[171][2] = static_cast<Scalar>(2) / order;
    out[172][0] = static_cast<Scalar>(1) / order;
    out[172][1] = static_cast<Scalar>(2) / order;
    out[172][2] = static_cast<Scalar>(2) / order;
    out[173][0] = static_cast<Scalar>(2) / order;
    out[173][1] = static_cast<Scalar>(2) / order;
    out[173][2] = static_cast<Scalar>(2) / order;
    out[174][0] = static_cast<Scalar>(3) / order;
    out[174][1] = static_cast<Scalar>(2) / order;
    out[174][2] = static_cast<Scalar>(2) / order;
    out[175][0] = static_cast<Scalar>(4) / order;
    out[175][1] = static_cast<Scalar>(2) / order;
    out[175][2] = static_cast<Scalar>(2) / order;
    out[176][0] = static_cast<Scalar>(1) / order;
    out[176][1] = static_cast<Scalar>(3) / order;
    out[176][2] = static_cast<Scalar>(2) / order;
    out[177][0] = static_cast<Scalar>(2) / order;
    out[177][1] = static_cast<Scalar>(3) / order;
    out[177][2] = static_cast<Scalar>(2) / order;
    out[178][0] = static_cast<Scalar>(3) / order;
    out[178][1] = static_cast<Scalar>(3) / order;
    out[178][2] = static_cast<Scalar>(2) / order;
    out[179][0] = static_cast<Scalar>(4) / order;
    out[179][1] = static_cast<Scalar>(3) / order;
    out[179][2] = static_cast<Scalar>(2) / order;
    out[180][0] = static_cast<Scalar>(1) / order;
    out[180][1] = static_cast<Scalar>(4) / order;
    out[180][2] = static_cast<Scalar>(2) / order;
    out[181][0] = static_cast<Scalar>(2) / order;
    out[181][1] = static_cast<Scalar>(4) / order;
    out[181][2] = static_cast<Scalar>(2) / order;
    out[182][0] = static_cast<Scalar>(3) / order;
    out[182][1] = static_cast<Scalar>(4) / order;
    out[182][2] = static_cast<Scalar>(2) / order;
    out[183][0] = static_cast<Scalar>(4) / order;
    out[183][1] = static_cast<Scalar>(4) / order;
    out[183][2] = static_cast<Scalar>(2) / order;
    out[184][0] = static_cast<Scalar>(1) / order;
    out[184][1] = static_cast<Scalar>(1) / order;
    out[184][2] = static_cast<Scalar>(3) / order;
    out[185][0] = static_cast<Scalar>(2) / order;
    out[185][1] = static_cast<Scalar>(1) / order;
    out[185][2] = static_cast<Scalar>(3) / order;
    out[186][0] = static_cast<Scalar>(3) / order;
    out[186][1] = static_cast<Scalar>(1) / order;
    out[186][2] = static_cast<Scalar>(3) / order;
    out[187][0] = static_cast<Scalar>(4) / order;
    out[187][1] = static_cast<Scalar>(1) / order;
    out[187][2] = static_cast<Scalar>(3) / order;
    out[188][0] = static_cast<Scalar>(1) / order;
    out[188][1] = static_cast<Scalar>(2) / order;
    out[188][2] = static_cast<Scalar>(3) / order;
    out[189][0] = static_cast<Scalar>(2) / order;
    out[189][1] = static_cast<Scalar>(2) / order;
    out[189][2] = static_cast<Scalar>(3) / order;
    out[190][0] = static_cast<Scalar>(3) / order;
    out[190][1] = static_cast<Scalar>(2) / order;
    out[190][2] = static_cast<Scalar>(3) / order;
    out[191][0] = static_cast<Scalar>(4) / order;
    out[191][1] = static_cast<Scalar>(2) / order;
    out[191][2] = static_cast<Scalar>(3) / order;
    out[192][0] = static_cast<Scalar>(1) / order;
    out[192][1] = static_cast<Scalar>(3) / order;
    out[192][2] = static_cast<Scalar>(3) / order;
    out[193][0] = static_cast<Scalar>(2) / order;
    out[193][1] = static_cast<Scalar>(3) / order;
    out[193][2] = static_cast<Scalar>(3) / order;
    out[194][0] = static_cast<Scalar>(3) / order;
    out[194][1] = static_cast<Scalar>(3) / order;
    out[194][2] = static_cast<Scalar>(3) / order;
    out[195][0] = static_cast<Scalar>(4) / order;
    out[195][1] = static_cast<Scalar>(3) / order;
    out[195][2] = static_cast<Scalar>(3) / order;
    out[196][0] = static_cast<Scalar>(1) / order;
    out[196][1] = static_cast<Scalar>(4) / order;
    out[196][2] = static_cast<Scalar>(3) / order;
    out[197][0] = static_cast<Scalar>(2) / order;
    out[197][1] = static_cast<Scalar>(4) / order;
    out[197][2] = static_cast<Scalar>(3) / order;
    out[198][0] = static_cast<Scalar>(3) / order;
    out[198][1] = static_cast<Scalar>(4) / order;
    out[198][2] = static_cast<Scalar>(3) / order;
    out[199][0] = static_cast<Scalar>(4) / order;
    out[199][1] = static_cast<Scalar>(4) / order;
    out[199][2] = static_cast<Scalar>(3) / order;
    out[200][0] = static_cast<Scalar>(1) / order;
    out[200][1] = static_cast<Scalar>(1) / order;
    out[200][2] = static_cast<Scalar>(4) / order;
    out[201][0] = static_cast<Scalar>(2) / order;
    out[201][1] = static_cast<Scalar>(1) / order;
    out[201][2] = static_cast<Scalar>(4) / order;
    out[202][0] = static_cast<Scalar>(3) / order;
    out[202][1] = static_cast<Scalar>(1) / order;
    out[202][2] = static_cast<Scalar>(4) / order;
    out[203][0] = static_cast<Scalar>(4) / order;
    out[203][1] = static_cast<Scalar>(1) / order;
    out[203][2] = static_cast<Scalar>(4) / order;
    out[204][0] = static_cast<Scalar>(1) / order;
    out[204][1] = static_cast<Scalar>(2) / order;
    out[204][2] = static_cast<Scalar>(4) / order;
    out[205][0] = static_cast<Scalar>(2) / order;
    out[205][1] = static_cast<Scalar>(2) / order;
    out[205][2] = static_cast<Scalar>(4) / order;
    out[206][0] = static_cast<Scalar>(3) / order;
    out[206][1] = static_cast<Scalar>(2) / order;
    out[206][2] = static_cast<Scalar>(4) / order;
    out[207][0] = static_cast<Scalar>(4) / order;
    out[207][1] = static_cast<Scalar>(2) / order;
    out[207][2] = static_cast<Scalar>(4) / order;
    out[208][0] = static_cast<Scalar>(1) / order;
    out[208][1] = static_cast<Scalar>(3) / order;
    out[208][2] = static_cast<Scalar>(4) / order;
    out[209][0] = static_cast<Scalar>(2) / order;
    out[209][1] = static_cast<Scalar>(3) / order;
    out[209][2] = static_cast<Scalar>(4) / order;
    out[210][0] = static_cast<Scalar>(3) / order;
    out[210][1] = static_cast<Scalar>(3) / order;
    out[210][2] = static_cast<Scalar>(4) / order;
    out[211][0] = static_cast<Scalar>(4) / order;
    out[211][1] = static_cast<Scalar>(3) / order;
    out[211][2] = static_cast<Scalar>(4) / order;
    out[212][0] = static_cast<Scalar>(1) / order;
    out[212][1] = static_cast<Scalar>(4) / order;
    out[212][2] = static_cast<Scalar>(4) / order;
    out[213][0] = static_cast<Scalar>(2) / order;
    out[213][1] = static_cast<Scalar>(4) / order;
    out[213][2] = static_cast<Scalar>(4) / order;
    out[214][0] = static_cast<Scalar>(3) / order;
    out[214][1] = static_cast<Scalar>(4) / order;
    out[214][2] = static_cast<Scalar>(4) / order;
    out[215][0] = static_cast<Scalar>(4) / order;
    out[215][1] = static_cast<Scalar>(4) / order;
    out[215][2] = static_cast<Scalar>(4) / order;
  }
  template <typename Scalar>
  static constexpr void interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
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
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 10:
      out[0] = 3;
      out[1] = 0;
      out[2] = 0;
      break;
    case 11:
      out[0] = 4;
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
      out[1] = 2;
      out[2] = 0;
      break;
    case 14:
      out[0] = 5;
      out[1] = 3;
      out[2] = 0;
      break;
    case 15:
      out[0] = 5;
      out[1] = 4;
      out[2] = 0;
      break;
    case 16:
      out[0] = 1;
      out[1] = 5;
      out[2] = 0;
      break;
    case 17:
      out[0] = 2;
      out[1] = 5;
      out[2] = 0;
      break;
    case 18:
      out[0] = 3;
      out[1] = 5;
      out[2] = 0;
      break;
    case 19:
      out[0] = 4;
      out[1] = 5;
      out[2] = 0;
      break;
    case 20:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 21:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 22:
      out[0] = 0;
      out[1] = 3;
      out[2] = 0;
      break;
    case 23:
      out[0] = 0;
      out[1] = 4;
      out[2] = 0;
      break;
    case 24:
      out[0] = 1;
      out[1] = 0;
      out[2] = 5;
      break;
    case 25:
      out[0] = 2;
      out[1] = 0;
      out[2] = 5;
      break;
    case 26:
      out[0] = 3;
      out[1] = 0;
      out[2] = 5;
      break;
    case 27:
      out[0] = 4;
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
      out[1] = 2;
      out[2] = 5;
      break;
    case 30:
      out[0] = 5;
      out[1] = 3;
      out[2] = 5;
      break;
    case 31:
      out[0] = 5;
      out[1] = 4;
      out[2] = 5;
      break;
    case 32:
      out[0] = 1;
      out[1] = 5;
      out[2] = 5;
      break;
    case 33:
      out[0] = 2;
      out[1] = 5;
      out[2] = 5;
      break;
    case 34:
      out[0] = 3;
      out[1] = 5;
      out[2] = 5;
      break;
    case 35:
      out[0] = 4;
      out[1] = 5;
      out[2] = 5;
      break;
    case 36:
      out[0] = 0;
      out[1] = 1;
      out[2] = 5;
      break;
    case 37:
      out[0] = 0;
      out[1] = 2;
      out[2] = 5;
      break;
    case 38:
      out[0] = 0;
      out[1] = 3;
      out[2] = 5;
      break;
    case 39:
      out[0] = 0;
      out[1] = 4;
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
      out[2] = 2;
      break;
    case 42:
      out[0] = 0;
      out[1] = 0;
      out[2] = 3;
      break;
    case 43:
      out[0] = 0;
      out[1] = 0;
      out[2] = 4;
      break;
    case 44:
      out[0] = 5;
      out[1] = 0;
      out[2] = 1;
      break;
    case 45:
      out[0] = 5;
      out[1] = 0;
      out[2] = 2;
      break;
    case 46:
      out[0] = 5;
      out[1] = 0;
      out[2] = 3;
      break;
    case 47:
      out[0] = 5;
      out[1] = 0;
      out[2] = 4;
      break;
    case 48:
      out[0] = 5;
      out[1] = 5;
      out[2] = 1;
      break;
    case 49:
      out[0] = 5;
      out[1] = 5;
      out[2] = 2;
      break;
    case 50:
      out[0] = 5;
      out[1] = 5;
      out[2] = 3;
      break;
    case 51:
      out[0] = 5;
      out[1] = 5;
      out[2] = 4;
      break;
    case 52:
      out[0] = 0;
      out[1] = 5;
      out[2] = 1;
      break;
    case 53:
      out[0] = 0;
      out[1] = 5;
      out[2] = 2;
      break;
    case 54:
      out[0] = 0;
      out[1] = 5;
      out[2] = 3;
      break;
    case 55:
      out[0] = 0;
      out[1] = 5;
      out[2] = 4;
      break;
    case 56:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 57:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 58:
      out[0] = 0;
      out[1] = 3;
      out[2] = 1;
      break;
    case 59:
      out[0] = 0;
      out[1] = 4;
      out[2] = 1;
      break;
    case 60:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 61:
      out[0] = 0;
      out[1] = 2;
      out[2] = 2;
      break;
    case 62:
      out[0] = 0;
      out[1] = 3;
      out[2] = 2;
      break;
    case 63:
      out[0] = 0;
      out[1] = 4;
      out[2] = 2;
      break;
    case 64:
      out[0] = 0;
      out[1] = 1;
      out[2] = 3;
      break;
    case 65:
      out[0] = 0;
      out[1] = 2;
      out[2] = 3;
      break;
    case 66:
      out[0] = 0;
      out[1] = 3;
      out[2] = 3;
      break;
    case 67:
      out[0] = 0;
      out[1] = 4;
      out[2] = 3;
      break;
    case 68:
      out[0] = 0;
      out[1] = 1;
      out[2] = 4;
      break;
    case 69:
      out[0] = 0;
      out[1] = 2;
      out[2] = 4;
      break;
    case 70:
      out[0] = 0;
      out[1] = 3;
      out[2] = 4;
      break;
    case 71:
      out[0] = 0;
      out[1] = 4;
      out[2] = 4;
      break;
    case 72:
      out[0] = 5;
      out[1] = 1;
      out[2] = 1;
      break;
    case 73:
      out[0] = 5;
      out[1] = 2;
      out[2] = 1;
      break;
    case 74:
      out[0] = 5;
      out[1] = 3;
      out[2] = 1;
      break;
    case 75:
      out[0] = 5;
      out[1] = 4;
      out[2] = 1;
      break;
    case 76:
      out[0] = 5;
      out[1] = 1;
      out[2] = 2;
      break;
    case 77:
      out[0] = 5;
      out[1] = 2;
      out[2] = 2;
      break;
    case 78:
      out[0] = 5;
      out[1] = 3;
      out[2] = 2;
      break;
    case 79:
      out[0] = 5;
      out[1] = 4;
      out[2] = 2;
      break;
    case 80:
      out[0] = 5;
      out[1] = 1;
      out[2] = 3;
      break;
    case 81:
      out[0] = 5;
      out[1] = 2;
      out[2] = 3;
      break;
    case 82:
      out[0] = 5;
      out[1] = 3;
      out[2] = 3;
      break;
    case 83:
      out[0] = 5;
      out[1] = 4;
      out[2] = 3;
      break;
    case 84:
      out[0] = 5;
      out[1] = 1;
      out[2] = 4;
      break;
    case 85:
      out[0] = 5;
      out[1] = 2;
      out[2] = 4;
      break;
    case 86:
      out[0] = 5;
      out[1] = 3;
      out[2] = 4;
      break;
    case 87:
      out[0] = 5;
      out[1] = 4;
      out[2] = 4;
      break;
    case 88:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 89:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 90:
      out[0] = 3;
      out[1] = 0;
      out[2] = 1;
      break;
    case 91:
      out[0] = 4;
      out[1] = 0;
      out[2] = 1;
      break;
    case 92:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 93:
      out[0] = 2;
      out[1] = 0;
      out[2] = 2;
      break;
    case 94:
      out[0] = 3;
      out[1] = 0;
      out[2] = 2;
      break;
    case 95:
      out[0] = 4;
      out[1] = 0;
      out[2] = 2;
      break;
    case 96:
      out[0] = 1;
      out[1] = 0;
      out[2] = 3;
      break;
    case 97:
      out[0] = 2;
      out[1] = 0;
      out[2] = 3;
      break;
    case 98:
      out[0] = 3;
      out[1] = 0;
      out[2] = 3;
      break;
    case 99:
      out[0] = 4;
      out[1] = 0;
      out[2] = 3;
      break;
    case 100:
      out[0] = 1;
      out[1] = 0;
      out[2] = 4;
      break;
    case 101:
      out[0] = 2;
      out[1] = 0;
      out[2] = 4;
      break;
    case 102:
      out[0] = 3;
      out[1] = 0;
      out[2] = 4;
      break;
    case 103:
      out[0] = 4;
      out[1] = 0;
      out[2] = 4;
      break;
    case 104:
      out[0] = 1;
      out[1] = 5;
      out[2] = 1;
      break;
    case 105:
      out[0] = 2;
      out[1] = 5;
      out[2] = 1;
      break;
    case 106:
      out[0] = 3;
      out[1] = 5;
      out[2] = 1;
      break;
    case 107:
      out[0] = 4;
      out[1] = 5;
      out[2] = 1;
      break;
    case 108:
      out[0] = 1;
      out[1] = 5;
      out[2] = 2;
      break;
    case 109:
      out[0] = 2;
      out[1] = 5;
      out[2] = 2;
      break;
    case 110:
      out[0] = 3;
      out[1] = 5;
      out[2] = 2;
      break;
    case 111:
      out[0] = 4;
      out[1] = 5;
      out[2] = 2;
      break;
    case 112:
      out[0] = 1;
      out[1] = 5;
      out[2] = 3;
      break;
    case 113:
      out[0] = 2;
      out[1] = 5;
      out[2] = 3;
      break;
    case 114:
      out[0] = 3;
      out[1] = 5;
      out[2] = 3;
      break;
    case 115:
      out[0] = 4;
      out[1] = 5;
      out[2] = 3;
      break;
    case 116:
      out[0] = 1;
      out[1] = 5;
      out[2] = 4;
      break;
    case 117:
      out[0] = 2;
      out[1] = 5;
      out[2] = 4;
      break;
    case 118:
      out[0] = 3;
      out[1] = 5;
      out[2] = 4;
      break;
    case 119:
      out[0] = 4;
      out[1] = 5;
      out[2] = 4;
      break;
    case 120:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 121:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 122:
      out[0] = 3;
      out[1] = 1;
      out[2] = 0;
      break;
    case 123:
      out[0] = 4;
      out[1] = 1;
      out[2] = 0;
      break;
    case 124:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 125:
      out[0] = 2;
      out[1] = 2;
      out[2] = 0;
      break;
    case 126:
      out[0] = 3;
      out[1] = 2;
      out[2] = 0;
      break;
    case 127:
      out[0] = 4;
      out[1] = 2;
      out[2] = 0;
      break;
    case 128:
      out[0] = 1;
      out[1] = 3;
      out[2] = 0;
      break;
    case 129:
      out[0] = 2;
      out[1] = 3;
      out[2] = 0;
      break;
    case 130:
      out[0] = 3;
      out[1] = 3;
      out[2] = 0;
      break;
    case 131:
      out[0] = 4;
      out[1] = 3;
      out[2] = 0;
      break;
    case 132:
      out[0] = 1;
      out[1] = 4;
      out[2] = 0;
      break;
    case 133:
      out[0] = 2;
      out[1] = 4;
      out[2] = 0;
      break;
    case 134:
      out[0] = 3;
      out[1] = 4;
      out[2] = 0;
      break;
    case 135:
      out[0] = 4;
      out[1] = 4;
      out[2] = 0;
      break;
    case 136:
      out[0] = 1;
      out[1] = 1;
      out[2] = 5;
      break;
    case 137:
      out[0] = 2;
      out[1] = 1;
      out[2] = 5;
      break;
    case 138:
      out[0] = 3;
      out[1] = 1;
      out[2] = 5;
      break;
    case 139:
      out[0] = 4;
      out[1] = 1;
      out[2] = 5;
      break;
    case 140:
      out[0] = 1;
      out[1] = 2;
      out[2] = 5;
      break;
    case 141:
      out[0] = 2;
      out[1] = 2;
      out[2] = 5;
      break;
    case 142:
      out[0] = 3;
      out[1] = 2;
      out[2] = 5;
      break;
    case 143:
      out[0] = 4;
      out[1] = 2;
      out[2] = 5;
      break;
    case 144:
      out[0] = 1;
      out[1] = 3;
      out[2] = 5;
      break;
    case 145:
      out[0] = 2;
      out[1] = 3;
      out[2] = 5;
      break;
    case 146:
      out[0] = 3;
      out[1] = 3;
      out[2] = 5;
      break;
    case 147:
      out[0] = 4;
      out[1] = 3;
      out[2] = 5;
      break;
    case 148:
      out[0] = 1;
      out[1] = 4;
      out[2] = 5;
      break;
    case 149:
      out[0] = 2;
      out[1] = 4;
      out[2] = 5;
      break;
    case 150:
      out[0] = 3;
      out[1] = 4;
      out[2] = 5;
      break;
    case 151:
      out[0] = 4;
      out[1] = 4;
      out[2] = 5;
      break;
    case 152:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    case 153:
      out[0] = 2;
      out[1] = 1;
      out[2] = 1;
      break;
    case 154:
      out[0] = 3;
      out[1] = 1;
      out[2] = 1;
      break;
    case 155:
      out[0] = 4;
      out[1] = 1;
      out[2] = 1;
      break;
    case 156:
      out[0] = 1;
      out[1] = 2;
      out[2] = 1;
      break;
    case 157:
      out[0] = 2;
      out[1] = 2;
      out[2] = 1;
      break;
    case 158:
      out[0] = 3;
      out[1] = 2;
      out[2] = 1;
      break;
    case 159:
      out[0] = 4;
      out[1] = 2;
      out[2] = 1;
      break;
    case 160:
      out[0] = 1;
      out[1] = 3;
      out[2] = 1;
      break;
    case 161:
      out[0] = 2;
      out[1] = 3;
      out[2] = 1;
      break;
    case 162:
      out[0] = 3;
      out[1] = 3;
      out[2] = 1;
      break;
    case 163:
      out[0] = 4;
      out[1] = 3;
      out[2] = 1;
      break;
    case 164:
      out[0] = 1;
      out[1] = 4;
      out[2] = 1;
      break;
    case 165:
      out[0] = 2;
      out[1] = 4;
      out[2] = 1;
      break;
    case 166:
      out[0] = 3;
      out[1] = 4;
      out[2] = 1;
      break;
    case 167:
      out[0] = 4;
      out[1] = 4;
      out[2] = 1;
      break;
    case 168:
      out[0] = 1;
      out[1] = 1;
      out[2] = 2;
      break;
    case 169:
      out[0] = 2;
      out[1] = 1;
      out[2] = 2;
      break;
    case 170:
      out[0] = 3;
      out[1] = 1;
      out[2] = 2;
      break;
    case 171:
      out[0] = 4;
      out[1] = 1;
      out[2] = 2;
      break;
    case 172:
      out[0] = 1;
      out[1] = 2;
      out[2] = 2;
      break;
    case 173:
      out[0] = 2;
      out[1] = 2;
      out[2] = 2;
      break;
    case 174:
      out[0] = 3;
      out[1] = 2;
      out[2] = 2;
      break;
    case 175:
      out[0] = 4;
      out[1] = 2;
      out[2] = 2;
      break;
    case 176:
      out[0] = 1;
      out[1] = 3;
      out[2] = 2;
      break;
    case 177:
      out[0] = 2;
      out[1] = 3;
      out[2] = 2;
      break;
    case 178:
      out[0] = 3;
      out[1] = 3;
      out[2] = 2;
      break;
    case 179:
      out[0] = 4;
      out[1] = 3;
      out[2] = 2;
      break;
    case 180:
      out[0] = 1;
      out[1] = 4;
      out[2] = 2;
      break;
    case 181:
      out[0] = 2;
      out[1] = 4;
      out[2] = 2;
      break;
    case 182:
      out[0] = 3;
      out[1] = 4;
      out[2] = 2;
      break;
    case 183:
      out[0] = 4;
      out[1] = 4;
      out[2] = 2;
      break;
    case 184:
      out[0] = 1;
      out[1] = 1;
      out[2] = 3;
      break;
    case 185:
      out[0] = 2;
      out[1] = 1;
      out[2] = 3;
      break;
    case 186:
      out[0] = 3;
      out[1] = 1;
      out[2] = 3;
      break;
    case 187:
      out[0] = 4;
      out[1] = 1;
      out[2] = 3;
      break;
    case 188:
      out[0] = 1;
      out[1] = 2;
      out[2] = 3;
      break;
    case 189:
      out[0] = 2;
      out[1] = 2;
      out[2] = 3;
      break;
    case 190:
      out[0] = 3;
      out[1] = 2;
      out[2] = 3;
      break;
    case 191:
      out[0] = 4;
      out[1] = 2;
      out[2] = 3;
      break;
    case 192:
      out[0] = 1;
      out[1] = 3;
      out[2] = 3;
      break;
    case 193:
      out[0] = 2;
      out[1] = 3;
      out[2] = 3;
      break;
    case 194:
      out[0] = 3;
      out[1] = 3;
      out[2] = 3;
      break;
    case 195:
      out[0] = 4;
      out[1] = 3;
      out[2] = 3;
      break;
    case 196:
      out[0] = 1;
      out[1] = 4;
      out[2] = 3;
      break;
    case 197:
      out[0] = 2;
      out[1] = 4;
      out[2] = 3;
      break;
    case 198:
      out[0] = 3;
      out[1] = 4;
      out[2] = 3;
      break;
    case 199:
      out[0] = 4;
      out[1] = 4;
      out[2] = 3;
      break;
    case 200:
      out[0] = 1;
      out[1] = 1;
      out[2] = 4;
      break;
    case 201:
      out[0] = 2;
      out[1] = 1;
      out[2] = 4;
      break;
    case 202:
      out[0] = 3;
      out[1] = 1;
      out[2] = 4;
      break;
    case 203:
      out[0] = 4;
      out[1] = 1;
      out[2] = 4;
      break;
    case 204:
      out[0] = 1;
      out[1] = 2;
      out[2] = 4;
      break;
    case 205:
      out[0] = 2;
      out[1] = 2;
      out[2] = 4;
      break;
    case 206:
      out[0] = 3;
      out[1] = 2;
      out[2] = 4;
      break;
    case 207:
      out[0] = 4;
      out[1] = 2;
      out[2] = 4;
      break;
    case 208:
      out[0] = 1;
      out[1] = 3;
      out[2] = 4;
      break;
    case 209:
      out[0] = 2;
      out[1] = 3;
      out[2] = 4;
      break;
    case 210:
      out[0] = 3;
      out[1] = 3;
      out[2] = 4;
      break;
    case 211:
      out[0] = 4;
      out[1] = 3;
      out[2] = 4;
      break;
    case 212:
      out[0] = 1;
      out[1] = 4;
      out[2] = 4;
      break;
    case 213:
      out[0] = 2;
      out[1] = 4;
      out[2] = 4;
      break;
    case 214:
      out[0] = 3;
      out[1] = 4;
      out[2] = 4;
      break;
    case 215:
      out[0] = 4;
      out[1] = 4;
      out[2] = 4;
      break;
    }
  }

  template <typename Element>
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs) {
    subelement_node_idxs(subelement, idxs, meta::type_tag<Element>{});
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElPoint>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      break;
    case 1:
      idxs[0] = 1;
      break;
    case 2:
      idxs[0] = 2;
      break;
    case 3:
      idxs[0] = 3;
      break;
    case 4:
      idxs[0] = 4;
      break;
    case 5:
      idxs[0] = 5;
      break;
    case 6:
      idxs[0] = 6;
      break;
    case 7:
      idxs[0] = 7;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 8;
      idxs[3] = 9;
      idxs[4] = 10;
      idxs[5] = 11;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 12;
      idxs[3] = 13;
      idxs[4] = 14;
      idxs[5] = 15;
      break;
    case 2:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 16;
      idxs[3] = 17;
      idxs[4] = 18;
      idxs[5] = 19;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 20;
      idxs[3] = 21;
      idxs[4] = 22;
      idxs[5] = 23;
      break;
    case 4:
      idxs[0] = 4;
      idxs[1] = 5;
      idxs[2] = 24;
      idxs[3] = 25;
      idxs[4] = 26;
      idxs[5] = 27;
      break;
    case 5:
      idxs[0] = 5;
      idxs[1] = 6;
      idxs[2] = 28;
      idxs[3] = 29;
      idxs[4] = 30;
      idxs[5] = 31;
      break;
    case 6:
      idxs[0] = 7;
      idxs[1] = 6;
      idxs[2] = 32;
      idxs[3] = 33;
      idxs[4] = 34;
      idxs[5] = 35;
      break;
    case 7:
      idxs[0] = 4;
      idxs[1] = 7;
      idxs[2] = 36;
      idxs[3] = 37;
      idxs[4] = 38;
      idxs[5] = 39;
      break;
    case 8:
      idxs[0] = 0;
      idxs[1] = 4;
      idxs[2] = 40;
      idxs[3] = 41;
      idxs[4] = 42;
      idxs[5] = 43;
      break;
    case 9:
      idxs[0] = 1;
      idxs[1] = 5;
      idxs[2] = 44;
      idxs[3] = 45;
      idxs[4] = 46;
      idxs[5] = 47;
      break;
    case 10:
      idxs[0] = 2;
      idxs[1] = 6;
      idxs[2] = 48;
      idxs[3] = 49;
      idxs[4] = 50;
      idxs[5] = 51;
      break;
    case 11:
      idxs[0] = 3;
      idxs[1] = 7;
      idxs[2] = 52;
      idxs[3] = 53;
      idxs[4] = 54;
      idxs[5] = 55;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 7;
      idxs[3] = 4;
      idxs[4] = 20;
      idxs[5] = 21;
      idxs[6] = 22;
      idxs[7] = 23;
      idxs[8] = 52;
      idxs[9] = 53;
      idxs[10] = 54;
      idxs[11] = 55;
      idxs[12] = 36;
      idxs[13] = 37;
      idxs[14] = 38;
      idxs[15] = 39;
      idxs[16] = 40;
      idxs[17] = 41;
      idxs[18] = 42;
      idxs[19] = 43;
      idxs[20] = 56;
      idxs[21] = 57;
      idxs[22] = 58;
      idxs[23] = 59;
      idxs[24] = 60;
      idxs[25] = 61;
      idxs[26] = 62;
      idxs[27] = 63;
      idxs[28] = 64;
      idxs[29] = 65;
      idxs[30] = 66;
      idxs[31] = 67;
      idxs[32] = 68;
      idxs[33] = 69;
      idxs[34] = 70;
      idxs[35] = 71;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 5;
      idxs[4] = 12;
      idxs[5] = 13;
      idxs[6] = 14;
      idxs[7] = 15;
      idxs[8] = 48;
      idxs[9] = 49;
      idxs[10] = 50;
      idxs[11] = 51;
      idxs[12] = 28;
      idxs[13] = 29;
      idxs[14] = 30;
      idxs[15] = 31;
      idxs[16] = 44;
      idxs[17] = 45;
      idxs[18] = 46;
      idxs[19] = 47;
      idxs[20] = 72;
      idxs[21] = 73;
      idxs[22] = 74;
      idxs[23] = 75;
      idxs[24] = 76;
      idxs[25] = 77;
      idxs[26] = 78;
      idxs[27] = 79;
      idxs[28] = 80;
      idxs[29] = 81;
      idxs[30] = 82;
      idxs[31] = 83;
      idxs[32] = 84;
      idxs[33] = 85;
      idxs[34] = 86;
      idxs[35] = 87;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 5;
      idxs[3] = 4;
      idxs[4] = 8;
      idxs[5] = 9;
      idxs[6] = 10;
      idxs[7] = 11;
      idxs[8] = 44;
      idxs[9] = 45;
      idxs[10] = 46;
      idxs[11] = 47;
      idxs[12] = 24;
      idxs[13] = 25;
      idxs[14] = 26;
      idxs[15] = 27;
      idxs[16] = 40;
      idxs[17] = 41;
      idxs[18] = 42;
      idxs[19] = 43;
      idxs[20] = 88;
      idxs[21] = 89;
      idxs[22] = 90;
      idxs[23] = 91;
      idxs[24] = 92;
      idxs[25] = 93;
      idxs[26] = 94;
      idxs[27] = 95;
      idxs[28] = 96;
      idxs[29] = 97;
      idxs[30] = 98;
      idxs[31] = 99;
      idxs[32] = 100;
      idxs[33] = 101;
      idxs[34] = 102;
      idxs[35] = 103;
      break;
    case 3:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 7;
      idxs[4] = 16;
      idxs[5] = 17;
      idxs[6] = 18;
      idxs[7] = 19;
      idxs[8] = 48;
      idxs[9] = 49;
      idxs[10] = 50;
      idxs[11] = 51;
      idxs[12] = 32;
      idxs[13] = 33;
      idxs[14] = 34;
      idxs[15] = 35;
      idxs[16] = 52;
      idxs[17] = 53;
      idxs[18] = 54;
      idxs[19] = 55;
      idxs[20] = 104;
      idxs[21] = 105;
      idxs[22] = 106;
      idxs[23] = 107;
      idxs[24] = 108;
      idxs[25] = 109;
      idxs[26] = 110;
      idxs[27] = 111;
      idxs[28] = 112;
      idxs[29] = 113;
      idxs[30] = 114;
      idxs[31] = 115;
      idxs[32] = 116;
      idxs[33] = 117;
      idxs[34] = 118;
      idxs[35] = 119;
      break;
    case 4:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 2;
      idxs[3] = 3;
      idxs[4] = 8;
      idxs[5] = 9;
      idxs[6] = 10;
      idxs[7] = 11;
      idxs[8] = 12;
      idxs[9] = 13;
      idxs[10] = 14;
      idxs[11] = 15;
      idxs[12] = 16;
      idxs[13] = 17;
      idxs[14] = 18;
      idxs[15] = 19;
      idxs[16] = 20;
      idxs[17] = 21;
      idxs[18] = 22;
      idxs[19] = 23;
      idxs[20] = 120;
      idxs[21] = 121;
      idxs[22] = 122;
      idxs[23] = 123;
      idxs[24] = 124;
      idxs[25] = 125;
      idxs[26] = 126;
      idxs[27] = 127;
      idxs[28] = 128;
      idxs[29] = 129;
      idxs[30] = 130;
      idxs[31] = 131;
      idxs[32] = 132;
      idxs[33] = 133;
      idxs[34] = 134;
      idxs[35] = 135;
      break;
    case 5:
      idxs[0] = 4;
      idxs[1] = 5;
      idxs[2] = 6;
      idxs[3] = 7;
      idxs[4] = 24;
      idxs[5] = 25;
      idxs[6] = 26;
      idxs[7] = 27;
      idxs[8] = 28;
      idxs[9] = 29;
      idxs[10] = 30;
      idxs[11] = 31;
      idxs[12] = 32;
      idxs[13] = 33;
      idxs[14] = 34;
      idxs[15] = 35;
      idxs[16] = 36;
      idxs[17] = 37;
      idxs[18] = 38;
      idxs[19] = 39;
      idxs[20] = 136;
      idxs[21] = 137;
      idxs[22] = 138;
      idxs[23] = 139;
      idxs[24] = 140;
      idxs[25] = 141;
      idxs[26] = 142;
      idxs[27] = 143;
      idxs[28] = 144;
      idxs[29] = 145;
      idxs[30] = 146;
      idxs[31] = 147;
      idxs[32] = 148;
      idxs[33] = 149;
      idxs[34] = 150;
      idxs[35] = 151;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTetra>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElCube>) {
    switch (subelement) {
    default:
      break;
    }
  }
};

} // namespace numeric::math

#endif
