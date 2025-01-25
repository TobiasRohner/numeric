#ifndef NUMERIC_MATH_BASIS_LAGRANGE_TETRA_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_TETRA_HPP_

#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math {

template <> struct BasisLagrange<mesh::RefElTetra, 1> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 4;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    return -coeffs[0] * (x[0] + x[1] + x[2] - 1) + coeffs[1] * x[0] +
           coeffs[2] * x[1] + coeffs[3] * x[2];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    out[0] = -coeffs[0] + coeffs[1];
    out[1] = -coeffs[0] + coeffs[2];
    out[2] = -coeffs[0] + coeffs[3];
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
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 0;
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
      idxs[0] = 0;
      idxs[1] = 2;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      break;
    case 4:
      idxs[0] = 2;
      idxs[1] = 3;
      break;
    case 5:
      idxs[0] = 3;
      idxs[1] = 1;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      break;
    case 1:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 3;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
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

template <> struct BasisLagrange<mesh::RefElTetra, 2> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 10;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 2 * x[0];
    const Scalar x1 = 2 * x[1];
    const Scalar x2 = x0 - 1;
    const Scalar x3 = 2 * x[2];
    const Scalar x4 = x[0] + x[1] + x[2] - 1;
    return coeffs[0] * x4 * (x1 + x2 + x3) + coeffs[1] * x2 * x[0] +
           coeffs[2] * x[1] * (x1 - 1) + coeffs[3] * x[2] * (x3 - 1) -
           2 * coeffs[4] * x0 * x4 - 2 * coeffs[5] * x1 * x4 -
           2 * coeffs[6] * x3 * x4 + 2 * coeffs[7] * x0 * x[1] +
           2 * coeffs[8] * x1 * x[2] + 2 * coeffs[9] * x0 * x[2];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x[2] - 1;
    const Scalar x2 = 4 * x[1];
    const Scalar x3 = -coeffs[5] * x2;
    const Scalar x4 = 4 * x[2];
    const Scalar x5 = coeffs[0] * (x0 + x2 + x4 - 3);
    const Scalar x6 = -coeffs[6] * x4 + x5;
    const Scalar x7 = -coeffs[4] * x0;
    out[0] = coeffs[1] * (x0 - 1) - 4 * coeffs[4] * (x1 + 2 * x[0] + x[1]) +
             coeffs[7] * x2 + coeffs[9] * x4 + x3 + x6;
    out[1] = coeffs[2] * (x2 - 1) - 4 * coeffs[5] * (x1 + x[0] + 2 * x[1]) +
             coeffs[7] * x0 + coeffs[8] * x4 + x6 + x7;
    out[2] = coeffs[3] * (x4 - 1) -
             4 * coeffs[6] * (x[0] + x[1] + 2 * x[2] - 1) + coeffs[8] * x2 +
             coeffs[9] * x0 + x3 + x5 + x7;
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
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 0;
      out[2] = 2;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 5:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 6:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 7:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 8:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 9:
      out[0] = 1;
      out[1] = 0;
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
      idxs[2] = 4;
      break;
    case 1:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 5;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 6;
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 7;
      break;
    case 4:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 8;
      break;
    case 5:
      idxs[0] = 3;
      idxs[1] = 1;
      idxs[2] = 9;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      idxs[3] = 5;
      idxs[4] = 7;
      idxs[5] = 4;
      break;
    case 1:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      idxs[3] = 6;
      idxs[4] = 8;
      idxs[5] = 5;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      idxs[3] = 4;
      idxs[4] = 9;
      idxs[5] = 6;
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 3;
      idxs[3] = 7;
      idxs[4] = 8;
      idxs[5] = 9;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
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

template <> struct BasisLagrange<mesh::RefElTetra, 3> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 20;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 9 * x[1];
    const Scalar x1 = x0 * x[2];
    const Scalar x2 = 3 * x[0];
    const Scalar x3 = x2 - 1;
    const Scalar x4 = (3.0 / 2.0) * x3;
    const Scalar x5 = x4 * x[0];
    const Scalar x6 = 3 * x[1];
    const Scalar x7 = x[1] * (x6 - 1);
    const Scalar x8 = (3.0 / 2.0) * x7;
    const Scalar x9 = 3 * x[2];
    const Scalar x10 = x[2] * (x9 - 1);
    const Scalar x11 = (3.0 / 2.0) * x10;
    const Scalar x12 = x[0] + x[1] + x[2] - 1;
    const Scalar x13 = x12 * x[0];
    const Scalar x14 = x2 - 2;
    const Scalar x15 = (3.0 / 2.0) * x12;
    const Scalar x16 = x6 + x9;
    const Scalar x17 = x14 + x16;
    const Scalar x18 = x15 * x17;
    return -1.0 / 2.0 * coeffs[0] * x12 * x17 * (x16 + x3) +
           3 * coeffs[10] * x5 * x[1] + 3 * coeffs[11] * x8 * x[0] +
           3 * coeffs[12] * x8 * x[2] + 3 * coeffs[13] * x11 * x[1] +
           3 * coeffs[14] * x11 * x[0] + 3 * coeffs[15] * x5 * x[2] -
           3 * coeffs[16] * x0 * x13 - 3 * coeffs[17] * x1 * x12 -
           27 * coeffs[18] * x13 * x[2] + 3 * coeffs[19] * x1 * x[0] +
           (1.0 / 2.0) * coeffs[1] * x14 * x3 * x[0] +
           (1.0 / 2.0) * coeffs[2] * x7 * (x6 - 2) +
           (1.0 / 2.0) * coeffs[3] * x10 * (x9 - 2) +
           (9.0 / 2.0) * coeffs[4] * x13 * x17 - 3 * coeffs[5] * x13 * x4 +
           3 * coeffs[6] * x18 * x[1] - 3 * coeffs[7] * x15 * x7 +
           3 * coeffs[8] * x18 * x[2] - 3 * coeffs[9] * x10 * x15;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 3 * x[0];
    const Scalar x1 = x0 - 2;
    const Scalar x2 = x0 - 1;
    const Scalar x3 = 3 * x[1];
    const Scalar x4 = 3 * x[2];
    const Scalar x5 = x3 + x4;
    const Scalar x6 = -x1 - x5;
    const Scalar x7 = x[1] + x[2] - 1;
    const Scalar x8 = x7 + x[0];
    const Scalar x9 = -x8;
    const Scalar x10 = x6 * x9;
    const Scalar x11 = -x10;
    const Scalar x12 = (3.0 / 2.0) * coeffs[4];
    const Scalar x13 = x2 * x[0];
    const Scalar x14 = (3.0 / 2.0) * coeffs[5];
    const Scalar x15 = x7 + 2 * x[0];
    const Scalar x16 = 9 * x[1];
    const Scalar x17 = coeffs[16] * x16;
    const Scalar x18 = 9 * x[2];
    const Scalar x19 = coeffs[18] * x18;
    const Scalar x20 = 6 * x[0];
    const Scalar x21 = x20 - 1;
    const Scalar x22 = x3 - 1;
    const Scalar x23 = x4 - 1;
    const Scalar x24 = -x2 - x5;
    const Scalar x25 =
        (1.0 / 6.0) * coeffs[0] * (3 * x10 + x24 * x6 + 3 * x24 * x9);
    const Scalar x26 = 6 * x[1];
    const Scalar x27 = 6 * x[2];
    const Scalar x28 = -x20 - x26 - x27 + 5;
    const Scalar x29 = (3.0 / 2.0) * coeffs[8];
    const Scalar x30 = x23 * x[2];
    const Scalar x31 = (3.0 / 2.0) * coeffs[9];
    const Scalar x32 = x25 + x28 * x29 * x[2] + x30 * x31;
    const Scalar x33 = (3.0 / 2.0) * coeffs[6];
    const Scalar x34 = x22 * x[1];
    const Scalar x35 = (3.0 / 2.0) * coeffs[7];
    const Scalar x36 = x28 * x33 * x[1] + x34 * x35;
    const Scalar x37 = x3 - 2;
    const Scalar x38 = x[0] - 1;
    const Scalar x39 = x38 + 2 * x[1] + x[2];
    const Scalar x40 = 9 * x[0];
    const Scalar x41 = x26 - 1;
    const Scalar x42 = x12 * x28 * x[0] + x13 * x14;
    const Scalar x43 = x4 - 2;
    const Scalar x44 = x38 + x[1] + 2 * x[2];
    const Scalar x45 = x27 - 1;
    out[0] = (9.0 / 2.0) * coeffs[10] * x21 * x[1] +
             (9.0 / 2.0) * coeffs[11] * x22 * x[1] +
             (9.0 / 2.0) * coeffs[14] * x23 * x[2] +
             (9.0 / 2.0) * coeffs[15] * x21 * x[2] -
             3 * coeffs[17] * x16 * x[2] + 27 * coeffs[19] * x[1] * x[2] +
             (1.0 / 2.0) * coeffs[1] * (x0 * x1 + x0 * x2 + x1 * x2) -
             3 * x12 * (x0 * x9 + x11 + x6 * x[0]) -
             3 * x14 * (x0 * x8 + x13 + x2 * x8) - 3 * x15 * x17 -
             3 * x15 * x19 - 3 * x32 - 3 * x36;
    out[1] = (9.0 / 2.0) * coeffs[10] * x2 * x[0] +
             (9.0 / 2.0) * coeffs[11] * x41 * x[0] +
             (9.0 / 2.0) * coeffs[12] * x41 * x[2] +
             (9.0 / 2.0) * coeffs[13] * x23 * x[2] -
             3 * coeffs[16] * x39 * x40 - 3 * coeffs[17] * x18 * x39 +
             27 * coeffs[19] * x[0] * x[2] +
             (1.0 / 2.0) * coeffs[2] * (x22 * x3 + x22 * x37 + x3 * x37) -
             3 * x19 * x[0] - 3 * x32 - 3 * x33 * (x11 + x3 * x9 + x6 * x[1]) -
             3 * x35 * (x22 * x8 + x3 * x8 + x34) - 3 * x42;
    out[2] = (9.0 / 2.0) * coeffs[12] * x22 * x[1] +
             (9.0 / 2.0) * coeffs[13] * x45 * x[1] +
             (9.0 / 2.0) * coeffs[14] * x45 * x[0] +
             (9.0 / 2.0) * coeffs[15] * x2 * x[0] - 3 * coeffs[17] * x16 * x44 -
             3 * coeffs[18] * x40 * x44 + 27 * coeffs[19] * x[0] * x[1] +
             (1.0 / 2.0) * coeffs[3] * (x23 * x4 + x23 * x43 + x4 * x43) -
             3 * x17 * x[0] - 3 * x25 - 3 * x29 * (x11 + x4 * x9 + x6 * x[2]) -
             3 * x31 * (x23 * x8 + x30 + x4 * x8) - 3 * x36 - 3 * x42;
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
      out[0] = 0;
      out[1] = 3;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 0;
      out[2] = 3;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 6:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 7:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 8:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 9:
      out[0] = 0;
      out[1] = 0;
      out[2] = 2;
      break;
    case 10:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 11:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 12:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 13:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 14:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 15:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 16:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 17:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 18:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 19:
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
      idxs[2] = 4;
      idxs[3] = 5;
      break;
    case 1:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 7;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 8;
      idxs[3] = 9;
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 10;
      idxs[3] = 11;
      break;
    case 4:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 12;
      idxs[3] = 13;
      break;
    case 5:
      idxs[0] = 3;
      idxs[1] = 1;
      idxs[2] = 14;
      idxs[3] = 15;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      idxs[3] = 6;
      idxs[4] = 7;
      idxs[5] = 11;
      idxs[6] = 10;
      idxs[7] = 5;
      idxs[8] = 4;
      idxs[9] = 16;
      break;
    case 1:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      idxs[3] = 8;
      idxs[4] = 9;
      idxs[5] = 13;
      idxs[6] = 12;
      idxs[7] = 7;
      idxs[8] = 6;
      idxs[9] = 17;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      idxs[3] = 4;
      idxs[4] = 5;
      idxs[5] = 15;
      idxs[6] = 14;
      idxs[7] = 9;
      idxs[8] = 8;
      idxs[9] = 18;
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 3;
      idxs[3] = 10;
      idxs[4] = 11;
      idxs[5] = 12;
      idxs[6] = 13;
      idxs[7] = 14;
      idxs[8] = 15;
      idxs[9] = 19;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
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

template <> struct BasisLagrange<mesh::RefElTetra, 4> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 4;
  static constexpr dim_t num_basis_functions = 35;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 4 * x[2];
    const Scalar x1 = 4 * x[0];
    const Scalar x2 = x1 - 1;
    const Scalar x3 = x2 * x[0];
    const Scalar x4 = x3 * x[1];
    const Scalar x5 = 4 * x[1];
    const Scalar x6 = x[1] * (x5 - 1);
    const Scalar x7 = x0 * x[0];
    const Scalar x8 = x0 - 1;
    const Scalar x9 = x[0] * x[1];
    const Scalar x10 = x[0] + x[1] + x[2] - 1;
    const Scalar x11 = x10 * x[2];
    const Scalar x12 = 2 * x[0];
    const Scalar x13 = x12 - 1;
    const Scalar x14 = 2 * x[1];
    const Scalar x15 = x6 * (x14 - 1);
    const Scalar x16 = (2.0 / 3.0) * x15;
    const Scalar x17 = (1.0 / 2.0) * x3;
    const Scalar x18 = x8 * x[1];
    const Scalar x19 = 2 * x[2] - 1;
    const Scalar x20 = (2.0 / 3.0) * x[2];
    const Scalar x21 = x19 * x20;
    const Scalar x22 = (1.0 / 2.0) * x8;
    const Scalar x23 = x22 * x[2];
    const Scalar x24 = x8 * x[0];
    const Scalar x25 = x13 * x3;
    const Scalar x26 = x1 * x10;
    const Scalar x27 = x0 * x10;
    const Scalar x28 = x19 * x8;
    const Scalar x29 = x1 - 3;
    const Scalar x30 = (2.0 / 3.0) * x10;
    const Scalar x31 = x0 + x29 + x5;
    const Scalar x32 = x10 * x31;
    const Scalar x33 = x32 * x[1];
    const Scalar x34 = x11 * x31;
    const Scalar x35 = x12 + x14 + x19;
    const Scalar x36 = (2.0 / 3.0) * x35;
    return (1.0 / 3.0) * coeffs[0] * x32 * x35 * (x1 + x5 + x8) -
           8 * coeffs[10] * x34 * x36 - 16.0 / 3.0 * coeffs[11] * x11 * x28 +
           8 * coeffs[12] * x22 * x34 + (16.0 / 3.0) * coeffs[13] * x13 * x4 +
           8 * coeffs[14] * x16 * x[0] + 8 * coeffs[15] * x17 * x6 +
           8 * coeffs[16] * x16 * x[2] + 8 * coeffs[17] * x18 * x21 +
           8 * coeffs[18] * x23 * x6 + 8 * coeffs[19] * x21 * x24 +
           (1.0 / 3.0) * coeffs[1] * x25 * x29 + 8 * coeffs[20] * x20 * x25 +
           8 * coeffs[21] * x23 * x3 + 8 * coeffs[22] * x1 * x33 -
           8 * coeffs[23] * x26 * x6 - 8 * coeffs[24] * x2 * x26 * x[1] +
           8 * coeffs[25] * x0 * x33 - 8 * coeffs[26] * x18 * x27 -
           8 * coeffs[27] * x27 * x6 + 8 * coeffs[28] * x32 * x7 -
           8 * coeffs[29] * x27 * x3 +
           (1.0 / 3.0) * coeffs[2] * x15 * (x5 - 3) -
           8 * coeffs[30] * x24 * x27 + 8 * coeffs[31] * x0 * x4 +
           8 * coeffs[32] * x6 * x7 + 8 * coeffs[33] * x0 * x8 * x9 -
           256 * coeffs[34] * x11 * x9 +
           (1.0 / 3.0) * coeffs[3] * x28 * x[2] * (x0 - 3) -
           8 * coeffs[4] * x32 * x36 * x[0] - 8 * coeffs[5] * x25 * x30 +
           8 * coeffs[6] * x17 * x32 - 8 * coeffs[7] * x33 * x36 -
           8 * coeffs[8] * x15 * x30 + 4 * coeffs[9] * x32 * x6;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = x0 - 3;
    const Scalar x3 = 4 * x[1];
    const Scalar x4 = 4 * x[2];
    const Scalar x5 = x3 + x4;
    const Scalar x6 = -x2 - x5;
    const Scalar x7 = x6 * x[0];
    const Scalar x8 = x[1] + x[2];
    const Scalar x9 = x8 + x[0] - 1;
    const Scalar x10 = -x9;
    const Scalar x11 = x0 * x10;
    const Scalar x12 = x10 * x6;
    const Scalar x13 = (1.0 / 2.0) * coeffs[6];
    const Scalar x14 = 2 * x[0];
    const Scalar x15 = x14 - 1;
    const Scalar x16 = x0 * x15;
    const Scalar x17 = x1 * x15;
    const Scalar x18 = x1 * x14;
    const Scalar x19 = 2 * x[2];
    const Scalar x20 = 2 * x[1];
    const Scalar x21 = -x15 - x19 - x20;
    const Scalar x22 = x12 * x21;
    const Scalar x23 = -x22;
    const Scalar x24 = (2.0 / 3.0) * coeffs[4];
    const Scalar x25 = x1 * x[0];
    const Scalar x26 = x15 * x25;
    const Scalar x27 = x0 * x9;
    const Scalar x28 = x1 * x9;
    const Scalar x29 = (2.0 / 3.0) * coeffs[5];
    const Scalar x30 = -x12;
    const Scalar x31 = x11 + x30 + x7;
    const Scalar x32 = x25 + x27 + x28;
    const Scalar x33 = x16 + x17 + x18;
    const Scalar x34 = 8 * x[0];
    const Scalar x35 = x34 - 1;
    const Scalar x36 = x3 - 1;
    const Scalar x37 = x4 - 1;
    const Scalar x38 = x15 + x8;
    const Scalar x39 = x3 * x36;
    const Scalar x40 = coeffs[25] * x3;
    const Scalar x41 = 8 * x[1];
    const Scalar x42 = 8 * x[2];
    const Scalar x43 = -x34 - x41 - x42 + 7;
    const Scalar x44 = x43 * x[2];
    const Scalar x45 = x37 * x[2];
    const Scalar x46 = coeffs[26] * x3;
    const Scalar x47 = x37 * x4;
    const Scalar x48 = 32 * coeffs[34] * x[2];
    const Scalar x49 = x20 - 1;
    const Scalar x50 = x19 - 1;
    const Scalar x51 = -x1 - x5;
    const Scalar x52 = x21 * x6;
    const Scalar x53 = 2 * x12;
    const Scalar x54 = x10 * x21;
    const Scalar x55 = 4 * x54;
    const Scalar x56 = (1.0 / 24.0) * coeffs[0] *
                       (4 * x22 + x51 * x52 + x51 * x53 + x51 * x55);
    const Scalar x57 = x52 + x53 + x55;
    const Scalar x58 = (2.0 / 3.0) * coeffs[10];
    const Scalar x59 = (1.0 / 2.0) * coeffs[12];
    const Scalar x60 = x45 * x50;
    const Scalar x61 = (2.0 / 3.0) * coeffs[11];
    const Scalar x62 = x43 * x45 * x59 + x56 + x57 * x58 * x[2] + x60 * x61;
    const Scalar x63 = (2.0 / 3.0) * coeffs[7];
    const Scalar x64 = x36 * x[1];
    const Scalar x65 = (1.0 / 2.0) * coeffs[9];
    const Scalar x66 = x49 * x64;
    const Scalar x67 = (2.0 / 3.0) * coeffs[8];
    const Scalar x68 = x43 * x64 * x65 + x57 * x63 * x[1] + x66 * x67;
    const Scalar x69 = x6 * x[1];
    const Scalar x70 = x10 * x3;
    const Scalar x71 = x3 - 3;
    const Scalar x72 = x3 * x49;
    const Scalar x73 = x36 * x49;
    const Scalar x74 = x20 * x36;
    const Scalar x75 = x3 * x9;
    const Scalar x76 = x36 * x9;
    const Scalar x77 = x30 + x69 + x70;
    const Scalar x78 = coeffs[22] * x0;
    const Scalar x79 = x64 + x75 + x76;
    const Scalar x80 = coeffs[23] * x0;
    const Scalar x81 = x72 + x73 + x74;
    const Scalar x82 = x41 - 1;
    const Scalar x83 = x49 + x[0] + x[2];
    const Scalar x84 = x0 * x1;
    const Scalar x85 = coeffs[24] * x84;
    const Scalar x86 = coeffs[28] * x0;
    const Scalar x87 = coeffs[29] * x84;
    const Scalar x88 = coeffs[30] * x0;
    const Scalar x89 = x13 * x25 * x43 + x24 * x57 * x[0] + x26 * x29;
    const Scalar x90 = x6 * x[2];
    const Scalar x91 = x10 * x4;
    const Scalar x92 = x4 - 3;
    const Scalar x93 = x4 * x50;
    const Scalar x94 = x37 * x50;
    const Scalar x95 = x19 * x37;
    const Scalar x96 = x4 * x9;
    const Scalar x97 = x37 * x9;
    const Scalar x98 = x30 + x90 + x91;
    const Scalar x99 = x45 + x96 + x97;
    const Scalar x100 = x93 + x94 + x95;
    const Scalar x101 = x42 - 1;
    const Scalar x102 = x50 + x[0] + x[1];
    out[0] =
        (16.0 / 3.0) * coeffs[13] * x33 * x[1] +
        (16.0 / 3.0) * coeffs[14] * x36 * x49 * x[1] +
        4 * coeffs[15] * x35 * x36 * x[1] +
        (16.0 / 3.0) * coeffs[19] * x37 * x50 * x[2] +
        (1.0 / 3.0) * coeffs[1] * (x0 * x17 + x16 * x2 + x17 * x2 + x18 * x2) +
        (16.0 / 3.0) * coeffs[20] * x33 * x[2] +
        4 * coeffs[21] * x35 * x37 * x[2] - 8 * coeffs[22] * x3 * x31 -
        8 * coeffs[23] * x38 * x39 - 8 * coeffs[24] * x3 * x32 -
        8 * coeffs[27] * x39 * x[2] - 8 * coeffs[28] * x31 * x4 -
        8 * coeffs[29] * x32 * x4 - 8 * coeffs[30] * x38 * x47 +
        32 * coeffs[31] * x35 * x[1] * x[2] +
        32 * coeffs[32] * x36 * x[1] * x[2] +
        32 * coeffs[33] * x37 * x[1] * x[2] -
        8 * x13 * (-x0 * x12 + x1 * x11 - x1 * x12 + x1 * x7) -
        8 * x24 * (x11 * x21 + x12 * x14 + x21 * x7 + x23) -
        8 * x29 * (x14 * x28 + x15 * x27 + x15 * x28 + x26) -
        8 * x38 * x48 * x[1] - 8 * x40 * x44 - 8 * x45 * x46 - 8 * x62 -
        8 * x68;
    out[1] = (16.0 / 3.0) * coeffs[13] * x1 * x15 * x[0] +
             (16.0 / 3.0) * coeffs[14] * x81 * x[0] +
             4 * coeffs[15] * x1 * x82 * x[0] +
             (16.0 / 3.0) * coeffs[16] * x81 * x[2] +
             (16.0 / 3.0) * coeffs[17] * x37 * x50 * x[2] +
             4 * coeffs[18] * x37 * x82 * x[2] - 8 * coeffs[25] * x4 * x77 -
             8 * coeffs[26] * x47 * x83 - 8 * coeffs[27] * x4 * x79 +
             (1.0 / 3.0) * coeffs[2] *
                 (x3 * x73 + x71 * x72 + x71 * x73 + x71 * x74) +
             32 * coeffs[31] * x1 * x[0] * x[2] +
             32 * coeffs[32] * x82 * x[0] * x[2] +
             32 * coeffs[33] * x37 * x[0] * x[2] - 8 * x44 * x86 -
             8 * x45 * x88 - 8 * x48 * x83 * x[0] - 8 * x62 -
             8 * x63 * (x12 * x20 + x23 + x3 * x54 + x52 * x[1]) -
             8 * x65 * (-x12 * x3 - x12 * x36 + x36 * x69 + x36 * x70) -
             8 * x67 * (x20 * x76 + x49 * x75 + x49 * x76 + x66) -
             8 * x77 * x78 - 8 * x79 * x80 - 8 * x83 * x85 - 8 * x87 * x[2] -
             8 * x89;
    out[2] =
        (16.0 / 3.0) * coeffs[16] * x36 * x49 * x[1] +
        (16.0 / 3.0) * coeffs[17] * x100 * x[1] +
        4 * coeffs[18] * x101 * x36 * x[1] +
        (16.0 / 3.0) * coeffs[19] * x100 * x[0] +
        (16.0 / 3.0) * coeffs[20] * x1 * x15 * x[0] +
        4 * coeffs[21] * x1 * x101 * x[0] - 8 * coeffs[27] * x102 * x39 +
        32 * coeffs[31] * x1 * x[0] * x[1] +
        32 * coeffs[32] * x36 * x[0] * x[1] +
        32 * coeffs[33] * x101 * x[0] * x[1] -
        256 * coeffs[34] * x102 * x[0] * x[1] +
        (1.0 / 3.0) * coeffs[3] *
            (x4 * x94 + x92 * x93 + x92 * x94 + x92 * x95) -
        8 * x102 * x87 - 8 * x40 * x98 - 8 * x43 * x78 * x[1] - 8 * x46 * x99 -
        8 * x56 - 8 * x58 * (x12 * x19 + x23 + x4 * x54 + x52 * x[2]) -
        8 * x59 * (-x12 * x37 - x12 * x4 + x37 * x90 + x37 * x91) -
        8 * x61 * (x19 * x97 + x50 * x96 + x50 * x97 + x60) - 8 * x64 * x80 -
        8 * x68 - 8 * x85 * x[1] - 8 * x86 * x98 - 8 * x88 * x99 - 8 * x89;
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
      out[0] = 0;
      out[1] = 4;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 0;
      out[2] = 4;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 5:
      out[0] = 3;
      out[1] = 0;
      out[2] = 0;
      break;
    case 6:
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 7:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 8:
      out[0] = 0;
      out[1] = 3;
      out[2] = 0;
      break;
    case 9:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 10:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 11:
      out[0] = 0;
      out[1] = 0;
      out[2] = 3;
      break;
    case 12:
      out[0] = 0;
      out[1] = 0;
      out[2] = 2;
      break;
    case 13:
      out[0] = 3;
      out[1] = 1;
      out[2] = 0;
      break;
    case 14:
      out[0] = 1;
      out[1] = 3;
      out[2] = 0;
      break;
    case 15:
      out[0] = 2;
      out[1] = 2;
      out[2] = 0;
      break;
    case 16:
      out[0] = 0;
      out[1] = 3;
      out[2] = 1;
      break;
    case 17:
      out[0] = 0;
      out[1] = 1;
      out[2] = 3;
      break;
    case 18:
      out[0] = 0;
      out[1] = 2;
      out[2] = 2;
      break;
    case 19:
      out[0] = 1;
      out[1] = 0;
      out[2] = 3;
      break;
    case 20:
      out[0] = 3;
      out[1] = 0;
      out[2] = 1;
      break;
    case 21:
      out[0] = 2;
      out[1] = 0;
      out[2] = 2;
      break;
    case 22:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 23:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 24:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 25:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 26:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 27:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 28:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 29:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 30:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 31:
      out[0] = 2;
      out[1] = 1;
      out[2] = 1;
      break;
    case 32:
      out[0] = 1;
      out[1] = 2;
      out[2] = 1;
      break;
    case 33:
      out[0] = 1;
      out[1] = 1;
      out[2] = 2;
      break;
    case 34:
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
      idxs[2] = 4;
      idxs[3] = 5;
      idxs[4] = 6;
      break;
    case 1:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 7;
      idxs[3] = 8;
      idxs[4] = 9;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 10;
      idxs[3] = 11;
      idxs[4] = 12;
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 13;
      idxs[3] = 14;
      idxs[4] = 15;
      break;
    case 4:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 16;
      idxs[3] = 17;
      idxs[4] = 18;
      break;
    case 5:
      idxs[0] = 3;
      idxs[1] = 1;
      idxs[2] = 19;
      idxs[3] = 20;
      idxs[4] = 21;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      idxs[3] = 7;
      idxs[4] = 8;
      idxs[5] = 9;
      idxs[6] = 14;
      idxs[7] = 13;
      idxs[8] = 15;
      idxs[9] = 5;
      idxs[10] = 4;
      idxs[11] = 6;
      idxs[12] = 22;
      idxs[13] = 23;
      idxs[14] = 24;
      break;
    case 1:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      idxs[3] = 10;
      idxs[4] = 11;
      idxs[5] = 12;
      idxs[6] = 17;
      idxs[7] = 16;
      idxs[8] = 18;
      idxs[9] = 8;
      idxs[10] = 7;
      idxs[11] = 9;
      idxs[12] = 25;
      idxs[13] = 26;
      idxs[14] = 27;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      idxs[3] = 4;
      idxs[4] = 5;
      idxs[5] = 6;
      idxs[6] = 20;
      idxs[7] = 19;
      idxs[8] = 21;
      idxs[9] = 11;
      idxs[10] = 10;
      idxs[11] = 12;
      idxs[12] = 28;
      idxs[13] = 29;
      idxs[14] = 30;
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 3;
      idxs[3] = 13;
      idxs[4] = 14;
      idxs[5] = 15;
      idxs[6] = 16;
      idxs[7] = 17;
      idxs[8] = 18;
      idxs[9] = 19;
      idxs[10] = 20;
      idxs[11] = 21;
      idxs[12] = 31;
      idxs[13] = 32;
      idxs[14] = 33;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
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

template <> struct BasisLagrange<mesh::RefElTetra, 5> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 5;
  static constexpr dim_t num_basis_functions = 56;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 5 * x[0];
    const Scalar x1 = x0 - 2;
    const Scalar x2 = x0 - 1;
    const Scalar x3 = x2 * x[0];
    const Scalar x4 = x1 * x3;
    const Scalar x5 = x[1] * x[2];
    const Scalar x6 = 5 * x[1];
    const Scalar x7 = x[1] * (x6 - 1);
    const Scalar x8 = x7 * (x6 - 2);
    const Scalar x9 = x[0] * x[2];
    const Scalar x10 = 5 * x[2];
    const Scalar x11 = x[2] * (x10 - 1);
    const Scalar x12 = x11 * (x10 - 2);
    const Scalar x13 = x[0] * x[1];
    const Scalar x14 = x7 * x[2];
    const Scalar x15 = 750 * x11;
    const Scalar x16 = x7 * x[0];
    const Scalar x17 = x[0] + x[1] + x[2] - 1;
    const Scalar x18 = x17 * x3;
    const Scalar x19 = 7500 * x5;
    const Scalar x20 = x17 * x7;
    const Scalar x21 = x0 - 3;
    const Scalar x22 = x21 * x4;
    const Scalar x23 = 25 * x[1];
    const Scalar x24 = x8 * (x6 - 3);
    const Scalar x25 = 25 * x[0];
    const Scalar x26 = 50 * x7;
    const Scalar x27 = 50 * x8;
    const Scalar x28 = 25 * x[2];
    const Scalar x29 = x12 * (x10 - 3);
    const Scalar x30 = 500 * x17;
    const Scalar x31 = x30 * x8;
    const Scalar x32 = x30 * x[1];
    const Scalar x33 = 750 * x18;
    const Scalar x34 = x0 - 4;
    const Scalar x35 = x10 + x6;
    const Scalar x36 = x17 * (x34 + x35);
    const Scalar x37 = 25 * x17;
    const Scalar x38 = 750 * x36;
    const Scalar x39 = x3 * x38;
    const Scalar x40 = x15 * x36;
    const Scalar x41 = 50 * x36;
    const Scalar x42 = x36 * (x21 + x35);
    const Scalar x43 = 500 * x42;
    const Scalar x44 = 50 * x42;
    const Scalar x45 = x42 * (x1 + x35);
    return -1.0 / 24.0 * coeffs[0] * x45 * (x2 + x35) -
           1.0 / 24.0 * coeffs[10] * x44 * x7 +
           (1.0 / 24.0) * coeffs[11] * x41 * x8 +
           (1.0 / 24.0) * coeffs[12] * x28 * x45 -
           1.0 / 24.0 * coeffs[13] * x29 * x37 -
           1.0 / 24.0 * coeffs[14] * x11 * x44 +
           (1.0 / 24.0) * coeffs[15] * x12 * x41 +
           (1.0 / 24.0) * coeffs[16] * x22 * x23 +
           (1.0 / 24.0) * coeffs[17] * x24 * x25 +
           (1.0 / 24.0) * coeffs[18] * x26 * x4 +
           (1.0 / 24.0) * coeffs[19] * x27 * x3 +
           (1.0 / 24.0) * coeffs[1] * x22 * x34 +
           (1.0 / 24.0) * coeffs[20] * x24 * x28 +
           (1.0 / 24.0) * coeffs[21] * x23 * x29 +
           (1.0 / 24.0) * coeffs[22] * x11 * x27 +
           (1.0 / 24.0) * coeffs[23] * x12 * x26 +
           (1.0 / 24.0) * coeffs[24] * x25 * x29 +
           (1.0 / 24.0) * coeffs[25] * x22 * x28 +
           (25.0 / 12.0) * coeffs[26] * x12 * x3 +
           (25.0 / 12.0) * coeffs[27] * x11 * x4 -
           1.0 / 24.0 * coeffs[28] * x13 * x43 -
           1.0 / 24.0 * coeffs[29] * x31 * x[0] +
           (1.0 / 24.0) * coeffs[2] * x24 * (x6 - 4) -
           1.0 / 24.0 * coeffs[30] * x32 * x4 +
           (1.0 / 24.0) * coeffs[31] * x16 * x38 -
           1.0 / 24.0 * coeffs[32] * x33 * x7 +
           (1.0 / 24.0) * coeffs[33] * x39 * x[1] -
           1.0 / 24.0 * coeffs[34] * x43 * x5 -
           1.0 / 24.0 * coeffs[35] * x12 * x32 -
           1.0 / 24.0 * coeffs[36] * x31 * x[2] +
           (1.0 / 24.0) * coeffs[37] * x40 * x[1] -
           1.0 / 24.0 * coeffs[38] * x15 * x20 +
           (1.0 / 24.0) * coeffs[39] * x14 * x38 +
           (1.0 / 24.0) * coeffs[3] * x29 * (x10 - 4) -
           1.0 / 24.0 * coeffs[40] * x43 * x9 -
           1.0 / 24.0 * coeffs[41] * x30 * x4 * x[2] -
           1.0 / 24.0 * coeffs[42] * x12 * x30 * x[0] +
           (1.0 / 24.0) * coeffs[43] * x39 * x[2] -
           1.0 / 24.0 * coeffs[44] * x11 * x33 +
           (1.0 / 24.0) * coeffs[45] * x40 * x[0] +
           (125.0 / 6.0) * coeffs[46] * x4 * x5 +
           (125.0 / 6.0) * coeffs[47] * x8 * x9 +
           (125.0 / 6.0) * coeffs[48] * x12 * x13 +
           (125.0 / 4.0) * coeffs[49] * x14 * x3 +
           (1.0 / 24.0) * coeffs[4] * x25 * x45 +
           (1.0 / 24.0) * coeffs[50] * x15 * x16 +
           (1.0 / 24.0) * coeffs[51] * x15 * x3 * x[1] +
           (1.0 / 24.0) * coeffs[52] * x19 * x36 * x[0] -
           1.0 / 24.0 * coeffs[53] * x18 * x19 -
           625.0 / 2.0 * coeffs[54] * x20 * x9 -
           625.0 / 2.0 * coeffs[55] * x11 * x13 * x17 -
           1.0 / 24.0 * coeffs[5] * x22 * x37 -
           1.0 / 24.0 * coeffs[6] * x3 * x44 +
           (1.0 / 24.0) * coeffs[7] * x4 * x41 +
           (1.0 / 24.0) * coeffs[8] * x23 * x45 -
           1.0 / 24.0 * coeffs[9] * x24 * x37;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 5 * x[0];
    const Scalar x1 = x0 - 4;
    const Scalar x2 = x0 - 3;
    const Scalar x3 = x0 - 2;
    const Scalar x4 = x0 * x3;
    const Scalar x5 = x2 * x4;
    const Scalar x6 = x0 - 1;
    const Scalar x7 = x0 * x6;
    const Scalar x8 = x2 * x7;
    const Scalar x9 = x3 * x6;
    const Scalar x10 = x0 * x9;
    const Scalar x11 = x2 * x9;
    const Scalar x12 = x[1] + x[2] - 1;
    const Scalar x13 = x12 + x[0];
    const Scalar x14 = x13 * x6;
    const Scalar x15 = x14 * x3;
    const Scalar x16 = x0 * x15;
    const Scalar x17 = 5 * x[1];
    const Scalar x18 = 5 * x[2];
    const Scalar x19 = x17 + x18;
    const Scalar x20 = x1 + x19;
    const Scalar x21 = x6 * x[0];
    const Scalar x22 = x21 * x3;
    const Scalar x23 = x0 * x13;
    const Scalar x24 = x23 * x3;
    const Scalar x25 = x0 * x14;
    const Scalar x26 = -x19 - x3;
    const Scalar x27 = -x19 - x2;
    const Scalar x28 = -x20;
    const Scalar x29 = x28 * x[0];
    const Scalar x30 = x27 * x29;
    const Scalar x31 = -x13;
    const Scalar x32 = x0 * x31;
    const Scalar x33 = x27 * x32;
    const Scalar x34 = x28 * x31;
    const Scalar x35 = x0 * x34;
    const Scalar x36 = x27 * x34;
    const Scalar x37 = x0 * x36;
    const Scalar x38 = x26 * x36;
    const Scalar x39 = -x38;
    const Scalar x40 = 25 * coeffs[4];
    const Scalar x41 = x2 * x22;
    const Scalar x42 = 25 * coeffs[5];
    const Scalar x43 = x29 * x6;
    const Scalar x44 = x32 * x6;
    const Scalar x45 = x34 * x6;
    const Scalar x46 = 50 * coeffs[6];
    const Scalar x47 = x10 + x11 + x5 + x8;
    const Scalar x48 = -x36;
    const Scalar x49 = x30 + x33 + x35 + x48;
    const Scalar x50 = 500 * x[1];
    const Scalar x51 = coeffs[28] * x50;
    const Scalar x52 = x15 + x22 + x24 + x25;
    const Scalar x53 = coeffs[30] * x50;
    const Scalar x54 = 500 * x[2];
    const Scalar x55 = coeffs[40] * x54;
    const Scalar x56 = coeffs[41] * x54;
    const Scalar x57 = -750 * x35 + 750 * x43 + 750 * x44 - 750 * x45;
    const Scalar x58 = coeffs[33] * x[1];
    const Scalar x59 = coeffs[43] * x[2];
    const Scalar x60 = x4 + x7 + x9;
    const Scalar x61 = x17 - 1;
    const Scalar x62 = x18 - 1;
    const Scalar x63 = x27 * x28;
    const Scalar x64 = x27 * x31;
    const Scalar x65 = 5 * x64;
    const Scalar x66 = 5 * x34;
    const Scalar x67 = x63 + x65 + x66;
    const Scalar x68 = coeffs[34] * x50;
    const Scalar x69 = -x34;
    const Scalar x70 = x29 + x32 + x69;
    const Scalar x71 = x61 * x[1];
    const Scalar x72 = 750 * x71;
    const Scalar x73 = x14 + x21 + x23;
    const Scalar x74 = coeffs[32] * x72;
    const Scalar x75 = x62 * x[2];
    const Scalar x76 = 750 * x75;
    const Scalar x77 = coeffs[44] * x76;
    const Scalar x78 = 7500 * x[2];
    const Scalar x79 = x78 * x[1];
    const Scalar x80 = 10 * x[0];
    const Scalar x81 = x80 - 1;
    const Scalar x82 = x17 - 2;
    const Scalar x83 = x18 - 2;
    const Scalar x84 = x17 - 3;
    const Scalar x85 = x18 - 3;
    const Scalar x86 = x71 * x82;
    const Scalar x87 = x12 + 2 * x[0];
    const Scalar x88 = 500 * x87;
    const Scalar x89 = x75 * x83;
    const Scalar x90 = coeffs[35] * x50;
    const Scalar x91 = coeffs[36] * x54;
    const Scalar x92 = coeffs[37] * x[1];
    const Scalar x93 = 10 * x[1];
    const Scalar x94 = 10 * x[2];
    const Scalar x95 = -x80 - x93 - x94 + 9;
    const Scalar x96 = x76 * x95;
    const Scalar x97 = coeffs[38] * x72;
    const Scalar x98 = coeffs[39] * x[2];
    const Scalar x99 = x72 * x95;
    const Scalar x100 = coeffs[54] * x71;
    const Scalar x101 = 7500 * x[1];
    const Scalar x102 = coeffs[55] * x75;
    const Scalar x103 = -x19 - x6;
    const Scalar x104 = x26 * x63;
    const Scalar x105 = x26 * x65;
    const Scalar x106 = x26 * x66;
    const Scalar x107 = 5 * x36;
    const Scalar x108 = coeffs[0] * (x103 * x104 + x103 * x105 + x103 * x106 +
                                     x103 * x107 + 5 * x38);
    const Scalar x109 = x104 + x105 + x106 + x107;
    const Scalar x110 = 25 * coeffs[12];
    const Scalar x111 = 50 * coeffs[14];
    const Scalar x112 = 25 * coeffs[13];
    const Scalar x113 = x85 * x89;
    const Scalar x114 = 50 * coeffs[15] * x89 * x95 + x108 +
                        x109 * x110 * x[2] + x111 * x67 * x75 + x112 * x113;
    const Scalar x115 = 25 * coeffs[8];
    const Scalar x116 = 50 * coeffs[10];
    const Scalar x117 = 25 * coeffs[9];
    const Scalar x118 = x84 * x86;
    const Scalar x119 = 50 * coeffs[11] * x86 * x95 + x109 * x115 * x[1] +
                        x116 * x67 * x71 + x117 * x118;
    const Scalar x120 = x17 - 4;
    const Scalar x121 = x17 * x82;
    const Scalar x122 = x121 * x84;
    const Scalar x123 = x17 * x61;
    const Scalar x124 = x123 * x84;
    const Scalar x125 = x61 * x82;
    const Scalar x126 = x125 * x17;
    const Scalar x127 = x125 * x84;
    const Scalar x128 = x13 * x61;
    const Scalar x129 = x128 * x82;
    const Scalar x130 = x129 * x17;
    const Scalar x131 = x13 * x17;
    const Scalar x132 = x131 * x82;
    const Scalar x133 = x128 * x17;
    const Scalar x134 = x17 * x64;
    const Scalar x135 = x17 * x34;
    const Scalar x136 = x17 * x36;
    const Scalar x137 = x34 * x61;
    const Scalar x138 = x122 + x124 + x126 + x127;
    const Scalar x139 = x134 + x135 + x48 + x63 * x[1];
    const Scalar x140 = 500 * x[0];
    const Scalar x141 = x129 + x132 + x133 + x86;
    const Scalar x142 = coeffs[29] * x140;
    const Scalar x143 = x28 * x[1];
    const Scalar x144 = x17 * x31;
    const Scalar x145 =
        -750 * x135 - 750 * x137 + 750 * x143 * x61 + 750 * x144 * x61;
    const Scalar x146 = coeffs[31] * x[0];
    const Scalar x147 = x121 + x123 + x125;
    const Scalar x148 = x67 * x[0];
    const Scalar x149 = x128 + x131 + x71;
    const Scalar x150 = 750 * x21;
    const Scalar x151 = x143 + x144 + x69;
    const Scalar x152 = x78 * x[0];
    const Scalar x153 = x93 - 1;
    const Scalar x154 = x[0] - 1;
    const Scalar x155 = x154 + 2 * x[1] + x[2];
    const Scalar x156 = 500 * x155;
    const Scalar x157 = coeffs[42] * x140;
    const Scalar x158 = x150 * x95;
    const Scalar x159 = coeffs[45] * x[0];
    const Scalar x160 = coeffs[53] * x21;
    const Scalar x161 = 7500 * x[0];
    const Scalar x162 = 50 * coeffs[7] * x22 * x95 + x109 * x40 * x[0] +
                        x21 * x46 * x67 + x41 * x42;
    const Scalar x163 = x18 - 4;
    const Scalar x164 = x18 * x83;
    const Scalar x165 = x164 * x85;
    const Scalar x166 = x18 * x62;
    const Scalar x167 = x166 * x85;
    const Scalar x168 = x62 * x83;
    const Scalar x169 = x168 * x18;
    const Scalar x170 = x168 * x85;
    const Scalar x171 = x13 * x62;
    const Scalar x172 = x171 * x83;
    const Scalar x173 = x172 * x18;
    const Scalar x174 = x13 * x18;
    const Scalar x175 = x174 * x83;
    const Scalar x176 = x171 * x18;
    const Scalar x177 = x18 * x64;
    const Scalar x178 = x18 * x34;
    const Scalar x179 = x18 * x36;
    const Scalar x180 = x34 * x62;
    const Scalar x181 = x165 + x167 + x169 + x170;
    const Scalar x182 = x177 + x178 + x48 + x63 * x[2];
    const Scalar x183 = x172 + x175 + x176 + x89;
    const Scalar x184 = x28 * x[2];
    const Scalar x185 = x18 * x31;
    const Scalar x186 =
        -750 * x178 - 750 * x180 + 750 * x184 * x62 + 750 * x185 * x62;
    const Scalar x187 = x164 + x166 + x168;
    const Scalar x188 = x171 + x174 + x75;
    const Scalar x189 = x184 + x185 + x69;
    const Scalar x190 = x101 * x[0];
    const Scalar x191 = x94 - 1;
    const Scalar x192 = x154 + x[1] + 2 * x[2];
    const Scalar x193 = 500 * x192;
    out[0] =
        (25.0 / 24.0) * coeffs[16] * x47 * x[1] +
        (25.0 / 24.0) * coeffs[17] * x61 * x82 * x84 * x[1] +
        (25.0 / 12.0) * coeffs[18] * x60 * x61 * x[1] +
        (25.0 / 12.0) * coeffs[19] * x61 * x81 * x82 * x[1] +
        (1.0 / 24.0) * coeffs[1] *
            (x0 * x11 + x1 * x10 + x1 * x11 + x1 * x5 + x1 * x8) +
        (25.0 / 24.0) * coeffs[24] * x62 * x83 * x85 * x[2] +
        (25.0 / 24.0) * coeffs[25] * x47 * x[2] +
        (25.0 / 12.0) * coeffs[26] * x62 * x81 * x83 * x[2] +
        (25.0 / 12.0) * coeffs[27] * x60 * x62 * x[2] -
        1.0 / 24.0 * coeffs[29] * x86 * x88 -
        1.0 / 24.0 * coeffs[31] * x70 * x72 -
        1.0 / 24.0 * coeffs[42] * x88 * x89 -
        1.0 / 24.0 * coeffs[45] * x70 * x76 +
        (125.0 / 6.0) * coeffs[46] * x60 * x[1] * x[2] +
        (125.0 / 6.0) * coeffs[47] * x61 * x82 * x[1] * x[2] +
        (125.0 / 6.0) * coeffs[48] * x62 * x83 * x[1] * x[2] +
        (125.0 / 4.0) * coeffs[49] * x61 * x81 * x[1] * x[2] +
        (125.0 / 4.0) * coeffs[50] * x61 * x62 * x[1] * x[2] +
        (125.0 / 4.0) * coeffs[51] * x62 * x81 * x[1] * x[2] -
        1.0 / 24.0 * coeffs[52] * x70 * x79 -
        1.0 / 24.0 * coeffs[53] * x73 * x79 +
        (25.0 / 12.0) * coeffs[7] *
            (x15 * x20 + x16 + x20 * x22 + x20 * x24 + x20 * x25) -
        1.0 / 24.0 * x100 * x78 * x87 - 1.0 / 24.0 * x101 * x102 * x87 -
        1.0 / 24.0 * x114 - 1.0 / 24.0 * x119 -
        1.0 / 24.0 * x40 * (x26 * x30 + x26 * x33 + x26 * x35 + x37 + x39) -
        1.0 / 24.0 * x42 * (x15 * x2 + x16 + x2 * x24 + x2 * x25 + x41) -
        1.0 / 24.0 * x46 *
            (x0 * x45 + x27 * x43 + x27 * x44 - x27 * x45 - x37) -
        1.0 / 24.0 * x49 * x51 - 1.0 / 24.0 * x49 * x55 -
        1.0 / 24.0 * x52 * x53 - 1.0 / 24.0 * x52 * x56 -
        1.0 / 24.0 * x57 * x58 - 1.0 / 24.0 * x57 * x59 -
        1.0 / 24.0 * x67 * x68 * x[2] - 1.0 / 24.0 * x73 * x74 -
        1.0 / 24.0 * x73 * x77 - 1.0 / 24.0 * x75 * x97 -
        1.0 / 24.0 * x86 * x91 - 1.0 / 24.0 * x89 * x90 -
        1.0 / 24.0 * x92 * x96 - 1.0 / 24.0 * x98 * x99;
    out[1] = (25.0 / 12.0) * coeffs[11] *
                 (x129 * x20 + x130 + x132 * x20 + x133 * x20 + x20 * x86) +
             (25.0 / 24.0) * coeffs[16] * x2 * x3 * x6 * x[0] +
             (25.0 / 24.0) * coeffs[17] * x138 * x[0] +
             (25.0 / 12.0) * coeffs[18] * x153 * x3 * x6 * x[0] +
             (25.0 / 12.0) * coeffs[19] * x147 * x6 * x[0] +
             (25.0 / 24.0) * coeffs[20] * x138 * x[2] +
             (25.0 / 24.0) * coeffs[21] * x62 * x83 * x85 * x[2] +
             (25.0 / 12.0) * coeffs[22] * x147 * x62 * x[2] +
             (25.0 / 12.0) * coeffs[23] * x153 * x62 * x83 * x[2] -
             1.0 / 24.0 * coeffs[28] * x139 * x140 +
             (1.0 / 24.0) * coeffs[2] *
                 (x120 * x122 + x120 * x124 + x120 * x126 + x120 * x127 +
                  x127 * x17) -
             1.0 / 24.0 * coeffs[30] * x156 * x22 -
             1.0 / 24.0 * coeffs[32] * x149 * x150 -
             1.0 / 24.0 * coeffs[33] * x150 * x151 -
             1.0 / 24.0 * coeffs[34] * x139 * x54 -
             1.0 / 24.0 * coeffs[35] * x156 * x89 -
             1.0 / 24.0 * coeffs[37] * x151 * x76 -
             1.0 / 24.0 * coeffs[38] * x149 * x76 +
             (125.0 / 6.0) * coeffs[46] * x3 * x6 * x[0] * x[2] +
             (125.0 / 6.0) * coeffs[47] * x147 * x[0] * x[2] +
             (125.0 / 6.0) * coeffs[48] * x62 * x83 * x[0] * x[2] +
             (125.0 / 4.0) * coeffs[49] * x153 * x6 * x[0] * x[2] +
             (125.0 / 4.0) * coeffs[50] * x153 * x62 * x[0] * x[2] +
             (125.0 / 4.0) * coeffs[51] * x6 * x62 * x[0] * x[2] -
             1.0 / 24.0 * coeffs[52] * x151 * x152 -
             1.0 / 24.0 * coeffs[54] * x149 * x152 -
             1.0 / 24.0 * x102 * x155 * x161 - 1.0 / 24.0 * x114 -
             1.0 / 24.0 * x115 *
                 (x104 * x[1] + x134 * x26 + x135 * x26 + x136 + x39) -
             1.0 / 24.0 * x116 *
                 (x123 * x64 - x136 + x137 * x17 - x36 * x61 + x63 * x71) -
             1.0 / 24.0 * x117 *
                 (x118 + x129 * x84 + x130 + x132 * x84 + x133 * x84) -
             1.0 / 24.0 * x141 * x142 - 1.0 / 24.0 * x141 * x91 -
             1.0 / 24.0 * x145 * x146 - 1.0 / 24.0 * x145 * x98 -
             1.0 / 24.0 * x148 * x55 - 1.0 / 24.0 * x155 * x160 * x78 -
             1.0 / 24.0 * x157 * x89 - 1.0 / 24.0 * x158 * x59 -
             1.0 / 24.0 * x159 * x96 - 1.0 / 24.0 * x162 -
             1.0 / 24.0 * x21 * x77 - 1.0 / 24.0 * x22 * x56;
    out[2] = (25.0 / 12.0) * coeffs[15] *
                 (x172 * x20 + x173 + x175 * x20 + x176 * x20 + x20 * x89) +
             (25.0 / 24.0) * coeffs[20] * x61 * x82 * x84 * x[1] +
             (25.0 / 24.0) * coeffs[21] * x181 * x[1] +
             (25.0 / 12.0) * coeffs[22] * x191 * x61 * x82 * x[1] +
             (25.0 / 12.0) * coeffs[23] * x187 * x61 * x[1] +
             (25.0 / 24.0) * coeffs[24] * x181 * x[0] +
             (25.0 / 24.0) * coeffs[25] * x2 * x3 * x6 * x[0] +
             (25.0 / 12.0) * coeffs[26] * x187 * x6 * x[0] +
             (25.0 / 12.0) * coeffs[27] * x191 * x3 * x6 * x[0] -
             1.0 / 24.0 * coeffs[36] * x193 * x86 -
             1.0 / 24.0 * coeffs[39] * x189 * x72 +
             (1.0 / 24.0) * coeffs[3] *
                 (x163 * x165 + x163 * x167 + x163 * x169 + x163 * x170 +
                  x170 * x18) -
             1.0 / 24.0 * coeffs[40] * x140 * x182 -
             1.0 / 24.0 * coeffs[41] * x193 * x22 -
             1.0 / 24.0 * coeffs[43] * x150 * x189 -
             1.0 / 24.0 * coeffs[44] * x150 * x188 +
             (125.0 / 6.0) * coeffs[46] * x3 * x6 * x[0] * x[1] +
             (125.0 / 6.0) * coeffs[47] * x61 * x82 * x[0] * x[1] +
             (125.0 / 6.0) * coeffs[48] * x187 * x[0] * x[1] +
             (125.0 / 4.0) * coeffs[49] * x6 * x61 * x[0] * x[1] +
             (125.0 / 4.0) * coeffs[50] * x191 * x61 * x[0] * x[1] +
             (125.0 / 4.0) * coeffs[51] * x191 * x6 * x[0] * x[1] -
             1.0 / 24.0 * coeffs[52] * x189 * x190 -
             1.0 / 24.0 * coeffs[55] * x188 * x190 -
             1.0 / 24.0 * x100 * x161 * x192 - 1.0 / 24.0 * x101 * x160 * x192 -
             1.0 / 24.0 * x108 -
             1.0 / 24.0 * x110 *
                 (x104 * x[2] + x177 * x26 + x178 * x26 + x179 + x39) -
             1.0 / 24.0 * x111 *
                 (x166 * x64 - x179 + x18 * x180 - x36 * x62 + x63 * x75) -
             1.0 / 24.0 * x112 *
                 (x113 + x172 * x85 + x173 + x175 * x85 + x176 * x85) -
             1.0 / 24.0 * x119 - 1.0 / 24.0 * x142 * x86 -
             1.0 / 24.0 * x146 * x99 - 1.0 / 24.0 * x148 * x51 -
             1.0 / 24.0 * x157 * x183 - 1.0 / 24.0 * x158 * x58 -
             1.0 / 24.0 * x159 * x186 - 1.0 / 24.0 * x162 -
             1.0 / 24.0 * x182 * x68 - 1.0 / 24.0 * x183 * x90 -
             1.0 / 24.0 * x186 * x92 - 1.0 / 24.0 * x188 * x97 -
             1.0 / 24.0 * x21 * x74 - 1.0 / 24.0 * x22 * x53;
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
      out[0] = 0;
      out[1] = 5;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 0;
      out[2] = 5;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 5:
      out[0] = 4;
      out[1] = 0;
      out[2] = 0;
      break;
    case 6:
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 7:
      out[0] = 3;
      out[1] = 0;
      out[2] = 0;
      break;
    case 8:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 9:
      out[0] = 0;
      out[1] = 4;
      out[2] = 0;
      break;
    case 10:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 11:
      out[0] = 0;
      out[1] = 3;
      out[2] = 0;
      break;
    case 12:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 13:
      out[0] = 0;
      out[1] = 0;
      out[2] = 4;
      break;
    case 14:
      out[0] = 0;
      out[1] = 0;
      out[2] = 2;
      break;
    case 15:
      out[0] = 0;
      out[1] = 0;
      out[2] = 3;
      break;
    case 16:
      out[0] = 4;
      out[1] = 1;
      out[2] = 0;
      break;
    case 17:
      out[0] = 1;
      out[1] = 4;
      out[2] = 0;
      break;
    case 18:
      out[0] = 3;
      out[1] = 2;
      out[2] = 0;
      break;
    case 19:
      out[0] = 2;
      out[1] = 3;
      out[2] = 0;
      break;
    case 20:
      out[0] = 0;
      out[1] = 4;
      out[2] = 1;
      break;
    case 21:
      out[0] = 0;
      out[1] = 1;
      out[2] = 4;
      break;
    case 22:
      out[0] = 0;
      out[1] = 3;
      out[2] = 2;
      break;
    case 23:
      out[0] = 0;
      out[1] = 2;
      out[2] = 3;
      break;
    case 24:
      out[0] = 1;
      out[1] = 0;
      out[2] = 4;
      break;
    case 25:
      out[0] = 4;
      out[1] = 0;
      out[2] = 1;
      break;
    case 26:
      out[0] = 2;
      out[1] = 0;
      out[2] = 3;
      break;
    case 27:
      out[0] = 3;
      out[1] = 0;
      out[2] = 2;
      break;
    case 28:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 29:
      out[0] = 1;
      out[1] = 3;
      out[2] = 0;
      break;
    case 30:
      out[0] = 3;
      out[1] = 1;
      out[2] = 0;
      break;
    case 31:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 32:
      out[0] = 2;
      out[1] = 2;
      out[2] = 0;
      break;
    case 33:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 34:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 35:
      out[0] = 0;
      out[1] = 1;
      out[2] = 3;
      break;
    case 36:
      out[0] = 0;
      out[1] = 3;
      out[2] = 1;
      break;
    case 37:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 38:
      out[0] = 0;
      out[1] = 2;
      out[2] = 2;
      break;
    case 39:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 40:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 41:
      out[0] = 3;
      out[1] = 0;
      out[2] = 1;
      break;
    case 42:
      out[0] = 1;
      out[1] = 0;
      out[2] = 3;
      break;
    case 43:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 44:
      out[0] = 2;
      out[1] = 0;
      out[2] = 2;
      break;
    case 45:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 46:
      out[0] = 3;
      out[1] = 1;
      out[2] = 1;
      break;
    case 47:
      out[0] = 1;
      out[1] = 3;
      out[2] = 1;
      break;
    case 48:
      out[0] = 1;
      out[1] = 1;
      out[2] = 3;
      break;
    case 49:
      out[0] = 2;
      out[1] = 2;
      out[2] = 1;
      break;
    case 50:
      out[0] = 1;
      out[1] = 2;
      out[2] = 2;
      break;
    case 51:
      out[0] = 2;
      out[1] = 1;
      out[2] = 2;
      break;
    case 52:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    case 53:
      out[0] = 2;
      out[1] = 1;
      out[2] = 1;
      break;
    case 54:
      out[0] = 1;
      out[1] = 2;
      out[2] = 1;
      break;
    case 55:
      out[0] = 1;
      out[1] = 1;
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
      idxs[2] = 4;
      idxs[3] = 5;
      idxs[4] = 6;
      idxs[5] = 7;
      break;
    case 1:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 8;
      idxs[3] = 9;
      idxs[4] = 10;
      idxs[5] = 11;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 12;
      idxs[3] = 13;
      idxs[4] = 14;
      idxs[5] = 15;
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 16;
      idxs[3] = 17;
      idxs[4] = 18;
      idxs[5] = 19;
      break;
    case 4:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 20;
      idxs[3] = 21;
      idxs[4] = 22;
      idxs[5] = 23;
      break;
    case 5:
      idxs[0] = 3;
      idxs[1] = 1;
      idxs[2] = 24;
      idxs[3] = 25;
      idxs[4] = 26;
      idxs[5] = 27;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      idxs[3] = 8;
      idxs[4] = 9;
      idxs[5] = 10;
      idxs[6] = 11;
      idxs[7] = 17;
      idxs[8] = 16;
      idxs[9] = 19;
      idxs[10] = 18;
      idxs[11] = 5;
      idxs[12] = 4;
      idxs[13] = 7;
      idxs[14] = 6;
      idxs[15] = 28;
      idxs[16] = 29;
      idxs[17] = 30;
      idxs[18] = 31;
      idxs[19] = 32;
      idxs[20] = 33;
      break;
    case 1:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      idxs[3] = 12;
      idxs[4] = 13;
      idxs[5] = 14;
      idxs[6] = 15;
      idxs[7] = 21;
      idxs[8] = 20;
      idxs[9] = 23;
      idxs[10] = 22;
      idxs[11] = 9;
      idxs[12] = 8;
      idxs[13] = 11;
      idxs[14] = 10;
      idxs[15] = 34;
      idxs[16] = 35;
      idxs[17] = 36;
      idxs[18] = 37;
      idxs[19] = 38;
      idxs[20] = 39;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      idxs[3] = 4;
      idxs[4] = 5;
      idxs[5] = 6;
      idxs[6] = 7;
      idxs[7] = 25;
      idxs[8] = 24;
      idxs[9] = 27;
      idxs[10] = 26;
      idxs[11] = 13;
      idxs[12] = 12;
      idxs[13] = 15;
      idxs[14] = 14;
      idxs[15] = 40;
      idxs[16] = 41;
      idxs[17] = 42;
      idxs[18] = 43;
      idxs[19] = 44;
      idxs[20] = 45;
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 3;
      idxs[3] = 16;
      idxs[4] = 17;
      idxs[5] = 18;
      idxs[6] = 19;
      idxs[7] = 20;
      idxs[8] = 21;
      idxs[9] = 22;
      idxs[10] = 23;
      idxs[11] = 24;
      idxs[12] = 25;
      idxs[13] = 26;
      idxs[14] = 27;
      idxs[15] = 46;
      idxs[16] = 47;
      idxs[17] = 48;
      idxs[18] = 49;
      idxs[19] = 50;
      idxs[20] = 51;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
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
