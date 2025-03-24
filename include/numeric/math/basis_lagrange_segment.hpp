#ifndef NUMERIC_MATH_BASIS_LAGRANGE_SEGMENT_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_SEGMENT_HPP_

#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math {

template <> struct BasisLagrange<mesh::RefElSegment, 1> {
  using ref_el_t = mesh::RefElSegment;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 2;
  static constexpr dim_t num_interpolation_nodes = 2;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    return -coeffs[0] * (x[0] - 1) + coeffs[1] * x[0];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    out[0] = 1 - x[0];
    out[1] = x[0];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    out[0] = -coeffs[0] + coeffs[1];
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[1]) {
    out[0][0] = -1;
    out[1][0] = 1;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[1];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[1]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(1) / order;
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
      break;
    case 1:
      out[0] = 1;
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
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
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

template <> struct BasisLagrange<mesh::RefElSegment, 2> {
  using ref_el_t = mesh::RefElSegment;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 3;
  static constexpr dim_t num_interpolation_nodes = 3;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 2 * x[0];
    const Scalar x2 = (1.0 / 2.0) * x1 - 1.0 / 2.0;
    return 2 * coeffs[0] * x0 * x2 + 2 * coeffs[1] * x2 * x[0] -
           2 * coeffs[2] * x0 * x1;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 2 * x[0] - 1;
    out[0] = x0 * x1;
    out[1] = x1 * x[0];
    out[2] = -4 * x0 * x[0];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 4 * x[0];
    out[0] = coeffs[0] * (x0 - 3) + coeffs[1] * (x0 - 1) -
             4 * coeffs[2] * (2 * x[0] - 1);
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[1]) {
    const Scalar x0 = 4 * x[0];
    out[0][0] = x0 - 3;
    out[1][0] = x0 - 1;
    out[2][0] = 4 * (1 - 2 * x[0]);
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[1];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[1]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(2) / order;
    out[2][0] = static_cast<Scalar>(1) / order;
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
      break;
    case 1:
      out[0] = 2;
      break;
    case 2:
      out[0] = 1;
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
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
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

template <> struct BasisLagrange<mesh::RefElSegment, 3> {
  using ref_el_t = mesh::RefElSegment;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 4;
  static constexpr dim_t num_interpolation_nodes = 4;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 3 * x[0];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = x[0] - 1;
    const Scalar x3 = x0 - 2;
    return -1.0 / 2.0 * coeffs[0] * x1 * x2 * x3 +
           (1.0 / 2.0) * coeffs[1] * x1 * x3 * x[0] +
           (9.0 / 2.0) * coeffs[2] * x2 * x3 * x[0] -
           9.0 / 2.0 * coeffs[3] * x1 * x2 * x[0];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 3 * x[0];
    const Scalar x2 = x1 - 2;
    const Scalar x3 = x0 * x2;
    const Scalar x4 = x1 - 1;
    const Scalar x5 = (1.0 / 2.0) * x4;
    const Scalar x6 = (9.0 / 2.0) * x[0];
    out[0] = -x3 * x5;
    out[1] = x2 * x5 * x[0];
    out[2] = x3 * x6;
    out[3] = -x0 * x4 * x6;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 3 * x[0];
    const Scalar x2 = x0 * x1;
    const Scalar x3 = x1 - 1;
    const Scalar x4 = x0 * x3;
    const Scalar x5 = x1 - 2;
    const Scalar x6 = x0 * x5;
    const Scalar x7 = x3 * x5;
    out[0] = -1.0 / 2.0 * coeffs[0] * (3 * x4 + 3 * x6 + x7) +
             (1.0 / 2.0) * coeffs[1] * (x1 * x3 + x1 * x5 + x7) +
             (9.0 / 2.0) * coeffs[2] * (x2 + x5 * x[0] + x6) -
             9.0 / 2.0 * coeffs[3] * (x2 + x3 * x[0] + x4);
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[1]) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 3 * x[0];
    const Scalar x2 = x1 - 2;
    const Scalar x3 = x0 * x2;
    const Scalar x4 = x1 - 1;
    const Scalar x5 = x0 * x4;
    const Scalar x6 = x2 * x4;
    const Scalar x7 = x0 * x1;
    out[0][0] = -3.0 / 2.0 * x3 - 3.0 / 2.0 * x5 - 1.0 / 2.0 * x6;
    out[1][0] =
        (1.0 / 2.0) * x1 * x2 + (1.0 / 2.0) * x1 * x4 + (1.0 / 2.0) * x6;
    out[2][0] = (9.0 / 2.0) * x2 * x[0] + (9.0 / 2.0) * x3 + (9.0 / 2.0) * x7;
    out[3][0] = -9.0 / 2.0 * x4 * x[0] - 9.0 / 2.0 * x5 - 9.0 / 2.0 * x7;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[1];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[1]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(3) / order;
    out[2][0] = static_cast<Scalar>(1) / order;
    out[3][0] = static_cast<Scalar>(2) / order;
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
      break;
    case 1:
      out[0] = 3;
      break;
    case 2:
      out[0] = 1;
      break;
    case 3:
      out[0] = 2;
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
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
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

template <> struct BasisLagrange<mesh::RefElSegment, 4> {
  using ref_el_t = mesh::RefElSegment;
  static constexpr dim_t order = 4;
  static constexpr dim_t num_basis_functions = 5;
  static constexpr dim_t num_interpolation_nodes = 5;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 4 * x[0];
    const Scalar x2 = x1 - 3;
    const Scalar x3 = x0 * x2 * x[0];
    const Scalar x4 = 2 * x[0] - 1;
    const Scalar x5 = 16 * x4;
    const Scalar x6 = x1 - 1;
    const Scalar x7 = x2 * x4 * x6;
    return (1.0 / 3.0) * coeffs[0] * x0 * x7 +
           (1.0 / 3.0) * coeffs[1] * x7 * x[0] -
           1.0 / 3.0 * coeffs[2] * x3 * x5 + 4 * coeffs[3] * x3 * x6 -
           1.0 / 3.0 * coeffs[4] * x0 * x5 * x6 * x[0];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 2 * x[0] - 1;
    const Scalar x2 = 4 * x[0];
    const Scalar x3 = x2 - 3;
    const Scalar x4 = x0 * x1 * x3;
    const Scalar x5 = x2 - 1;
    const Scalar x6 = (1.0 / 3.0) * x5;
    const Scalar x7 = x1 * x[0];
    const Scalar x8 = x0 * x5;
    out[0] = x4 * x6;
    out[1] = x3 * x6 * x7;
    out[2] = -16.0 / 3.0 * x4 * x[0];
    out[3] = x2 * x3 * x8;
    out[4] = -16.0 / 3.0 * x7 * x8;
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
    const Scalar x9 = x2 * x8;
    const Scalar x10 = x3 - 1;
    const Scalar x11 = x10 * x6;
    const Scalar x12 = x11 * x2;
    const Scalar x13 = x1 * x10;
    const Scalar x14 = x13 * x2;
    const Scalar x15 = x10 * x8;
    out[0] = (1.0 / 3.0) * coeffs[0] * (2 * x12 + 4 * x14 + x15 + 4 * x9) +
             (1.0 / 3.0) * coeffs[1] * (x0 * x11 + x13 * x3 + x15 + x3 * x8) -
             16.0 / 3.0 * coeffs[2] * (x5 + x6 * x7 + x8 * x[0] + x9) +
             4 * coeffs[3] * (x10 * x4 + x11 * x[0] + x12 + x4 * x6) -
             16.0 / 3.0 * coeffs[4] * (x10 * x7 + x13 * x[0] + x14 + x5);
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[1]) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 2 * x[0];
    const Scalar x2 = x1 - 1;
    const Scalar x3 = 4 * x[0];
    const Scalar x4 = x3 - 3;
    const Scalar x5 = x2 * x4;
    const Scalar x6 = x0 * x5;
    const Scalar x7 = x3 - 1;
    const Scalar x8 = x2 * x7;
    const Scalar x9 = x0 * x8;
    const Scalar x10 = x4 * x7;
    const Scalar x11 = x0 * x10;
    const Scalar x12 = x5 * x7;
    const Scalar x13 = x0 * x3;
    const Scalar x14 = x13 * x2;
    const Scalar x15 = x0 * x4;
    out[0][0] = (2.0 / 3.0) * x11 + (1.0 / 3.0) * x12 + (4.0 / 3.0) * x6 +
                (4.0 / 3.0) * x9;
    out[1][0] = (1.0 / 3.0) * x1 * x10 + (1.0 / 3.0) * x12 +
                (1.0 / 3.0) * x3 * x5 + (1.0 / 3.0) * x3 * x8;
    out[2][0] = -16.0 / 3.0 * x1 * x15 - 16.0 / 3.0 * x14 -
                16.0 / 3.0 * x5 * x[0] - 16.0 / 3.0 * x6;
    out[3][0] = 4 * x10 * x[0] + 4 * x11 + 4 * x13 * x7 + 4 * x15 * x3;
    out[4][0] = -16.0 / 3.0 * x0 * x1 * x7 - 16.0 / 3.0 * x14 -
                16.0 / 3.0 * x8 * x[0] - 16.0 / 3.0 * x9;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[1];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[1]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(4) / order;
    out[2][0] = static_cast<Scalar>(1) / order;
    out[3][0] = static_cast<Scalar>(2) / order;
    out[4][0] = static_cast<Scalar>(3) / order;
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
      break;
    case 1:
      out[0] = 4;
      break;
    case 2:
      out[0] = 1;
      break;
    case 3:
      out[0] = 2;
      break;
    case 4:
      out[0] = 3;
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
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
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

template <> struct BasisLagrange<mesh::RefElSegment, 5> {
  using ref_el_t = mesh::RefElSegment;
  static constexpr dim_t order = 5;
  static constexpr dim_t num_basis_functions = 6;
  static constexpr dim_t num_interpolation_nodes = 6;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 5 * x[0];
    const Scalar x2 = x1 - 4;
    const Scalar x3 = x1 - 3;
    const Scalar x4 = x1 - 2;
    const Scalar x5 = x1 - 1;
    const Scalar x6 = x0 * x3 * x5 * x[0];
    return -1.0 / 24.0 * coeffs[0] * x0 * x2 * x3 * x4 * x5 +
           (1.0 / 24.0) * coeffs[1] * x2 * x3 * x4 * x5 * x[0] +
           (25.0 / 24.0) * coeffs[2] * x0 * x2 * x3 * x4 * x[0] -
           25.0 / 12.0 * coeffs[3] * x2 * x6 +
           (25.0 / 12.0) * coeffs[4] * x0 * x2 * x4 * x5 * x[0] -
           25.0 / 24.0 * coeffs[5] * x4 * x6;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 5 * x[0];
    const Scalar x2 = x1 - 4;
    const Scalar x3 = x1 - 3;
    const Scalar x4 = x1 - 2;
    const Scalar x5 = x0 * x2 * x3 * x4;
    const Scalar x6 = x1 - 1;
    const Scalar x7 = (1.0 / 24.0) * x6;
    const Scalar x8 = x2 * x3 * x[0];
    const Scalar x9 = (25.0 / 24.0) * x[0];
    const Scalar x10 = x0 * x6;
    const Scalar x11 = (25.0 / 12.0) * x10;
    out[0] = -x5 * x7;
    out[1] = x4 * x7 * x8;
    out[2] = x5 * x9;
    out[3] = -x11 * x8;
    out[4] = x11 * x2 * x4 * x[0];
    out[5] = -x10 * x3 * x4 * x9;
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
    const Scalar x13 = x12 * x3;
    const Scalar x14 = x0 - 1;
    const Scalar x15 = x1 * x14;
    const Scalar x16 = x15 * x4;
    const Scalar x17 = x14 * x5;
    const Scalar x18 = x15 * x2;
    const Scalar x19 = x18 * x3;
    const Scalar x20 = x14 * x8;
    const Scalar x21 = x15 * x7;
    const Scalar x22 = x21 * x3;
    const Scalar x23 = x10 * x14;
    const Scalar x24 = x23 * x3;
    const Scalar x25 = x10 * x15;
    out[0] = -1.0 / 24.0 * coeffs[0] *
                 (5 * x13 + 5 * x19 + 5 * x22 + 5 * x24 + x25) +
             (1.0 / 24.0) * coeffs[1] *
                 (x0 * x12 + x0 * x18 + x0 * x21 + x0 * x23 + x25) +
             (25.0 / 24.0) * coeffs[2] * (x11 + x12 * x[0] + x13 + x6 + x9) -
             25.0 / 12.0 * coeffs[3] * (x16 + x17 + x18 * x[0] + x19 + x6) +
             (25.0 / 12.0) * coeffs[4] * (x16 + x20 + x21 * x[0] + x22 + x9) -
             25.0 / 24.0 * coeffs[5] * (x11 + x17 + x20 + x23 * x[0] + x24);
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[1]) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 5 * x[0];
    const Scalar x2 = x1 - 2;
    const Scalar x3 = x1 - 4;
    const Scalar x4 = x1 - 3;
    const Scalar x5 = x3 * x4;
    const Scalar x6 = x2 * x5;
    const Scalar x7 = x0 * x6;
    const Scalar x8 = x1 - 1;
    const Scalar x9 = x5 * x8;
    const Scalar x10 = x0 * x9;
    const Scalar x11 = x2 * x8;
    const Scalar x12 = x11 * x3;
    const Scalar x13 = x0 * x12;
    const Scalar x14 = x11 * x4;
    const Scalar x15 = x0 * x14;
    const Scalar x16 = x11 * x5;
    const Scalar x17 = x0 * x1;
    const Scalar x18 = x17 * x5;
    const Scalar x19 = x17 * x2;
    const Scalar x20 = x19 * x3;
    const Scalar x21 = x19 * x4;
    const Scalar x22 = x17 * x8;
    const Scalar x23 = x22 * x3;
    const Scalar x24 = x22 * x4;
    const Scalar x25 = x11 * x17;
    out[0][0] = -5.0 / 24.0 * x10 - 5.0 / 24.0 * x13 - 5.0 / 24.0 * x15 -
                1.0 / 24.0 * x16 - 5.0 / 24.0 * x7;
    out[1][0] = (1.0 / 24.0) * x1 * x12 + (1.0 / 24.0) * x1 * x14 +
                (1.0 / 24.0) * x1 * x6 + (1.0 / 24.0) * x1 * x9 +
                (1.0 / 24.0) * x16;
    out[2][0] = (25.0 / 24.0) * x18 + (25.0 / 24.0) * x20 +
                (25.0 / 24.0) * x21 + (25.0 / 24.0) * x6 * x[0] +
                (25.0 / 24.0) * x7;
    out[3][0] = -25.0 / 12.0 * x10 - 25.0 / 12.0 * x18 - 25.0 / 12.0 * x23 -
                25.0 / 12.0 * x24 - 25.0 / 12.0 * x9 * x[0];
    out[4][0] = (25.0 / 12.0) * x12 * x[0] + (25.0 / 12.0) * x13 +
                (25.0 / 12.0) * x20 + (25.0 / 12.0) * x23 + (25.0 / 12.0) * x25;
    out[5][0] = -25.0 / 24.0 * x14 * x[0] - 25.0 / 24.0 * x15 -
                25.0 / 24.0 * x21 - 25.0 / 24.0 * x24 - 25.0 / 24.0 * x25;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[1];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[1]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(5) / order;
    out[2][0] = static_cast<Scalar>(1) / order;
    out[3][0] = static_cast<Scalar>(2) / order;
    out[4][0] = static_cast<Scalar>(3) / order;
    out[5][0] = static_cast<Scalar>(4) / order;
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
      break;
    case 1:
      out[0] = 5;
      break;
    case 2:
      out[0] = 1;
      break;
    case 3:
      out[0] = 2;
      break;
    case 4:
      out[0] = 3;
      break;
    case 5:
      out[0] = 4;
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
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
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
