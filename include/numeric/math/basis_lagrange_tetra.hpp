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

template <dim_t Order> struct BasisLagrange<mesh::RefElTetra, Order> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 4;
  static constexpr dim_t num_interpolation_nodes = 4;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    return -coeffs[0] * (x[0] + x[1] + x[2] - 1) + coeffs[1] * x[0] +
           coeffs[2] * x[1] + coeffs[3] * x[2];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    out[0] = -x[0] - x[1] - x[2] + 1;
    out[1] = x[0];
    out[2] = x[1];
    out[3] = x[2];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    out[0] = -coeffs[0] + coeffs[1];
    out[1] = -coeffs[0] + coeffs[2];
    out[2] = -coeffs[0] + coeffs[3];
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
    out[0][0] = -1;
    out[0][1] = -1;
    out[0][2] = -1;
    out[1][0] = 1;
    out[1][1] = 0;
    out[1][2] = 0;
    out[2][0] = 0;
    out[2][1] = 1;
    out[2][2] = 0;
    out[3][0] = 0;
    out[3][1] = 0;
    out[3][2] = 1;
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
    out[2][0] = static_cast<Scalar>(0) / order;
    out[2][1] = static_cast<Scalar>(1) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(0) / order;
    out[3][2] = static_cast<Scalar>(1) / order;
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
      idxs[0] = 1;
      idxs[1] = 2;
      break;
    case 2:
      idxs[0] = 2;
      idxs[1] = 0;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      break;
    case 4:
      idxs[0] = 1;
      idxs[1] = 3;
      break;
    case 5:
      idxs[0] = 2;
      idxs[1] = 3;
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
      idxs[1] = 1;
      idxs[2] = 3;
      break;
    case 1:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 1;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
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
