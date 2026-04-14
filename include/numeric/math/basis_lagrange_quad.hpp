#ifndef NUMERIC_MATH_BASIS_LAGRANGE_QUAD_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_QUAD_HPP_

#include <numeric/math/dihedral_group.hpp>
#include <numeric/math/polynomial.hpp>
#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>
#include <numeric/utils/error.hpp>

namespace numeric::math {

template <dim_t Order> struct BasisLagrange<mesh::RefElQuad, Order> {
  using ref_el_t = mesh::RefElQuad;
  using interpolation_nodes_t = NodesEquispaced<Order>;
  using poly_t = Lagrange<interpolation_nodes_t>;
  static constexpr dim_t order = Order;
  static constexpr dim_t num_basis_functions = (Order + 1) * (Order + 1);
  static constexpr dim_t num_interpolation_nodes = (Order + 1) * (Order + 1);

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    Scalar basis_x[order + 1];
    Scalar basis_y[order + 1];
    for (dim_t i = 0; i < order + 1; ++i) {
      basis_x[i] = poly_t::basis(i, x[0]);
      basis_y[i] = poly_t::basis(i, x[1]);
    }
    Scalar y = 0;
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      dim_t idxs[2];
      node_idxs(i, idxs);
      y += coeffs[i] * basis_x[idxs[0]] * basis_y[idxs[1]];
    }
    return y;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    Scalar basis_x[order + 1];
    Scalar basis_y[order + 1];
    for (dim_t i = 0; i < order + 1; ++i) {
      basis_x[i] = poly_t::basis(i, x[0]);
      basis_y[i] = poly_t::basis(i, x[1]);
    }
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      dim_t idxs[2];
      node_idxs(i, idxs);
      out[i] = basis_x[idxs[0]] * basis_y[idxs[1]];
    }
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    Scalar basis_x[order + 1];
    Scalar basis_y[order + 1];
    Scalar grad_x[order + 1];
    Scalar grad_y[order + 1];
    for (dim_t i = 0; i < order + 1; ++i) {
      basis_x[i] = poly_t::basis(i, x[0]);
      basis_y[i] = poly_t::basis(i, x[1]);
      grad_x[i] = poly_t::basis_diff(i, x[0]);
      grad_y[i] = poly_t::basis_diff(i, x[1]);
    }
    out[0] = 0;
    out[1] = 0;
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      dim_t idxs[2];
      node_idxs(i, idxs);
      out[0] += coeffs[i] * grad_x[idxs[0]] * basis_y[idxs[1]];
      out[1] += coeffs[i] * basis_x[idxs[0]] * grad_y[idxs[1]];
    }
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[2]) {
    Scalar basis_x[order + 1];
    Scalar basis_y[order + 1];
    Scalar grad_x[order + 1];
    Scalar grad_y[order + 1];
    for (dim_t i = 0; i < order + 1; ++i) {
      basis_x[i] = poly_t::basis(i, x[0]);
      basis_y[i] = poly_t::basis(i, x[1]);
      grad_x[i] = poly_t::basis_diff(i, x[0]);
      grad_y[i] = poly_t::basis_diff(i, x[1]);
    }
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      dim_t idxs[2];
      node_idxs(i, idxs);
      out[i][0] = grad_x[idxs[0]] * basis_y[idxs[1]];
      out[i][1] = basis_x[idxs[0]] * grad_y[idxs[1]];
    }
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = interpolation_nodes_t::template node<Scalar>(idxs[0]);
    out[1] = interpolation_nodes_t::template node<Scalar>(idxs[1]);
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolation_nodes(Scalar (*out)[2]) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      Scalar node_coords[2];
      node(i, node_coords);
      out[i][0] = node_coords[0];
      out[i][1] = node_coords[1];
    }
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
    if (i == 0) {
      out[0] = 0;
      out[1] = 0;
    } else if (i == 1) {
      out[0] = order;
      out[1] = 0;
    } else if (i == 2) {
      out[0] = order;
      out[1] = order;
    } else if (i == 3) {
      out[0] = 0;
      out[1] = order;
    } else if (i < 4 + 1 * (order - 1)) {
      out[0] = i - (3 + 0 * (order - 1));
      out[1] = 0;
    } else if (i < 4 + 2 * (order - 1)) {
      out[0] = order;
      out[1] = i - (3 + 1 * (order - 1));
    } else if (i < 4 + 3 * (order - 1)) {
      out[0] = i - (3 + 2 * (order - 1));
      out[1] = order;
    } else if (i < 4 + 4 * (order - 1)) {
      out[0] = 0;
      out[1] = i - (3 + 3 * (order - 1));
    } else {
      out[0] = 1 + (i - 4 * order) % (order - 1);
      out[1] = 1 + (i - 4 * order) / (order - 1);
    }
  }

  static constexpr NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
    NUMERIC_ERROR("Not yet implemented");
    return -1;
  }

  template <typename Element>
  static NUMERIC_HOST_DEVICE void subelement_node_idxs(dim_t subelement,
                                                       dim_t *idxs) {
    subelement_node_idxs(subelement, idxs, meta::type_tag<Element>{});
  }

  static NUMERIC_HOST_DEVICE void
  subelement_node_idxs(dim_t subelement, dim_t *idxs,
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

  static NUMERIC_HOST_DEVICE void
  subelement_node_idxs(dim_t subelement, dim_t *idxs,
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
    default:
      break;
    }
    for (dim_t i = 0; i < order - 1; ++i) {
      idxs[i + 2] = i + 4 + subelement * (order - 1);
    }
  }

  static NUMERIC_HOST_DEVICE void
  subelement_node_idxs(dim_t subelement, dim_t *idxs,
                       meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static NUMERIC_HOST_DEVICE void
  subelement_node_idxs(dim_t subelement, dim_t *idxs,
                       meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static NUMERIC_HOST_DEVICE void
  subelement_node_idxs(dim_t subelement, dim_t *idxs,
                       meta::type_tag<mesh::RefElTetra>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static NUMERIC_HOST_DEVICE void
  subelement_node_idxs(dim_t subelement, dim_t *idxs,
                       meta::type_tag<mesh::RefElCube>) {
    switch (subelement) {
    default:
      break;
    }
  }
};

} // namespace numeric::math

#endif
