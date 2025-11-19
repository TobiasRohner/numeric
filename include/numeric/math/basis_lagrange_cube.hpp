#ifndef NUMERIC_MATH_BASIS_LAGRANGE_CUBE_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_CUBE_HPP_

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

template <dim_t Order> struct BasisLagrange<mesh::RefElCube, Order> {
  using ref_el_t = mesh::RefElCube;
  using interpolation_nodes_t = NodesEquispaced<Order>;
  using poly_t = Lagrange<interpolation_nodes_t>;
  static constexpr dim_t order = Order;
  static constexpr dim_t num_basis_functions =
      (Order + 1) * (Order + 1) * (Order + 1);
  static constexpr dim_t num_interpolation_nodes =
      (Order + 1) * (Order + 1) * (Order + 1);

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    Scalar basis_x[order + 1];
    Scalar basis_y[order + 1];
    Scalar basis_z[order + 1];
    for (dim_t i = 0; i < order + 1; ++i) {
      basis_x[i] = poly_t::basis(i, x[0]);
      basis_y[i] = poly_t::basis(i, x[1]);
      basis_z[i] = poly_t::basis(i, x[2]);
    }
    Scalar y = 0;
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      dim_t idxs[3];
      node_idxs(i, idxs);
      y += coeffs[i] * basis_x[idxs[0]] * basis_y[idxs[1]] * basis_z[idxs[2]];
    }
    return y;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    Scalar basis_x[order + 1];
    Scalar basis_y[order + 1];
    Scalar basis_z[order + 1];
    for (dim_t i = 0; i < order + 1; ++i) {
      basis_x[i] = poly_t::basis(i, x[0]);
      basis_y[i] = poly_t::basis(i, x[1]);
      basis_z[i] = poly_t::basis(i, x[2]);
    }
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      dim_t idxs[3];
      node_idxs(i, idxs);
      out[i] = basis_x[idxs[0]] * basis_y[idxs[1]] * basis_z[idxs[2]];
    }
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    Scalar basis_x[order + 1];
    Scalar basis_y[order + 1];
    Scalar basis_z[order + 1];
    Scalar grad_x[order + 1];
    Scalar grad_y[order + 1];
    Scalar grad_z[order + 1];
    for (dim_t i = 0; i < order + 1; ++i) {
      basis_x[i] = poly_t::basis(i, x[0]);
      basis_y[i] = poly_t::basis(i, x[1]);
      basis_z[i] = poly_t::basis(i, x[2]);
      grad_x[i] = poly_t::basis_diff(i, x[0]);
      grad_y[i] = poly_t::basis_diff(i, x[1]);
      grad_z[i] = poly_t::basis_diff(i, x[2]);
    }
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      dim_t idxs[3];
      node_idxs(i, idxs);
      out[0] +=
          coeffs[i] * grad_x[idxs[0]] * basis_y[idxs[1]] * basis_z[idxs[2]];
      out[1] +=
          coeffs[i] * basis_x[idxs[0]] * grad_y[idxs[1]] * basis_z[idxs[2]];
      out[2] +=
          coeffs[i] * basis_x[idxs[0]] * basis_y[idxs[1]] * grad_z[idxs[2]];
    }
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[3]) {
    Scalar basis_x[order + 1];
    Scalar basis_y[order + 1];
    Scalar basis_z[order + 1];
    Scalar grad_x[order + 1];
    Scalar grad_y[order + 1];
    Scalar grad_z[order + 1];
    for (dim_t i = 0; i < order + 1; ++i) {
      basis_x[i] = poly_t::basis(i, x[0]);
      basis_y[i] = poly_t::basis(i, x[1]);
      basis_z[i] = poly_t::basis(i, x[2]);
      grad_x[i] = poly_t::basis_diff(i, x[0]);
      grad_y[i] = poly_t::basis_diff(i, x[1]);
      grad_z[i] = poly_t::basis_diff(i, x[2]);
    }
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      dim_t idxs[3];
      node_idxs(i, idxs);
      out[i][0] = grad_x[idxs[0]] * basis_y[idxs[1]] * basis_z[idxs[2]];
      out[i][1] = basis_x[idxs[0]] * grad_y[idxs[1]] * basis_z[idxs[2]];
      out[i][2] = basis_x[idxs[0]] * basis_y[idxs[1]] * grad_z[idxs[2]];
    }
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = interpolation_nodes_t::template node<Scalar>(idxs[0]);
    out[1] = interpolation_nodes_t::template node<Scalar>(idxs[1]);
    out[2] = interpolation_nodes_t::template node<Scalar>(idxs[2]);
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolation_nodes(Scalar (*out)[3]) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      Scalar node_coords[3];
      node(i, node_coords);
      out[i][0] = node_coords[0];
      out[i][1] = node_coords[1];
      out[i][2] = node_coords[2];
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
      out[2] = 0;
    } else if (i == 1) {
      out[0] = order;
      out[1] = 0;
      out[2] = 0;
    } else if (i == 2) {
      out[0] = order;
      out[1] = order;
      out[2] = 0;
    } else if (i == 3) {
      out[0] = 0;
      out[1] = order;
      out[2] = 0;
    } else if (i == 4) {
      out[0] = 0;
      out[1] = 0;
      out[2] = order;
    } else if (i == 5) {
      out[0] = order;
      out[1] = 0;
      out[2] = order;
    } else if (i == 6) {
      out[0] = order;
      out[1] = order;
      out[2] = order;
    } else if (i == 7) {
      out[0] = 0;
      out[1] = order;
      out[2] = order;
    } else if (i < 8 + 1 * (order - 1)) {
      out[0] = i - (7 + 0 * (order - 1));
      out[1] = 0;
      out[2] = 0;
    } else if (i < 8 + 2 * (order - 1)) {
      out[0] = order;
      out[1] = i - (7 + 1 * (order - 1));
      out[2] = 0;
    } else if (i < 8 + 3 * (order - 1)) {
      out[0] = i - (7 + 2 * (order - 1));
      out[1] = order;
      out[2] = 0;
    } else if (i < 8 + 4 * (order - 1)) {
      out[0] = 0;
      out[1] = i - (7 + 3 * (order - 1));
      out[2] = 0;
    } else if (i < 8 + 5 * (order - 1)) {
      out[0] = i - (7 + 4 * (order - 1));
      out[1] = 0;
      out[2] = order;
    } else if (i < 8 + 6 * (order - 1)) {
      out[0] = order;
      out[1] = i - (7 + 5 * (order - 1));
      out[2] = order;
    } else if (i < 8 + 7 * (order - 1)) {
      out[0] = i - (7 + 6 * (order - 1));
      out[1] = order;
      out[2] = order;
    } else if (i < 8 + 8 * (order - 1)) {
      out[0] = 0;
      out[1] = i - (7 + 7 * (order - 1));
      out[2] = order;
    } else if (i < 8 + 9 * (order - 1)) {
      out[0] = 0;
      out[1] = 0;
      out[2] = i - (7 + 8 * (order - 1));
    } else if (i < 8 + 10 * (order - 1)) {
      out[0] = order;
      out[1] = 0;
      out[2] = i - (7 + 9 * (order - 1));
    } else if (i < 8 + 11 * (order - 1)) {
      out[0] = order;
      out[1] = order;
      out[2] = i - (7 + 10 * (order - 1));
    } else if (i < 8 + 12 * (order - 1)) {
      out[0] = 0;
      out[1] = order;
      out[2] = i - (7 + 11 * (order - 1));
    } else if (i < 8 + 12 * (order - 1) + 1 * (order - 1) * (order - 1)) {
      const dim_t iloc =
          i - (8 + 12 * (order - 1) + 0 * (order - 1) * (order - 1));
      out[0] = 0;
      out[1] = 1 + iloc % (order - 1);
      out[2] = 1 + iloc / (order - 1);
    } else if (i < 8 + 12 * (order - 1) + 2 * (order - 1) * (order - 1)) {
      const dim_t iloc =
          i - (8 + 12 * (order - 1) + 1 * (order - 1) * (order - 1));
      out[0] = order;
      out[1] = 1 + iloc % (order - 1);
      out[2] = 1 + iloc / (order - 1);
    } else if (i < 8 + 12 * (order - 1) + 3 * (order - 1) * (order - 1)) {
      const dim_t iloc =
          i - (8 + 12 * (order - 1) + 2 * (order - 1) * (order - 1));
      out[0] = 1 + iloc % (order - 1);
      out[1] = 0;
      out[2] = 1 + iloc / (order - 1);
    } else if (i < 8 + 12 * (order - 1) + 4 * (order - 1) * (order - 1)) {
      const dim_t iloc =
          i - (8 + 12 * (order - 1) + 3 * (order - 1) * (order - 1));
      out[0] = 1 + iloc % (order - 1);
      out[1] = order;
      out[2] = 1 + iloc / (order - 1);
    } else if (i < 8 + 12 * (order - 1) + 5 * (order - 1) * (order - 1)) {
      const dim_t iloc =
          i - (8 + 12 * (order - 1) + 4 * (order - 1) * (order - 1));
      out[0] = 1 + iloc % (order - 1);
      out[1] = 1 + iloc / (order - 1);
      out[2] = 0;
    } else if (i < 8 + 12 * (order - 1) + 6 * (order - 1) * (order - 1)) {
      const dim_t iloc =
          i - (8 + 12 * (order - 1) + 5 * (order - 1) * (order - 1));
      out[0] = 1 + iloc % (order - 1);
      out[1] = 1 + iloc / (order - 1);
      out[2] = order;
    } else {
      const dim_t iloc =
          i - (8 + 12 * (order - 1) + 6 * (order - 1) * (order - 1));
      out[0] = 1 + iloc % (order - 1);
      out[1] = 1 + (iloc / (order - 1)) % (order - 1);
      out[2] = 1 + iloc / ((order - 1) * (order - 1));
    }
  }

  static constexpr NUMERIC_HOST_DEVICE dim_t
  node_idx_under_permutation(dim_t i, dim_t *perm) {
    NUMERIC_ERROR("Not yet implemented");
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
    for (dim_t i = 0; i < order - 1; ++i) {
      idxs[i + 2] = i + 8 + subelement * (order - 1);
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
    case 0:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 7;
      idxs[3] = 4;
      for (dim_t i = 0; i < order - 1; ++i) {
        idxs[i + 4 + 0 * (order - 1)] = i + 8 + 3 * (order - 1);
        idxs[i + 4 + 1 * (order - 1)] = i + 8 + 11 * (order - 1);
        idxs[i + 4 + 2 * (order - 1)] = i + 8 + 7 * (order - 1);
        idxs[i + 4 + 3 * (order - 1)] = i + 8 + 8 * (order - 1);
      }
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 5;
      for (dim_t i = 0; i < order - 1; ++i) {
        idxs[i + 4 + 0 * (order - 1)] = i + 8 + 1 * (order - 1);
        idxs[i + 4 + 1 * (order - 1)] = i + 8 + 10 * (order - 1);
        idxs[i + 4 + 2 * (order - 1)] = i + 8 + 5 * (order - 1);
        idxs[i + 4 + 3 * (order - 1)] = i + 8 + 9 * (order - 1);
      }
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 5;
      idxs[3] = 4;
      for (dim_t i = 0; i < order - 1; ++i) {
        idxs[i + 4 + 0 * (order - 1)] = i + 8 + 0 * (order - 1);
        idxs[i + 4 + 1 * (order - 1)] = i + 8 + 9 * (order - 1);
        idxs[i + 4 + 2 * (order - 1)] = i + 8 + 4 * (order - 1);
        idxs[i + 4 + 3 * (order - 1)] = i + 8 + 8 * (order - 1);
      }
      break;
    case 3:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 7;
      for (dim_t i = 0; i < order - 1; ++i) {
        idxs[i + 4 + 0 * (order - 1)] = i + 8 + 2 * (order - 1);
        idxs[i + 4 + 1 * (order - 1)] = i + 8 + 10 * (order - 1);
        idxs[i + 4 + 2 * (order - 1)] = i + 8 + 6 * (order - 1);
        idxs[i + 4 + 3 * (order - 1)] = i + 8 + 11 * (order - 1);
      }
      break;
    case 4:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 2;
      idxs[3] = 3;
      for (dim_t i = 0; i < order - 1; ++i) {
        idxs[i + 4 + 0 * (order - 1)] = i + 8 + 0 * (order - 1);
        idxs[i + 4 + 1 * (order - 1)] = i + 8 + 1 * (order - 1);
        idxs[i + 4 + 2 * (order - 1)] = i + 8 + 2 * (order - 1);
        idxs[i + 4 + 3 * (order - 1)] = i + 8 + 3 * (order - 1);
      }
      break;
    case 5:
      idxs[0] = 4;
      idxs[1] = 5;
      idxs[2] = 6;
      idxs[3] = 7;
      for (dim_t i = 0; i < order - 1; ++i) {
        idxs[i + 4 + 0 * (order - 1)] = i + 8 + 4 * (order - 1);
        idxs[i + 4 + 1 * (order - 1)] = i + 8 + 5 * (order - 1);
        idxs[i + 4 + 2 * (order - 1)] = i + 8 + 6 * (order - 1);
        idxs[i + 4 + 3 * (order - 1)] = i + 8 + 7 * (order - 1);
      }
      break;
    default:
      break;
    }
    for (dim_t i = 0; i < (order - 1) * (order - 1); ++i) {
      idxs[i + 4 * order] =
          i + 8 + 12 * (order - 1) + subelement * (order - 1) * (order - 1);
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
