#ifndef NUMERIC_MATH_BASIS_LAGRANGE_SEGMENT_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_SEGMENT_HPP_

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

namespace {

template <typename InterpolationNodes> struct BasisLagrangeSegmentNodeAdaptor {
  using interpolation_nodes_t = InterpolationNodes;
  static constexpr dim_t order = interpolation_nodes_t::order;

  static constexpr dim_t reordered_node(dim_t i) {
    switch (i) {
    case 0:
      return 0;
    case 1:
      return order;
    default:
      return i - 1;
    }
  }

  template <typename Scalar> static constexpr Scalar node(dim_t i) {
    return interpolation_nodes_t::template node<Scalar>(reordered_node(i));
  }

  template <typename Scalar> static Scalar weight(dim_t i) {
    return interpolation_nodes_t::template weight<Scalar>(reordered_node(i));
  }
};

} // namespace

template <dim_t Order> struct BasisLagrange<mesh::RefElSegment, Order> {
  using ref_el_t = mesh::RefElSegment;
  using interpolation_nodes_t = NodesEquispaced<Order>;
  using poly_t =
      Lagrange<BasisLagrangeSegmentNodeAdaptor<interpolation_nodes_t>>;
  static constexpr dim_t order = Order;
  static constexpr dim_t num_basis_functions = Order + 1;
  static constexpr dim_t num_interpolation_nodes = Order + 1;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    return poly_t::eval(coeffs, x[0]);
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      out[i] = poly_t::basis(i, x[0]);
    }
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    out[0] = poly_t::diff(coeffs, x[0]);
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[1]) {
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      out[i][0] = poly_t::basis_diff(i, x[0]);
    }
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    out[0] = poly_t::template node<Scalar>(i);
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[1]) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      out[i][0] = poly_t::template node<Scalar>(i);
    }
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
      out[0] = order;
      break;
    default:
      out[0] = i - 1;
      break;
    }
  }

  static constexpr dim_t node_idx_under_permutation(dim_t i, dim_t *perm) {
    if (perm[0] == 0 && perm[1] == 1) {
      return i;
    } else if (perm[0] == 1 && perm[1] == 0) {
      switch (i) {
      case 0:
        return order;
      case 1:
        return 0;
      default:
        return order + 2 - i;
      }
    } else {
      NUMERIC_ERROR("What a weird permutation you have");
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
