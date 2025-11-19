#ifndef NUMERIC_MATH_BASIS_LAGRANGE_TRIA_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_TRIA_HPP_

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

template <dim_t Order> struct BasisLagrange<mesh::RefElTria, Order> {
  using ref_el_t = mesh::RefElTria;
  static constexpr dim_t order = Order;
  static constexpr dim_t num_basis_functions = (Order + 1) * (Order + 2) / 2;
  static constexpr dim_t num_interpolation_nodes =
      (Order + 1) * (Order + 2) / 2;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar P(dim_t m, Scalar z) {
    Scalar result = 1;
    for (dim_t i = 1; i <= m; ++i) {
      result *= (order * z - i + 1) / i;
    }
    return result;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar diff_P(dim_t m, Scalar z) {
    Scalar result = 0;
    for (dim_t i = 1; i <= m; ++i) {
      Scalar prod = static_cast<Scalar>(order) / i;
      for (dim_t j = 1; j <= m; ++j) {
        if (i == j) {
          continue;
        }
        prod *= (order * z - j + 1) / j;
      }
      result += prod;
    }
    return result;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    Scalar y = 0;
    const Scalar l1 = x[0];
    const Scalar l2 = x[1];
    const Scalar l3 = 1 - l1 - l2;
    for (dim_t b = 0; b < num_basis_functions; ++b) {
      dim_t ijk[3];
      node_idxs(b, ijk);
      ijk[2] = order - ijk[0] - ijk[1];
      y += coeffs[b] * P(ijk[0], l1) * P(ijk[1], l2) * P(ijk[2], l3);
    }
    return y;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    const Scalar l1 = x[0];
    const Scalar l2 = x[1];
    const Scalar l3 = 1 - l1 - l2;
    for (dim_t b = 0; b < num_basis_functions; ++b) {
      dim_t ijk[3];
      node_idxs(b, ijk);
      ijk[2] = order - ijk[0] - ijk[1];
      out[b] = P(ijk[0], l1) * P(ijk[1], l2) * P(ijk[2], l3);
    }
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    const Scalar l1 = x[0];
    const Scalar l2 = x[1];
    const Scalar l3 = 1 - l1 - l2;
    out[0] = 0;
    out[1] = 0;
    for (dim_t b = 0; b < num_basis_functions; ++b) {
      dim_t ijk[3];
      node_idxs(b, ijk);
      ijk[2] = order - ijk[0] - ijk[1];
      const Scalar P1 = P(ijk[0], l1);
      const Scalar P2 = P(ijk[1], l2);
      const Scalar P3 = P(ijk[2], l3);
      const Scalar dP1 = diff_P(ijk[0], l1);
      const Scalar dP2 = diff_P(ijk[1], l2);
      const Scalar dP3 = diff_P(ijk[2], l3);
      out[0] += coeffs[b] * (dP1 * P2 * P3 - P1 * P2 * dP3);
      out[1] += coeffs[b] * (P1 * dP2 * P3 - P1 * P2 * dP3);
    }
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[2]) {
    const Scalar l1 = x[0];
    const Scalar l2 = x[1];
    const Scalar l3 = 1 - l1 - l2;
    for (dim_t b = 0; b < num_basis_functions; ++b) {
      dim_t ijk[3];
      node_idxs(b, ijk);
      ijk[2] = order - ijk[0] - ijk[1];
      const Scalar P1 = P(ijk[0], l1);
      const Scalar P2 = P(ijk[1], l2);
      const Scalar P3 = P(ijk[2], l3);
      const Scalar dP1 = diff_P(ijk[0], l1);
      const Scalar dP2 = diff_P(ijk[1], l2);
      const Scalar dP3 = diff_P(ijk[2], l3);
      out[b][0] = dP1 * P2 * P3 - P1 * P2 * dP3;
      out[b][1] = P1 * dP2 * P3 - P1 * P2 * dP3;
    }
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolation_nodes(Scalar (*out)[2]) {
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      dim_t idxs[2];
      node_idxs(i, idxs);
      out[i][0] = static_cast<Scalar>(idxs[0]) / order;
      out[i][1] = static_cast<Scalar>(idxs[1]) / order;
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
      out[0] = 0;
      out[1] = order;
    } else if (i < 3 + 1 * (order - 1)) {
      out[0] = i - (2 + 0 * (order - 1));
      out[1] = 0;
    } else if (i < 3 + 2 * (order - 1)) {
      out[0] = order - (i - (2 + 1 * (order - 1)));
      out[1] = i - (2 + 1 * (order - 1));
    } else if (i < 3 + 3 * (order - 1)) {
      out[0] = 0;
      out[1] = order - (i - (2 + 2 * (order - 1)));
    } else {
      if constexpr (order >= 3) {
        using tria_low_t = BasisLagrange<mesh::RefElTria, order - 3>;
        tria_low_t::node_idxs(i - 3 * order, out);
        ++out[0];
        ++out[1];
      }
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
      idxs[0] = 2;
      idxs[1] = 0;
      break;
    default:
      break;
    }
    for (dim_t i = 0; i < order - 1; ++i) {
      idxs[i + 2] = i + 3 + subelement * (order - 1);
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
