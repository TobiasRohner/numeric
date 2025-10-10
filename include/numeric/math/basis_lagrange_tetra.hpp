#ifndef NUMERIC_MATH_BASIS_LAGRANGE_TETRA_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_TETRA_HPP_

#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/math/basis_lagrange_tria.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math {

template <dim_t Order> struct BasisLagrange<mesh::RefElTetra, Order> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = Order;
  static constexpr dim_t num_basis_functions = (Order + 1) * (Order + 2) * (Order + 3) / 6;
  static constexpr dim_t num_interpolation_nodes = (Order + 1) * (Order + 2) * (Order + 3) / 6;

  template <typename Scalar>
  static constexpr Scalar P(dim_t m, Scalar z) {
    Scalar result = 1;
    for (dim_t i = 1 ; i <= m ; ++i) {
      result *= (order * z - i + 1) / i;
    }
    return result;
  }

  template <typename Scalar>
  static constexpr Scalar diff_P(dim_t m, Scalar z) {
    Scalar result = 0;
    for (dim_t i = 1 ; i <= m ; ++i) {
      Scalar prod = static_cast<Scalar>(order) / i;
      for (dim_t j = 1 ; j <= m ; ++j) {
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
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    Scalar y = 0;
    const Scalar l1 = x[0];
    const Scalar l2 = x[1];
    const Scalar l3 = x[2];
    const Scalar l4 = 1 - l1 - l2 - l3;
    for (dim_t b = 0 ; b < num_basis_functions ; ++b) {
      dim_t ijkl[4];
      node_idxs(b, ijkl);
      ijkl[3] = order - ijkl[0] - ijkl[1] - ijkl[2];
      y += coeffs[b] * P(ijkl[0], l1) * P(ijkl[1], l2) * P(ijkl[2], l3) * P(ijkl[3], l4);
    }
    return y;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar l1 = x[0];
    const Scalar l2 = x[1];
    const Scalar l3 = x[2];
    const Scalar l4 = 1 - l1 - l2 - l3;
    for (dim_t b = 0 ; b < num_basis_functions ; ++b) {
      dim_t ijkl[4];
      node_idxs(b, ijkl);
      ijkl[3] = order - ijkl[0] - ijkl[1] - ijkl[2];
      out[b] = P(ijkl[0], l1) * P(ijkl[1], l2) * P(ijkl[2], l3) * P(ijkl[3], l4);
    }
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar l1 = x[0];
    const Scalar l2 = x[1];
    const Scalar l3 = x[2];
    const Scalar l4 = 1 - l1 - l2 - l3;
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    for (dim_t b = 0 ; b < num_basis_functions ; ++b) {
      dim_t ijkl[4];
      node_idxs(b, ijkl);
      ijkl[3] = order - ijkl[0] - ijkl[1] - ijkl[2];
      const Scalar P1 = P(ijkl[0], l1);
      const Scalar P2 = P(ijkl[1], l2);
      const Scalar P3 = P(ijkl[2], l3);
      const Scalar P4 = P(ijkl[3], l4);
      const Scalar dP1 = diff_P(ijkl[0], l1);
      const Scalar dP2 = diff_P(ijkl[1], l2);
      const Scalar dP3 = diff_P(ijkl[2], l3);
      const Scalar dP4 = diff_P(ijkl[3], l4);
      out[0] += coeffs[b] * (dP1 * P2 * P3 * P4 - P1 * P2 * P3 * dP4);
      out[1] += coeffs[b] * (P1 * dP2 * P3 * P4 - P1 * P2 * P3 * dP4);
      out[2] += coeffs[b] * (P1 * P2 * dP3 * P4 - P1 * P2 * P3 * dP4);
    }
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
    const Scalar l1 = x[0];
    const Scalar l2 = x[1];
    const Scalar l3 = x[2];
    const Scalar l4 = 1 - l1 - l2 - l3;
    for (dim_t b = 0 ; b < num_basis_functions ; ++b) {
      dim_t ijkl[4];
      node_idxs(b, ijkl);
      ijkl[3] = order - ijkl[0] - ijkl[1] - ijkl[2];
      const Scalar P1 = P(ijkl[0], l1);
      const Scalar P2 = P(ijkl[1], l2);
      const Scalar P3 = P(ijkl[2], l3);
      const Scalar P4 = P(ijkl[3], l4);
      const Scalar dP1 = diff_P(ijkl[0], l1);
      const Scalar dP2 = diff_P(ijkl[1], l2);
      const Scalar dP3 = diff_P(ijkl[2], l3);
      const Scalar dP4 = diff_P(ijkl[3], l4);
      out[b][0] = dP1 * P2 * P3 * P4 - P1 * P2 * P3 * dP4;
      out[b][1] = P1 * dP2 * P3 * P4 - P1 * P2 * P3 * dP4;
      out[b][2] = P1 * P2 * dP3 * P4 - P1 * P2 * P3 * dP4;
    }
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
    for (dim_t i = 0 ; i < num_basis_functions ; ++i) {
      dim_t idxs[3];
      node_idxs(i, idxs);
      out[i][0] = static_cast<Scalar>(idxs[0]) / order;
      out[i][1] = static_cast<Scalar>(idxs[1]) / order;
      out[i][2] = static_cast<Scalar>(idxs[2]) / order;
    }
  }

  template <typename Scalar>
  static constexpr void interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    if (i == 0) {
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
    } else if (i == 1) {
      out[0] = order;
      out[1] = 0;
      out[2] = 0;
    } else if (i == 2) {
      out[0] = 0;
      out[1] = order;
      out[2] = 0;
    } else if (i == 3) {
      out[0] = 0;
      out[1] = 0;
      out[2] = order;
    } else if (i < 4 + 1 * (order - 1)) {
      out[0] = i - (3 + 0 * (order - 1));
      out[1] = 0;
      out[2] = 0;
    } else if (i < 4 + 2 * (order - 1)) {
      out[0] = order - (i - (3 + 1 * (order - 1)));
      out[1] = i - (3 + 1 * (order - 1));
      out[2] = 0;
    } else if (i < 4 + 3 * (order - 1)) {
      out[0] = 0;
      out[1] = order - (i - (3 + 2 * (order - 1)));
      out[2] = 0;
    } else if (i < 4 + 4 * (order - 1)) {
      out[0] = 0;
      out[1] = 0;
      out[2] = i - (3 + 3 * (order - 1));
    } else if (i < 4 + 5 * (order - 1)) {
      out[0] = order - (i - (3 + 4 * (order - 1)));
      out[1] = 0;
      out[2] = i - (3 + 4 * (order - 1));
    } else if (i < 4 + 6 * (order - 1)) {
      out[0] = 0;
      out[1] = order - (i - (3 + 5 * (order - 1)));
      out[2] = i - (3 + 5 * (order - 1));
    } else if constexpr (order >= 3) {
      using tria_low_t = BasisLagrange<mesh::RefElTria, order - 3>;
      if (i < 4 + 6 * (order - 1) + 1 * tria_low_t::num_basis_functions) {
	const dim_t iloc = i - (4 + 6 * (order - 1) + 0 * tria_low_t::num_basis_functions);
	dim_t idxs_tria[2];
	tria_low_t::node_idxs(iloc, idxs_tria);
	out[0] = 1 + idxs_tria[0];
	out[1] = 0;
	out[2] = 1 + idxs_tria[1];
      } else if (i < 4 + 6 * (order - 1) + 2 * tria_low_t::num_basis_functions) {
	const dim_t iloc = i - (4 + 6 * (order - 1) + 1 * tria_low_t::num_basis_functions);
	dim_t idxs_tria[2];
	tria_low_t::node_idxs(iloc, idxs_tria);
	out[0] = order - 2 - idxs_tria[0] - idxs_tria[1];
	out[1] = 1 + (order - 3) - idxs_tria[0] - idxs_tria[1];
	out[2] = 1 + idxs_tria[0];
      } else if (i < 4 + 6 * (order - 1) + 3 * tria_low_t::num_basis_functions) {
	const dim_t iloc = i - (4 + 6 * (order - 1) + 2 * tria_low_t::num_basis_functions);
	dim_t idxs_tria[2];
	tria_low_t::node_idxs(iloc, idxs_tria);
	out[0] = 0;
	out[1] = 1 + idxs_tria[1];
	out[2] = 1 + idxs_tria[0];
      } else if (i < 4 + 6 * (order - 1) + 4 * tria_low_t::num_basis_functions) {
	const dim_t iloc = i - (4 + 6 * (order - 1) + 3 * tria_low_t::num_basis_functions);
	dim_t idxs_tria[2];
	tria_low_t::node_idxs(iloc, idxs_tria);
	out[0] = 1 + idxs_tria[1];
	out[1] = 1 + idxs_tria[0];
	out[2] = 0;
      } else if constexpr (order >= 4) {
	using tetra_low_t = BasisLagrange<mesh::RefElTetra, order - 4>;
	const dim_t iloc = i - (4 + 6 * (order - 1) + 4 * tria_low_t::num_basis_functions);
	dim_t idxs_low[3];
	tetra_low_t::node_idxs(iloc, idxs_low);
	out[0] = 1 + idxs_low[0];
	out[1] = 1 + idxs_low[1];
	out[2] = 1 + idxs_low[2];
      }
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
    for (dim_t i = 0 ; i < order - 1 ; ++i) {
      idxs[i + 2] = i + 4 + subelement * (order - 1);
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      for (dim_t i = 0 ; i < order - 1 ; ++i) {
	idxs[i + 2 + 0 * (order - 1)] = i + 4 + 0 * (order - 1);
	idxs[i + 2 + 1 * (order - 1)] = i + 4 + 4 * (order - 1);
	idxs[i + 2 + 2 * (order - 1)] = (order - 2 - i) + 4 + 3 * (order - 1);
      }
      break;
    case 1:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 1;
      for (dim_t i = 0 ; i < order - 1 ; ++i) {
	idxs[i + 2 + 0 * (order - 1)] = i + 4 + 5 * (order - 1);
	idxs[i + 2 + 1 * (order - 1)] = (order - 2 -i) + 4 + 4 * (order - 1);
	idxs[i + 2 + 2 * (order - 1)] = i + 4 + 1 * (order - 1);
      }
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      for (dim_t i = 0 ; i < order - 1 ; ++i) {
	idxs[i + 2 + 0 * (order - 1)] = i + 4 + 3 * (order - 1);
	idxs[i + 2 + 1 * (order - 1)] = (order - 2 - i) + 4 + 5 * (order - 1);
	idxs[i + 2 + 2 * (order - 1)] = i + 4 + 2 * (order - 1);
      }
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      for (dim_t i = 0 ; i < order - 1 ; ++i) {
	idxs[i + 2 + 0 * (order - 1)] = (order - 2 - i) + 4 + 2 * (order - 1);
	idxs[i + 2 + 1 * (order - 1)] = (order - 2 - i) + 4 + 1 * (order - 1);
	idxs[i + 2 + 2 * (order - 1)] = (order - 2 - i) + 4 + 0 * (order - 1);
      }
      break;
    default:
      break;
    }
    if constexpr (order >= 3) {
      using tria_low_t = BasisLagrange<mesh::RefElTria, order - 3>;
      for (dim_t i = 0 ; i < tria_low_t::num_basis_functions ; ++i) {
	idxs[i + 3 * order] = i + subelement * tria_low_t::num_basis_functions + 4 + 6 * (order - 1);
      }
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
