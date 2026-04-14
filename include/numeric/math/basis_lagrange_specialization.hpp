#ifndef NUMERIC_MATH_BASIS_LAGRANGE_SPECIALIZATION_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_SPECIALIZATION_HPP_

namespace numeric::math {

template <> struct BasisLagrange<mesh::RefElSegment, 1> {
  using ref_el_t = mesh::RefElSegment;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 2;
  static constexpr dim_t num_interpolation_nodes = 2;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    return -coeffs[0] * (x[0] - 1) + coeffs[1] * x[0];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    out[0] = 1 - x[0];
    out[1] = x[0];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    out[0] = -coeffs[0] + coeffs[1];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[1]) {
    out[0][0] = -1;
    out[1][0] = 1;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void node(dim_t i, Scalar *out) {
    dim_t idxs[1];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolation_nodes(Scalar (*out)[1]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(1) / order;
  }
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      break;
    case 1:
      out[0] = 1;
      break;
    }
  }

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
    if (action.is_identity()) {
      return i;
    } else {
      switch (i) {
      case 0:
        return 1;
      case 1:
        return 0;
      default:
        return order + 2 - i;
      }
    }
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
    default:
      break;
    }
  }

  static NUMERIC_HOST_DEVICE void
  subelement_node_idxs(dim_t subelement, dim_t *idxs,
                       meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
    default:
      break;
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

template <> struct BasisLagrange<mesh::RefElSegment, 2> {
  using ref_el_t = mesh::RefElSegment;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 3;
  static constexpr dim_t num_interpolation_nodes = 3;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 2 * x[0];
    const Scalar x2 = (1.0 / 2.0) * x1 - 1.0 / 2.0;
    return 2 * coeffs[0] * x0 * x2 + 2 * coeffs[1] * x2 * x[0] -
           2 * coeffs[2] * x0 * x1;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 2 * x[0] - 1;
    out[0] = x0 * x1;
    out[1] = x1 * x[0];
    out[2] = -4 * x0 * x[0];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    const Scalar x0 = 4 * x[0];
    out[0] = coeffs[0] * (x0 - 3) + coeffs[1] * (x0 - 1) -
             4 * coeffs[2] * (2 * x[0] - 1);
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[1]) {
    const Scalar x0 = 4 * x[0];
    out[0][0] = x0 - 3;
    out[1][0] = x0 - 1;
    out[2][0] = 4 * (1 - 2 * x[0]);
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void node(dim_t i, Scalar *out) {
    dim_t idxs[1];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolation_nodes(Scalar (*out)[1]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(2) / order;
    out[2][0] = static_cast<Scalar>(1) / order;
  }
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
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

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
    if (action.is_identity()) {
      return i;
    } else {
      switch (i) {
      case 0:
        return 1;
      case 1:
        return 0;
      default:
        return order + 2 - i;
      }
    }
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
    default:
      break;
    }
  }

  static NUMERIC_HOST_DEVICE void
  subelement_node_idxs(dim_t subelement, dim_t *idxs,
                       meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
    default:
      break;
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

template <> struct BasisLagrange<mesh::RefElSegment, 3> {
  using ref_el_t = mesh::RefElSegment;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 4;
  static constexpr dim_t num_interpolation_nodes = 4;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
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
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
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
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
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
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[1]) {
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

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void node(dim_t i, Scalar *out) {
    dim_t idxs[1];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolation_nodes(Scalar (*out)[1]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(3) / order;
    out[2][0] = static_cast<Scalar>(1) / order;
    out[3][0] = static_cast<Scalar>(2) / order;
  }
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
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

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
    if (action.is_identity()) {
      return i;
    } else {
      switch (i) {
      case 0:
        return 1;
      case 1:
        return 0;
      default:
        return order + 2 - i;
      }
    }
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
    default:
      break;
    }
  }

  static NUMERIC_HOST_DEVICE void
  subelement_node_idxs(dim_t subelement, dim_t *idxs,
                       meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
    default:
      break;
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

template <> struct BasisLagrange<mesh::RefElTria, 1> {
  using ref_el_t = mesh::RefElTria;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 3;
  static constexpr dim_t num_interpolation_nodes = 3;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    return -coeffs[0] * (x[0] + x[1] - 1) + coeffs[1] * x[0] + coeffs[2] * x[1];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    out[0] = -x[0] - x[1] + 1;
    out[1] = x[0];
    out[2] = x[1];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    out[0] = -coeffs[0] + coeffs[1];
    out[1] = -coeffs[0] + coeffs[2];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[2]) {
    out[0][0] = -1;
    out[0][1] = -1;
    out[1][0] = 1;
    out[1][1] = 0;
    out[2][0] = 0;
    out[2][1] = 1;
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
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(1) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(0) / order;
    out[2][1] = static_cast<Scalar>(1) / order;
  }
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      break;
    case 1:
      out[0] = 1;
      out[1] = 0;
      break;
    case 2:
      out[0] = 0;
      out[1] = 1;
      break;
    }
  }

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
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

template <> struct BasisLagrange<mesh::RefElTria, 2> {
  using ref_el_t = mesh::RefElTria;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 6;
  static constexpr dim_t num_interpolation_nodes = 6;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    const Scalar x0 = 2 * x[0];
    const Scalar x1 = x[0] + x[1] - 1;
    const Scalar x2 = 2 * x[1];
    const Scalar x3 = x0 - 1;
    return coeffs[0] * x1 * (x2 + x3) + coeffs[1] * x3 * x[0] +
           coeffs[2] * x[1] * (x2 - 1) - 2 * coeffs[3] * x0 * x1 +
           2 * coeffs[4] * x0 * x[1] - 2 * coeffs[5] * x1 * x2;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    const Scalar x0 = x[0] + x[1] - 1;
    const Scalar x1 = 2 * x[1];
    const Scalar x2 = 2 * x[0] - 1;
    const Scalar x3 = 4 * x[0];
    out[0] = x0 * (x1 + x2);
    out[1] = x2 * x[0];
    out[2] = x[1] * (x1 - 1);
    out[3] = -x0 * x3;
    out[4] = x3 * x[1];
    out[5] = -4 * x0 * x[1];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    const Scalar x0 = 4 * x[1];
    const Scalar x1 = 4 * x[0];
    const Scalar x2 = coeffs[0] * (x0 + x1 - 3);
    out[0] = coeffs[1] * (x1 - 1) - 4 * coeffs[3] * (2 * x[0] + x[1] - 1) +
             coeffs[4] * x0 - coeffs[5] * x0 + x2;
    out[1] = coeffs[2] * (x0 - 1) - coeffs[3] * x1 + coeffs[4] * x1 -
             4 * coeffs[5] * (x[0] + 2 * x[1] - 1) + x2;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[2]) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = 4 * x[1];
    const Scalar x2 = x0 + x1 - 3;
    out[0][0] = x2;
    out[0][1] = x2;
    out[1][0] = x0 - 1;
    out[1][1] = 0;
    out[2][0] = 0;
    out[2][1] = x1 - 1;
    out[3][0] = 4 * (-2 * x[0] - x[1] + 1);
    out[3][1] = -x0;
    out[4][0] = x1;
    out[4][1] = x0;
    out[5][0] = -x1;
    out[5][1] = 4 * (-x[0] - 2 * x[1] + 1);
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
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(2) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(0) / order;
    out[2][1] = static_cast<Scalar>(2) / order;
    out[3][0] = static_cast<Scalar>(1) / order;
    out[3][1] = static_cast<Scalar>(0) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(1) / order;
    out[5][0] = static_cast<Scalar>(0) / order;
    out[5][1] = static_cast<Scalar>(1) / order;
  }
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      break;
    case 1:
      out[0] = 2;
      out[1] = 0;
      break;
    case 2:
      out[0] = 0;
      out[1] = 2;
      break;
    case 3:
      out[0] = 1;
      out[1] = 0;
      break;
    case 4:
      out[0] = 1;
      out[1] = 1;
      break;
    case 5:
      out[0] = 0;
      out[1] = 1;
      break;
    }
  }

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
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
      idxs[2] = 3;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 4;
      break;
    case 2:
      idxs[0] = 2;
      idxs[1] = 0;
      idxs[2] = 5;
      break;
    default:
      break;
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

template <> struct BasisLagrange<mesh::RefElTria, 3> {
  using ref_el_t = mesh::RefElTria;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 10;
  static constexpr dim_t num_interpolation_nodes = 10;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    const Scalar x0 = x[0] + x[1] - 1;
    const Scalar x1 = x0 * x[0];
    const Scalar x2 = 3 * x[0];
    const Scalar x3 = x2 - 1;
    const Scalar x4 = x3 * x[0];
    const Scalar x5 = (3.0 / 2.0) * x[1];
    const Scalar x6 = 3 * x[1];
    const Scalar x7 = x[1] * (x6 - 1);
    const Scalar x8 = (3.0 / 2.0) * x7;
    const Scalar x9 = (3.0 / 2.0) * x1;
    const Scalar x10 = x2 - 2;
    const Scalar x11 = x10 + x6;
    const Scalar x12 = x0 * x11;
    return -1.0 / 2.0 * coeffs[0] * x12 * (x3 + x6) +
           (1.0 / 2.0) * coeffs[1] * x10 * x4 +
           (1.0 / 2.0) * coeffs[2] * x7 * (x6 - 2) + 3 * coeffs[3] * x11 * x9 -
           3 * coeffs[4] * x3 * x9 + 3 * coeffs[5] * x4 * x5 +
           3 * coeffs[6] * x8 * x[0] - 3 * coeffs[7] * x0 * x8 +
           3 * coeffs[8] * x12 * x5 - 27 * coeffs[9] * x1 * x[1];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    const Scalar x0 = 3 * x[1];
    const Scalar x1 = 3 * x[0];
    const Scalar x2 = x1 - 1;
    const Scalar x3 = x[0] + x[1] - 1;
    const Scalar x4 = x1 - 2;
    const Scalar x5 = x3 * (x0 + x4);
    const Scalar x6 = x2 * x[0];
    const Scalar x7 = x[1] * (x0 - 1);
    const Scalar x8 = (9.0 / 2.0) * x[0];
    const Scalar x9 = (9.0 / 2.0) * x6;
    out[0] = -1.0 / 2.0 * x5 * (x0 + x2);
    out[1] = (1.0 / 2.0) * x4 * x6;
    out[2] = (1.0 / 2.0) * x7 * (x0 - 2);
    out[3] = x5 * x8;
    out[4] = -x3 * x9;
    out[5] = x9 * x[1];
    out[6] = x7 * x8;
    out[7] = -9.0 / 2.0 * x3 * x7;
    out[8] = (9.0 / 2.0) * x5 * x[1];
    out[9] = -27 * x3 * x[0] * x[1];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    const Scalar x0 = 6 * x[0];
    const Scalar x1 = 3 * x[1];
    const Scalar x2 = x1 - 1;
    const Scalar x3 = x2 * x[1];
    const Scalar x4 = (3.0 / 2.0) * coeffs[7];
    const Scalar x5 = x[1] - 1;
    const Scalar x6 = 9 * coeffs[9];
    const Scalar x7 = 6 * x[1];
    const Scalar x8 = -x0 - x7 + 5;
    const Scalar x9 = (3.0 / 2.0) * coeffs[8];
    const Scalar x10 = 3 * x[0];
    const Scalar x11 = x10 - 1;
    const Scalar x12 = x11 * x[0];
    const Scalar x13 = x5 + x[0];
    const Scalar x14 = (3.0 / 2.0) * coeffs[4];
    const Scalar x15 = x10 - 2;
    const Scalar x16 = -x1 - x15;
    const Scalar x17 = -x13;
    const Scalar x18 = x16 * x17;
    const Scalar x19 = -x18;
    const Scalar x20 = (3.0 / 2.0) * coeffs[3];
    const Scalar x21 = -x1 - x11;
    const Scalar x22 =
        (1.0 / 6.0) * coeffs[0] * (x16 * x21 + 3 * x17 * x21 + 3 * x18);
    const Scalar x23 = x1 - 2;
    out[0] = (1.0 / 2.0) * coeffs[1] * (x10 * x11 + x10 * x15 + x11 * x15) +
             (9.0 / 2.0) * coeffs[5] * x[1] * (x0 - 1) +
             (9.0 / 2.0) * coeffs[6] * x2 * x[1] -
             3 * x14 * (x10 * x13 + x11 * x13 + x12) -
             3 * x20 * (x10 * x17 + x16 * x[0] + x19) - 3 * x22 - 3 * x3 * x4 -
             3 * x6 * x[1] * (x5 + 2 * x[0]) - 3 * x8 * x9 * x[1];
    out[1] = (1.0 / 2.0) * coeffs[2] * (x1 * x2 + x1 * x23 + x2 * x23) +
             (9.0 / 2.0) * coeffs[5] * x11 * x[0] +
             (9.0 / 2.0) * coeffs[6] * x[0] * (x7 - 1) - 3 * x12 * x14 -
             3 * x20 * x8 * x[0] - 3 * x22 -
             3 * x4 * (x1 * x13 + x13 * x2 + x3) -
             3 * x6 * x[0] * (x[0] + 2 * x[1] - 1) -
             3 * x9 * (x1 * x17 + x16 * x[1] + x19);
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[2]) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x0 + x[0];
    const Scalar x2 = 3 * x[1];
    const Scalar x3 = 3 * x[0];
    const Scalar x4 = x3 - 2;
    const Scalar x5 = x2 + x4;
    const Scalar x6 = x1 * x5;
    const Scalar x7 = x3 - 1;
    const Scalar x8 = x2 + x7;
    const Scalar x9 =
        -3.0 / 2.0 * x1 * x8 - 1.0 / 2.0 * x5 * x8 - 3.0 / 2.0 * x6;
    const Scalar x10 = x2 - 2;
    const Scalar x11 = x2 - 1;
    const Scalar x12 = x1 * x3;
    const Scalar x13 = 6 * x[0];
    const Scalar x14 = 6 * x[1];
    const Scalar x15 = x13 + x14 - 5;
    const Scalar x16 = (9.0 / 2.0) * x[0];
    const Scalar x17 = x7 * x[0];
    const Scalar x18 = (9.0 / 2.0) * x17;
    const Scalar x19 = (9.0 / 2.0) * x[1];
    const Scalar x20 = x11 * x[1];
    const Scalar x21 = (9.0 / 2.0) * x20;
    const Scalar x22 = x1 * x2;
    out[0][0] = x9;
    out[0][1] = x9;
    out[1][0] =
        (1.0 / 2.0) * x3 * x4 + (1.0 / 2.0) * x3 * x7 + (1.0 / 2.0) * x4 * x7;
    out[1][1] = 0;
    out[2][0] = 0;
    out[2][1] = (1.0 / 2.0) * x10 * x11 + (1.0 / 2.0) * x10 * x2 +
                (1.0 / 2.0) * x11 * x2;
    out[3][0] = (9.0 / 2.0) * x12 + (9.0 / 2.0) * x5 * x[0] + (9.0 / 2.0) * x6;
    out[3][1] = x15 * x16;
    out[4][0] = -9.0 / 2.0 * x1 * x7 - 9.0 / 2.0 * x12 - 9.0 / 2.0 * x17;
    out[4][1] = -x18;
    out[5][0] = x19 * (x13 - 1);
    out[5][1] = x18;
    out[6][0] = x21;
    out[6][1] = x16 * (x14 - 1);
    out[7][0] = -x21;
    out[7][1] = -9.0 / 2.0 * x1 * x11 - 9.0 / 2.0 * x20 - 9.0 / 2.0 * x22;
    out[8][0] = x15 * x19;
    out[8][1] = (9.0 / 2.0) * x22 + (9.0 / 2.0) * x5 * x[1] + (9.0 / 2.0) * x6;
    out[9][0] = -27 * x[1] * (x0 + 2 * x[0]);
    out[9][1] = -27 * x[0] * (x[0] + 2 * x[1] - 1);
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
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(3) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(0) / order;
    out[2][1] = static_cast<Scalar>(3) / order;
    out[3][0] = static_cast<Scalar>(1) / order;
    out[3][1] = static_cast<Scalar>(0) / order;
    out[4][0] = static_cast<Scalar>(2) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(2) / order;
    out[5][1] = static_cast<Scalar>(1) / order;
    out[6][0] = static_cast<Scalar>(1) / order;
    out[6][1] = static_cast<Scalar>(2) / order;
    out[7][0] = static_cast<Scalar>(0) / order;
    out[7][1] = static_cast<Scalar>(2) / order;
    out[8][0] = static_cast<Scalar>(0) / order;
    out[8][1] = static_cast<Scalar>(1) / order;
    out[9][0] = static_cast<Scalar>(1) / order;
    out[9][1] = static_cast<Scalar>(1) / order;
  }
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      break;
    case 1:
      out[0] = 3;
      out[1] = 0;
      break;
    case 2:
      out[0] = 0;
      out[1] = 3;
      break;
    case 3:
      out[0] = 1;
      out[1] = 0;
      break;
    case 4:
      out[0] = 2;
      out[1] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 1;
      break;
    case 6:
      out[0] = 1;
      out[1] = 2;
      break;
    case 7:
      out[0] = 0;
      out[1] = 2;
      break;
    case 8:
      out[0] = 0;
      out[1] = 1;
      break;
    case 9:
      out[0] = 1;
      out[1] = 1;
      break;
    }
  }

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
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
      idxs[2] = 3;
      idxs[3] = 4;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 5;
      idxs[3] = 6;
      break;
    case 2:
      idxs[0] = 2;
      idxs[1] = 0;
      idxs[2] = 7;
      idxs[3] = 8;
      break;
    default:
      break;
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

template <> struct BasisLagrange<mesh::RefElQuad, 1> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 4;
  static constexpr dim_t num_interpolation_nodes = 4;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[0] - 1;
    return coeffs[0] * x0 * x1 - coeffs[1] * x0 * x[0] +
           coeffs[2] * x[0] * x[1] - coeffs[3] * x1 * x[1];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    out[0] = x0 * x1;
    out[1] = -x1 * x[0];
    out[2] = x[0] * x[1];
    out[3] = -x0 * x[1];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[0] - 1;
    out[0] =
        coeffs[0] * x0 - coeffs[1] * x0 + coeffs[2] * x[1] - coeffs[3] * x[1];
    out[1] =
        coeffs[0] * x1 - coeffs[1] * x[0] + coeffs[2] * x[0] - coeffs[3] * x1;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[2]) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[0] - 1;
    out[0][0] = x0;
    out[0][1] = x1;
    out[1][0] = -x0;
    out[1][1] = -x[0];
    out[2][0] = x[1];
    out[2][1] = x[0];
    out[3][0] = -x[1];
    out[3][1] = -x1;
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
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(1) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(1) / order;
    out[2][1] = static_cast<Scalar>(1) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(1) / order;
  }
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      break;
    case 1:
      out[0] = 1;
      out[1] = 0;
      break;
    case 2:
      out[0] = 1;
      out[1] = 1;
      break;
    case 3:
      out[0] = 0;
      out[1] = 1;
      break;
    }
  }

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
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

template <> struct BasisLagrange<mesh::RefElQuad, 2> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 9;
  static constexpr dim_t num_interpolation_nodes = 9;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x1 * x[0];
    const Scalar x3 = 2 * x[0] - 1;
    const Scalar x4 = x3 * x[1];
    const Scalar x5 = 2 * x[1] - 1;
    const Scalar x6 = x0 * x5;
    const Scalar x7 = (1.0 / 4.0) * x4;
    const Scalar x8 = (1.0 / 4.0) * x3;
    return 4 * coeffs[0] * x1 * x6 * x8 + 4 * coeffs[1] * x2 * x5 * x8 +
           4 * coeffs[2] * x5 * x7 * x[0] + 4 * coeffs[3] * x6 * x7 -
           4 * coeffs[4] * x2 * x6 - 4 * coeffs[5] * x2 * x4 -
           4 * coeffs[6] * x6 * x[0] * x[1] - 4 * coeffs[7] * x0 * x1 * x4 +
           16 * coeffs[8] * x0 * x2 * x[1];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    const Scalar x0 = 2 * x[0] - 1;
    const Scalar x1 = 2 * x[1] - 1;
    const Scalar x2 = x0 * x1;
    const Scalar x3 = x[0] - 1;
    const Scalar x4 = x[1] - 1;
    const Scalar x5 = x3 * x4;
    const Scalar x6 = x2 * x[0];
    const Scalar x7 = x3 * x[1];
    const Scalar x8 = 4 * x[0];
    const Scalar x9 = x1 * x8;
    const Scalar x10 = x0 * x[1];
    out[0] = x2 * x5;
    out[1] = x4 * x6;
    out[2] = x6 * x[1];
    out[3] = x2 * x7;
    out[4] = -x5 * x9;
    out[5] = -x10 * x4 * x8;
    out[6] = -x7 * x9;
    out[7] = -4 * x10 * x5;
    out[8] = 16 * x5 * x[0] * x[1];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = x[1] - 1;
    const Scalar x3 = x2 * x[1];
    const Scalar x4 = x0 - 3;
    const Scalar x5 = 2 * x[0] - 1;
    const Scalar x6 = 4 * x[1];
    const Scalar x7 = 2 * x[1] - 1;
    const Scalar x8 = (1.0 / 4.0) * x7;
    const Scalar x9 = x8 * x[1];
    const Scalar x10 = x5 * x7;
    const Scalar x11 = x2 * x8;
    const Scalar x12 = x6 - 3;
    const Scalar x13 = x[0] - 1;
    const Scalar x14 = x13 * x[0];
    const Scalar x15 = x6 - 1;
    const Scalar x16 = (1.0 / 4.0) * x5;
    const Scalar x17 = x12 * x16;
    const Scalar x18 = x15 * x16;
    out[0] = 4 * coeffs[0] * x11 * x4 + 4 * coeffs[1] * x1 * x11 +
             4 * coeffs[2] * x1 * x9 + 4 * coeffs[3] * x4 * x9 -
             4 * coeffs[4] * x10 * x2 - 4 * coeffs[5] * x1 * x3 -
             4 * coeffs[6] * x10 * x[1] - 4 * coeffs[7] * x3 * x4 +
             4 * coeffs[8] * x2 * x5 * x6;
    out[1] = 4 * coeffs[0] * x13 * x17 + 4 * coeffs[1] * x17 * x[0] +
             4 * coeffs[2] * x18 * x[0] + 4 * coeffs[3] * x13 * x18 -
             4 * coeffs[4] * x12 * x14 - 4 * coeffs[5] * x10 * x[0] -
             4 * coeffs[6] * x14 * x15 - 4 * coeffs[7] * x10 * x13 +
             4 * coeffs[8] * x0 * x13 * x7;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[2]) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x0 - 3;
    const Scalar x2 = x[1] - 1;
    const Scalar x3 = 2 * x[1] - 1;
    const Scalar x4 = x2 * x3;
    const Scalar x5 = x[0] - 1;
    const Scalar x6 = 2 * x[0] - 1;
    const Scalar x7 = 4 * x[1];
    const Scalar x8 = x7 - 3;
    const Scalar x9 = x6 * x8;
    const Scalar x10 = x0 - 1;
    const Scalar x11 = x3 * x[1];
    const Scalar x12 = x7 - 1;
    const Scalar x13 = x12 * x6;
    const Scalar x14 = 4 * x6;
    const Scalar x15 = x0 * x5;
    const Scalar x16 = x2 * x7;
    const Scalar x17 = x3 * x6;
    const Scalar x18 = x3 * x5;
    out[0][0] = x1 * x4;
    out[0][1] = x5 * x9;
    out[1][0] = x10 * x4;
    out[1][1] = x9 * x[0];
    out[2][0] = x10 * x11;
    out[2][1] = x13 * x[0];
    out[3][0] = x1 * x11;
    out[3][1] = x13 * x5;
    out[4][0] = -x14 * x4;
    out[4][1] = -x15 * x8;
    out[5][0] = -x10 * x16;
    out[5][1] = -x0 * x17;
    out[6][0] = -x17 * x7;
    out[6][1] = -x12 * x15;
    out[7][0] = -x1 * x16;
    out[7][1] = -x14 * x18;
    out[8][0] = 16 * x2 * x6 * x[1];
    out[8][1] = 16 * x18 * x[0];
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
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(2) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(2) / order;
    out[2][1] = static_cast<Scalar>(2) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(2) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(2) / order;
    out[5][1] = static_cast<Scalar>(1) / order;
    out[6][0] = static_cast<Scalar>(1) / order;
    out[6][1] = static_cast<Scalar>(2) / order;
    out[7][0] = static_cast<Scalar>(0) / order;
    out[7][1] = static_cast<Scalar>(1) / order;
    out[8][0] = static_cast<Scalar>(1) / order;
    out[8][1] = static_cast<Scalar>(1) / order;
  }
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      break;
    case 1:
      out[0] = 2;
      out[1] = 0;
      break;
    case 2:
      out[0] = 2;
      out[1] = 2;
      break;
    case 3:
      out[0] = 0;
      out[1] = 2;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 1;
      break;
    case 6:
      out[0] = 1;
      out[1] = 2;
      break;
    case 7:
      out[0] = 0;
      out[1] = 1;
      break;
    case 8:
      out[0] = 1;
      out[1] = 1;
      break;
    }
  }

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
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
      idxs[2] = 4;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 5;
      break;
    case 2:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 6;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 7;
      break;
    default:
      break;
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

template <> struct BasisLagrange<mesh::RefElQuad, 3> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 16;
  static constexpr dim_t num_interpolation_nodes = 16;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x0 * x1 * x[0] * x[1];
    const Scalar x3 = 3 * x[0];
    const Scalar x4 = x3 - 2;
    const Scalar x5 = 3 * x[1];
    const Scalar x6 = x5 - 2;
    const Scalar x7 = x4 * x6;
    const Scalar x8 = x5 - 1;
    const Scalar x9 = x3 - 1;
    const Scalar x10 = x2 * x9;
    const Scalar x11 = x4 * x8;
    const Scalar x12 = x1 * x7 * x9;
    const Scalar x13 = x1 * x[0];
    const Scalar x14 = x9 * x[1];
    const Scalar x15 = (1.0 / 9.0) * x0;
    const Scalar x16 = x15 * x[1];
    const Scalar x17 = x7 * x8;
    const Scalar x18 = x8 * x[0];
    const Scalar x19 = x16 * x9;
    const Scalar x20 = (1.0 / 81.0) * x14 * x17;
    const Scalar x21 = x13 * x15;
    const Scalar x22 = (1.0 / 81.0) * x12;
    return (81.0 / 4.0) * coeffs[0] * x0 * x22 * x8 -
           81.0 / 4.0 * coeffs[10] * x12 * x16 +
           (81.0 / 4.0) * coeffs[11] * x1 * x11 * x19 +
           (81.0 / 4.0) * coeffs[12] * x2 * x7 -
           81.0 / 4.0 * coeffs[13] * x10 * x6 -
           81.0 / 4.0 * coeffs[14] * x11 * x2 +
           (81.0 / 4.0) * coeffs[15] * x10 * x8 -
           81.0 / 4.0 * coeffs[1] * x18 * x22 +
           (81.0 / 4.0) * coeffs[2] * x20 * x[0] -
           81.0 / 4.0 * coeffs[3] * x0 * x20 -
           81.0 / 4.0 * coeffs[4] * x17 * x21 +
           (81.0 / 4.0) * coeffs[5] * x21 * x6 * x8 * x9 +
           (9.0 / 4.0) * coeffs[6] * x12 * x[0] * x[1] -
           9.0 / 4.0 * coeffs[7] * x11 * x13 * x14 +
           (81.0 / 4.0) * coeffs[8] * x16 * x17 * x[0] -
           81.0 / 4.0 * coeffs[9] * x18 * x19 * x6;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 3 * x[1];
    const Scalar x2 = x1 - 1;
    const Scalar x3 = x0 * x2;
    const Scalar x4 = x[1] - 1;
    const Scalar x5 = 3 * x[0];
    const Scalar x6 = x5 - 1;
    const Scalar x7 = x5 - 2;
    const Scalar x8 = x1 - 2;
    const Scalar x9 = x4 * x6 * x7 * x8;
    const Scalar x10 = (1.0 / 4.0) * x9;
    const Scalar x11 = x2 * x[0];
    const Scalar x12 = x6 * x[1];
    const Scalar x13 = x11 * x12;
    const Scalar x14 = x7 * x8;
    const Scalar x15 = (1.0 / 4.0) * x14;
    const Scalar x16 = x12 * x3;
    const Scalar x17 = x14 * x4;
    const Scalar x18 = (9.0 / 4.0) * x[0];
    const Scalar x19 = x18 * x3;
    const Scalar x20 = x4 * x8;
    const Scalar x21 = x9 * x[1];
    const Scalar x22 = x4 * x7;
    const Scalar x23 = (9.0 / 4.0) * x22;
    const Scalar x24 = (81.0 / 4.0) * x[0];
    const Scalar x25 = x0 * x24;
    out[0] = x10 * x3;
    out[1] = -x10 * x11;
    out[2] = x13 * x15;
    out[3] = -x15 * x16;
    out[4] = -x17 * x19;
    out[5] = x19 * x20 * x6;
    out[6] = x18 * x21;
    out[7] = -x13 * x23;
    out[8] = x14 * x19 * x[1];
    out[9] = -x16 * x18 * x8;
    out[10] = -9.0 / 4.0 * x0 * x21;
    out[11] = x16 * x23;
    out[12] = x17 * x25 * x[1];
    out[13] = -x12 * x20 * x25;
    out[14] = -x22 * x24 * x3 * x[1];
    out[15] = x16 * x24 * x4;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 3 * x[0];
    const Scalar x2 = x0 * x1;
    const Scalar x3 = x1 - 2;
    const Scalar x4 = x3 * x[0];
    const Scalar x5 = x0 * x3;
    const Scalar x6 = x2 + x4 + x5;
    const Scalar x7 = x[1] - 1;
    const Scalar x8 = 3 * x[1];
    const Scalar x9 = x8 - 2;
    const Scalar x10 = x9 * x[1];
    const Scalar x11 = x10 * x7;
    const Scalar x12 = x1 - 1;
    const Scalar x13 = x12 * x[0];
    const Scalar x14 = x0 * x12;
    const Scalar x15 = x13 + x14 + x2;
    const Scalar x16 = x8 - 1;
    const Scalar x17 = x16 * x[1];
    const Scalar x18 = x17 * x7;
    const Scalar x19 = (1.0 / 9.0) * x16;
    const Scalar x20 = x10 * x19;
    const Scalar x21 = x7 * x9;
    const Scalar x22 = x19 * x21;
    const Scalar x23 = x12 * x3;
    const Scalar x24 = x1 * x12 + x1 * x3 + x23;
    const Scalar x25 = (1.0 / 9.0) * x24;
    const Scalar x26 = (1.0 / 81.0) * x16;
    const Scalar x27 = x10 * x26;
    const Scalar x28 = 3 * x14 + x23 + 3 * x5;
    const Scalar x29 = (1.0 / 9.0) * x28;
    const Scalar x30 = x21 * x26;
    const Scalar x31 = x7 * x8;
    const Scalar x32 = x10 + x21 + x31;
    const Scalar x33 = x0 * x32;
    const Scalar x34 = x16 * x7;
    const Scalar x35 = x17 + x31 + x34;
    const Scalar x36 = x0 * x35;
    const Scalar x37 = (1.0 / 9.0) * x12;
    const Scalar x38 = x37 * x4;
    const Scalar x39 = x37 * x5;
    const Scalar x40 = (1.0 / 9.0) * x0;
    const Scalar x41 = x16 * x9;
    const Scalar x42 = x16 * x8 + x41 + x8 * x9;
    const Scalar x43 = x4 * x42;
    const Scalar x44 = x13 * x40;
    const Scalar x45 = (1.0 / 81.0) * x12;
    const Scalar x46 = x45 * x5;
    const Scalar x47 = 3 * x21 + 3 * x34 + x41;
    const Scalar x48 = x4 * x47;
    out[0] = (81.0 / 4.0) * coeffs[0] * x28 * x30 -
             81.0 / 4.0 * coeffs[10] * x11 * x29 +
             (81.0 / 4.0) * coeffs[11] * x18 * x29 +
             (81.0 / 4.0) * coeffs[12] * x11 * x6 -
             81.0 / 4.0 * coeffs[13] * x11 * x15 -
             81.0 / 4.0 * coeffs[14] * x18 * x6 +
             (81.0 / 4.0) * coeffs[15] * x15 * x18 -
             81.0 / 4.0 * coeffs[1] * x24 * x30 +
             (81.0 / 4.0) * coeffs[2] * x24 * x27 -
             81.0 / 4.0 * coeffs[3] * x27 * x28 -
             81.0 / 4.0 * coeffs[4] * x22 * x6 +
             (81.0 / 4.0) * coeffs[5] * x15 * x22 +
             (81.0 / 4.0) * coeffs[6] * x11 * x25 -
             81.0 / 4.0 * coeffs[7] * x18 * x25 +
             (81.0 / 4.0) * coeffs[8] * x20 * x6 -
             81.0 / 4.0 * coeffs[9] * x15 * x20;
    out[1] = (81.0 / 4.0) * coeffs[0] * x46 * x47 -
             81.0 / 4.0 * coeffs[10] * x32 * x39 +
             (81.0 / 4.0) * coeffs[11] * x35 * x39 +
             (81.0 / 4.0) * coeffs[12] * x33 * x4 -
             81.0 / 4.0 * coeffs[13] * x13 * x33 -
             81.0 / 4.0 * coeffs[14] * x36 * x4 +
             (81.0 / 4.0) * coeffs[15] * x13 * x36 -
             81.0 / 4.0 * coeffs[1] * x45 * x48 +
             (81.0 / 4.0) * coeffs[2] * x43 * x45 -
             81.0 / 4.0 * coeffs[3] * x42 * x46 -
             81.0 / 4.0 * coeffs[4] * x40 * x48 +
             (81.0 / 4.0) * coeffs[5] * x44 * x47 +
             (81.0 / 4.0) * coeffs[6] * x32 * x38 -
             81.0 / 4.0 * coeffs[7] * x35 * x38 +
             (81.0 / 4.0) * coeffs[8] * x40 * x43 -
             81.0 / 4.0 * coeffs[9] * x42 * x44;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[2]) {
    const Scalar x0 = 3 * x[1];
    const Scalar x1 = x0 - 2;
    const Scalar x2 = x0 - 1;
    const Scalar x3 = x1 * x2;
    const Scalar x4 = (1.0 / 4.0) * x3;
    const Scalar x5 = x[1] - 1;
    const Scalar x6 = x[0] - 1;
    const Scalar x7 = 3 * x[0];
    const Scalar x8 = x7 - 2;
    const Scalar x9 = x6 * x8;
    const Scalar x10 = x7 - 1;
    const Scalar x11 = x10 * x6;
    const Scalar x12 = x10 * x8;
    const Scalar x13 = 3 * x11 + x12 + 3 * x9;
    const Scalar x14 = x13 * x5;
    const Scalar x15 = (1.0 / 4.0) * x12;
    const Scalar x16 = x1 * x5;
    const Scalar x17 = x2 * x5;
    const Scalar x18 = 3 * x16 + 3 * x17 + x3;
    const Scalar x19 = x18 * x6;
    const Scalar x20 = x10 * x7 + x12 + x7 * x8;
    const Scalar x21 = x20 * x4;
    const Scalar x22 = x15 * x[0];
    const Scalar x23 = x0 * x1 + x0 * x2 + x3;
    const Scalar x24 = x23 * x6;
    const Scalar x25 = x6 * x7;
    const Scalar x26 = x8 * x[0];
    const Scalar x27 = x25 + x26 + x9;
    const Scalar x28 = (9.0 / 4.0) * x5;
    const Scalar x29 = x28 * x3;
    const Scalar x30 = (9.0 / 4.0) * x19;
    const Scalar x31 = x10 * x[0];
    const Scalar x32 = x11 + x25 + x31;
    const Scalar x33 = x1 * x[1];
    const Scalar x34 = x20 * x28;
    const Scalar x35 = x0 * x5;
    const Scalar x36 = x16 + x33 + x35;
    const Scalar x37 = (9.0 / 4.0) * x12;
    const Scalar x38 = x37 * x[0];
    const Scalar x39 = x2 * x[1];
    const Scalar x40 = x17 + x35 + x39;
    const Scalar x41 = (9.0 / 4.0) * x3 * x[1];
    const Scalar x42 = (9.0 / 4.0) * x24;
    const Scalar x43 = (9.0 / 4.0) * x14;
    const Scalar x44 = x37 * x6;
    const Scalar x45 = (81.0 / 4.0) * x5;
    const Scalar x46 = x33 * x45;
    const Scalar x47 = (81.0 / 4.0) * x6;
    const Scalar x48 = x36 * x47;
    const Scalar x49 = x39 * x45;
    const Scalar x50 = x40 * x47;
    out[0][0] = x14 * x4;
    out[0][1] = x15 * x19;
    out[1][0] = -x21 * x5;
    out[1][1] = -x18 * x22;
    out[2][0] = x21 * x[1];
    out[2][1] = x22 * x23;
    out[3][0] = -x13 * x4 * x[1];
    out[3][1] = -x15 * x24;
    out[4][0] = -x27 * x29;
    out[4][1] = -x26 * x30;
    out[5][0] = x29 * x32;
    out[5][1] = x30 * x31;
    out[6][0] = x33 * x34;
    out[6][1] = x36 * x38;
    out[7][0] = -x34 * x39;
    out[7][1] = -x38 * x40;
    out[8][0] = x27 * x41;
    out[8][1] = x26 * x42;
    out[9][0] = -x32 * x41;
    out[9][1] = -x31 * x42;
    out[10][0] = -x33 * x43;
    out[10][1] = -x36 * x44;
    out[11][0] = x39 * x43;
    out[11][1] = x40 * x44;
    out[12][0] = x27 * x46;
    out[12][1] = x26 * x48;
    out[13][0] = -x32 * x46;
    out[13][1] = -x31 * x48;
    out[14][0] = -x27 * x49;
    out[14][1] = -x26 * x50;
    out[15][0] = x32 * x49;
    out[15][1] = x31 * x50;
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
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(3) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(3) / order;
    out[2][1] = static_cast<Scalar>(3) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(3) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(2) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[6][0] = static_cast<Scalar>(3) / order;
    out[6][1] = static_cast<Scalar>(1) / order;
    out[7][0] = static_cast<Scalar>(3) / order;
    out[7][1] = static_cast<Scalar>(2) / order;
    out[8][0] = static_cast<Scalar>(1) / order;
    out[8][1] = static_cast<Scalar>(3) / order;
    out[9][0] = static_cast<Scalar>(2) / order;
    out[9][1] = static_cast<Scalar>(3) / order;
    out[10][0] = static_cast<Scalar>(0) / order;
    out[10][1] = static_cast<Scalar>(1) / order;
    out[11][0] = static_cast<Scalar>(0) / order;
    out[11][1] = static_cast<Scalar>(2) / order;
    out[12][0] = static_cast<Scalar>(1) / order;
    out[12][1] = static_cast<Scalar>(1) / order;
    out[13][0] = static_cast<Scalar>(2) / order;
    out[13][1] = static_cast<Scalar>(1) / order;
    out[14][0] = static_cast<Scalar>(1) / order;
    out[14][1] = static_cast<Scalar>(2) / order;
    out[15][0] = static_cast<Scalar>(2) / order;
    out[15][1] = static_cast<Scalar>(2) / order;
  }
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      break;
    case 1:
      out[0] = 3;
      out[1] = 0;
      break;
    case 2:
      out[0] = 3;
      out[1] = 3;
      break;
    case 3:
      out[0] = 0;
      out[1] = 3;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 0;
      break;
    case 6:
      out[0] = 3;
      out[1] = 1;
      break;
    case 7:
      out[0] = 3;
      out[1] = 2;
      break;
    case 8:
      out[0] = 1;
      out[1] = 3;
      break;
    case 9:
      out[0] = 2;
      out[1] = 3;
      break;
    case 10:
      out[0] = 0;
      out[1] = 1;
      break;
    case 11:
      out[0] = 0;
      out[1] = 2;
      break;
    case 12:
      out[0] = 1;
      out[1] = 1;
      break;
    case 13:
      out[0] = 2;
      out[1] = 1;
      break;
    case 14:
      out[0] = 1;
      out[1] = 2;
      break;
    case 15:
      out[0] = 2;
      out[1] = 2;
      break;
    }
  }

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
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
      idxs[2] = 4;
      idxs[3] = 5;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 7;
      break;
    case 2:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 8;
      idxs[3] = 9;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 10;
      idxs[3] = 11;
      break;
    default:
      break;
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

template <> struct BasisLagrange<mesh::RefElTetra, 1> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 4;
  static constexpr dim_t num_interpolation_nodes = 4;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    return -coeffs[0] * (x[0] + x[1] + x[2] - 1) + coeffs[1] * x[0] +
           coeffs[2] * x[1] + coeffs[3] * x[2];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    out[0] = -x[0] - x[1] - x[2] + 1;
    out[1] = x[0];
    out[2] = x[1];
    out[3] = x[2];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    out[0] = -coeffs[0] + coeffs[1];
    out[1] = -coeffs[0] + coeffs[2];
    out[2] = -coeffs[0] + coeffs[3];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[3]) {
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

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolation_nodes(Scalar (*out)[3]) {
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
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
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

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
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

  static NUMERIC_HOST_DEVICE void
  subelement_node_idxs(dim_t subelement, dim_t *idxs,
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

template <> struct BasisLagrange<mesh::RefElTetra, 2> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 10;
  static constexpr dim_t num_interpolation_nodes = 10;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    const Scalar x0 = 2 * x[0];
    const Scalar x1 = 2 * x[1];
    const Scalar x2 = x0 - 1;
    const Scalar x3 = 2 * x[2];
    const Scalar x4 = x[0] + x[1] + x[2] - 1;
    return coeffs[0] * x4 * (x1 + x2 + x3) + coeffs[1] * x2 * x[0] +
           coeffs[2] * x[1] * (x1 - 1) + coeffs[3] * x[2] * (x3 - 1) -
           2 * coeffs[4] * x0 * x4 + 2 * coeffs[5] * x0 * x[1] -
           2 * coeffs[6] * x1 * x4 - 2 * coeffs[7] * x3 * x4 +
           2 * coeffs[8] * x0 * x[2] + 2 * coeffs[9] * x1 * x[2];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    const Scalar x0 = x[0] + x[1] + x[2] - 1;
    const Scalar x1 = 2 * x[1];
    const Scalar x2 = 2 * x[2];
    const Scalar x3 = 2 * x[0] - 1;
    const Scalar x4 = 4 * x[0];
    const Scalar x5 = 4 * x0;
    out[0] = x0 * (x1 + x2 + x3);
    out[1] = x3 * x[0];
    out[2] = x[1] * (x1 - 1);
    out[3] = x[2] * (x2 - 1);
    out[4] = -x0 * x4;
    out[5] = x4 * x[1];
    out[6] = -x5 * x[1];
    out[7] = -x5 * x[2];
    out[8] = x4 * x[2];
    out[9] = 4 * x[1] * x[2];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x[2] - 1;
    const Scalar x2 = 4 * x[1];
    const Scalar x3 = -coeffs[6] * x2;
    const Scalar x4 = 4 * x[2];
    const Scalar x5 = coeffs[0] * (x0 + x2 + x4 - 3);
    const Scalar x6 = -coeffs[7] * x4 + x5;
    const Scalar x7 = -coeffs[4] * x0;
    out[0] = coeffs[1] * (x0 - 1) - 4 * coeffs[4] * (x1 + 2 * x[0] + x[1]) +
             coeffs[5] * x2 + coeffs[8] * x4 + x3 + x6;
    out[1] = coeffs[2] * (x2 - 1) + coeffs[5] * x0 -
             4 * coeffs[6] * (x1 + x[0] + 2 * x[1]) + coeffs[9] * x4 + x6 + x7;
    out[2] = coeffs[3] * (x4 - 1) -
             4 * coeffs[7] * (x[0] + x[1] + 2 * x[2] - 1) + coeffs[8] * x0 +
             coeffs[9] * x2 + x3 + x5 + x7;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[3]) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = 4 * x[1];
    const Scalar x2 = 4 * x[2];
    const Scalar x3 = x0 + x1 + x2 - 3;
    const Scalar x4 = x[2] - 1;
    const Scalar x5 = -x0;
    const Scalar x6 = -x1;
    const Scalar x7 = -x2;
    out[0][0] = x3;
    out[0][1] = x3;
    out[0][2] = x3;
    out[1][0] = x0 - 1;
    out[1][1] = 0;
    out[1][2] = 0;
    out[2][0] = 0;
    out[2][1] = x1 - 1;
    out[2][2] = 0;
    out[3][0] = 0;
    out[3][1] = 0;
    out[3][2] = x2 - 1;
    out[4][0] = -4 * x4 - 8 * x[0] - 4 * x[1];
    out[4][1] = x5;
    out[4][2] = x5;
    out[5][0] = x1;
    out[5][1] = x0;
    out[5][2] = 0;
    out[6][0] = x6;
    out[6][1] = -4 * x4 - 4 * x[0] - 8 * x[1];
    out[6][2] = x6;
    out[7][0] = x7;
    out[7][1] = x7;
    out[7][2] = 4 * (-x[0] - x[1] - 2 * x[2] + 1);
    out[8][0] = x2;
    out[8][1] = 0;
    out[8][2] = x0;
    out[9][0] = 0;
    out[9][1] = x2;
    out[9][2] = x1;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolation_nodes(Scalar (*out)[3]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[0][2] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(2) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[1][2] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(0) / order;
    out[2][1] = static_cast<Scalar>(2) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(0) / order;
    out[3][2] = static_cast<Scalar>(2) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[4][2] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(1) / order;
    out[5][1] = static_cast<Scalar>(1) / order;
    out[5][2] = static_cast<Scalar>(0) / order;
    out[6][0] = static_cast<Scalar>(0) / order;
    out[6][1] = static_cast<Scalar>(1) / order;
    out[6][2] = static_cast<Scalar>(0) / order;
    out[7][0] = static_cast<Scalar>(0) / order;
    out[7][1] = static_cast<Scalar>(0) / order;
    out[7][2] = static_cast<Scalar>(1) / order;
    out[8][0] = static_cast<Scalar>(1) / order;
    out[8][1] = static_cast<Scalar>(0) / order;
    out[8][2] = static_cast<Scalar>(1) / order;
    out[9][0] = static_cast<Scalar>(0) / order;
    out[9][1] = static_cast<Scalar>(1) / order;
    out[9][2] = static_cast<Scalar>(1) / order;
  }
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
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
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 6:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 7:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 8:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 9:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    }
  }

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
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
      idxs[2] = 4;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 5;
      break;
    case 2:
      idxs[0] = 2;
      idxs[1] = 0;
      idxs[2] = 6;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 7;
      break;
    case 4:
      idxs[0] = 1;
      idxs[1] = 3;
      idxs[2] = 8;
      break;
    case 5:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 9;
      break;
    default:
      break;
    }
  }

  static NUMERIC_HOST_DEVICE void
  subelement_node_idxs(dim_t subelement, dim_t *idxs,
                       meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      idxs[3] = 4;
      idxs[4] = 8;
      idxs[5] = 7;
      break;
    case 1:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 1;
      idxs[3] = 9;
      idxs[4] = 8;
      idxs[5] = 5;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      idxs[3] = 7;
      idxs[4] = 9;
      idxs[5] = 6;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      idxs[3] = 6;
      idxs[4] = 5;
      idxs[5] = 4;
      break;
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

template <> struct BasisLagrange<mesh::RefElTetra, 3> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 20;
  static constexpr dim_t num_interpolation_nodes = 20;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    const Scalar x0 = 9 * x[1];
    const Scalar x1 = x[0] * x[2];
    const Scalar x2 = 3 * x[0];
    const Scalar x3 = x2 - 1;
    const Scalar x4 = (3.0 / 2.0) * x3;
    const Scalar x5 = 3 * x[2];
    const Scalar x6 = x5 - 1;
    const Scalar x7 = 3 * x[1];
    const Scalar x8 = x[1] * (x7 - 1);
    const Scalar x9 = (3.0 / 2.0) * x8;
    const Scalar x10 = x6 * x[2];
    const Scalar x11 = (3.0 / 2.0) * x[1];
    const Scalar x12 = x[0] + x[1] + x[2] - 1;
    const Scalar x13 = x12 * x[2];
    const Scalar x14 = 9 * x13;
    const Scalar x15 = x12 * x[0];
    const Scalar x16 = (3.0 / 2.0) * x13;
    const Scalar x17 = x2 - 2;
    const Scalar x18 = x17 + x5 + x7;
    const Scalar x19 = x12 * x18;
    return -1.0 / 2.0 * coeffs[0] * x19 * (x2 + x6 + x7) +
           3 * coeffs[10] * x16 * x18 - 3 * coeffs[11] * x16 * x6 +
           3 * coeffs[12] * x1 * x4 + (9.0 / 2.0) * coeffs[13] * x1 * x6 +
           3 * coeffs[14] * x9 * x[2] + 3 * coeffs[15] * x10 * x11 -
           3 * coeffs[16] * x14 * x[0] + 3 * coeffs[17] * x0 * x1 -
           3 * coeffs[18] * x14 * x[1] - 3 * coeffs[19] * x0 * x15 +
           (1.0 / 2.0) * coeffs[1] * x17 * x3 * x[0] +
           (1.0 / 2.0) * coeffs[2] * x8 * (x7 - 2) +
           (1.0 / 2.0) * coeffs[3] * x10 * (x5 - 2) +
           (9.0 / 2.0) * coeffs[4] * x15 * x18 - 3 * coeffs[5] * x15 * x4 +
           3 * coeffs[6] * x4 * x[0] * x[1] + 3 * coeffs[7] * x9 * x[0] -
           3 * coeffs[8] * x12 * x9 + 3 * coeffs[9] * x11 * x19;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
    const Scalar x0 = 3 * x[0];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = 3 * x[1];
    const Scalar x3 = 3 * x[2];
    const Scalar x4 = x2 + x3;
    const Scalar x5 = x[0] + x[1] + x[2] - 1;
    const Scalar x6 = x0 - 2;
    const Scalar x7 = x5 * (x4 + x6);
    const Scalar x8 = x1 * x[0];
    const Scalar x9 = x[1] * (x2 - 1);
    const Scalar x10 = x[2] * (x3 - 1);
    const Scalar x11 = (9.0 / 2.0) * x[0];
    const Scalar x12 = (9.0 / 2.0) * x8;
    const Scalar x13 = (9.0 / 2.0) * x5;
    const Scalar x14 = (9.0 / 2.0) * x7;
    const Scalar x15 = 27 * x[0] * x[2];
    const Scalar x16 = 27 * x5 * x[1];
    out[0] = -1.0 / 2.0 * x7 * (x1 + x4);
    out[1] = (1.0 / 2.0) * x6 * x8;
    out[2] = (1.0 / 2.0) * x9 * (x2 - 2);
    out[3] = (1.0 / 2.0) * x10 * (x3 - 2);
    out[4] = x11 * x7;
    out[5] = -x12 * x5;
    out[6] = x12 * x[1];
    out[7] = x11 * x9;
    out[8] = -x13 * x9;
    out[9] = x14 * x[1];
    out[10] = x14 * x[2];
    out[11] = -x10 * x13;
    out[12] = x12 * x[2];
    out[13] = x10 * x11;
    out[14] = (9.0 / 2.0) * x9 * x[2];
    out[15] = (9.0 / 2.0) * x10 * x[1];
    out[16] = -x15 * x5;
    out[17] = x15 * x[1];
    out[18] = -x16 * x[2];
    out[19] = -x16 * x[0];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
    const Scalar x0 = 3 * x[0];
    const Scalar x1 = x0 - 2;
    const Scalar x2 = x0 - 1;
    const Scalar x3 = 3 * x[2];
    const Scalar x4 = 3 * x[1];
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
    const Scalar x16 = 9 * x[2];
    const Scalar x17 = coeffs[16] * x16;
    const Scalar x18 = 9 * coeffs[19];
    const Scalar x19 = x18 * x[1];
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
    const Scalar x29 = (3.0 / 2.0) * coeffs[10];
    const Scalar x30 = x22 * x[2];
    const Scalar x31 = (3.0 / 2.0) * coeffs[11];
    const Scalar x32 = x25 + x28 * x29 * x[2] + x30 * x31;
    const Scalar x33 = x23 * x[1];
    const Scalar x34 = (3.0 / 2.0) * coeffs[8];
    const Scalar x35 = (3.0 / 2.0) * coeffs[9];
    const Scalar x36 = x28 * x35 * x[1] + x33 * x34;
    const Scalar x37 = x4 - 2;
    const Scalar x38 = x[0] - 1;
    const Scalar x39 = x38 + 2 * x[1] + x[2];
    const Scalar x40 = x26 - 1;
    const Scalar x41 = x12 * x28 * x[0] + x13 * x14;
    const Scalar x42 = x3 - 2;
    const Scalar x43 = 9 * x38 + 9 * x[1] + 18 * x[2];
    const Scalar x44 = x27 - 1;
    out[0] = (9.0 / 2.0) * coeffs[12] * x21 * x[2] +
             (9.0 / 2.0) * coeffs[13] * x22 * x[2] +
             27 * coeffs[17] * x[1] * x[2] - 3 * coeffs[18] * x16 * x[1] +
             (1.0 / 2.0) * coeffs[1] * (x0 * x1 + x0 * x2 + x1 * x2) +
             (9.0 / 2.0) * coeffs[6] * x21 * x[1] +
             (9.0 / 2.0) * coeffs[7] * x23 * x[1] -
             3 * x12 * (x0 * x9 + x11 + x6 * x[0]) -
             3 * x14 * (x0 * x8 + x13 + x2 * x8) - 3 * x15 * x17 -
             3 * x15 * x19 - 3 * x32 - 3 * x36;
    out[1] = (9.0 / 2.0) * coeffs[14] * x40 * x[2] +
             (9.0 / 2.0) * coeffs[15] * x22 * x[2] +
             27 * coeffs[17] * x[0] * x[2] - 3 * coeffs[18] * x16 * x39 +
             (1.0 / 2.0) * coeffs[2] * (x23 * x37 + x23 * x4 + x37 * x4) +
             (9.0 / 2.0) * coeffs[6] * x2 * x[0] +
             (9.0 / 2.0) * coeffs[7] * x40 * x[0] - 3 * x17 * x[0] -
             3 * x18 * x39 * x[0] - 3 * x32 -
             3 * x34 * (x23 * x8 + x33 + x4 * x8) -
             3 * x35 * (x11 + x4 * x9 + x6 * x[1]) - 3 * x41;
    out[2] = (9.0 / 2.0) * coeffs[12] * x2 * x[0] +
             (9.0 / 2.0) * coeffs[13] * x44 * x[0] +
             (9.0 / 2.0) * coeffs[14] * x23 * x[1] +
             (9.0 / 2.0) * coeffs[15] * x44 * x[1] -
             3 * coeffs[16] * x43 * x[0] + 27 * coeffs[17] * x[0] * x[1] -
             3 * coeffs[18] * x43 * x[1] +
             (1.0 / 2.0) * coeffs[3] * (x22 * x3 + x22 * x42 + x3 * x42) -
             3 * x19 * x[0] - 3 * x25 - 3 * x29 * (x11 + x3 * x9 + x6 * x[2]) -
             3 * x31 * (x22 * x8 + x3 * x8 + x30) - 3 * x36 - 3 * x41;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[3]) {
    const Scalar x0 = x[1] + x[2] - 1;
    const Scalar x1 = x0 + x[0];
    const Scalar x2 = 3 * x[0];
    const Scalar x3 = x2 - 2;
    const Scalar x4 = 3 * x[1];
    const Scalar x5 = 3 * x[2];
    const Scalar x6 = x4 + x5;
    const Scalar x7 = x3 + x6;
    const Scalar x8 = x1 * x7;
    const Scalar x9 = x2 - 1;
    const Scalar x10 = x6 + x9;
    const Scalar x11 =
        -3.0 / 2.0 * x1 * x10 - 1.0 / 2.0 * x10 * x7 - 3.0 / 2.0 * x8;
    const Scalar x12 = x4 - 2;
    const Scalar x13 = x4 - 1;
    const Scalar x14 = x5 - 2;
    const Scalar x15 = x5 - 1;
    const Scalar x16 = x1 * x2;
    const Scalar x17 = 6 * x[0];
    const Scalar x18 = 6 * x[1];
    const Scalar x19 = 6 * x[2];
    const Scalar x20 = x17 + x18 + x19 - 5;
    const Scalar x21 = (9.0 / 2.0) * x[0];
    const Scalar x22 = x20 * x21;
    const Scalar x23 = x9 * x[0];
    const Scalar x24 = (9.0 / 2.0) * x23;
    const Scalar x25 = -x24;
    const Scalar x26 = x17 - 1;
    const Scalar x27 = (9.0 / 2.0) * x[1];
    const Scalar x28 = x13 * x[1];
    const Scalar x29 = (9.0 / 2.0) * x28;
    const Scalar x30 = x18 - 1;
    const Scalar x31 = -x29;
    const Scalar x32 = x1 * x4;
    const Scalar x33 = x20 * x27;
    const Scalar x34 = (9.0 / 2.0) * x[2];
    const Scalar x35 = x20 * x34;
    const Scalar x36 = x1 * x5;
    const Scalar x37 = x15 * x[2];
    const Scalar x38 = (9.0 / 2.0) * x37;
    const Scalar x39 = -x38;
    const Scalar x40 = x19 - 1;
    const Scalar x41 = x0 + 2 * x[0];
    const Scalar x42 = 27 * x[2];
    const Scalar x43 = x42 * x[0];
    const Scalar x44 = x[0] - 1;
    const Scalar x45 = x44 + x[1] + 2 * x[2];
    const Scalar x46 = 27 * x[0];
    const Scalar x47 = x42 * x[1];
    const Scalar x48 = x46 * x[1];
    const Scalar x49 = x44 + 2 * x[1] + x[2];
    const Scalar x50 = 27 * x[1];
    out[0][0] = x11;
    out[0][1] = x11;
    out[0][2] = x11;
    out[1][0] =
        (1.0 / 2.0) * x2 * x3 + (1.0 / 2.0) * x2 * x9 + (1.0 / 2.0) * x3 * x9;
    out[1][1] = 0;
    out[1][2] = 0;
    out[2][0] = 0;
    out[2][1] = (1.0 / 2.0) * x12 * x13 + (1.0 / 2.0) * x12 * x4 +
                (1.0 / 2.0) * x13 * x4;
    out[2][2] = 0;
    out[3][0] = 0;
    out[3][1] = 0;
    out[3][2] = (1.0 / 2.0) * x14 * x15 + (1.0 / 2.0) * x14 * x5 +
                (1.0 / 2.0) * x15 * x5;
    out[4][0] = (9.0 / 2.0) * x16 + (9.0 / 2.0) * x7 * x[0] + (9.0 / 2.0) * x8;
    out[4][1] = x22;
    out[4][2] = x22;
    out[5][0] = -9.0 / 2.0 * x1 * x9 - 9.0 / 2.0 * x16 - 9.0 / 2.0 * x23;
    out[5][1] = x25;
    out[5][2] = x25;
    out[6][0] = x26 * x27;
    out[6][1] = x24;
    out[6][2] = 0;
    out[7][0] = x29;
    out[7][1] = x21 * x30;
    out[7][2] = 0;
    out[8][0] = x31;
    out[8][1] = -9.0 / 2.0 * x1 * x13 - 9.0 / 2.0 * x28 - 9.0 / 2.0 * x32;
    out[8][2] = x31;
    out[9][0] = x33;
    out[9][1] = (9.0 / 2.0) * x32 + (9.0 / 2.0) * x7 * x[1] + (9.0 / 2.0) * x8;
    out[9][2] = x33;
    out[10][0] = x35;
    out[10][1] = x35;
    out[10][2] = (9.0 / 2.0) * x36 + (9.0 / 2.0) * x7 * x[2] + (9.0 / 2.0) * x8;
    out[11][0] = x39;
    out[11][1] = x39;
    out[11][2] = -9.0 / 2.0 * x1 * x15 - 9.0 / 2.0 * x36 - 9.0 / 2.0 * x37;
    out[12][0] = x26 * x34;
    out[12][1] = 0;
    out[12][2] = x24;
    out[13][0] = x38;
    out[13][1] = 0;
    out[13][2] = x21 * x40;
    out[14][0] = 0;
    out[14][1] = x30 * x34;
    out[14][2] = x29;
    out[15][0] = 0;
    out[15][1] = x38;
    out[15][2] = x27 * x40;
    out[16][0] = -x41 * x42;
    out[16][1] = -x43;
    out[16][2] = -x45 * x46;
    out[17][0] = x47;
    out[17][1] = x43;
    out[17][2] = x48;
    out[18][0] = -x47;
    out[18][1] = -x42 * x49;
    out[18][2] = -x45 * x50;
    out[19][0] = -x41 * x50;
    out[19][1] = -x46 * x49;
    out[19][2] = -x48;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolation_nodes(Scalar (*out)[3]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[0][2] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(3) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[1][2] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(0) / order;
    out[2][1] = static_cast<Scalar>(3) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(0) / order;
    out[3][2] = static_cast<Scalar>(3) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[4][2] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(2) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[5][2] = static_cast<Scalar>(0) / order;
    out[6][0] = static_cast<Scalar>(2) / order;
    out[6][1] = static_cast<Scalar>(1) / order;
    out[6][2] = static_cast<Scalar>(0) / order;
    out[7][0] = static_cast<Scalar>(1) / order;
    out[7][1] = static_cast<Scalar>(2) / order;
    out[7][2] = static_cast<Scalar>(0) / order;
    out[8][0] = static_cast<Scalar>(0) / order;
    out[8][1] = static_cast<Scalar>(2) / order;
    out[8][2] = static_cast<Scalar>(0) / order;
    out[9][0] = static_cast<Scalar>(0) / order;
    out[9][1] = static_cast<Scalar>(1) / order;
    out[9][2] = static_cast<Scalar>(0) / order;
    out[10][0] = static_cast<Scalar>(0) / order;
    out[10][1] = static_cast<Scalar>(0) / order;
    out[10][2] = static_cast<Scalar>(1) / order;
    out[11][0] = static_cast<Scalar>(0) / order;
    out[11][1] = static_cast<Scalar>(0) / order;
    out[11][2] = static_cast<Scalar>(2) / order;
    out[12][0] = static_cast<Scalar>(2) / order;
    out[12][1] = static_cast<Scalar>(0) / order;
    out[12][2] = static_cast<Scalar>(1) / order;
    out[13][0] = static_cast<Scalar>(1) / order;
    out[13][1] = static_cast<Scalar>(0) / order;
    out[13][2] = static_cast<Scalar>(2) / order;
    out[14][0] = static_cast<Scalar>(0) / order;
    out[14][1] = static_cast<Scalar>(2) / order;
    out[14][2] = static_cast<Scalar>(1) / order;
    out[15][0] = static_cast<Scalar>(0) / order;
    out[15][1] = static_cast<Scalar>(1) / order;
    out[15][2] = static_cast<Scalar>(2) / order;
    out[16][0] = static_cast<Scalar>(1) / order;
    out[16][1] = static_cast<Scalar>(0) / order;
    out[16][2] = static_cast<Scalar>(1) / order;
    out[17][0] = static_cast<Scalar>(1) / order;
    out[17][1] = static_cast<Scalar>(1) / order;
    out[17][2] = static_cast<Scalar>(1) / order;
    out[18][0] = static_cast<Scalar>(0) / order;
    out[18][1] = static_cast<Scalar>(1) / order;
    out[18][2] = static_cast<Scalar>(1) / order;
    out[19][0] = static_cast<Scalar>(1) / order;
    out[19][1] = static_cast<Scalar>(1) / order;
    out[19][2] = static_cast<Scalar>(0) / order;
  }
  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
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
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 7:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 8:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 9:
      out[0] = 0;
      out[1] = 1;
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
      out[2] = 2;
      break;
    case 12:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 13:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 14:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 15:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 16:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 17:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    case 18:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 19:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    }
  }

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
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
      idxs[2] = 4;
      idxs[3] = 5;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 7;
      break;
    case 2:
      idxs[0] = 2;
      idxs[1] = 0;
      idxs[2] = 8;
      idxs[3] = 9;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 10;
      idxs[3] = 11;
      break;
    case 4:
      idxs[0] = 1;
      idxs[1] = 3;
      idxs[2] = 12;
      idxs[3] = 13;
      break;
    case 5:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 14;
      idxs[3] = 15;
      break;
    default:
      break;
    }
  }

  static NUMERIC_HOST_DEVICE void
  subelement_node_idxs(dim_t subelement, dim_t *idxs,
                       meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      idxs[3] = 4;
      idxs[4] = 5;
      idxs[5] = 12;
      idxs[6] = 13;
      idxs[7] = 11;
      idxs[8] = 10;
      idxs[9] = 16;
      break;
    case 1:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 1;
      idxs[3] = 14;
      idxs[4] = 15;
      idxs[5] = 13;
      idxs[6] = 12;
      idxs[7] = 6;
      idxs[8] = 7;
      idxs[9] = 17;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      idxs[3] = 10;
      idxs[4] = 11;
      idxs[5] = 15;
      idxs[6] = 14;
      idxs[7] = 8;
      idxs[8] = 9;
      idxs[9] = 18;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      idxs[3] = 9;
      idxs[4] = 8;
      idxs[5] = 7;
      idxs[6] = 6;
      idxs[7] = 5;
      idxs[8] = 4;
      idxs[9] = 19;
      break;
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

template <> struct BasisLagrange<mesh::RefElCube, 1> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 8;
  static constexpr dim_t num_interpolation_nodes = 8;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
    const Scalar x0 = x[2] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x[0] - 1;
    return -coeffs[0] * x0 * x1 * x2 + coeffs[1] * x0 * x1 * x[0] -
           coeffs[2] * x0 * x[0] * x[1] + coeffs[3] * x0 * x2 * x[1] +
           coeffs[4] * x1 * x2 * x[2] - coeffs[5] * x1 * x[0] * x[2] +
           coeffs[6] * x[0] * x[1] * x[2] - coeffs[7] * x2 * x[1] * x[2];
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
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
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
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
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[3]) {
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

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolation_nodes(Scalar (*out)[3]) {
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
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
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

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
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

template <> struct BasisLagrange<mesh::RefElCube, 2> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 27;
  static constexpr dim_t num_interpolation_nodes = 27;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
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
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
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
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
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
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[3]) {
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

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolation_nodes(Scalar (*out)[3]) {
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
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
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

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
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

template <> struct BasisLagrange<mesh::RefElCube, 3> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 64;
  static constexpr dim_t num_interpolation_nodes = 64;

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE Scalar eval(const Scalar *x,
                                                   const Scalar *coeffs) {
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
  static constexpr NUMERIC_HOST_DEVICE void eval_basis(const Scalar *x,
                                                       Scalar *out) {
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
  static constexpr NUMERIC_HOST_DEVICE void
  grad(const Scalar *x, const Scalar *coeffs, Scalar *out) {
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
  static constexpr NUMERIC_HOST_DEVICE void grad_basis(const Scalar *x,
                                                       Scalar (*out)[3]) {
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

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void
  interpolation_nodes(Scalar (*out)[3]) {
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
  static constexpr NUMERIC_HOST_DEVICE void
  interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr NUMERIC_HOST_DEVICE void node_idxs(dim_t i, dim_t *out) {
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

  static NUMERIC_HOST_DEVICE dim_t node_idx_under_group_action(
      dim_t i, const DihedralGroupElement<ref_el_t::num_nodes> &action) {
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
