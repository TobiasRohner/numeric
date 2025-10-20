#ifndef NUMERIC_MATH_BASIS_LAGRANGE_SPECIALIZATION_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_SPECIALIZATION_HPP_

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

template <> struct BasisLagrange<mesh::RefElTria, 1> {
  using ref_el_t = mesh::RefElTria;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 3;
  static constexpr dim_t num_interpolation_nodes = 3;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    return -coeffs[0] * (x[0] + x[1] - 1) + coeffs[1] * x[0] + coeffs[2] * x[1];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    out[0] = -x[0] - x[1] + 1;
    out[1] = x[0];
    out[2] = x[1];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    out[0] = -coeffs[0] + coeffs[1];
    out[1] = -coeffs[0] + coeffs[2];
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[2]) {
    out[0][0] = -1;
    out[0][1] = -1;
    out[1][0] = 1;
    out[1][1] = 0;
    out[2][0] = 0;
    out[2][1] = 1;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[2]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(1) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(0) / order;
    out[2][1] = static_cast<Scalar>(1) / order;
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

template <> struct BasisLagrange<mesh::RefElTria, 2> {
  using ref_el_t = mesh::RefElTria;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 6;
  static constexpr dim_t num_interpolation_nodes = 6;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 2 * x[0];
    const Scalar x1 = x[0] + x[1] - 1;
    const Scalar x2 = 2 * x[1];
    const Scalar x3 = x0 - 1;
    return coeffs[0] * x1 * (x2 + x3) + coeffs[1] * x3 * x[0] +
           coeffs[2] * x[1] * (x2 - 1) - 2 * coeffs[3] * x0 * x1 +
           2 * coeffs[4] * x0 * x[1] - 2 * coeffs[5] * x1 * x2;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
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
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 4 * x[1];
    const Scalar x1 = 4 * x[0];
    const Scalar x2 = coeffs[0] * (x0 + x1 - 3);
    out[0] = coeffs[1] * (x1 - 1) - 4 * coeffs[3] * (2 * x[0] + x[1] - 1) +
             coeffs[4] * x0 - coeffs[5] * x0 + x2;
    out[1] = coeffs[2] * (x0 - 1) - coeffs[3] * x1 + coeffs[4] * x1 -
             4 * coeffs[5] * (x[0] + 2 * x[1] - 1) + x2;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[2]) {
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

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[2]) {
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

template <> struct BasisLagrange<mesh::RefElQuad, 1> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 4;
  static constexpr dim_t num_interpolation_nodes = 4;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[0] - 1;
    return coeffs[0] * x0 * x1 - coeffs[1] * x0 * x[0] +
           coeffs[2] * x[0] * x[1] - coeffs[3] * x1 * x[1];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    out[0] = x0 * x1;
    out[1] = -x1 * x[0];
    out[2] = x[0] * x[1];
    out[3] = -x0 * x[1];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[0] - 1;
    out[0] =
        coeffs[0] * x0 - coeffs[1] * x0 + coeffs[2] * x[1] - coeffs[3] * x[1];
    out[1] =
        coeffs[0] * x1 - coeffs[1] * x[0] + coeffs[2] * x[0] - coeffs[3] * x1;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[2]) {
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

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[2]) {
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

template <> struct BasisLagrange<mesh::RefElQuad, 2> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 9;
  static constexpr dim_t num_interpolation_nodes = 9;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
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
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
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
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
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
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[2]) {
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

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[2]) {
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

template <> struct BasisLagrange<mesh::RefElTetra, 1> {
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

template <> struct BasisLagrange<mesh::RefElTetra, 2> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 10;
  static constexpr dim_t num_interpolation_nodes = 10;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
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
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
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
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
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
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
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

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
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

} // namespace numeric::math

#endif
