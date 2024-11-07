#ifndef NUMERIC_MATH_BASIS_LAGRANGE_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_HPP_

#include <numeric/math/functions.hpp>
#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>

namespace numeric::math {

namespace detail {

template <typename Scalar>
constexpr Scalar lagrange(const Scalar *xi, Scalar x, dim_t p, dim_t j) {
  // TODO: This implementation can be improved a lot!
  Scalar lj = 1;
  for (dim_t m = 0; m <= p; ++m) {
    if (m == j) {
      continue;
    }
    lj *= (x - xi[m]) / (xi[j] - xi[m]);
  }
  return lj;
}

template <typename Scalar>
constexpr void lagrange(const Scalar *xi, Scalar x, dim_t p, Scalar *out) {
  // TODO: This implementation can be improved a lot!
  for (dim_t j = 0; j <= p; ++j) {
    out[j] = lagrange(xi, x, p, j);
  }
}

template <typename Scalar>
constexpr Scalar diff_lagrange(const Scalar *xi, Scalar x, dim_t p, dim_t j) {
  // TODO: This implementation can be improved a lot!
  Scalar dlj = 0;
  for (dim_t i = 0; i <= p; ++i) {
    if (i == j) {
      continue;
    }
    Scalar prod = 1;
    for (dim_t m = 0; m <= p; ++m) {
      if (m == i || m == j) {
        continue;
      }
      prod *= (x - xi[m]) / (xi[j] - xi[m]);
    }
    dlj += prod / (xi[j] - xi[i]);
  }
  return dlj;
}

template <typename Scalar>
constexpr void diff_lagrange(const Scalar *xi, Scalar x, dim_t p, Scalar *out) {
  // TODO: This implementation can be improved a lot!
  for (dim_t j = 0; j <= p; ++j) {
    out[j] = diff_lagrange(xi, x, p, j);
  }
}

} // namespace detail

template <typename RefEl, dim_t Order> struct BasisLagrange {
  using ref_el_t = RefEl;
  static constexpr dim_t order = Order;
  static_assert(
      !meta::is_same_v<RefEl, RefEl>,
      "Lagrangian Basis is not implemented for the given element and order");
};

template <dim_t Ord> struct BasisLagrange<mesh::RefElPoint, Ord> {
  using ref_el_t = mesh::RefElPoint;
  static constexpr dim_t order = Ord;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    return coeffs[0];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    // Nothing to do here
  }
};

template <dim_t Ord> struct BasisLagrange<mesh::RefElSegment, Ord> {
  using ref_el_t = mesh::RefElSegment;
  static constexpr dim_t order = Ord;
  static constexpr dim_t num_basis_functions = order + 1;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_t i, const Scalar *x) {
    Scalar xj[num_basis_functions];
    for (dim_t j = 0; j < num_basis_functions; ++j) {
      node(j, xj + j);
    }
    return detail::lagrange(xj, x[0], order, i);
  }

  template <typename Scalar>
  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    Scalar xj[num_basis_functions];
    for (dim_t j = 0; j < num_basis_functions; ++j) {
      node(j, xj + j);
    }
    out[0] = detail::diff_lagrange(xj, x[0], order, i);
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    Scalar xj[num_basis_functions];
    for (dim_t j = 0; j < num_basis_functions; ++j) {
      node(j, xj + j);
    }
    Scalar bj[num_basis_functions];
    detail::lagrange(xj, x[0], order, bj);
    Scalar result = 0;
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      result += coeffs[i] * bj[i];
    }
    return result;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    Scalar xj[num_basis_functions];
    for (dim_t j = 0; j < num_basis_functions; ++j) {
      node(j, xj + j);
    }
    Scalar dbj[num_basis_functions];
    detail::diff_lagrange(xj, x[0], order, dbj);
    out[0] = 0;
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      out[0] += coeffs[i] * dbj[i];
    }
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[1];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    if (i == 0) {
      out[0] = 0;
    } else if (i == 1) {
      out[0] = order;
    } else {
      out[0] = i - 1;
    }
  }
};

/**
 * These basis functions are taken from P. Silvester, High-Order Polynomial
 * Triangular Finite Elements for Potential Problems, Int. J. Engng Sci. Vol. 7,
 * pp 849-861
 */
template <dim_t Ord> struct BasisLagrange<mesh::RefElTria, Ord> {
  using ref_el_t = mesh::RefElTria;
  static constexpr dim_t order = Ord;
  static constexpr dim_t num_basis_functions = (order + 1) * (order + 2) / 2;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_t i, const Scalar *x) {
    const Scalar l0 = 1 - x[0] - x[1];
    const Scalar l1 = x[0];
    const Scalar l2 = x[1];
    dim_t ijk[2];
    node_idxs(i, ijk);
    ijk[2] = order - ijk[0] - ijk[1];
    return P(ijk[0], l0) * P(ijk[1], l1) * P(ijk[2], l2);
  }

  template <typename Scalar>
  static constexpr Scalar grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    out[0] = 0;
    out[1] = 0;
    const Scalar l0 = 1 - x[0] - x[1];
    const Scalar l1 = x[0];
    const Scalar l2 = x[1];
    dim_t ijk[2];
    node_idxs(i, ijk);
    ijk[2] = order - ijk[0] - ijk[1];
    const Scalar Pi = P(ijk[0], l0);
    const Scalar Pj = P(ijk[1], l1);
    const Scalar Pk = P(ijk[2], l2);
    const Scalar dPi = dPdz(ijk[0], l0);
    const Scalar dPj = dPdz(ijk[1], l1);
    const Scalar dPk = dPdz(ijk[2], l2);
    out[0] += -1 * dPi * Pj * Pk;
    out[0] += Pi * dPj * Pk;
    out[1] += -1 * dPi * Pj * Pk;
    out[1] += Pi * Pj * dPk;
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    Scalar result = 0;
    for (dim_t idx = 0; idx < num_basis_functions; ++idx) {
      result += coeffs[idx] * eval_basis(idx, x);
    }
    return result;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    out[0] = 0;
    out[1] = 0;
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      Scalar dbdx[2];
      grad_basis(i, x, dbdx);
      out[0] += coeffs[i] * dbdx[0];
      out[1] += coeffs[i] * dbdx[1];
    }
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    if (i == 0) {
      out[0] = 0;
      out[1] = 0;
    } else if (i == 1) {
      out[0] = order;
      out[1] = 0;
    } else if (i == 2) {
      out[0] = 0;
      out[1] = order;
    } else if (i < order + 2) {
      dim_t lin_idx;
      BasisLagrange<mesh::RefElSegment, order>::node_idxs(i - 1, &lin_idx);
      out[0] = lin_idx;
      out[1] = 0;
    } else if (i < 2 * order + 1) {
      dim_t lin_idx;
      BasisLagrange<mesh::RefElSegment, order>::node_idxs(i - order - 2,
                                                          &lin_idx);
      out[0] = order - lin_idx;
      out[1] = lin_idx;
    } else if (i < 3 * order) {
      dim_t lin_idx;
      BasisLagrange<mesh::RefElSegment, order>::node_idxs(i - 2 * order - 1,
                                                          &lin_idx);
      out[0] = 0;
      out[1] = order - lin_idx;
    } else {
      dim_t ix = i - 3 * order;
      dim_t iy = 0;
      while (ix + iy > order - 2) {
        ix -= order - 2 - iy;
        ++iy;
      }
      out[0] = 1 + ix;
      out[1] = 1 + iy;
    }
  }

  template <typename Scalar> static constexpr Scalar P(dim_t m, Scalar z) {
    constexpr dim_t N = order;
    Scalar result = 1;
    for (dim_t i = 1; i <= m; ++i) {
      result *= (N * z - i + 1) / i;
    }
    return result;
  }

  template <typename Scalar> static constexpr Scalar dPdz(dim_t m, Scalar z) {
    // TODO: This can be optimized a lot!
    constexpr dim_t N = order;
    Scalar result = 0;
    for (dim_t i = 1; i <= m; ++i) {
      Scalar prod = 0;
      for (dim_t j = 1; j <= m; ++j) {
        if (i == j) {
          prod *= static_cast<Scalar>(N) / j;
        } else {
          prod *= (N * z - j + 1) / j;
        }
      }
      result += prod;
    }
    return result;
  }
};

template <dim_t Ord> struct BasisLagrange<mesh::RefElQuad, Ord> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = Ord;
  static constexpr dim_t num_basis_functions = (order + 1) * (order + 1);

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_t i, const Scalar *x) {
    Scalar xj[order + 1];
    for (dim_t j = 0; j <= order; ++j) {
      xj[j] = idx_to_node_1d<Scalar>(j);
    }
    dim_t idxs[2];
    node_idxs(i, idxs);
    const Scalar lix = detail::lagrange(xj, x[0], order, idxs[0]);
    const Scalar liy = detail::lagrange(xj, x[1], order, idxs[1]);
    return lix * liy;
  }

  template <typename Scalar>
  static constexpr void grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    Scalar xj[order + 1];
    for (dim_t j = 0; j <= order; ++j) {
      xj[j] = idx_to_node_1d<Scalar>(j);
    }
    dim_t idxs[2];
    node_idxs(i, idxs);
    const Scalar lix = detail::lagrange(xj, x[0], order, idxs[0]);
    const Scalar dlix = detail::diff_lagrange(xj, x[0], order, idxs[0]);
    const Scalar liy = detail::lagrange(xj, x[1], order, idxs[1]);
    const Scalar dliy = detail::diff_lagrange(xj, x[1], order, idxs[1]);
    out[0] = dlix * liy;
    out[1] = lix * dliy;
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    Scalar xj[order + 1];
    for (dim_t j = 0; j <= order; ++j) {
      xj[j] = idx_to_node_1d<Scalar>(j);
    }
    Scalar lix[order + 1];
    Scalar liy[order + 1];
    detail::lagrange(xj, x[0], order, lix);
    detail::lagrange(xj, x[1], order, liy);
    Scalar value = 0;
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      dim_t idxs[2];
      node_idxs(i, idxs);
      value += coeffs[i] * lix[idxs[0]] * liy[idxs[1]];
    }
    return value;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    Scalar xj[order + 1];
    for (dim_t j = 0; j <= order; ++j) {
      xj[j] = idx_to_node_1d<Scalar>(j);
    }
    Scalar lix[order + 1];
    Scalar dlix[order + 1];
    Scalar liy[order + 1];
    Scalar dliy[order + 1];
    detail::lagrange(xj, x[0], order, lix);
    detail::diff_lagrange(xj, x[0], order, dlix);
    detail::lagrange(xj, x[1], order, liy);
    detail::diff_lagrange(xj, x[1], order, dliy);
    Scalar value = 0;
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      dim_t idxs[2];
      node_idxs(i, idxs);
      out[0] += coeffs[i] * dlix[idxs[0]] * liy[idxs[1]];
      out[1] += coeffs[i] * lix[idxs[0]] * dliy[idxs[1]];
    }
  }

  template <typename Scalar> static constexpr Scalar idx_to_node_1d(dim_t i) {
    return static_cast<Scalar>(i) / order;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = idx_to_node_1d<Scalar>(idxs[0]);
    out[1] = idx_to_node_1d<Scalar>(idxs[1]);
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
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
    } else if (i < order + 3) {
      dim_t idx_segment[1];
      BasisLagrange<mesh::RefElSegment, Ord>::node_idxs(i - 2, idx_segment);
      out[0] = idx_segment[0];
      out[1] = 0;
    } else if (i < 2 * order - 2) {
      dim_t idx_segment[1];
      BasisLagrange<mesh::RefElSegment, Ord>::node_idxs(i - order - 3,
                                                        idx_segment);
      out[0] = order;
      out[1] = idx_segment[0];
    } else if (i - 4 < 3 * order - 1) {
      dim_t idx_segment[1];
      BasisLagrange<mesh::RefElSegment, Ord>::node_idxs(i - 2 * order - 4,
                                                        idx_segment);
      out[0] = order - 1 - idx_segment[0];
      out[1] = order;
    } else if (i < 4 * order) {
      dim_t idx_segment[1];
      BasisLagrange<mesh::RefElSegment, Ord>::node_idxs(i - 3 * order - 5,
                                                        idx_segment);
      out[0] = 0;
      out[1] = order - 1 - idx_segment[0];
    } else {
      out[0] = (i - 4 * order) % (order - 1);
      out[1] = (i - 4 * order) / (order - 1);
    }
  }
};

/**
 * These basis functions are taken from P. Silvester, Tetrahedral Polynomial
 * Finite Elements for the Helmholtz Equation, International Journal for
 * Numerical Methods in Engineering, Vol 4, pp. 405-413
 */
template <dim_t Ord> struct BasisLagrange<mesh::RefElTetra, Ord> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = Ord;
  static constexpr dim_t num_basis_functions =
      (order + 1) * (order + 2) * (order + 3) / 6;

  template <typename Scalar>
  static constexpr Scalar eval_basis(dim_t i, const Scalar *x) {
    const Scalar l0 = 1 - x[0] - x[1] - x[2];
    const Scalar l1 = x[0];
    const Scalar l2 = x[1];
    const Scalar l3 = x[2];
    dim_t ijkl[3];
    node_idxs(i, ijkl);
    ijkl[3] = order - ijkl[0] - ijkl[1] - ijkl[2];
    return P(ijkl[0], l0) * P(ijkl[1], l1) * P(ijkl[2], l2) * P(ijkl[3], l3);
  }

  template <typename Scalar>
  static constexpr Scalar grad_basis(dim_t i, const Scalar *x, Scalar *out) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    const Scalar l0 = 1 - x[0] - x[1] - x[2];
    const Scalar l1 = x[0];
    const Scalar l2 = x[1];
    const Scalar l3 = x[2];
    dim_t ijkl[3];
    node_idxs(i, ijkl);
    ijkl[3] = order - ijkl[0] - ijkl[1] - ijkl[2];
    const Scalar Pi = P(ijkl[0], l0);
    const Scalar Pj = P(ijkl[1], l1);
    const Scalar Pk = P(ijkl[2], l2);
    const Scalar Pl = P(ijkl[3], l3);
    const Scalar dPi = dPdz(ijkl[0], l0);
    const Scalar dPj = dPdz(ijkl[1], l1);
    const Scalar dPk = dPdz(ijkl[2], l2);
    const Scalar dPl = dPdz(ijkl[3], l3);
    out[0] += -1 * dPi * Pj * Pk * Pl;
    out[0] += Pi * dPj * Pk * Pl;
    out[1] += -1 * dPi * Pj * Pk * Pl;
    out[1] += Pi * Pj * dPk * Pl;
    out[2] += -1 * dPi * Pj * Pk * Pl;
    out[2] += Pi * Pj * Pk * dPl;
  }

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    Scalar result = 0;
    for (dim_t idx = 0; idx < num_basis_functions; ++idx) {
      result += coeffs[idx] * eval_basis(idx, x);
    }
    return result;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      Scalar dbdx[3];
      grad_basis(i, x, dbdx);
      out[0] += coeffs[i] * dbdx[0];
      out[1] += coeffs[i] * dbdx[1];
      out[2] += coeffs[i] * dbdx[2];
    }
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    if (i == 0) {
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
    } else if (i == 1) {
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
    } else if (i == 2) {
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
    } else if (i == 3) {
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
    } else if (i < order + 3) {
      dim_t idx_segment;
      BasisLagrange<mesh::RefElSegment, Ord>::node_idxs(i - 2, &idx_segment);
      out[0] = idx_segment;
      out[1] = 0;
      out[2] = 0;
    } else if (i < 2 * order + 2) {
      dim_t idx_segment;
      BasisLagrange<mesh::RefElSegment, Ord>::node_idxs(i - order - 1,
                                                        &idx_segment);
      out[0] = 0;
      out[1] = idx_segment;
      out[2] = 0;
    } else if (i < 3 * order + 1) {
      dim_t idx_segment;
      BasisLagrange<mesh::RefElSegment, Ord>::node_idxs(i - 2 * order,
                                                        &idx_segment);
      out[0] = 0;
      out[1] = 0;
      out[2] = idx_segment;
    } else if (i < 4 * order) {
      dim_t idx_segment;
      BasisLagrange<mesh::RefElSegment, Ord>::node_idxs(i - 3 * order + 1,
                                                        &idx_segment);
      out[0] = order - idx_segment;
      out[1] = idx_segment;
      out[2] = 0;
    } else if (i < 5 * order - 1) {
      dim_t idx_segment;
      BasisLagrange<mesh::RefElSegment, Ord>::node_idxs(i - 4 * order + 2,
                                                        &idx_segment);
      out[0] = 0;
      out[1] = order - idx_segment;
      out[2] = idx_segment;
    } else if (i < 6 * order - 2) {
      dim_t idx_segment;
      BasisLagrange<mesh::RefElSegment, Ord>::node_idxs(i - 5 * order + 3,
                                                        &idx_segment);
      out[0] = idx_segment;
      out[1] = 0;
      out[2] = order - idx_segment;
    } else if (i < 6 * order - 2 + (order - 1) * order / 2) {
      dim_t idx_tria[2];
      BasisLagrange<mesh::RefElTria, Ord>::node_idxs(i - 3 * order - 10,
                                                     idx_tria);
      out[0] = 0;
      out[1] = idx_tria[0];
      out[2] = idx_tria[1];
    } else if (i < 6 * order - 2 + 2 * (order - 1) * order / 2) {
      dim_t idx_tria[2];
      BasisLagrange<mesh::RefElTria, Ord>::node_idxs(
          i - (order - 1) * order / 2 - 3 * order - 10, idx_tria);
      out[0] = idx_tria[0];
      out[1] = 0;
      out[2] = idx_tria[1];
    } else if (i < 6 * order - 2 + 3 * (order - 1) * order / 2) {
      dim_t idx_tria[2];
      BasisLagrange<mesh::RefElTria, Ord>::node_idxs(
          i - 2 * (order - 1) * order / 2 - 3 * order - 10, idx_tria);
      out[0] = idx_tria[0];
      out[1] = idx_tria[1];
      out[2] = 0;
    } else if (i < 6 * order - 2 + 4 * (order - 1) * order / 2) {
      dim_t idx_tria[2];
      BasisLagrange<mesh::RefElTria, Ord>::node_idxs(
          i - 3 * (order - 1) * order / 2 - 3 * order - 10, idx_tria);
      out[0] = idx_tria[0];
      out[1] = idx_tria[1];
      out[2] = order - idx_tria[0] - idx_tria[1];
    } else {
      i -= 4 + 6 * (order - 1) + 4 * (order - 1) * order / 2;
      dim_t ix = 0;
      dim_t iy = 0;
      dim_t iz = 0;
      // TODO: Implement Interior
    }
  }

  template <typename Scalar> static constexpr Scalar P(dim_t m, Scalar z) {
    constexpr dim_t M = order;
    Scalar result = 1;
    for (dim_t i = 1; i <= m; ++i) {
      result *= (M * z - i + 1) / i;
    }
    return result;
  }

  template <typename Scalar> static constexpr Scalar dPdz(dim_t m, Scalar z) {
    // TODO: This can be optimized a lot!
    constexpr dim_t M = order;
    Scalar result = 0;
    for (dim_t i = 1; i <= m; ++i) {
      Scalar prod = 0;
      for (dim_t j = 1; j <= m; ++j) {
        if (i == j) {
          prod *= static_cast<Scalar>(M) / j;
        } else {
          prod *= (M * z - j + 1) / j;
        }
      }
      result += prod;
    }
    return result;
  }
};

} // namespace numeric::math

#endif
