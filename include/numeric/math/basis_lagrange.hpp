#ifndef NUMERIC_MATH_BASIS_LAGRANGE_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_HPP_

#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>

namespace numeric::math {

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
  static constexpr dim_t num_basis_functions = 1;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    return coeffs[0];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    out[0] = 1;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    // Nothing to do here
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[1]) {
    // Nothing to do here
  }
};

} // namespace numeric::math

#include <numeric/math/basis_lagrange_cube.hpp>
#include <numeric/math/basis_lagrange_quad.hpp>
#include <numeric/math/basis_lagrange_segment.hpp>
#include <numeric/math/basis_lagrange_tetra.hpp>
#include <numeric/math/basis_lagrange_tria.hpp>

#include <numeric/math/basis_lagrange_specialization.hpp>

#endif
