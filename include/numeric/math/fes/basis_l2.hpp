#ifndef NUMERIC_MATH_FES_BASIS_L2_HPP_
#define NUMERIC_MATH_FES_BASIS_L2_HPP_

#include <numeric/math/basis_lagrange.hpp>
#include <numeric/math/fes/basis_base.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math::fes {

/**
 * @brief L2 finite element basis using Lagrange polynomials of a given order.
 *
 * This class represents an L2 basis space built from Lagrange basis functions
 * on reference elements. It inherits from BasisBase and provides concrete
 * implementations of the interface for interior basis function counts,
 * evaluation, and gradients.
 *
 * @tparam Order Polynomial order of the Lagrange basis.
 */
template <dim_t Order> struct BasisL2 : public BasisBase<BasisL2<Order>> {
  using super = BasisBase<BasisL2<Order>>;

  /// Type alias for the Lagrange basis on a specific reference element
  template <typename RefEl> using element_basis_t = BasisLagrange<RefEl, Order>;

  /// Polynomial order of this basis
  static constexpr dim_t order = Order;

  /**
   * @brief Get the number of interior basis functions on a reference element.
   *
   * @tparam RefEl Reference element type.
   * @param tag Type tag for the reference element.
   * @return Number of basis functions defined on the interior of the reference
   * element.
   */
  template <typename RefEl>
  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_interior_basis_functions(meta::type_tag<RefEl>) {
    return element_basis_t<RefEl>::num_basis_functions;
  }

  /**
   * @brief Get the total number of basis functions on a reference element.
   *
   * Since L2 basis functions only have interior degrees of freedom, this is the
   * same as num_interior_basis_functions.
   *
   * @tparam RefEl Reference element type.
   * @param tt Type tag for the reference element.
   * @return Number of basis functions.
   */
  template <typename RefEl>
  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_basis_functions(meta::type_tag<RefEl> tt) {
    return num_interior_basis_functions(tt);
  }

  /**
   * @brief Evaluate basis functions at a point on a reference element.
   *
   * @tparam Scalar Scalar type.
   * @tparam RefEl Reference element type.
   * @param out Output array for basis function values.
   * @param x Input point in reference coordinates.
   * @param tag Type tag for the reference element.
   */
  template <typename Scalar, typename RefEl>
  static NUMERIC_HOST_DEVICE void eval(Scalar *out, const Scalar *x,
                                       meta::type_tag<RefEl>) {
    element_basis_t<RefEl>::eval_basis(x, out);
  }

  /**
   * @brief Evaluate gradients of basis functions at a point on a reference
   * element.
   *
   * @tparam Scalar Scalar type.
   * @tparam RefEl Reference element type.
   * @param out Output array for gradient values.
   * @param x Input point in reference coordinates.
   * @param tag Type tag for the reference element.
   */
  template <typename Scalar, typename RefEl>
  static NUMERIC_HOST_DEVICE void
  gradient(Scalar (*out)[RefEl::dim == 0 ? dim_t(1) : RefEl::dim],
           const Scalar *x, meta::type_tag<RefEl>) {
    element_basis_t<RefEl>::grad_basis(x, out);
  }

  using super::eval;
  using super::gradient;
  using super::interior_dof_idx_under_group_action;
  using super::num_basis_functions;
  using super::num_interior_basis_functions;
  using super::total_num_basis_functions;
};

} // namespace numeric::math::fes

#endif
