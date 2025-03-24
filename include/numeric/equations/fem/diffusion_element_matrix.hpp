#ifndef NUMERIC_EQUATIONS_FEM_DIFFUSION_ELEMENT_MATRIX_HPP_
#define NUMERIC_EQUATIONS_FEM_DIFFUSION_ELEMENT_MATRIX_HPP_

#include <numeric/math/fes/fe_space.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view.hpp>
#include <numeric/mesh/elements.hpp>

namespace numeric::equations::fem {

/**
 * @brief A class representing the diffusion element matrix for FEM.
 *
 * @tparam Scalar The scalar type used for computations.
 * @tparam Basis The basis function type used in the finite element space.
 * @tparam Element The element type defining the shape and properties of the
 * finite element.
 */
template <typename Scalar, typename Basis, typename Element>
class DiffusionElementMatrix {
public:
  using scalar_t = Scalar; ///< The scalar type used for computations.
  using basis_t =
      Basis; ///< The basis function type used in the finite element space.
  using element_t = Element; ///< The element type defining the shape and
                             ///< properties of the finite element.
  using ref_el_t = typename element_t::ref_el_t;
  template <typename OtherElement>
  using rebind = DiffusionElementMatrix<Scalar, Basis, OtherElement>;

  static constexpr dim_t num_nodes = element_t::num_nodes;
  static constexpr dim_t num_basis_functions =
      basis_t::template num_basis_functions<ref_el_t>();

  /**
   * @brief Constructs a DiffusionElementMatrix with given quadrature points and
   * weights.
   *
   * @param qr_points The quadrature points for integration.
   * @param qr_weights The quadrature weights for integration.
   */
  DiffusionElementMatrix(const memory::ArrayConstView<scalar_t, 2> &qr_points,
                         const memory::ArrayConstView<scalar_t, 1> &qr_weights)
      : qr_points_(qr_points), qr_weights_(qr_weights) {}

  DiffusionElementMatrix(const DiffusionElementMatrix &) = default;
  DiffusionElementMatrix(DiffusionElementMatrix &&) = default;
  DiffusionElementMatrix &operator=(const DiffusionElementMatrix &) = default;
  DiffusionElementMatrix &operator=(DiffusionElementMatrix &&) = default;

  /**
   * @brief Computes the work size required for applying the diffusion element
   * matrix.
   *
   * @param world_dim The dimension of the world space.
   * @return The work size required.
   */
  static constexpr dim_t apply_work_size(dim_t world_dim) {
    const dim_t jinvt_work_size =
        element_t::template jacobian_inverse_gramian_work_size<scalar_t>(
            world_dim);
    const dim_t ie_work_size =
        element_t::template integration_element_work_size<scalar_t>(world_dim);
    const dim_t jinvt_size = sizeof(Scalar) * world_dim * element_t::dim;
    const dim_t grad_trans_size = world_dim * num_basis_functions;
    return math::max(ie_work_size,
                     jinvt_size + math::max(grad_trans_size, jinvt_work_size));
  }

  /**
   * @brief Applies the diffusion element matrix to the given basis function
   * coefficients.
   *
   * @param nodes The nodes of the element.
   * @param coeffs The coefficients for the basis functions.
   * @param world_dim The dimension of the world space.
   * @param out The output array to store the results.
   * @param work The workspace for intermediate computations.
   */
  void apply(const scalar_t (*nodes)[num_nodes],
             const scalar_t (&coeffs)[num_basis_functions], dim_t world_dim,
             scalar_t (&out)[num_basis_functions], void *work) const {
    if constexpr (element_t::is_affine) {
      apply_affine(nodes, coeffs, world_dim, out, work);
    } else {
      apply_non_affine(nodes, coeffs, world_dim, out, work);
    }
  }

private:
  memory::ArrayConstView<scalar_t, 2> qr_points_;
  memory::ArrayConstView<scalar_t, 1> qr_weights_;

  void apply_local(const scalar_t (*nodes)[num_nodes],
                   const scalar_t (&coeffs)[num_basis_functions],
                   dim_t world_dim, scalar_t (&out)[num_basis_functions],
                   dim_t point, scalar_t ie,
                   const scalar_t (*JinvT)[element_t::dim],
                   scalar_t (&grads)[num_basis_functions][element_t::dim],
                   scalar_t (&grad)[element_t::dim], scalar_t *grad_trans,
                   const scalar_t (&x)[element_t::dim]) const {
    basis_t::template gradient<ref_el_t>(grads, x);
    for (dim_t j = 0; j < element_t::dim; ++j) {
      grad[j] = 0;
      for (dim_t k = 0; k < num_basis_functions; ++k) {
        grad[j] += coeffs[k] * grads[k][j];
      }
    }
    for (dim_t j = 0; j < world_dim; ++j) {
      grad_trans[j] = 0;
      for (dim_t k = 0; k < element_t::dim; ++k) {
        grad_trans[j] += JinvT[j][k] * grad[k];
      }
    }
    for (dim_t j = 0; j < element_t::dim; ++j) {
      grad[j] = 0;
      for (dim_t k = 0; k < world_dim; ++k) {
        grad[j] += JinvT[k][j] * grad_trans[k];
      }
    }
    for (dim_t j = 0; j < num_basis_functions; ++j) {
      for (dim_t k = 0; k < element_t::dim; ++k) {
        out[j] += ie * qr_weights_(point) * grads[j][k] * grad[k];
      }
    }
  }

  void apply_affine(const scalar_t (*nodes)[num_nodes],
                    const scalar_t (&coeffs)[num_basis_functions],
                    dim_t world_dim, scalar_t (&out)[num_basis_functions],
                    void *work) const {
    const dim_t jinvt_work_size =
        element_t::template jacobian_inverse_gramian_work_size<scalar_t>(
            world_dim);
    const dim_t jinvt_size = world_dim * element_t::dim;
    scalar_t(*JinvT)[element_t::dim] =
        static_cast<scalar_t(*)[element_t::dim]>(work);
    scalar_t grads[num_basis_functions][element_t::dim];
    scalar_t grad[element_t::dim];
    scalar_t *grad_trans = static_cast<scalar_t *>(work) + jinvt_size;
    const scalar_t x[element_t::dim] = {0};
    // TODO: Compute these two quantities in one function call
    const scalar_t ie =
        element_t::integration_element(nodes, x, world_dim, work);
    element_t::template jacobian_inverse_gramian<scalar_t>(
        nodes, x, JinvT, world_dim, static_cast<scalar_t *>(work) + jinvt_size);
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      out[i] = 0;
    }
    const dim_t num_points = qr_weights_.shape(0);
    for (dim_t i = 0; i < num_points; ++i) {
      apply_local(nodes, coeffs, world_dim, out, i, ie, JinvT, grads, grad,
                  grad_trans, x);
    }
  }

  void apply_non_affine(const scalar_t (*nodes)[num_nodes],
                        const scalar_t (&coeffs)[num_basis_functions],
                        dim_t world_dim, scalar_t (&out)[num_basis_functions],
                        void *work) const {
    const dim_t jinvt_work_size =
        element_t::template jacobian_inverse_gramian_work_size<scalar_t>(
            world_dim);
    const dim_t jinvt_size = world_dim * element_t::dim;
    scalar_t(*JinvT)[element_t::dim] =
        static_cast<scalar_t(*)[element_t::dim]>(work);
    scalar_t grads[num_basis_functions][element_t::dim];
    scalar_t grad[element_t::dim];
    scalar_t *grad_trans = static_cast<scalar_t *>(work) + jinvt_size;
    scalar_t x[element_t::dim];
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      out[i] = 0;
    }
    const dim_t num_points = qr_weights_.shape(0);
    for (dim_t i = 0; i < num_points; ++i) {
      for (dim_t j = 0; j < element_t::dim; ++j) {
        x[j] = qr_points_(i, j);
      }
      // TODO: Compute these two quantities in one function call
      const scalar_t ie =
          element_t::integration_element(nodes, x, world_dim, work);
      element_t::template jacobian_inverse_gramian<scalar_t>(
          nodes, x, JinvT, world_dim,
          static_cast<scalar_t *>(work) + jinvt_size);
      apply_local(nodes, coeffs, world_dim, out, i, ie, JinvT, grads, grad,
                  grad_trans, x);
    }
  }
};

template <typename Scalar, typename Basis>
struct DiffusionElementMatrixFactory {
  using scalar_t = Scalar;
  using basis_t = Basis;
  template <typename Element>
  using create = DiffusionElementMatrix<Scalar, Basis, Element>;
};

} // namespace numeric::equations::fem

#endif
