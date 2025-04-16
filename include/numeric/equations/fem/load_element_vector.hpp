#ifndef NUMERIC_EQUATIONS_FEM_LOAD_ELEMENT_VECTOR_HPP_
#define NUMERIC_EQUATIONS_FEM_LOAD_ELEMENT_VECTOR_HPP_

#include <numeric/math/fes/fe_space.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view.hpp>
#include <numeric/mesh/elements.hpp>

namespace numeric::equations::fem {

/**
 * @brief Computes the local element vector (load vector) for a finite element.
 *
 * This class evaluates the integral of a function against basis functions over
 * a reference element, using quadrature.
 *
 * @tparam Scalar Scalar type (e.g., float or double).
 * @tparam Basis Type of the basis functions.
 * @tparam Element Type of the element (e.g., triangle, tetrahedron).
 */
template <typename Scalar, typename Basis, typename Element>
class LoadElementVector {
public:
  using scalar_t = Scalar;                       ///< Scalar type
  using basis_t = Basis;                         ///< Basis function type
  using element_t = Element;                     ///< Mesh element type
  using ref_el_t = typename element_t::ref_el_t; ///< Reference element type
  using element_basis_t =
      basis_t::template element_basis_t<ref_el_t>; ///< Specific basis for this
                                                   ///< element

  /// Rebind this class to a different element type
  template <typename OtherElement>
  using rebind = LoadElementVector<Scalar, Basis, OtherElement>;

  static constexpr dim_t num_nodes =
      element_t::num_nodes; ///< Number of nodes in the element
  static constexpr dim_t num_basis_functions =
      basis_t::template num_basis_functions<ref_el_t>(); ///< Number of basis
                                                         ///< functions

  /**
   * @brief Constructor
   * @param qr_points Quadrature points in reference element.
   * @param qr_weights Corresponding quadrature weights.
   */
  LoadElementVector(const memory::ArrayConstView<scalar_t, 2> &qr_points,
                    const memory::ArrayConstView<scalar_t, 1> &qr_weights)
      : qr_points_(qr_points), qr_weights_(qr_weights) {}
  LoadElementVector(const LoadElementVector &) = default;
  LoadElementVector(LoadElementVector &&) = default;
  LoadElementVector &operator=(const LoadElementVector &) = default;
  LoadElementVector &operator=(LoadElementVector &&) = default;

  /**
   * @brief Computes temporary workspace size needed for apply().
   * @param world_dim Spatial dimension of the mesh (e.g., 2D or 3D).
   * @return Size in bytes of the workspace.
   */
  static constexpr dim_t apply_work_size(dim_t world_dim) {
    const dim_t global_work_size = world_dim * sizeof(scalar_t);
    const dim_t ie_work_size =
        element_t::template integration_element_work_size<scalar_t>(world_dim);
    return math::max(global_work_size, ie_work_size);
  }

  /**
   * @brief Applies the load vector computation to a given function.
   *
   * Computes:
   * \f$ out_j = \int_{\Omega_e} f(x) \phi_j(x) dx \f$
   * using quadrature for numerical integration.
   *
   * @tparam Func Type of the function f(x).
   * @param f Function to evaluate at quadrature points.
   * @param nodes Global coordinates of the element nodes.
   * @param world_dim Dimension of the space (e.g., 2 or 3).
   * @param out Output array for the local load vector.
   * @param work Temporary workspace.
   */
  template <typename Func>
  void apply(Func &&f, const scalar_t (*nodes)[num_nodes], dim_t world_dim,
             scalar_t (&out)[num_basis_functions], void *work) const {
    scalar_t local[element_t::dim];      // Coordinates in reference element
    scalar_t basis[num_basis_functions]; // Evaluated basis functions
    scalar_t *global = static_cast<scalar_t *>(work); // Global coordinates
    scalar_t *ie_work =
        static_cast<scalar_t *>(work); // Workspace for integration element

    // Initialize output vector to zero
    for (dim_t i = 0; i < num_basis_functions; ++i) {
      out[i] = 0;
    }

    const dim_t num_points = qr_weights_.shape(0);

    // Loop over all quadrature points
    for (dim_t i = 0; i < num_points; ++i) {
      // Extract local coordinates of quadrature point
      for (dim_t j = 0; j < element_t::dim; ++j) {
        local[j] = qr_points_(i, j);
      }

      // Compute integration element (Jacobian determinant * weight)
      const scalar_t ie =
          element_t::integration_element(nodes, local, world_dim, ie_work);
      const scalar_t weight = ie * qr_weights_(i);

      // Map local point to global coordinates
      element_t::local_to_global(nodes, local, global, world_dim);

      // Evaluate function at global quadrature point
      const scalar_t fval = f(global);

      // Evaluate basis functions at local point
      element_basis_t::eval_basis(local, basis);

      // Accumulate weighted value into each entry of output
      for (dim_t j = 0; j < num_basis_functions; ++j) {
        out[j] += weight * fval * basis[j];
      }
    }
  }

private:
  memory::ArrayConstView<scalar_t, 2> qr_points_;  ///< Quadrature points
  memory::ArrayConstView<scalar_t, 1> qr_weights_; ///< Quadrature weights
};

/**
 * @brief Factory for creating `LoadElementVector` instances for different
 * elements.
 */
template <typename Scalar, typename Basis> struct LoadElementVectorFactory {
  using scalar_t = Scalar;
  using basis_t = Basis;

  /// Create a LoadElementVector for a specific element type
  template <typename Element>
  using create = LoadElementVector<Scalar, Basis, Element>;
};

} // namespace numeric::equations::fem

#endif
