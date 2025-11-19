#ifndef NUMERIC_MATH_FES_BASIS_H1_HPP_
#define NUMERIC_MATH_FES_BASIS_H1_HPP_

#include <numeric/math/basis_lagrange.hpp>
#include <numeric/math/fes/basis_base.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math::fes {

/**
 * @brief H1-conforming finite element basis using Lagrange polynomials of a
 * given order.
 *
 * This basis class implements logic specific to H1 spaces, where continuity
 * across elements is enforced. It subtracts subelement contributions to compute
 * the number of interior degrees of freedom.
 *
 * @tparam Order Polynomial order of the basis functions.
 */
template <dim_t Order> struct BasisH1 : public BasisBase<BasisH1<Order>> {
  using super = BasisBase<BasisH1<Order>>;

  /// Type alias for the reference element Lagrange basis
  template <typename RefEl> using element_basis_t = BasisLagrange<RefEl, Order>;

  /// Polynomial order of the basis
  static constexpr dim_t order = Order;

  /**
   * @brief Returns number of interior basis functions for a reference point.
   */
  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_interior_basis_functions(meta::type_tag<mesh::RefElPoint>) {
    return element_basis_t<mesh::RefElPoint>::num_basis_functions;
  }

  /**
   * @brief Returns number of interior basis functions for a reference segment.
   * Excludes vertex DoFs shared with RefElPoint.
   */
  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_interior_basis_functions(meta::type_tag<mesh::RefElSegment>) {
    return element_basis_t<mesh::RefElSegment>::num_basis_functions -
           2 * num_interior_basis_functions<mesh::RefElPoint>();
  }

  /**
   * @brief Returns number of interior basis functions for a reference triangle.
   * Excludes DoFs shared with vertices and edges.
   */
  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_interior_basis_functions(meta::type_tag<mesh::RefElTria>) {
    return element_basis_t<mesh::RefElTria>::num_basis_functions -
           3 * num_interior_basis_functions<mesh::RefElPoint>() -
           3 * num_interior_basis_functions<mesh::RefElSegment>();
  }

  /**
   * @brief Returns number of interior basis functions for a reference
   * quadrilateral. Excludes DoFs on vertices and edges.
   */
  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_interior_basis_functions(meta::type_tag<mesh::RefElQuad>) {
    return element_basis_t<mesh::RefElQuad>::num_basis_functions -
           4 * num_interior_basis_functions<mesh::RefElPoint>() -
           4 * num_interior_basis_functions<mesh::RefElSegment>();
  }

  /**
   * @brief Returns number of interior basis functions for a reference
   * tetrahedron. Excludes contributions from vertices, edges, and faces.
   */
  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_interior_basis_functions(meta::type_tag<mesh::RefElTetra>) {
    return element_basis_t<mesh::RefElTetra>::num_basis_functions -
           4 * num_interior_basis_functions<mesh::RefElPoint>() -
           6 * num_interior_basis_functions<mesh::RefElSegment>() -
           4 * num_interior_basis_functions<mesh::RefElTria>();
  }

  /**
   * @brief Returns number of interior basis functions for a reference cube.
   * Excludes contributions from vertices, edges, and faces.
   */
  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_interior_basis_functions(meta::type_tag<mesh::RefElCube>) {
    return element_basis_t<mesh::RefElCube>::num_basis_functions -
           8 * num_interior_basis_functions<mesh::RefElPoint>() -
           12 * num_interior_basis_functions<mesh::RefElSegment>() -
           6 * num_interior_basis_functions<mesh::RefElQuad>();
  }

  /**
   * @brief Returns total number of basis functions on a given reference
   * element.
   *
   * @tparam RefEl The reference element type.
   * @param tag Type tag of the reference element.
   */
  template <typename RefEl>
  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_basis_functions(meta::type_tag<RefEl>) {
    return element_basis_t<RefEl>::num_basis_functions;
  }

  /**
   * @brief Returns the number of basis functions associated with a child
   * subelement.
   *
   * This allows splitting basis functions by sub-entity (vertex, edge, face,
   * etc.).
   *
   * @tparam Parent Parent reference element.
   * @tparam Child Child reference element (subentity type).
   * @param subelement Index of the subelement (not used).
   * @param tag_parent Type tag for the parent element.
   * @param tag_child Type tag for the child element.
   */
  template <typename Parent, typename Child>
  static constexpr NUMERIC_HOST_DEVICE dim_t num_basis_functions(
      dim_t /*subelement*/, meta::type_tag<Parent>, meta::type_tag<Child>) {
    return num_interior_basis_functions(meta::type_tag<Child>{});
  }

  /**
   * @brief Evaluate basis functions at a given point on a reference element.
   *
   * @tparam Scalar Scalar type.
   * @tparam RefEl Reference element type.
   * @param out Output array for function values.
   * @param x Point in reference coordinates.
   * @param tag Type tag of the reference element.
   */
  template <typename Scalar, typename RefEl>
  static NUMERIC_HOST_DEVICE void eval(Scalar *out, const Scalar *x,
                                       meta::type_tag<RefEl>) {
    element_basis_t<RefEl>::eval_basis(x, out);
  }

  /**
   * @brief Evaluate gradients of basis functions at a point.
   *
   * @tparam Scalar Scalar type.
   * @tparam RefEl Reference element type.
   * @param out Output gradient array.
   * @param x Point in reference coordinates.
   * @param tag Type tag of the reference element.
   */
  template <typename Scalar, typename RefEl>
  static NUMERIC_HOST_DEVICE void
  gradient(Scalar (*out)[RefEl::dim == 0 ? dim_t(1) : RefEl::dim],
           const Scalar *x, meta::type_tag<RefEl> tag) {
    element_basis_t<RefEl>::grad_basis(x, out);
  }

  using super::eval;
  using super::gradient;
  using super::interior_dof_idx_under_permutation;
  using super::num_basis_functions;
  using super::num_interior_basis_functions;
  using super::total_num_basis_functions;
};

} // namespace numeric::math::fes

#endif
