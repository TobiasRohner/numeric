#ifndef NUMERIC_MATH_FES_BASIS_BASE_HPP_
#define NUMERIC_MATH_FES_BASIS_BASE_HPP_

#include <numeric/config.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math::fes {

/**
 * @brief CRTP base class for finite element basis definitions.
 *
 * This class provides static interface methods that delegate to the
 * derived basis class (using the Curiously Recurring Template Pattern).
 * It supports querying and evaluating basis functions for a given element
 * or subelement.
 *
 * @tparam Derived The concrete basis implementation.
 */
template <typename Derived> struct BasisBase {
  /**
   * @brief Trait to detect if a Derived class provides subelement-specific
   * basis function counts.
   */
  template <typename Base, typename Element, typename Subelement>
  using has_subelement_num_basis_functions_t =
      decltype(Base::num_basis_functions(meta::declval<dim_t>(),
                                         meta::type_tag<Element>(),
                                         meta::type_tag<Subelement>()));

  /**
   * @brief Get the number of interior basis functions for a given element.
   *
   * @tparam Element The element type.
   * @return Number of interior basis functions.
   */
  template <typename Element>
  static constexpr NUMERIC_HOST_DEVICE dim_t num_interior_basis_functions() {
    return Derived::num_interior_basis_functions(meta::type_tag<Element>{});
  }

  /**
   * @brief Get the total number of basis functions for an element.
   *
   * Includes both interior and possibly boundary or subelement-associated
   * functions.
   *
   * @tparam Element The element type.
   * @return Number of basis functions.
   */
  template <typename Element>
  static constexpr NUMERIC_HOST_DEVICE dim_t num_basis_functions() {
    return Derived::num_basis_functions(meta::type_tag<Element>{});
  }

  /**
   * @brief Get the number of basis functions associated with a subelement.
   *
   * This will only return a non-zero value if the derived class defines the
   * corresponding overload.
   *
   * @tparam Element The parent element type.
   * @tparam Subelement The subelement type (e.g., edge, face).
   * @param subelement The subelement index.
   * @return Number of basis functions associated with the subelement.
   */
  template <typename Element, typename Subelement>
  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_basis_functions(dim_t subelement) {
    // Use SFINAE to determine whether the Derived type implements
    // subelement-specific methods.
    if constexpr (meta::is_detected_v<has_subelement_num_basis_functions_t,
                                      Derived, Element, Subelement>) {
      return Derived::num_basis_functions(subelement, meta::type_tag<Element>{},
                                          meta::type_tag<Subelement>{});
    } else {
      return 0;
    }
  }

  /**
   * @brief Compute the total number of basis functions associated with all
   * subelements.
   *
   * If Element == Subelement, this returns the number of interior basis
   * functions. Otherwise, it accumulates the number of basis functions over all
   * subelements of a given type.
   *
   * @tparam Element The parent element type.
   * @tparam Subelement The subelement type.
   * @return Total number of basis functions associated with all subelements.
   */
  template <typename Element, typename Subelement>
  static constexpr NUMERIC_HOST_DEVICE dim_t total_num_basis_functions() {
    // If the element and subelement are the same, only consider interior basis
    // functions.
    if constexpr (meta::is_same_v<Element, Subelement>) {
      return num_interior_basis_functions<Element>();
    } else {
      // Otherwise, accumulate over all subelements
      constexpr dim_t num_subs =
          Element::template num_subelements<Subelement>();
      dim_t n = 0;
      for (dim_t sub = 0; sub < num_subs; ++sub) {
        n += num_basis_functions<Element, Subelement>(sub);
      }
      return n;
    }
  }

  /**
   * @brief Evaluate all basis functions at a given point.
   *
   * @tparam Element The element type.
   * @tparam Scalar The scalar type (e.g., float, double).
   * @param out Output array to hold function values.
   * @param x Input coordinates.
   */
  template <typename Element, typename Scalar>
  static NUMERIC_HOST_DEVICE void eval(Scalar *out, const Scalar *x) {
    return Derived::eval(out, x, meta::type_tag<Element>{});
  }

  /**
   * @brief Evaluate the gradient of all basis functions at a given point.
   *
   * The gradient array shape depends on the element dimension.
   *
   * @tparam Element The element type.
   * @tparam Scalar The scalar type.
   * @param out Output array to hold gradients of the basis functions.
   * @param x Input coordinates.
   */
  template <typename Element, typename Scalar>
  static NUMERIC_HOST_DEVICE void
  gradient(Scalar (*out)[Element::dim == 0 ? dim_t(1) : Element::dim],
           const Scalar *x) {
    return Derived::gradient(out, x, meta::type_tag<Element>{});
  }

  template <typename Element>
  static constexpr NUMERIC_HOST_DEVICE dim_t
  interior_dof_idx_under_permutation(dim_t dof, dim_t *perm) {
    using element_basis_t =
        typename Derived::element_basis_t<typename Element::ref_el_t>;
    const dim_t nb = num_basis_functions<typename Element::ref_el_t>();
    const dim_t nib =
        num_interior_basis_functions<typename Element::ref_el_t>();
    return element_basis_t::node_idx_under_permutation(dof + nb - nib, perm) +
           nib - nb;
  }
};

} // namespace numeric::math::fes

#endif
