#ifndef NUMERIC_MATH_FES_BASIS_BASE_HPP_
#define NUMERIC_MATH_FES_BASIS_BASE_HPP_

#include <numeric/config.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math::fes {

template <typename Derived> struct BasisBase {
  template <typename Base, typename Element, typename Subelement>
  using has_subelement_num_basis_functions_t =
      decltype(Base::num_basis_functions(meta::declval<dim_t>(),
                                         meta::type_tag<Element>(),
                                         meta::type_tag<Subelement>()));

  template <typename Element>
  static constexpr dim_t num_interior_basis_functions() {
    return Derived::num_interior_basis_functions(meta::type_tag<Element>{});
  }

  template <typename Element> static constexpr dim_t num_basis_functions() {
    return Derived::num_basis_functions(meta::type_tag<Element>{});
  }

  template <typename Element, typename Subelement>
  static constexpr dim_t num_basis_functions(dim_t subelement) {
    if constexpr (meta::is_detected_v<has_subelement_num_basis_functions_t,
                                      Derived, Element, Subelement>) {
      return Derived::num_basis_functions(subelement, meta::type_tag<Element>{},
                                          meta::type_tag<Subelement>{});
    } else {
      return 0;
    }
  }

  template <typename Element, typename Subelement>
  static constexpr dim_t total_num_basis_functions() {
    if constexpr (meta::is_same_v<Element, Subelement>) {
      return num_interior_basis_functions<Element>();
    } else {
      constexpr dim_t num_subs =
          Element::template num_subelements<Subelement>();
      dim_t n = 0;
      for (dim_t sub = 0; sub < num_subs; ++sub) {
        n += num_basis_functions<Element, Subelement>(sub);
      }
      return n;
    }
  }

  template <typename Element, typename Scalar>
  static void eval(Scalar *out, const Scalar *x) {
    return Derived::eval(out, x, meta::type_tag<Element>{});
  }

  template <typename Element, typename Scalar>
  static void
  gradient(Scalar (*out)[Element::dim == 0 ? dim_t(1) : Element::dim],
           const Scalar *x) {
    return Derived::gradient(out, x, meta::type_tag<Element>{});
  }
};

} // namespace numeric::math::fes

#endif
