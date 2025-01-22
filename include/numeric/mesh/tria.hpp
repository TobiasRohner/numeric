#ifndef NUMERIC_MESH_TRIA_HPP_
#define NUMERIC_MESH_TRIA_HPP_

#include <numeric/mesh/element_base.hpp>
#include <numeric/mesh/element_traits.hpp>
#include <numeric/mesh/point.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/mesh/segment.hpp>

namespace numeric::mesh {

/**
 * @brief Mesh Triangle of arbitrary order
 *
 * ![Reference Element](reference_element_Tria.png "Reference Element")
 * ![Ordering of Points](subelement_indexing_Tria_Point.png "Ordering o Points")
 * ![Ordering of Segments](subelement_indexing_Tria_Segment.png "Ordering of
 * Segments")
 */
template <dim_t Order> struct Tria : public ElementBase<Tria<Order>> {
  using super = ElementBase<Tria<Order>>;

  using traits_t = ElementTraits<Tria<Order>>;
  using ref_el_t = typename traits_t::ref_el_t;
  using basis_t = typename traits_t::basis_t;
  static constexpr dim_t dim = traits_t::dim;
  static constexpr dim_t order = traits_t::order;
  static constexpr bool is_affine = traits_t::is_affine;
  static constexpr const char *name = traits_t::name;
  static constexpr dim_t num_nodes = traits_t::num_nodes;

  static constexpr dim_t num_subelements(meta::type_tag<Point<Order>>) {
    return 3;
  }
  static constexpr dim_t num_subelements(meta::type_tag<Segment<Order>>) {
    return 3;
  }
  using super::num_subelements;

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<Point<Order>>) {
    idxs[0] = subelement;
  }
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<Segment<Order>>) {
    idxs[0] = subelement;
    idxs[1] = (subelement + 1) % 3;
    for (dim_t i = 0; i < Order - 1; ++i) {
      idxs[i + 2] = 3 + (Order - 1) * i;
    }
  }
  using super::jacobian;
  using super::local_to_global;
  using super::subelement_node_idxs;
};

template <dim_t Order> struct ElementTraits<Tria<Order>> {
  using ref_el_t = RefElTria;
  using basis_t = math::BasisLagrange<ref_el_t, Order>;
  static constexpr dim_t dim = 2;
  static constexpr dim_t order = Order;
  static constexpr bool is_affine = true;
  static constexpr char name[] = "Tria";
  static constexpr dim_t num_nodes = (Order + 1) * (Order + 2) / 2;
};

} // namespace numeric::mesh

#endif
