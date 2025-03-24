#ifndef NUMERIC_MESH_TETRA_HPP_
#define NUMERIC_MESH_TETRA_HPP_

#include <numeric/mesh/element_base.hpp>
#include <numeric/mesh/element_traits.hpp>
#include <numeric/mesh/point.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/segment.hpp>
#include <numeric/mesh/tria.hpp>

namespace numeric::mesh {

/**
 * @brief Mesh Tetrahedron of arbitrary order
 *
 * ![Reference Element](reference_element_Tetra.png "Reference Element")
 * ![Ordering of Points](subelement_indexing_Tetra_Point.png "Ordering of
 * Points")
 * ![Ordering of Segments](subelement_indexing_Tetra_Segment.png "Ordering of
 * Segments")
 * ![Ordering of Triangles](subelement_indexing_Tetra_Tria.png "Ordering of
 * Triangles")
 */
template <dim_t Order> struct Tetra : public ElementBase<Tetra<Order>> {
  using super = ElementBase<Tetra<Order>>;

  using traits_t = ElementTraits<Tetra<Order>>;
  using ref_el_t = typename traits_t::ref_el_t;
  using basis_t = typename traits_t::basis_t;
  static constexpr dim_t dim = traits_t::dim;
  static constexpr dim_t order = traits_t::order;
  static constexpr bool is_affine = traits_t::is_affine;
  static constexpr const char *name = traits_t::name;
  static constexpr dim_t num_nodes = traits_t::num_nodes;

  static constexpr dim_t num_subelements(meta::type_tag<Point<Order>>) {
    return 4;
  }
  static constexpr dim_t num_subelements(meta::type_tag<Segment<Order>>) {
    return 6;
  }
  static constexpr dim_t num_subelements(meta::type_tag<Tria<Order>>) {
    return 4;
  }

  using super::jacobian;
  using super::local_to_global;
  using super::num_subelements;
  using super::subelement_node_idxs;
};

template <dim_t Order> struct ElementTraits<Tetra<Order>> {
  using ref_el_t = RefElTetra;
  using basis_t = math::BasisLagrange<ref_el_t, Order>;
  static constexpr dim_t dim = 3;
  static constexpr dim_t order = Order;
  static constexpr bool is_affine = true;
  static constexpr char name[] = "Tetra";
  static constexpr dim_t num_nodes =
      (Order + 1) * (Order + 2) * (Order + 3) / 6;
};

} // namespace numeric::mesh

#endif
