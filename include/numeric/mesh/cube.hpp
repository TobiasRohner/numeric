#ifndef NUMERIC_MESH_CUBE_HPP_
#define NUMERIC_MESH_CUBE_HPP_

#include <numeric/mesh/element_base.hpp>
#include <numeric/mesh/element_traits.hpp>
#include <numeric/mesh/point.hpp>
#include <numeric/mesh/quad.hpp>
#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/segment.hpp>

namespace numeric::mesh {

/**
 * @brief Mesh Cube of arbitrary order
 *
 * ![Reference Element](reference_element_Cube.png "Reference Element")
 * ![Ordering of Points](subelement_indexing_Cube_Point.png "Ordering of
 * Points")
 * ![Ordering of Segments](subelement_indexing_Cube_Segment.png "Ordering of
 * Segments")
 * ![Ordering of Quads](subelement_indexing_Cube_Quad.png "Ordering of Quads")
 */
template <dim_t Order> struct Cube : public ElementBase<Cube<Order>> {
  using super = ElementBase<Cube<Order>>;

  using traits_t = ElementTraits<Cube<Order>>;
  using ref_el_t = typename traits_t::ref_el_t;
  using basis_t = typename traits_t::basis_t;
  static constexpr dim_t dim = traits_t::dim;
  static constexpr dim_t order = traits_t::order;
  static constexpr bool is_affine = traits_t::is_affine;
  static constexpr const char *name = traits_t::name;
  static constexpr dim_t num_nodes = traits_t::num_nodes;

  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_subelements(meta::type_tag<Point<Order>>) {
    return 8;
  }
  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_subelements(meta::type_tag<Segment<Order>>) {
    return 12;
  }
  static constexpr NUMERIC_HOST_DEVICE dim_t
  num_subelements(meta::type_tag<Quad<Order>>) {
    return 6;
  }

  using super::jacobian;
  using super::local_to_global;
  using super::num_subelements;
  using super::subelement_node_idxs;
};

template <dim_t Order> struct ElementTraits<Cube<Order>> {
  using ref_el_t = RefElCube;
  using basis_t = math::BasisLagrange<ref_el_t, Order>;
  static constexpr dim_t dim = 3;
  static constexpr dim_t order = Order;
  static constexpr bool is_affine = false;
  static constexpr char name[] = "Cube";
  static constexpr dim_t num_nodes = (Order + 1) * (Order + 1) * (Order + 1);
};

} // namespace numeric::mesh

#endif
