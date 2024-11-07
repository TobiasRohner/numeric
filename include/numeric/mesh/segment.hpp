#ifndef NUMERIC_MESH_SEGMENT_HPP_
#define NUMERIC_MESH_SEGMENT_HPP_

#include <numeric/mesh/element_base.hpp>
#include <numeric/mesh/element_traits.hpp>
#include <numeric/mesh/point.hpp>
#include <numeric/mesh/ref_el_segment.hpp>

namespace numeric::mesh {

/**
 * @brief Mesh Segment of arbitrary order
 *
 * ![Reference Element](reference_element_Segment.png "Reference Element")
 * ![Ordering of Points](subelement_indexing_Segment_Point.png "Ordering of
 * Points")
 */
template <dim_t Order> struct Segment : public ElementBase<Segment<Order>> {
  using super = ElementBase<Segment<Order>>;

  using traits_t = ElementTraits<Segment<Order>>;
  using ref_el_t = typename traits_t::ref_el_t;
  static constexpr dim_t dim = traits_t::dim;
  static constexpr dim_t order = traits_t::order;
  static constexpr bool is_affine = traits_t::is_affine;
  static constexpr const char *name = traits_t::name;
  static constexpr dim_t num_nodes = traits_t::num_nodes;

  static constexpr dim_t num_subelements(meta::type_tag<Point<Order>>) {
    return 2;
  }
  using super::num_subelements;

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<Point<Order>>) {
    idxs[0] = subelement;
  }
  using super::local_to_global;
  using super::subelement_node_idxs;

  template <typename Scalar>
  static constexpr void jacobian(const Scalar (*nodes)[num_nodes],
                                 const Scalar *x, Scalar (*out)[dim],
                                 dim_t world_dim) {
    if constexpr (order == 1) {
      const Scalar x1 = x[0];
      for (dim_t i = 0; i < world_dim; ++i) {
        out[i][0] = nodes[i][1] - nodes[i][0];
      }
    } else {
      static_assert(order != order,
                    "Jacobian of segment not implemented for the given order");
    }
  }
};

template <dim_t Order> struct ElementTraits<Segment<Order>> {
  using ref_el_t = RefElSegment;
  static constexpr dim_t dim = 1;
  static constexpr dim_t order = Order;
  static constexpr bool is_affine = true;
  static constexpr char name[] = "Segment";
  static constexpr dim_t num_nodes = Order + 1;
};

} // namespace numeric::mesh

#endif
