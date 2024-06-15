#ifndef NUMERIC_MESH_TRIA_HPP_
#define NUMERIC_MESH_TRIA_HPP_

#include <numeric/mesh/element_base.hpp>
#include <numeric/mesh/element_traits.hpp>
#include <numeric/mesh/point.hpp>
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
  static constexpr dim_t dim = traits_t::dim;
  static constexpr dim_t order = traits_t::order;
  static constexpr bool is_affine = traits_t::is_affine;
  static constexpr const char *name = traits_t::name;

  static constexpr dim_t num_nodes() { return (Order + 1) * (Order + 2) / 2; }

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
  using super::subelement_node_idxs;

  template <typename Scalar>
  static constexpr void local_to_global(const Scalar *nodes[num_nodes()],
                                        const Scalar *x, Scalar *out,
                                        dim_t world_dim) {
    if constexpr (order == 1) {
      const Scalar x1 = x[0];
      const Scalar x2 = x[1];
      const Scalar l1 = 1 - x1 - x2;
      const Scalar l2 = x1;
      const Scalar l3 = x2;
      for (dim_t i = 0; i < world_dim; ++i) {
        out[i] = l1 * nodes[i][0] + l2 * nodes[i][1] + l3 * nodes[i][2];
      }
    } else {
      static_assert(order != order, "local to global map of triangle not "
                                    "implemented for the given order");
    }
  }

  template <typename Scalar>
  static constexpr void jacobian(const Scalar *nodes[num_nodes()],
                                 const Scalar *x, Scalar *out[dim],
                                 dim_t world_dim) {
    if constexpr (order == 1) {
      const Scalar x1 = x[0];
      const Scalar x2 = x[1];
      for (dim_t i = 0; i < world_dim; ++i) {
        out[i][0] = nodes[i][1] - nodes[i][0];
        out[i][1] = nodes[i][2] - nodes[i][0];
      }
    } else {
      static_assert(order != order,
                    "Jacobian of triangle not implemented for the given order");
    }
  }
};

template <dim_t Order> struct ElementTraits<Tria<Order>> {
  static constexpr dim_t dim = 2;
  static constexpr dim_t order = Order;
  static constexpr bool is_affine = true;
  static constexpr char name[] = "Tria";
};

} // namespace numeric::mesh

#endif
