#ifndef NUMERIC_MESH_QUAD_HPP_
#define NUMERIC_MESH_QUAD_HPP_

#include <numeric/mesh/element_base.hpp>
#include <numeric/mesh/element_traits.hpp>
#include <numeric/mesh/point.hpp>
#include <numeric/mesh/segment.hpp>

namespace numeric::mesh {

/**
 * @brief Mesh Quadrilateral of arbitrary order
 *
 * ![Reference Element]( reference_element_Quad.png "Reference Element")
 * ![Ordering of Points](subelement_indexing_Quad_Point.png "Ordering of
 * Points")
 * ![Ordering of Segments](subelement_indexing_Quad_Segment.png "Ordering of
 * Segments")
 */
template <dim_t Order> struct Quad : public ElementBase<Quad<Order>> {
  using super = ElementBase<Quad<Order>>;

  using traits_t = ElementTraits<Quad<Order>>;
  static constexpr dim_t dim = traits_t::dim;
  static constexpr dim_t order = traits_t::order;
  static constexpr bool is_affine = traits_t::is_affine;
  static constexpr const char *name = traits_t::name;

  static constexpr dim_t num_nodes() { return (Order + 1) * (Order + 1); }

  static constexpr dim_t num_subelements(meta::type_tag<Point<Order>>) {
    return 4;
  }
  static constexpr dim_t num_subelements(meta::type_tag<Segment<Order>>) {
    return 4;
  }
  using super::num_subelements;

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<Point<Order>>) {
    idxs[0] = subelement;
  }
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<Segment<Order>>) {
    idxs[0] = subelement;
    idxs[1] = (subelement + 1) % 4;
    for (dim_t i = 0; i < Order - 1; ++i) {
      idxs[i + 2] = 4 + (Order - 1) * i;
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
      for (dim_t i = 0; i < world_dim; ++i) {
        out[i] = (1 - x1) * (1 - x2) * nodes[i][0] +
                 x1 * (1 - x2) * nodes[i][1] + x1 * x2 * nodes[i][2] +
                 (1 - x1) * x2 * nodes[i][3];
      }
    } else {
      static_assert(
          order != order,
          "local to global map of quad not implemented for the given order");
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
        out[i][0] = (1 - x2) * (nodes[i][1] - nodes[i][0]) +
                    x2 * (nodes[i][2] - nodes[i][3]);
        out[i][1] = (1 - x1) * (nodes[i][3] - nodes[i][0]) +
                    x1 * (nodes[i][2] - nodes[i][1]);
      }
    } else {
      static_assert(order != order,
                    "Jacobian of quad not implemented for the given order");
    }
  }
};

template <dim_t Order> struct ElementTraits<Quad<Order>> {
  static constexpr dim_t dim = 2;
  static constexpr dim_t order = Order;
  static constexpr bool is_affine = false;
  static constexpr char name[] = "Quad";
};

} // namespace numeric::mesh

#endif
