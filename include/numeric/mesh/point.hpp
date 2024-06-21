#ifndef NUMERIC_MESH_POINT_HPP_
#define NUMERIC_MESH_POINT_HPP_

#include <numeric/mesh/element_base.hpp>
#include <numeric/mesh/element_traits.hpp>

namespace numeric::mesh {

/**
 * @brief Mesh Point of arbitrary order
 *
 * ![Reference Element](reference_element_Point.png "Reference Element")
 */
template <dim_t Order> struct Point : public ElementBase<Point<Order>> {
  using super = ElementBase<Point<Order>>;

  using traits_t = ElementTraits<Point<Order>>;
  static constexpr dim_t dim = traits_t::dim;
  static constexpr dim_t order = traits_t::order;
  static constexpr bool is_affine = traits_t::is_affine;
  static constexpr const char *name = traits_t::name;
  static constexpr dim_t num_nodes = traits_t::num_nodes;

  using super::num_subelements;
  using super::subelement_node_idxs;

  template <typename Scalar>
  static constexpr void local_to_global(const Scalar (*nodes)[1], const Scalar *x,
                                        Scalar *out, dim_t world_dim) {
    for (dim_t i = 0; i < world_dim; ++i) {
      out[i] = nodes[i][0];
    }
  }

  template <typename Scalar>
  static constexpr void jacobian(const Scalar (*nodes)[1], const Scalar *x,
                                 Scalar *out, dim_t world_dim) {
    // Nothing to be done here as a point is zero-dimensional
  }
};

template <dim_t Order> struct ElementTraits<Point<Order>> {
  static constexpr dim_t dim = 0;
  static constexpr dim_t order = Order;
  static constexpr bool is_affine = true;
  static constexpr char name[] = "Point";
  static constexpr dim_t num_nodes = 1;
};

} // namespace numeric::mesh

#endif
