#ifndef NUMERIC_MESH_TETRA_HPP_
#define NUMERIC_MESH_TETRA_HPP_

#include <numeric/mesh/element_base.hpp>
#include <numeric/mesh/element_traits.hpp>
#include <numeric/mesh/point.hpp>
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
  using super::num_subelements;

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<Point<Order>>) {
    idxs[0] = subelement;
  }
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<Segment<Order>>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      break;
    case 1:
      idxs[0] = 0;
      idxs[1] = 2;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      break;
    case 4:
      idxs[0] = 2;
      idxs[1] = 3;
      break;
    case 5:
      idxs[0] = 3;
      idxs[1] = 1;
      break;
    default:
      return;
    }
    for (dim_t i = 0; i < Order - 1; ++i) {
      idxs[i + 2] = 4 + subelement * (Order - 1) + i;
    }
  }
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<Tria<Order>>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 2;
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 3 + 0 * (Order - 1)] = 4 + 0 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 3 + 1 * (Order - 1)] = 4 + 3 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 3 + 2 * (Order - 1)] =
            4 + 1 * (Order - 1) + (Order - 1) - i - 1;
      }
      break;
    case 1:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 3;
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 3 + 0 * (Order - 1)] = 4 + 1 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 3 + 1 * (Order - 1)] = 4 + 4 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 3 + 2 * (Order - 1)] =
            4 + 2 * (Order - 1) + (Order - 1) - i - 1;
      }
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 1;
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 3 + 0 * (Order - 1)] = 4 + 2 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 3 + 1 * (Order - 1)] = 4 + 5 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 3 + 2 * (Order - 1)] =
            4 + 0 * (Order - 1) + (Order - 1) - i - 1;
      }
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 3;
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 3 + 0 * (Order - 1)] = 4 + 3 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 3 + 1 * (Order - 1)] = 4 + 4 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 3 + 2 * (Order - 1)] = 4 + 5 * (Order - 1) + i;
      }
      break;
    default:
      return;
    }
    static constexpr dim_t num_interior_nodes =
        (Order - 1) * Order * (Order + 1) / 6;
    for (dim_t i = 0; i < num_interior_nodes; ++i) {
      idxs[3 + 3 * (Order - 1)] =
          4 + 4 * (Order - 1) + subelement * num_interior_nodes + i;
    }
  }
  using super::subelement_node_idxs;

  template <typename Scalar>
  static constexpr void local_to_global(const Scalar (*nodes)[num_nodes],
                                        const Scalar *x, Scalar *out,
                                        dim_t world_dim) {
    if constexpr (order == 1) {
      const Scalar x1 = x[0];
      const Scalar x2 = x[1];
      const Scalar x3 = x[2];
      const Scalar l1 = 1 - x1 - x2 - x3;
      const Scalar l2 = x1;
      const Scalar l3 = x2;
      const Scalar l4 = x3;
      for (dim_t i = 0; i < world_dim; ++i) {
        out[i] = l1 * nodes[i][0] + l2 * nodes[i][1] + l3 * nodes[i][2] +
                 l4 * nodes[i][3];
      }
    } else {
      static_assert(order != order, "local to global map of tetrahedron not "
                                    "implemented for the given order");
    }
  }

  template <typename Scalar>
  static constexpr void jacobian(const Scalar (*nodes)[num_nodes],
                                 const Scalar *x, Scalar (*out)[dim],
                                 dim_t world_dim) {
    if constexpr (order == 1) {
      const Scalar x1 = x[0];
      const Scalar x2 = x[1];
      for (dim_t i = 0; i < world_dim; ++i) {
        out[i][0] = nodes[i][1] - nodes[i][0];
        out[i][1] = nodes[i][2] - nodes[i][0];
        out[i][2] = nodes[i][3] - nodes[i][0];
      }
    } else {
      static_assert(
          order != order,
          "Jacobian of tetrahedron not implemented for the given order");
    }
  }
};

template <dim_t Order> struct ElementTraits<Tetra<Order>> {
  static constexpr dim_t dim = 3;
  static constexpr dim_t order = Order;
  static constexpr bool is_affine = true;
  static constexpr char name[] = "Tetra";
  static constexpr dim_t num_nodes = (Order + 1) * (Order + 2) * (Order + 3) / 6;
};

} // namespace numeric::mesh

#endif
