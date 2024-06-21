#ifndef NUMERIC_MESH_CUBE_HPP_
#define NUMERIC_MESH_CUBE_HPP_

#include <numeric/mesh/element_base.hpp>
#include <numeric/mesh/element_traits.hpp>
#include <numeric/mesh/point.hpp>
#include <numeric/mesh/quad.hpp>
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
  static constexpr dim_t dim = traits_t::dim;
  static constexpr dim_t order = traits_t::order;
  static constexpr bool is_affine = traits_t::is_affine;
  static constexpr const char *name = traits_t::name;
  static constexpr dim_t num_nodes = traits_t::num_nodes;

  static constexpr dim_t num_subelements(meta::type_tag<Point<Order>>) {
    return 8;
  }
  static constexpr dim_t num_subelements(meta::type_tag<Segment<Order>>) {
    return 12;
  }
  static constexpr dim_t num_subelements(meta::type_tag<Quad<Order>>) {
    return 6;
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
      idxs[0] = 1;
      idxs[1] = 2;
      break;
    case 2:
      idxs[0] = 2;
      idxs[1] = 3;
      break;
    case 3:
      idxs[0] = 3;
      idxs[1] = 0;
      break;
    case 4:
      idxs[0] = 4;
      idxs[1] = 5;
      break;
    case 5:
      idxs[0] = 5;
      idxs[1] = 6;
      break;
    case 6:
      idxs[0] = 6;
      idxs[1] = 7;
      break;
    case 7:
      idxs[0] = 7;
      idxs[1] = 4;
      break;
    case 8:
      idxs[0] = 0;
      idxs[1] = 4;
      break;
    case 9:
      idxs[0] = 1;
      idxs[1] = 5;
      break;
    case 10:
      idxs[0] = 2;
      idxs[1] = 6;
      break;
    case 11:
      idxs[0] = 3;
      idxs[1] = 7;
      break;
    default:
      return;
    }
    for (dim_t i = 0; i < Order - 1; ++i) {
      idxs[i + 2] = 8 + subelement * (Order - 1) + i;
    }
  }
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<Quad<Order>>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 2;
      idxs[3] = 3;
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 0 * (Order - 1)] = 8 + 0 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 1 * (Order - 1)] = 8 + 1 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 2 * (Order - 1)] = 8 + 2 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 3 * (Order - 1)] = 8 + 3 * (Order - 1) + i;
      }
      break;
    case 1:
      idxs[0] = 4;
      idxs[1] = 5;
      idxs[2] = 6;
      idxs[3] = 7;
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 0 * (Order - 1)] = 8 + 4 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 1 * (Order - 1)] = 8 + 5 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 2 * (Order - 1)] = 8 + 6 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 3 * (Order - 1)] = 8 + 7 * (Order - 1) + i;
      }
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 7;
      idxs[3] = 4;
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 0 * (Order - 1)] =
            8 + 3 * (Order - 1) + (Order - 1) - i - 1;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 1 * (Order - 1)] = 8 + 11 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 2 * (Order - 1)] = 8 + 7 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 3 * (Order - 1)] =
            8 + 8 * (Order - 1) + (Order - 1) - i - 1;
      }
      break;
    case 3:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 5;
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 0 * (Order - 1)] = 8 + 1 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 1 * (Order - 1)] = 8 + 10 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 2 * (Order - 1)] =
            8 + 5 * (Order - 1) + (Order - 1) - i - 1;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 3 * (Order - 1)] =
            8 + 9 * (Order - 1) + (Order - 1) - i - 1;
      }
      break;
    case 4:
      idxs[0] = 0;
      idxs[1] = 4;
      idxs[2] = 5;
      idxs[3] = 1;
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 0 * (Order - 1)] = 8 + 8 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 1 * (Order - 1)] = 8 + 4 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 2 * (Order - 1)] =
            8 + 9 * (Order - 1) + (Order - 1) - i - 1;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 3 * (Order - 1)] =
            8 + 0 * (Order - 1) + (Order - 1) - i - 1;
      }
      break;
    case 5:
      idxs[0] = 3;
      idxs[1] = 7;
      idxs[2] = 6;
      idxs[3] = 2;
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 0 * (Order - 1)] = 8 + 11 * (Order - 1) + i;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 1 * (Order - 1)] =
            8 + 6 * (Order - 1) + (Order - 1) - i - 1;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 2 * (Order - 1)] =
            8 + 10 * (Order - 1) + (Order - 1) - i - 1;
      }
      for (dim_t i = 0; i < Order - 1; ++i) {
        idxs[i + 4 + 3 * (Order - 1)] = 8 + 2 * (Order - 1) + i;
      }
      break;
    default:
      return;
    }
    for (dim_t i = 0; i < (Order - 1) * (Order - 1); ++i) {
      idxs[8 + 12 * (Order - 1)] =
          8 + 12 * (Order - 1) + subelement * (Order - 1) * (Order - 1) + i;
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
      for (dim_t i = 0; i < world_dim; ++i) {
        out[i] = (1 - x1) * (1 - x2) * (1 - x3) * nodes[i][0] +
                 x1 * (1 - x2) * (1 - x3) * nodes[i][1] +
                 x1 * x2 * (1 - x3) * nodes[i][2] +
                 (1 - x1) * x2 * (1 - x3) * nodes[i][3] +
                 (1 - x1) * (1 - x2) * x3 * nodes[i][4] +
                 x1 * (1 - x2) * x3 * nodes[i][5] + x1 * x2 * x3 * nodes[i][6] +
                 (1 - x1) * x2 * x3 * nodes[i][7];
      }
    } else {
      static_assert(
          order != order,
          "local to global map of cube not implemented for the given order");
    }
  }

  template <typename Scalar>
  static constexpr void jacobian(const Scalar (*nodes)[num_nodes],
                                 const Scalar *x, Scalar (*out)[dim],
                                 dim_t world_dim) {
    if constexpr (order == 1) {
      const Scalar x1 = x[0];
      const Scalar x2 = x[1];
      const Scalar x3 = x[2];
      for (dim_t i = 0; i < world_dim; ++i) {
        out[i][0] = (1 - x2) * ((1 - x3) * (nodes[i][1] - nodes[i][0]) +
                                x3 * (nodes[i][5] - nodes[i][4])) +
                    x2 * ((1 - x3) * (nodes[i][2] - nodes[i][3]) +
                          x3 * (nodes[i][6] - nodes[i][7]));
        out[i][1] = (1 - x1) * ((1 - x3) * (nodes[i][3] - nodes[i][0]) +
                                x3 * (nodes[i][7] - nodes[i][4])) +
                    x1 * ((1 - x3) * (nodes[i][2] - nodes[i][1]) +
                          x3 * (nodes[i][6] - nodes[i][5]));
        out[i][2] = (1 - x1) * ((1 - x2) * (nodes[i][4] - nodes[i][0]) +
                                x2 * (nodes[i][7] - nodes[i][3])) +
                    x1 * ((1 - x2) * (nodes[i][5] - nodes[i][1]) +
                          x2 * (nodes[i][6] - nodes[i][2]));
      }
    } else {
      static_assert(order != order,
                    "Jacobian of cube not implemented for the given order");
    }
  }
};

template <dim_t Order> struct ElementTraits<Cube<Order>> {
  static constexpr dim_t dim = 3;
  static constexpr dim_t order = Order;
  static constexpr bool is_affine = false;
  static constexpr char name[] = "Cube";
  static constexpr dim_t num_nodes = (Order + 1) * (Order + 1) * (Order + 1);
};

} // namespace numeric::mesh

#endif
