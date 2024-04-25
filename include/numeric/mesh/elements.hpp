#ifndef NUMERIC_MESH_ELEMENTS_HPP_
#define NUMERIC_MESH_ELEMENTS_HPP_

#include <numeric/config.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::mesh {

template <typename Derived> struct ElementBase {
  template <typename Base, typename Element>
  using has_subelement_t =
      decltype(Base::num_subelements(meta::type_tag<Element>()));

  template <typename Base, typename Element>
  using has_subelement_node_idxs_t =
      decltype(Base::subelement_node_idxs(meta::declval<dim_t>(),
                                          meta::declval<dim_t *>(),
                                          meta::type_tag<Element>()));

  template <typename Element> static constexpr dim_t num_subelements() {
    if constexpr (meta::is_detected_v<has_subelement_t, Derived, Element>) {
      return Derived::num_subelements(meta::type_tag<Element>());
    } else {
      return 0;
    }
  }

  template <typename Element>
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs) {
    if constexpr (meta::is_detected_v<has_subelement_node_idxs_t, Derived,
                                      Element>) {
      Derived::subelement_node_idxs(subelement, idxs,
                                    meta::type_tag<Element>());
    }
  }
};

/**
 * @brief Mesh Point of arbitrary order
 *
 * @image html reference_element_Point.png "Reference Element"
 */
template <dim_t Order> struct Point : public ElementBase<Point<Order>> {
  using super = ElementBase<Point<Order>>;

  static constexpr dim_t dim = 0;
  static constexpr dim_t order = Order;
  static constexpr char name[] = "Point";

  static constexpr dim_t num_nodes() { return 1; }

  using super::num_subelements;
  using super::subelement_node_idxs;
};

/**
 * @brief Mesh Segment of arbitrary order
 *
 * @image html reference_element_Segment.png "Reference Element"
 * @image html subelement_indexing_Segment_Point.png "Ordering of Points"
 */
template <dim_t Order> struct Segment : public ElementBase<Segment<Order>> {
  using super = ElementBase<Segment<Order>>;

  static constexpr dim_t dim = 1;
  static constexpr dim_t order = Order;
  static constexpr char name[] = "Segment";

  static constexpr dim_t num_nodes() { return Order + 1; }

  static constexpr dim_t num_subelements(meta::type_tag<Point<Order>>) {
    return 2;
  }
  using super::num_subelements;

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<Point<Order>>) {
    idxs[0] = subelement;
  }
  using super::subelement_node_idxs;
};

/**
 * @brief Mesh Triangle of arbitrary order
 *
 * @image html reference_element_Tria.png "Reference Element"
 * @image html subelement_indexing_Tria_Point.png "Ordering o Points"
 * @image html subelement_indexing_Tria_Segment.png "Ordering of Segments"
 */
template <dim_t Order> struct Tria : public ElementBase<Tria<Order>> {
  using super = ElementBase<Tria<Order>>;

  static constexpr dim_t dim = 2;
  static constexpr dim_t order = Order;
  static constexpr char name[] = "Tria";

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
};

/**
 * @brief Mesh Quadrilateral of arbitrary order
 *
 * @image html reference_element_Quad.png "Reference Element"
 * @image html subelement_indexing_Quad_Point.png "Ordering of Points"
 * @image html subelement_indexing_Quad_Segment.png "Ordering of Segments"
 */
template <dim_t Order> struct Quad : public ElementBase<Quad<Order>> {
  using super = ElementBase<Quad<Order>>;

  static constexpr dim_t dim = 2;
  static constexpr dim_t order = Order;
  static constexpr char name[] = "Quad";

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
};

/**
 * @brief Mesh Tetrahedron of arbitrary order
 *
 * @image html reference_element_Tetra.png "Reference Element"
 * @image html subelement_indexing_Tetra_Point.png "Ordering of Points"
 * @image html subelement_indexing_Tetra_Segment.png "Ordering of Segments"
 * @image html subelement_indexing_Tetra_Tria.png "Ordering of Triangles"
 */
template <dim_t Order> struct Tetra : public ElementBase<Tetra<Order>> {
  using super = ElementBase<Tetra<Order>>;

  static constexpr dim_t dim = 3;
  static constexpr dim_t order = Order;
  static constexpr char name[] = "Tetra";

  static constexpr dim_t num_nodes() {
    return (Order + 1) * (Order + 2) * (Order + 3) / 6;
  }

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
};

/**
 * @brief Mesh Cube of arbitrary order
 *
 * @image html reference_element_Cube.png "Reference Element"
 * @image html subelement_indexing_Cube_Point.png "Ordering of Points"
 * @image html subelement_indexing_Cube_Segment.png "Ordering of Segments"
 * @image html subelement_indexing_Cube_Quad.png "Ordering of Quads"
 */
template <dim_t Order> struct Cube : public ElementBase<Cube<Order>> {
  using super = ElementBase<Cube<Order>>;

  static constexpr dim_t dim = 3;
  static constexpr dim_t order = Order;
  static constexpr char name[] = "Cube";

  static constexpr dim_t num_nodes() {
    return (Order + 1) * (Order + 1) * (Order + 1);
  }

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
};

} // namespace numeric::mesh

#endif
