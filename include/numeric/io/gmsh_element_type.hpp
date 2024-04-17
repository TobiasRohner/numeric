#ifndef NUMERIC_IO_GMSH_ELEMENT_TYPE_HPP_
#define NUMERIC_IO_GMSH_ELEMENT_TYPE_HPP_

#include <numeric/mesh/elements.hpp>
#include <numeric/meta/meta.hpp>
#include <string_view>

namespace numeric::io {

enum class GmshElementType : int {
  LINE_2_1 = 1,
  TRIA_3_1 = 2,
  QUAD_4_1 = 3,
  TETRA_4_1 = 4,
  HEXA_8_1 = 5,
  POINT_1_1 = 15
};

std::string_view to_string(GmshElementType type);

template <typename Element> struct to_gmsh_element_type {
  static_assert(!meta::is_same_v<Element, Element>, "Unsupported element type");
};

template <> struct to_gmsh_element_type<mesh::Point<1>> {
  static constexpr GmshElementType value = GmshElementType::LINE_2_1;
};

template <> struct to_gmsh_element_type<mesh::Segment<1>> {
  static constexpr GmshElementType value = GmshElementType::LINE_2_1;
};

template <> struct to_gmsh_element_type<mesh::Tria<1>> {
  static constexpr GmshElementType value = GmshElementType::TRIA_3_1;
};

template <> struct to_gmsh_element_type<mesh::Quad<1>> {
  static constexpr GmshElementType value = GmshElementType::QUAD_4_1;
};

template <> struct to_gmsh_element_type<mesh::Tetra<1>> {
  static constexpr GmshElementType value = GmshElementType::TETRA_4_1;
};

template <> struct to_gmsh_element_type<mesh::Cube<1>> {
  static constexpr GmshElementType value = GmshElementType::HEXA_8_1;
};

template <typename Element>
static constexpr GmshElementType to_gmsh_element_type_v =
    to_gmsh_element_type<Element>::value;

template <GmshElementType type> struct from_gmsh_element_type {
  static_assert(type != type, "Unsupported element type");
};

template <> struct from_gmsh_element_type<GmshElementType::POINT_1_1> {
  using type = mesh::Point<1>;
};

template <> struct from_gmsh_element_type<GmshElementType::LINE_2_1> {
  using type = mesh::Segment<1>;
};

template <> struct from_gmsh_element_type<GmshElementType::TRIA_3_1> {
  using type = mesh::Tria<1>;
};

template <> struct from_gmsh_element_type<GmshElementType::QUAD_4_1> {
  using type = mesh::Quad<1>;
};

template <> struct from_gmsh_element_type<GmshElementType::TETRA_4_1> {
  using type = mesh::Tetra<1>;
};

template <> struct from_gmsh_element_type<GmshElementType::HEXA_8_1> {
  using type = mesh::Cube<1>;
};

template <GmshElementType type>
using from_gmsh_element_type_t = typename from_gmsh_element_type<type>::type;

constexpr size_t num_nodes(GmshElementType type) {
  switch (type) {
  case GmshElementType::POINT_1_1:
    return 1;
  case GmshElementType::LINE_2_1:
    return 2;
  case GmshElementType::TRIA_3_1:
    return 3;
  case GmshElementType::QUAD_4_1:
    return 4;
  case GmshElementType::TETRA_4_1:
    return 4;
  case GmshElementType::HEXA_8_1:
    return 8;
  default:
    return 0;
  }
}

} // namespace numeric::io

#endif
