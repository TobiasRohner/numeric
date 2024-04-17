#include <numeric/io/gmsh_element_type.hpp>

namespace numeric::io {

std::string_view to_string(GmshElementType type) {
  switch (type) {
  case GmshElementType::POINT_1_1:
    return "POINT_1_1";
  case GmshElementType::LINE_2_1:
    return "LINE_2_1";
  case GmshElementType::TRIA_3_1:
    return "TRIA_3_1";
  case GmshElementType::QUAD_4_1:
    return "QUAD_4_1";
  case GmshElementType::TETRA_4_1:
    return "TETRA_4_1";
  case GmshElementType::HEXA_8_1:
    return "HEXA_8_1";
  default:
    return "UNKNOWN";
  }
}

} // namespace numeric::io
