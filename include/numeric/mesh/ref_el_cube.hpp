#ifndef NUMERIC_MESH_REF_EL_CUBE_HPP_
#define NUMERIC_MESH_REF_EL_CUBE_HPP_

#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::mesh {

struct RefElCube {
  static constexpr dim_t dim = 3;
  static constexpr dim_t num_nodes = 8;
  static constexpr char name[] = "Cube";

  template <typename Subelement>
  static constexpr NUMERIC_HOST_DEVICE dim_t num_subelements() {
    if constexpr (meta::is_same_v<Subelement, RefElPoint>) {
      return 8;
    }
    if constexpr (meta::is_same_v<Subelement, RefElSegment>) {
      return 12;
    }
    if constexpr (meta::is_same_v<Subelement, RefElQuad>) {
      return 6;
    }
    return 0;
  }

  template <typename Scalar>
  static constexpr NUMERIC_HOST_DEVICE void get_nodes(Scalar (*out)[3]) {
    out[0][0] = 0;
    out[0][1] = 0;
    out[0][2] = 0;
    out[1][0] = 1;
    out[1][1] = 0;
    out[1][2] = 0;
    out[2][0] = 1;
    out[2][1] = 1;
    out[2][2] = 0;
    out[3][0] = 0;
    out[3][1] = 1;
    out[3][2] = 0;
    out[4][0] = 0;
    out[4][1] = 0;
    out[4][2] = 1;
    out[5][0] = 1;
    out[5][1] = 0;
    out[5][2] = 1;
    out[6][0] = 1;
    out[6][1] = 1;
    out[6][2] = 1;
    out[7][0] = 0;
    out[7][1] = 1;
    out[7][2] = 1;
  }

  template <typename Subelement>
  static constexpr NUMERIC_HOST_DEVICE void subelement_node_idxs(dim_t idx,
                                                                 dim_t *out) {
    if constexpr (meta::is_same_v<Subelement, RefElPoint>) {
      switch (idx) {
      case 0:
        out[0] = 0;
        break;
      case 1:
        out[0] = 1;
        break;
      case 2:
        out[0] = 2;
        break;
      case 3:
        out[0] = 3;
        break;
      case 4:
        out[0] = 4;
        break;
      case 5:
        out[0] = 5;
        break;
      case 6:
        out[0] = 6;
        break;
      case 7:
        out[0] = 7;
        break;
      }
    }
    if constexpr (meta::is_same_v<Subelement, RefElSegment>) {
      switch (idx) {
      case 0:
        out[0] = 0;
        out[1] = 1;
        break;
      case 1:
        out[0] = 1;
        out[1] = 2;
        break;
      case 2:
        out[0] = 3;
        out[1] = 2;
        break;
      case 3:
        out[0] = 0;
        out[1] = 3;
        break;
      case 4:
        out[0] = 4;
        out[1] = 5;
        break;
      case 5:
        out[0] = 5;
        out[1] = 6;
        break;
      case 6:
        out[0] = 7;
        out[1] = 6;
        break;
      case 7:
        out[0] = 4;
        out[1] = 7;
        break;
      case 8:
        out[0] = 0;
        out[1] = 4;
        break;
      case 9:
        out[0] = 1;
        out[1] = 5;
        break;
      case 10:
        out[0] = 2;
        out[1] = 6;
        break;
      case 11:
        out[0] = 3;
        out[1] = 7;
        break;
      }
    }
    if constexpr (meta::is_same_v<Subelement, RefElQuad>) {
      switch (idx) {
      case 0:
        out[0] = 0;
        out[1] = 3;
        out[2] = 7;
        out[3] = 4;
        break;
      case 1:
        out[0] = 1;
        out[1] = 2;
        out[2] = 6;
        out[3] = 5;
        break;
      case 2:
        out[0] = 0;
        out[1] = 1;
        out[2] = 5;
        out[3] = 4;
        break;
      case 3:
        out[0] = 3;
        out[1] = 2;
        out[2] = 6;
        out[3] = 7;
        break;
      case 4:
        out[0] = 0;
        out[1] = 1;
        out[2] = 2;
        out[3] = 3;
        break;
      case 5:
        out[0] = 4;
        out[1] = 5;
        out[2] = 6;
        out[3] = 7;
        break;
      }
    }
  }
};

} // namespace numeric::mesh

#endif
