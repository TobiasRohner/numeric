#ifndef NUMERIC_MESH_REF_EL_TETRA_HPP_
#define NUMERIC_MESH_REF_EL_TETRA_HPP_

#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::mesh {

struct RefElTetra {
  static constexpr dim_t dim = 3;
  static constexpr dim_t num_nodes = 4;
  static constexpr char name[] = "Tetra";

  template <typename Subelement> static constexpr dim_t num_subelements() {
    if constexpr (meta::is_same_v<Subelement, RefElPoint>) {
      return 4;
    }
    if constexpr (meta::is_same_v<Subelement, RefElSegment>) {
      return 6;
    }
    if constexpr (meta::is_same_v<Subelement, RefElTria>) {
      return 4;
    }
    return 0;
  }

  template <typename Scalar> static constexpr void get_nodes(Scalar (*out)[3]) {
    out[0][0] = 0;
    out[0][1] = 0;
    out[0][2] = 0;
    out[1][0] = 1;
    out[1][1] = 0;
    out[1][2] = 0;
    out[2][0] = 0;
    out[2][1] = 1;
    out[2][2] = 0;
    out[3][0] = 0;
    out[3][1] = 0;
    out[3][2] = 1;
  }

  template <typename Subelement>
  static constexpr void subelement_node_idxs(dim_t idx, dim_t *out) {
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
        out[0] = 2;
        out[1] = 0;
        break;
      case 3:
        out[0] = 0;
        out[1] = 3;
        break;
      case 4:
        out[0] = 1;
        out[1] = 3;
        break;
      case 5:
        out[0] = 2;
        out[1] = 3;
        break;
      }
    }
    if constexpr (meta::is_same_v<Subelement, RefElTria>) {
      switch (idx) {
      case 0:
        out[0] = 0;
        out[1] = 1;
        out[2] = 3;
        break;
      case 1:
        out[0] = 2;
        out[1] = 3;
        out[2] = 1;
        break;
      case 2:
        out[0] = 0;
        out[1] = 3;
        out[2] = 2;
        break;
      case 3:
        out[0] = 0;
        out[1] = 2;
        out[2] = 1;
        break;
      }
    }
  }
};

} // namespace numeric::mesh

#endif
