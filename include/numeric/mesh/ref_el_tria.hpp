#ifndef NUMERIC_MESH_REF_EL_TRIA_HPP_
#define NUMERIC_MESH_REF_EL_TRIA_HPP_

#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::mesh {

struct RefElTria {
  static constexpr dim_t dim = 2;
  static constexpr dim_t num_nodes = 3;
  static constexpr char name[] = "Tria";

  template <typename Subelement> static constexpr dim_t num_subelements() {
    if constexpr (meta::is_same_v<Subelement, RefElPoint>) {
      return 3;
    }
    if constexpr (meta::is_same_v<Subelement, RefElSegment>) {
      return 3;
    }
    return 0;
  }

  template <typename Scalar> static constexpr void get_nodes(Scalar (*out)[2]) {
    out[0][0] = 0;
    out[0][1] = 0;
    out[1][0] = 1;
    out[1][1] = 0;
    out[2][0] = 0;
    out[2][1] = 1;
  }

  template <typename Subelement>
  static constexpr void subelement_node_idxs(dim_t idx, dim_t *out) {
    if constexpr (meta::is_same_v<Subelement, RefElPoint>) {
      switch (idx) {
      case 0:
        out[0] = 0;
      case 1:
        out[0] = 1;
      case 2:
        out[0] = 2;
      }
    }
    if constexpr (meta::is_same_v<Subelement, RefElSegment>) {
      switch (idx) {
      case 0:
        out[0] = 0;
        out[1] = 1;
      case 1:
        out[0] = 1;
        out[1] = 2;
      case 2:
        out[0] = 2;
        out[1] = 0;
      }
    }
  }
};

} // namespace numeric::mesh

#endif
