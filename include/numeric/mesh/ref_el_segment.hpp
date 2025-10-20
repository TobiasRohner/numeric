#ifndef NUMERIC_MESH_REF_EL_SEGMENT_HPP_
#define NUMERIC_MESH_REF_EL_SEGMENT_HPP_

#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::mesh {

struct RefElSegment {
  static constexpr dim_t dim = 1;
  static constexpr dim_t num_nodes = 2;
  static constexpr char name[] = "Segment";

  template <typename Subelement> static constexpr dim_t num_subelements() {
    if constexpr (meta::is_same_v<Subelement, RefElPoint>) {
      return 2;
    }
    return 0;
  }

  template <typename Scalar> static constexpr void get_nodes(Scalar (*out)[1]) {
    out[0][0] = 0;
    out[1][0] = 1;
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
      }
    }
  }
};

} // namespace numeric::mesh

#endif
