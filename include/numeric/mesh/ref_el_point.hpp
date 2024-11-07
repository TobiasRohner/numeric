#ifndef NUMERIC_MESH_REF_EL_POINT_HPP_
#define NUMERIC_MESH_REF_EL_POINT_HPP_

#include <numeric/meta/meta.hpp>

namespace numeric::mesh {

struct RefElPoint {
  static constexpr dim_t dim = 0;
  static constexpr dim_t num_nodes = 1;
  static constexpr char name[] = "Point";

  template <typename Subelement> static constexpr dim_t num_subelements() {
    return 0;
  }

  template <typename Scalar>
  static constexpr void get_nodes(Scalar (*out)[1]) {}

  template <typename Subelement>
  static constexpr void subelement_node_idxs(dim_t idx, dim_t *out) {}
};

} // namespace numeric::mesh

#endif
