#ifndef NUMERIC_MESH_NEIGHBOR_RELATION_HPP_
#define NUMERIC_MESH_NEIGHBOR_RELATION_HPP_

#include <numeric/math/graph/sparse_graph.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/mesh/subelement_relation.hpp>

namespace numeric::mesh {

math::graph::SparseGraph
neighboring(dim_t num_subelements,
            const memory::ArrayConstView<dim_t, 2> &relations);

template <typename BaseElement, typename ConnectingElement, typename Scalar,
          typename... ElementTypes>
math::graph::SparseGraph
neighboring_by(const UnstructuredMeshConstView<Scalar, ElementTypes...> &mesh) {
  const auto [subelements, relations] =
      subelement_relation<ConnectingElement>(mesh);
  return neighboring_by(subelements.shape(1),
                        relations.template get<BaseElement>());
}

} // namespace numeric::mesh

#endif
