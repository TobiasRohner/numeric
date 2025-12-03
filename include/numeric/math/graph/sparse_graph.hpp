#ifndef NUMERIC_MATH_GRAPH_SPARSE_GRAPH_HPP_
#define NUMERIC_MATH_GRAPH_SPARSE_GRAPH_HPP_

#include <numeric/memory/array.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::math::graph {

class SparseGraph {
public:
  SparseGraph(dim_t num_nodes, dim_t num_edges);
  SparseGraph(const memory::ArrayConstView<dim_t, 1> &degrees);
  SparseGraph(const SparseGraph &) = delete;
  SparseGraph(SparseGraph &&) = default;
  ~SparseGraph() = default;
  SparseGraph &operator=(const SparseGraph &) = delete;
  SparseGraph &operator=(SparseGraph &&) = default;

  dim_t num_nodes() const;
  dim_t num_edges() const;
  dim_t degree(dim_t node) const;
  auto degrees() const {
    return index_(memory::Slice(1, -1)) - index_(memory::Slice(0, -2));
  }
  memory::ArrayConstView<dim_t, 1> edges(dim_t node) const;
  memory::ArrayView<dim_t, 1> edges(dim_t node);

  memory::ArrayConstView<dim_t, 1> index_array() const;
  memory::ArrayView<dim_t, 1> index_array();
  memory::ArrayConstView<dim_t, 1> edge_array() const;
  memory::ArrayView<dim_t, 1> edge_array();

private:
  memory::Array<dim_t, 1> index_;
  memory::Array<dim_t, 1> edges_;
};

SparseGraph graph_union(const SparseGraph *graphs, dim_t N);

template <typename... Graphs, typename = meta::enable_if_t<(
                                  meta::is_same_v<Graphs, SparseGraph> && ...)>>
SparseGraph graph_union(const Graphs &...graphs) {
  SparseGraph G[] = {&graphs...};
  return graph_union(G, sizeof...(graphs));
}

} // namespace numeric::math::graph

#endif
