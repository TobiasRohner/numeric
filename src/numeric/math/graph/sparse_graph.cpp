#include <numeric/math/graph/sparse_graph.hpp>
#include <set>
#include <vector>

namespace numeric::math::graph {

SparseGraph::SparseGraph(dim_t num_nodes, dim_t num_edges)
    : index_(memory::Shape<1>(num_nodes + 1), memory::MemoryType::HOST),
      edges_(memory::Shape<1>(num_edges), memory::MemoryType::HOST) {}

SparseGraph::SparseGraph(const memory::ArrayConstView<dim_t, 1> &degrees)
    : index_(memory::Shape<1>(degrees.shape(0) + 1), memory::MemoryType::HOST) {
  index_(0) = 0;
  for (dim_t i = 0; i < num_nodes(); ++i) {
    index_(i + 1) = index_(i) + degrees(i);
  }
  edges_ = memory::Array<dim_t, 1>(memory::Shape<1>(index_(-1)),
                                   memory::MemoryType::HOST);
}

dim_t SparseGraph::num_nodes() const { return index_.shape(0) - 1; }

dim_t SparseGraph::num_edges() const { return edges_.shape(0); }

dim_t SparseGraph::degree(dim_t node) const {
  return index_(node + 1) - index_(node);
}

memory::ArrayConstView<dim_t, 1> SparseGraph::edges(dim_t node) const {
  return edges_(memory::Slice(index_(node), index_(node + 1)));
}

memory::ArrayView<dim_t, 1> SparseGraph::edges(dim_t node) {
  return edges_(memory::Slice(index_(node), index_(node + 1)));
}

memory::ArrayConstView<dim_t, 1> SparseGraph::index_array() const {
  return index_;
}

memory::ArrayView<dim_t, 1> SparseGraph::index_array() { return index_; }

memory::ArrayConstView<dim_t, 1> SparseGraph::edge_array() const {
  return edges_;
}

memory::ArrayView<dim_t, 1> SparseGraph::edge_array() { return edges_; }

SparseGraph graph_union(const SparseGraph *graphs, dim_t N) {
  const dim_t num_nodes = graphs[0].num_nodes();
  std::vector<std::set<dim_t>> edges(num_nodes);
  for (dim_t i = 0; i < N; ++i) {
    const SparseGraph &graph = graphs[i];
    for (dim_t node = 0; node < graph.num_nodes(); ++node) {
      const memory::ArrayConstView<dim_t, 1> E = graph.edges(node);
      const dim_t degree = E.shape(0);
      edges[node].insert(E.raw(), E.raw() + degree);
    }
  }
  dim_t num_edges = 0;
  for (dim_t i = 0; i < num_nodes; ++i) {
    num_edges += edges[i].size();
  }
  SparseGraph graph(num_nodes, num_edges);
  auto ia = graph.index_array();
  auto ea = graph.edge_array();
  ia(0) = 0;
  for (dim_t node = 0; node < num_nodes; ++node) {
    ia(node + 1) = ia(node) + edges[node].size();
    dim_t idx = 0;
    for (dim_t edge : edges[node]) {
      ea(ia(node) + idx) = edge;
      ++idx;
    }
  }
  return graph;
}

} // namespace numeric::math::graph
