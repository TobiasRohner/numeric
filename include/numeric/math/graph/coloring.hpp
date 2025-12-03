#ifndef NUMERIC_MATH_GRAPH_COLORING_HPP_
#define NUMERIC_MATH_GRAPH_COLORING_HPP_

#include <numeric/math/graph/sparse_graph.hpp>
#include <numeric/memory/array.hpp>

namespace numeric::math::graph {

memory::Array<dim_t, 1> dsatur(const SparseGraph &graph);

}

#endif
