#include <numeric/math/graph/coloring.hpp>
#include <numeric/memory/linspace.hpp>

namespace numeric::math::graph {

memory::Array<dim_t, 1> dsatur(const SparseGraph &graph) {
  memory::Array<dim_t, 1> nodes =
      memory::linspace(static_cast<dim_t>(0), graph.num_nodes(),
                       graph.num_nodes(), false, memory::MemoryType::HOST);
  memory::Array<dim_t, 1> saturation(memory::Shape<1>(graph.num_nodes()),
                                     memory::MemoryType::HOST);
  saturation = 0;
  memory::Array<dim_t, 1> uncolored_degree = graph.degrees();
  memory::Array<dim_t, 1> location_in_heap =
      memory::linspace(static_cast<dim_t>(0), graph.num_nodes(),
                       graph.num_nodes(), false, memory::MemoryType::HOST);
  memory::Array<dim_t, 1> color(memory::Shape<1>(graph.num_nodes()),
                                memory::MemoryType::HOST);
  color = -1;
  dim_t heap_size = 1;

  const auto print = [&](dim_t i) {
    std::cout << "(" << nodes(i) << ", " << saturation(i) << ", "
              << uncolored_degree(i) << ")";
  };
  const auto swap_elements = [&](dim_t i, dim_t j) {
    std::swap(nodes(i), nodes(j));
    std::swap(saturation(i), saturation(j));
    std::swap(uncolored_degree(i), uncolored_degree(j));
    location_in_heap(nodes(i)) = i;
    location_in_heap(nodes(j)) = j;
  };
  const auto greater = [&](dim_t i, dim_t j) {
    const dim_t si = saturation(i);
    const dim_t sj = saturation(j);
    if (si > sj) {
      return true;
    } else if (si < sj) {
      return false;
    }
    const dim_t udi = uncolored_degree(i);
    const dim_t udj = uncolored_degree(j);
    if (udi > udj) {
      return true;
    } else {
      return false;
    }
  };
  const auto parent = [](dim_t i) { return (i - 1) / 2; };
  const auto left_child = [](dim_t i) { return 2 * i + 1; };
  const auto right_child = [](dim_t i) { return 2 * i + 2; };
  const auto is_leaf = [&](dim_t i) { return left_child(i) >= heap_size; };
  const auto max_child = [&](dim_t i) {
    const dim_t lc = left_child(i);
    const dim_t rc = right_child(i);
    if (lc >= heap_size) {
      return i;
    } else if (rc >= heap_size) {
      return lc;
    } else if (greater(lc, rc)) {
      return lc;
    } else {
      return rc;
    }
  };
  const auto bubble_up = [&](dim_t i) {
    dim_t p = parent(i);
    while (i > 0 && greater(i, p)) {
      swap_elements(p, i);
      i = p;
      p = parent(i);
    }
  };
  const auto bubble_down = [&](dim_t i) {
    dim_t lc = left_child(i);
    dim_t rc = right_child(i);
    dim_t mc = max_child(i);
    while (!is_leaf(i) && greater(mc, i)) {
      swap_elements(mc, i);
      i = mc;
      mc = max_child(i);
    }
  };
  const auto bubble = [&](dim_t i) {
    if (greater(i, parent(i))) {
      bubble_up(i);
    } else if (greater(max_child(i), i)) {
      bubble_down(i);
    }
  };
  const auto insert = [&]() {
    heap_size += 1;
    bubble_up(heap_size - 1);
  };
  const auto extract = [&]() {
    heap_size -= 1;
    swap_elements(0, heap_size);
    bubble_down(0);
  };

  // Heapify everything
  for (dim_t i = 1; i < graph.num_nodes(); ++i) {
    insert();
  }

  // Main DSatur iteration
  dim_t num_colors = 0;
  for (dim_t iter = 0; iter < graph.num_nodes(); ++iter) {
    // Find node with highest degree of saturation
    extract();
    const dim_t node = nodes(heap_size);
    // Find smallest possible color
    std::vector<bool> neighboring_colors(num_colors, false);
    const memory::ArrayConstView<dim_t, 1> e = graph.edges(node);
    for (dim_t i = 0; i < e.shape(0); ++i) {
      const dim_t neighboring_color = color(e(i));
      if (neighboring_color != -1) {
        neighboring_colors[neighboring_color] = true;
      }
    }
    dim_t new_color;
    for (new_color = 0; new_color < num_colors; ++new_color) {
      if (!neighboring_colors[new_color]) {
        break;
      }
    }
    if (new_color == num_colors) {
      ++num_colors;
    }
    // Assign this color to the node and update all neighbors
    color(node) = new_color;
    for (dim_t i = 0; i < e.shape(0); ++i) {
      const dim_t neighbor_heap_idx = location_in_heap(e(i));
      if (neighbor_heap_idx >= heap_size) {
        continue;
      }
      saturation(neighbor_heap_idx) += 1;
      uncolored_degree(neighbor_heap_idx) -= 1;
      bubble(neighbor_heap_idx);
    }
  }
  return color;
}

} // namespace numeric::math::graph
