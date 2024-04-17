#ifndef NUMERIC_MESH_ELEMENTS_HPP_
#define NUMERIC_MESH_ELEMENTS_HPP_

#include <numeric/config.hpp>

namespace numeric::mesh {

template <dim_t Order> struct Point {
  static constexpr dim_t num_nodes() { return 1; }
};

template <dim_t Order> struct Segment {
  static constexpr dim_t num_nodes() { return Order + 1; }
};

template <dim_t Order> struct Tria {
  static constexpr dim_t num_nodes() { return (Order + 1) * (Order + 2) / 2; }
};

template <dim_t Order> struct Quad {
  static constexpr dim_t num_nodes() { return (Order + 1) * (Order + 1); }
};

template <dim_t Order> struct Tetra {
  static constexpr dim_t num_nodes() {
    return (Order + 1) * (Order + 2) * (Order + 3) / 2;
  }
};

template <dim_t Order> struct Cube {
  static constexpr dim_t num_nodes() {
    return (Order + 1) * (Order + 1) * (Order + 1);
  }
};

} // namespace numeric::mesh

#endif
