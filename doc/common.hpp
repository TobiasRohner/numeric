#ifndef NUMERIC_DOC_COMMON_HPP_
#define NUMERIC_DOC_COMMON_HPP_

#include <numeric/mesh/elements.hpp>
#include <numeric/meta/meta.hpp>
#include <string>
#include <vector>

template <typename Element> struct ReferenceVertices {
  static_assert(!numeric::meta::is_same_v<Element, Element>,
                "No overload found for the given element");
};

template <numeric::dim_t Order>
struct ReferenceVertices<numeric::mesh::Point<Order>> {
  static constexpr int num_verts = 1;
  static constexpr double verts[1][3] = {{0, 0, 0}};
};

template <numeric::dim_t Order>
struct ReferenceVertices<numeric::mesh::Segment<Order>> {
  static constexpr int num_verts = 2;
  static constexpr double verts[2][3] = {{0, 0, 0}, {1, 0, 0}};
};

template <numeric::dim_t Order>
struct ReferenceVertices<numeric::mesh::Tria<Order>> {
  static constexpr int num_verts = 3;
  static constexpr double verts[3][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
};

template <numeric::dim_t Order>
struct ReferenceVertices<numeric::mesh::Quad<Order>> {
  static constexpr int num_verts = 4;
  static constexpr double verts[4][3] = {
      {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
};

template <numeric::dim_t Order>
struct ReferenceVertices<numeric::mesh::Tetra<Order>> {
  static constexpr int num_verts = 4;
  static constexpr double verts[4][3] = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
};

template <numeric::dim_t Order>
struct ReferenceVertices<numeric::mesh::Cube<Order>> {
  static constexpr int num_verts = 8;
  static constexpr double verts[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                                         {0, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                         {1, 1, 1}, {0, 1, 1}};
};

std::string draw_in_coordinate_system(const std::string &src) {
  static constexpr char pre[] = R"(
    \documentclass[tikz,convert]{standalone}
    \usepackage{graphicx}
    \usetikzlibrary{decorations.markings}
    \begin{document}
    \tikzset{->-/.style={decoration={
      markings,
      mark=at position 0.5 with {\arrow{>}}},postaction={decorate}}}
    \begin{tikzpicture}[scale=2]
      \scalebox{0.5}{%
	\draw[->] (0,0,0) -- (1.5,0,0) node[right] {$x$};
	\draw[->] (0,0,0) -- (0,1.5,0) node[above] {$y$};
	\draw[->] (0,0,0) -- (0,0,1.5) node[below left] {$z$};)";
  static constexpr char post[] = R"(
    }
    \end{tikzpicture}
    \end{document}
  )";
  return pre + src + post;
}

template <typename Element> std::string reference_element_vertices_tikz_src() {
  using ref = ReferenceVertices<Element>;
  using point_t = numeric::mesh::Point<Element::order>;
  using segment_t = numeric::mesh::Segment<Element::order>;
  std::string src;
  std::vector<std::string> points;
  for (int i = 0; i < ref::num_verts; ++i) {
    const std::string point = "(" + std::to_string(ref::verts[i][0]) + "," +
                              std::to_string(ref::verts[i][1]) + "," +
                              std::to_string(ref::verts[i][2]) + ")";
    points.emplace_back(point);
  }
  if constexpr (numeric::meta::is_same_v<Element, point_t>) {
    src += "\n\\draw plot[mark=*, mark size=1] coordinates{" + points[0] + "};";
  }
  numeric::dim_t node_idxs[point_t::num_nodes];
  for (int point = 0; point < Element::template num_subelements<point_t>();
       ++point) {
    Element::template subelement_node_idxs<point_t>(point, node_idxs);
    src += "\n\\draw plot[mark=*, mark size=1] coordinates{" +
           points[node_idxs[0]] + "};";
  }
  return src;
}

template <typename Element>
std::string reference_element_edges_tikz_src(bool indicate_direction = true) {
  using ref = ReferenceVertices<Element>;
  using point_t = numeric::mesh::Point<Element::order>;
  using segment_t = numeric::mesh::Segment<Element::order>;
  std::string src;
  std::vector<std::string> points;
  for (int i = 0; i < ref::num_verts; ++i) {
    const std::string point = "(" + std::to_string(ref::verts[i][0]) + "," +
                              std::to_string(ref::verts[i][1]) + "," +
                              std::to_string(ref::verts[i][2]) + ")";
    points.emplace_back(point);
  }
  if constexpr (numeric::meta::is_same_v<Element, segment_t>) {
    if (indicate_direction) {
      src += "\n\\draw[->-,thick] ";
    } else {
      src += "\n\\draw[thick] ";
    }
    src += points[0] + " -- " + points[1] + ";";
  }
  numeric::dim_t node_idxs[segment_t::num_nodes];
  for (int segment = 0;
       segment < Element::template num_subelements<segment_t>(); ++segment) {
    Element::template subelement_node_idxs<segment_t>(segment, node_idxs);
    if (indicate_direction) {
      src += "\n\\draw[->-,thick] ";
    } else {
      src += "\n\\draw[thick] ";
    }
    src += points[node_idxs[0]] + " -- " + points[node_idxs[1]] + ";";
  }
  return src;
}

template <typename Element> std::string reference_element_tikz_src() {
  return reference_element_edges_tikz_src<Element>() +
         reference_element_vertices_tikz_src<Element>();
}

#endif
