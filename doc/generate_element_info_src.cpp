#include <fstream>
#include <iostream>
#include <numeric/mesh/elements.hpp>
#include <numeric/meta/meta.hpp>
#include <string>
#include <vector>

using namespace numeric;

template <typename Element> struct ReferenceVertices {
  static_assert(!meta::is_same_v<Element, Element>,
                "No overload found for the given element");
};

template <dim_t Order> struct ReferenceVertices<mesh::Point<Order>> {
  static constexpr int num_verts = 1;
  static constexpr double verts[1][3] = {{0, 0, 0}};
};

template <dim_t Order> struct ReferenceVertices<mesh::Segment<Order>> {
  static constexpr int num_verts = 2;
  static constexpr double verts[2][3] = {{0, 0, 0}, {1, 0, 0}};
};

template <dim_t Order> struct ReferenceVertices<mesh::Tria<Order>> {
  static constexpr int num_verts = 3;
  static constexpr double verts[3][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
};

template <dim_t Order> struct ReferenceVertices<mesh::Quad<Order>> {
  static constexpr int num_verts = 4;
  static constexpr double verts[4][3] = {
      {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
};

template <dim_t Order> struct ReferenceVertices<mesh::Tetra<Order>> {
  static constexpr int num_verts = 4;
  static constexpr double verts[4][3] = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
};

template <dim_t Order> struct ReferenceVertices<mesh::Cube<Order>> {
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
  using point_t = mesh::Point<Element::order>;
  using segment_t = mesh::Segment<Element::order>;
  std::string src;
  std::vector<std::string> points;
  for (int i = 0; i < ref::num_verts; ++i) {
    const std::string point = "(" + std::to_string(ref::verts[i][0]) + "," +
                              std::to_string(ref::verts[i][1]) + "," +
                              std::to_string(ref::verts[i][2]) + ")";
    points.emplace_back(point);
  }
  if constexpr (meta::is_same_v<Element, point_t>) {
    src += "\n\\draw plot[mark=*, mark size=1] coordinates{" + points[0] + "};";
  }
  dim_t node_idxs[point_t::num_nodes()];
  for (int point = 0; point < Element::template num_subelements<point_t>();
       ++point) {
    Element::template subelement_node_idxs<point_t>(point, node_idxs);
    src += "\n\\draw plot[mark=*, mark size=1] coordinates{" +
           points[node_idxs[0]] + "};";
  }
  return src;
}

template <typename Element> std::string reference_element_edges_tikz_src() {
  using ref = ReferenceVertices<Element>;
  using point_t = mesh::Point<Element::order>;
  using segment_t = mesh::Segment<Element::order>;
  std::string src;
  std::vector<std::string> points;
  for (int i = 0; i < ref::num_verts; ++i) {
    const std::string point = "(" + std::to_string(ref::verts[i][0]) + "," +
                              std::to_string(ref::verts[i][1]) + "," +
                              std::to_string(ref::verts[i][2]) + ")";
    points.emplace_back(point);
  }
  if constexpr (meta::is_same_v<Element, segment_t>) {
    src += "\n\\draw[->-,thick] " + points[0] + " -- " + points[1] + ";";
  }
  dim_t node_idxs[segment_t::num_nodes()];
  for (int segment = 0;
       segment < Element::template num_subelements<segment_t>(); ++segment) {
    Element::template subelement_node_idxs<segment_t>(segment, node_idxs);
    src += "\n\\draw[->-,thick] " + points[node_idxs[0]] + " -- " +
           points[node_idxs[1]] + ";";
  }
  return src;
}

template <typename Element> std::string reference_element_tikz_src() {
  return reference_element_edges_tikz_src<Element>() +
         reference_element_vertices_tikz_src<Element>();
}

template <typename Element, typename Subelement>
std::string subelement_tikz_src() {
  using ref = ReferenceVertices<Element>;
  using point_t = mesh::Point<Element::order>;
  std::string src;
  dim_t node_idxs[Subelement::num_nodes()];
  dim_t point_idxs[point_t::num_nodes()];
  for (int sub = 0; sub < Element::template num_subelements<Subelement>();
       ++sub) {
    Element::template subelement_node_idxs<Subelement>(sub, node_idxs);
    double x = 0;
    double y = 0;
    double z = 0;
    if constexpr (meta::is_same_v<Subelement, point_t>) {
      x = ref::verts[node_idxs[0]][0];
      y = ref::verts[node_idxs[0]][1];
      z = ref::verts[node_idxs[0]][2];
    } else {
      for (int vert = 0; vert < Subelement::template num_subelements<point_t>();
           ++vert) {
        Subelement::template subelement_node_idxs<point_t>(vert, point_idxs);
        x += ref::verts[node_idxs[point_idxs[0]]][0] /
             Subelement::template num_subelements<point_t>();
        y += ref::verts[node_idxs[point_idxs[0]]][1] /
             Subelement::template num_subelements<point_t>();
        z += ref::verts[node_idxs[point_idxs[0]]][2] /
             Subelement::template num_subelements<point_t>();
      }
    }
    const std::string pt = "(" + std::to_string(x) + "," + std::to_string(y) +
                           "," + std::to_string(z) + ")";
    src += "\n\\draw " + pt + " node{$" + std::to_string(sub) + "$};";
  }
  return src;
}

template <typename Element> void write_reference_element() {
  const std::string filename =
      std::string("reference_element_") + Element::name + ".tex";
  std::ofstream f(filename);
  const std::string ref_el_src = reference_element_edges_tikz_src<Element>();
  const std::string src = draw_in_coordinate_system(ref_el_src);
  f << src;
}

template <typename Element, typename Subelement>
void write_subelement_indexing() {
  if constexpr (Element::template num_subelements<Subelement>() > 0) {
    const std::string filename = std::string("subelement_indexing_") +
                                 Element::name + std::string("_") +
                                 Subelement::name + ".tex";
    std::ofstream f(filename);
    const std::string ref_el_src = reference_element_edges_tikz_src<Element>();
    const std::string subelement_src =
        subelement_tikz_src<Element, Subelement>();
    const std::string src =
        draw_in_coordinate_system(ref_el_src + subelement_src);
    f << src;
  }
}

template <typename Element> void write_all_subelement_indexing() {
  write_subelement_indexing<Element, mesh::Point<Element::order>>();
  write_subelement_indexing<Element, mesh::Segment<Element::order>>();
  write_subelement_indexing<Element, mesh::Tria<Element::order>>();
  write_subelement_indexing<Element, mesh::Quad<Element::order>>();
}

int main(int argc, char *argv[]) {
  write_reference_element<mesh::Point<1>>();
  write_reference_element<mesh::Segment<1>>();
  write_reference_element<mesh::Tria<1>>();
  write_reference_element<mesh::Quad<1>>();
  write_reference_element<mesh::Tetra<1>>();
  write_reference_element<mesh::Cube<1>>();
  write_all_subelement_indexing<mesh::Point<1>>();
  write_all_subelement_indexing<mesh::Segment<1>>();
  write_all_subelement_indexing<mesh::Tria<1>>();
  write_all_subelement_indexing<mesh::Quad<1>>();
  write_all_subelement_indexing<mesh::Tetra<1>>();
  write_all_subelement_indexing<mesh::Cube<1>>();
  return 0;
}
