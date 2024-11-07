#ifndef NUMERIC_DOC_COMMON_HPP_
#define NUMERIC_DOC_COMMON_HPP_

#include <numeric/mesh/elements.hpp>
#include <numeric/meta/meta.hpp>
#include <string>
#include <vector>

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
  std::string src;
  double ref_nodes[Element::num_nodes][Element::dim > 0 ? Element::dim : 1];
  Element::get_nodes(ref_nodes);
  std::vector<std::string> points;
  for (int i = 0; i < Element::num_nodes; ++i) {
    const std::string point = "(" + std::to_string(ref_nodes[i][0]) + "," +
                              std::to_string(ref_nodes[i][1]) + "," +
                              std::to_string(ref_nodes[i][2]) + ")";
    points.emplace_back(point);
  }
  if constexpr (numeric::meta::is_same_v<Element, numeric::mesh::RefElPoint>) {
    src += "\n\\draw plot[mark=*, mark size=1] coordinates{" + points[0] + "};";
  }
  numeric::dim_t node_idxs[numeric::mesh::RefElPoint::num_nodes];
  for (int point = 0;
       point < Element::template num_subelements<numeric::mesh::RefElPoint>();
       ++point) {
    Element::template subelement_node_idxs<numeric::mesh::RefElPoint>(
        point, node_idxs);
    src += "\n\\draw plot[mark=*, mark size=1] coordinates{" +
           points[node_idxs[0]] + "};";
  }
  return src;
}

template <typename Element>
std::string reference_element_edges_tikz_src(bool indicate_direction = true) {
  std::string src;
  double ref_nodes[Element::num_nodes][Element::dim > 0 ? Element::dim : 1];
  Element::get_nodes(ref_nodes);
  std::vector<std::string> points;
  for (int i = 0; i < Element::num_nodes; ++i) {
    const std::string point = "(" + std::to_string(ref_nodes[i][0]) + "," +
                              std::to_string(ref_nodes[i][1]) + "," +
                              std::to_string(ref_nodes[i][2]) + ")";
    points.emplace_back(point);
  }
  if constexpr (numeric::meta::is_same_v<Element,
                                         numeric::mesh::RefElSegment>) {
    if (indicate_direction) {
      src += "\n\\draw[->-,thick] ";
    } else {
      src += "\n\\draw[thick] ";
    }
    src += points[0] + " -- " + points[1] + ";";
  }
  numeric::dim_t node_idxs[numeric::mesh::RefElSegment::num_nodes];
  for (int segment = 0;
       segment <
       Element::template num_subelements<numeric::mesh::RefElSegment>();
       ++segment) {
    Element::template subelement_node_idxs<numeric::mesh::RefElSegment>(
        segment, node_idxs);
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
