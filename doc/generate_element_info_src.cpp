#include "common.hpp"
#include <fstream>
#include <iostream>
#include <numeric/mesh/elements.hpp>
#include <string>
#include <vector>

using namespace numeric;

template <typename Element, typename Subelement>
std::string subelement_tikz_src() {
  using ref = ReferenceVertices<Element>;
  using point_t = mesh::Point<Element::order>;
  std::string src;
  dim_t node_idxs[Subelement::num_nodes];
  dim_t point_idxs[point_t::num_nodes];
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
