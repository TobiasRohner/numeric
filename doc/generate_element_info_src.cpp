#include "common.hpp"
#include <fstream>
#include <iostream>
#include <numeric/mesh/elements.hpp>
#include <string>
#include <vector>

using namespace numeric;

template <typename Element, typename Subelement>
std::string subelement_tikz_src() {
  std::string src;
  double ref_nodes[Element::num_nodes][Element::dim];
  Element::get_nodes(ref_nodes);
  dim_t node_idxs[Subelement::num_nodes];
  dim_t point_idxs[mesh::RefElPoint::num_nodes];
  for (int sub = 0; sub < Element::template num_subelements<Subelement>();
       ++sub) {
    Element::template subelement_node_idxs<Subelement>(sub, node_idxs);
    double x = 0;
    double y = 0;
    double z = 0;
    if constexpr (meta::is_same_v<Subelement, mesh::RefElPoint>) {
      x = ref_nodes[node_idxs[0]][0];
      y = ref_nodes[node_idxs[0]][1];
      z = ref_nodes[node_idxs[0]][2];
    } else {
      for (int vert = 0;
           vert < Subelement::template num_subelements<mesh::RefElPoint>();
           ++vert) {
        Subelement::template subelement_node_idxs<mesh::RefElPoint>(vert,
                                                                    point_idxs);
        x += ref_nodes[node_idxs[point_idxs[0]]][0] /
             Subelement::template num_subelements<mesh::RefElPoint>();
        y += ref_nodes[node_idxs[point_idxs[0]]][1] /
             Subelement::template num_subelements<mesh::RefElPoint>();
        z += ref_nodes[node_idxs[point_idxs[0]]][2] /
             Subelement::template num_subelements<mesh::RefElPoint>();
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
  write_subelement_indexing<Element, mesh::RefElPoint>();
  write_subelement_indexing<Element, mesh::RefElSegment>();
  write_subelement_indexing<Element, mesh::RefElTria>();
  write_subelement_indexing<Element, mesh::RefElQuad>();
}

int main(int argc, char *argv[]) {
  write_reference_element<mesh::RefElPoint>();
  write_reference_element<mesh::RefElSegment>();
  write_reference_element<mesh::RefElTria>();
  write_reference_element<mesh::RefElQuad>();
  write_reference_element<mesh::RefElTetra>();
  write_reference_element<mesh::RefElCube>();
  write_all_subelement_indexing<mesh::RefElPoint>();
  write_all_subelement_indexing<mesh::RefElSegment>();
  write_all_subelement_indexing<mesh::RefElTria>();
  write_all_subelement_indexing<mesh::RefElQuad>();
  write_all_subelement_indexing<mesh::RefElTetra>();
  write_all_subelement_indexing<mesh::RefElCube>();
  return 0;
}
