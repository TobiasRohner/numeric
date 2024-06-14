#include "common.hpp"
#include <fstream>
#include <iostream>
#include <numeric/math/quad/quad_rule.hpp>
#include <numeric/mesh/elements.hpp>
#include <string>
#include <vector>

using namespace numeric;

template <typename Element> std::string quad_rule_tikz_src(dim_t order) {
  const auto [points, weights] = math::quad::quad_rule<Element>(order);
  const dim_t d = points.shape(1);
  const dim_t N = points.shape(0);
  double max_weight = weights(0);
  for (dim_t i = 0; i < N; ++i) {
    const double w = weights(i);
    if (w > max_weight) {
      max_weight = w;
    }
  }
  std::string src;
  for (dim_t i = 0; i < N; ++i) {
    const double x = points(i, 0);
    const double y = d > 1 ? points(i, 1) : 0;
    const double z = d > 2 ? points(i, 2) : 0;
    const double size = weights(i) / max_weight;
    src += "\n\\draw plot[mark=*, mark size=" + std::to_string(size) +
           "] coordinates{(" + std::to_string(x) + ", " + std::to_string(y) +
           ", " + std::to_string(z) + ")};";
  }
  return src;
}

template <typename Element> void write_quad_rule(dim_t order) {
  const std::string filename = std::string("quad_rule_") + Element::name + "_" +
                               std::to_string(order) + ".tex";
  std::ofstream f(filename);
  const std::string ref_el_src =
      reference_element_edges_tikz_src<Element>(false);
  const std::string qr_src = quad_rule_tikz_src<Element>(order);
  const std::string src = draw_in_coordinate_system(ref_el_src + qr_src);
  f << src;
}

int main(int argc, char *argv[]) {
  static constexpr dim_t max_order = 6;
  for (dim_t order = 1; order <= max_order; ++order) {
    write_quad_rule<mesh::Segment<1>>(order);
    write_quad_rule<mesh::Tria<1>>(order);
    write_quad_rule<mesh::Quad<1>>(order);
    write_quad_rule<mesh::Tetra<1>>(order);
    write_quad_rule<mesh::Cube<1>>(order);
  }
  return 0;
}
