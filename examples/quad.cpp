#include <iostream>
#include <numeric/math/quad/quad_rule.hpp>

int main(int argc, char *argv[]) {
  const int order = std::stoi(argv[1]);

  const auto [points, weights] =
      numeric::math::quad::quad_rule<numeric::mesh::RefElTria>(order);

  const numeric::dim_t num_points = points.shape(0);
  std::cout << "points = [(" << points(0, 0) << ", " << points(0, 1) << ")";
  for (numeric::dim_t i = 1; i < num_points; ++i) {
    std::cout << ", (" << points(i, 0) << ", " << points(i, 1) << ")";
  }
  std::cout << "]\nweights = [" << weights(0);
  for (numeric::dim_t i = 1; i < num_points; ++i) {
    std::cout << ", " << weights(i);
  }
  std::cout << "]\n";
  std::cout << std::endl;

  return 0;
}
