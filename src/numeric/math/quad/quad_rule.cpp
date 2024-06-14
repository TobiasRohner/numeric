#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <numeric/math/functions.hpp>
#include <numeric/math/quad/quad_rule.hpp>

namespace numeric::math::quad::detail {

extern std::tuple<size_t, const double *, const double *, const double *>
get_qr_tria(size_t order);
extern std::tuple<size_t, const double *, const double *, const double *,
                  const double *>
get_qr_tetra(size_t order);

utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
gauss_legendre(dim_t num_points) {
  using scalar_t =
      boost::multiprecision::number<boost::multiprecision::cpp_bin_float<
          57, boost::multiprecision::digit_base_2>>;
  const scalar_t pi = boost::math::constants::pi<scalar_t>();

  memory::Array<double, 2> points(memory::Shape<2>(num_points, 1),
                                  memory::MemoryType::HOST);
  memory::Array<double, 1> weights(memory::Shape<1>(num_points),
                                   memory::MemoryType::HOST);

  const dim_t m = (num_points + 1) / 2;
  for (dim_t i = 0; i < m; ++i) {
    scalar_t z = cos(pi * (i + 0.75) / (num_points + 0.5));
    scalar_t z1;
    scalar_t pp;
    do {
      scalar_t p1 = 1;
      scalar_t p2 = 0;
      scalar_t p3;
      for (dim_t j = 0; j < num_points; ++j) {
        p3 = p2;
        p2 = p1;
        p1 = ((2 * j + 1) * z * p2 - j * p3) / (j + 1);
      }
      pp = num_points * (z * p1 - p2) / (z * z - 1);
      z1 = z;
      z = z1 - p1 / pp;
    } while (abs(z - z1) > 1e-17);
    points(i, 0) = (0.5 * (1 - z)).convert_to<double>();
    points(num_points - i - 1, 0) = (0.5 * (1 + z)).convert_to<double>();
    weights(i) = (1 / ((1 - z * z) * pp * pp)).convert_to<double>();
    weights(num_points - i - 1) = weights(i);
  }

  return utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>(
      std::move(points), std::move(weights));
}

utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
quad_rule_segment(dim_t order) {
  const dim_t num_points = math::div_up(order + 1, 2);
  return gauss_legendre(num_points);
}

utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
quad_rule_tria(dim_t order) {
  const auto [num_points, x, y, w] = get_qr_tria(order);
  memory::Array<double, 2> points(memory::Shape<2>(num_points, 2),
                                  memory::MemoryType::HOST);
  memory::Array<double, 1> weights(memory::Shape<1>(num_points),
                                   memory::MemoryType::HOST);
  for (dim_t i = 0; i < num_points; ++i) {
    points(i, 0) = x[i];
    points(i, 1) = y[i];
    weights(i) = w[i];
  }
  return utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>(
      std::move(points), std::move(weights));
}

utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
quad_rule_quad(dim_t order) {
  const dim_t num_points = math::div_up(order + 1, 2);
  const auto [pgl, wgl] = gauss_legendre(num_points);
  memory::Array<double, 2> points(memory::Shape<2>(num_points * num_points, 2),
                                  memory::MemoryType::HOST);
  memory::Array<double, 1> weights(memory::Shape<1>(num_points * num_points),
                                   memory::MemoryType::HOST);
  for (dim_t i = 0; i < num_points; ++i) {
    for (dim_t j = 0; j < num_points; ++j) {
      const dim_t pt = i * num_points + j;
      points(pt, 0) = pgl(i, 0);
      points(pt, 1) = pgl(j, 0);
      weights(pt) = wgl(i) * wgl(j);
    }
  }
  return utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>(
      std::move(points), std::move(weights));
}

utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
quad_rule_tetra(dim_t order) {
  const auto [num_points, x, y, z, w] = get_qr_tetra(order);
  memory::Array<double, 2> points(memory::Shape<2>(num_points, 3),
                                  memory::MemoryType::HOST);
  memory::Array<double, 1> weights(memory::Shape<1>(num_points),
                                   memory::MemoryType::HOST);
  for (dim_t i = 0; i < num_points; ++i) {
    points(i, 0) = x[i];
    points(i, 1) = y[i];
    points(i, 2) = z[i];
    weights(i) = w[i];
  }
  return utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>(
      std::move(points), std::move(weights));
}

utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
quad_rule_cube(dim_t order) {
  const dim_t num_points = math::div_up(order + 1, 2);
  const auto [pgl, wgl] = gauss_legendre(num_points);
  memory::Array<double, 2> points(
      memory::Shape<2>(num_points * num_points * num_points, 3),
      memory::MemoryType::HOST);
  memory::Array<double, 1> weights(
      memory::Shape<1>(num_points * num_points * num_points),
      memory::MemoryType::HOST);
  for (dim_t i = 0; i < num_points; ++i) {
    for (dim_t j = 0; j < num_points; ++j) {
      for (dim_t k = 0; k < num_points; ++k) {
        const dim_t pt = i * num_points * num_points + j * num_points + k;
        points(pt, 0) = pgl(i, 0);
        points(pt, 1) = pgl(j, 0);
        points(pt, 2) = pgl(k, 0);
        weights(pt) = wgl(i) * wgl(j) * wgl(k);
      }
    }
  }
  return utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>(
      std::move(points), std::move(weights));
}

} // namespace numeric::math::quad::detail
