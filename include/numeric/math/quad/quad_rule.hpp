#ifndef NUMERIC_MATH_QUAD_QUAD_RULE_HPP_
#define NUMERIC_MATH_QUAD_QUAD_RULE_HPP_

#include <numeric/memory/array.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/meta/type_tag.hpp>
#include <numeric/utils/tuple.hpp>

namespace numeric::math::quad {

namespace detail {

utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
gauss_legendre(dim_t num_points);

} // namespace detail

/**
 * @brief Computes the quadrature points and weights for a given element type
 * and order.
 *
 * This function template serves as a generic interface to compute the
 * quadrature points and weights for different types of mesh elements (e.g.,
 * segments, triangles, quadrilaterals, tetrahedrons, and cubes) based on the
 * specified order of accuracy. It utilizes the appropriate specialized
 * implementation from the `detail` namespace.
 *
 * @tparam Element The type of the mesh element for which to compute the
 * quadrature rule. This could be one of `mesh::Segment<Ord>`,
 * `mesh::Tria<Ord>`, `mesh::Quad<Ord>`, `mesh::Tetra<Ord>`, or
 * `mesh::Cube<Ord>`.
 * @param order The order of accuracy for the quadrature rule.
 *
 * @return A `utils::Tuple` containing:
 *   - A `memory::Array<double, 2>` representing the quadrature points, where
 * each row is a point.
 *   - A `memory::Array<double, 1>` representing the corresponding weights for
 * the quadrature points.
 */
template <typename RefEl>
utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
quad_rule(dim_t order) {
  static_assert(!meta::is_same_v<RefEl, RefEl>,
                "Numerical quadrature is not supported for the given element");
  return utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>();
}

template <>
utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
quad_rule<mesh::RefElSegment>(dim_t order);

template <>
utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
quad_rule<mesh::RefElTria>(dim_t order);

template <>
utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
quad_rule<mesh::RefElQuad>(dim_t order);

template <>
utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
quad_rule<mesh::RefElTetra>(dim_t order);

template <>
utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
quad_rule<mesh::RefElCube>(dim_t order);

} // namespace numeric::math::quad

#endif
