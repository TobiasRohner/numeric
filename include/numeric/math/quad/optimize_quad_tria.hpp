#ifndef NUMERIC_MATH_QUAD_OPTIMIZE_QUAD_TRIA_HPP_
#define NUMERIC_MATH_QUAD_OPTIMIZE_QUAD_TRIA_HPP_

#include <numeric/memory/array.hpp>
#include <numeric/utils/tuple.hpp>

namespace numeric::math::quad {

utils::Tuple<memory::Array<double, 2>, memory::Array<double, 1>>
optimize_quad_tria(dim_t order, dim_t N0, dim_t N1, dim_t N2);

}

#endif
