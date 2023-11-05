#include <numeric/io/variable.hpp>

namespace numeric::io {

utils::Datatype Variable::datatype() const { return do_datatype(); }

std::vector<dim_t> Variable::dims() const { return do_dims(); }

} // namespace numeric::io
