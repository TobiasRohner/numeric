#ifndef NUMERIC_MESH_ELEMENT_LIST_BASE_HPP_
#define NUMERIC_MESH_ELEMENT_LIST_BASE_HPP_

#include <numeric/config.hpp>
#include <numeric/mesh/element_list_traits.hpp>

namespace numeric::mesh {

template <typename Derived> class ElementListBase {
public:
  using element_t = typename ElementListTraits<Derived>::element_t;

  dim_t num_elements() const noexcept { return derived().elements().shape(1); }
  static constexpr dim_t num_nodes_per_element() noexcept {
    return element_t::num_nodes();
  }

private:
  const Derived &derived() const noexcept {
    return static_cast<const Derived &>(*this);
  }
  Derived &derived() noexcept { return static_cast<Derived &>(*this); }
};

} // namespace numeric::mesh

#endif
