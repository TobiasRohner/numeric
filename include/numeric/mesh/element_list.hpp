#ifndef NUMERIC_MESH_ELEMENT_LIST_HPP_
#define NUMERIC_MESH_ELEMENT_LIST_HPP_

#include <numeric/mesh/element_list_base.hpp>
#include <numeric/mesh/element_list_traits.hpp>

namespace numeric::mesh {

template <typename ElementType>
class ElementList : public ElementListBase<ElementList<ElementType>> {
  using super = ElementListBase<ElementList<ElementType>>;

public:
  using element_t = ElementType;

  ElementList() = default;
  ElementList(dim_t num_elements)
      : elements_(memory::Shape<2>(num_nodes_per_element(), num_elements)) {}
  ElementList(const ElementList &) = delete;
  ElementList(ElementList &&) = default;
  ElementList &operator=(const ElementList &) = delete;
  ElementList &operator=(ElementList &&) = default;

  memory::ArrayConstView<dim_t, 2> indices() const noexcept {
    return elements_;
  }
  memory::ArrayView<dim_t, 2> indices() noexcept { return elements_; }

  using super::num_elements;
  using super::num_nodes_per_element;

private:
  memory::Array<dim_t, 2> elements_;
};

template <typename ElementType>
struct ElementListTraits<ElementList<ElementType>> {
  using element_t = ElementType;
};

} // namespace numeric::mesh

#endif
