#ifndef NUMERIC_MESH_ELEMENT_LIST_VIEW_HPP_
#define NUMERIC_MESH_ELEMENT_LIST_VIEW_HPP_

#include <numeric/mesh/element_list.hpp>
#include <numeric/mesh/element_list_base.hpp>
#include <numeric/mesh/element_list_traits.hpp>

namespace numeric::mesh {

template <typename ElementType>
class ElementListView : public ElementListBase<ElementListView<ElementType>> {
  using super = ElementListBase<ElementListView<ElementType>>;

public:
  using element_t = ElementType;

  ElementListView() = default;
  ElementListView(ElementList<ElementType> &list) : elements_(list.indices()) {}
  ElementListView(const ElementListView &) = default;
  ElementListView(ElementListView &&) = default;
  ElementListView &operator=(const ElementListView &) = default;
  ElementListView &operator=(ElementListView &&) = default;

  memory::ArrayConstView<dim_t, 2> indices() const noexcept {
    return elements_;
  }
  memory::ArrayView<dim_t, 2> indices() noexcept { return elements_; }

  using super::num_elements;
  using super::num_nodes_per_element;

private:
  memory::ArrayView<dim_t, 2> elements_;
};

template <typename ElementType>
struct ElementListTraits<ElementListView<ElementType>> {
  using element_t = ElementType;
};

} // namespace numeric::mesh

#endif
