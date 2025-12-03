#ifndef NUMERIC_EQUATIONS_FEM_FINITE_ELEMENT_VECTOR_HPP_
#define NUMERIC_EQUATIONS_FEM_FINITE_ELEMENT_VECTOR_HPP_

#include <numeric/hip/program.hpp>
#include <numeric/math/fes/fe_space.hpp>
#include <numeric/math/quad/quad_rule.hpp>
#include <numeric/utils/lambda.hpp>
#include <numeric/utils/type_name.hpp>
#include <string_view>

namespace numeric::equations::fem {

namespace internal {

hip::Kernel element_vector_build_kernel(std::string_view scalar,
                                        std::string_view scalar_mesh,
                                        std::string_view element_vector,
                                        std::string_view f);

}

/**
 * @brief Base template for FiniteElementVector. Causes a static assertion
 *        failure if used with an unsupported FE space.
 */
template <typename FES, typename ElementVectorFactory>
class FiniteElementVector {
  static_assert(
      !meta::is_same_v<FES, FES>,
      "FiniteElementVector is not specialized for the given FE space");
};

/**
 * @brief Assembles a global finite element vector (e.g., load vector)
 *        using element-wise contributions.
 *
 * @tparam Basis Basis function type.
 * @tparam ScalarMesh Scalar type used for mesh coordinates.
 * @tparam ElementTypes Variadic list of element types in the mesh.
 * @tparam ElementVectorFactory Factory for generating element vector operators.
 */
template <typename Basis, typename ScalarMesh, typename... ElementTypes,
          typename ElementVectorFactory>
class FiniteElementVector<
    math::fes::FESpace<Basis,
                       mesh::UnstructuredMesh<ScalarMesh, ElementTypes...>>,
    ElementVectorFactory> {
public:
  using scalar_t = typename ElementVectorFactory::scalar_t;
  using scalar_mesh_t = ScalarMesh;
  using basis_t = Basis;
  using mesh_t = mesh::UnstructuredMesh<ScalarMesh, ElementTypes...>;
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;
  template <typename Element>
  using element_vector_t =
      typename ElementVectorFactory::template create<Element>;

  // TODO: Is this order correct?
  /**
   * @brief Construct the FiniteElementVector with default quadrature order.
   *        Uses 2 * basis polynomial order for accuracy.
   *
   * @param fes Finite element space.
   */
  FiniteElementVector(const std::shared_ptr<fes_t> &fes)
      : FiniteElementVector(fes, 2 * basis_t::order) {}

  /**
   * @brief Construct the FiniteElementVector with specified quadrature order.
   *
   * @param fes Finite element space.
   * @param order Polynomial integration order for quadrature.
   */
  FiniteElementVector(const std::shared_ptr<fes_t> &fes, dim_t order)
      : FiniteElementVector(fes, build_qr(order, fes->memory_type())) {}

  void to(memory::MemoryType memory_type) {
    fes_->to(memory_type);
    work_.to(memory_type);
    (qr_points_.template get<ElementTypes>().to(memory_type), ...);
    (qr_weights_.template get<ElementTypes>().to(memory_type), ...);
    ((elem_vecs_.template get<element_vector_t<ElementTypes>>() =
          std::move(element_vector_t<ElementTypes>(
              qr_points_.template get<ElementTypes>(),
              qr_weights_.template get<ElementTypes>()))),
     ...);
  }

  memory::MemoryType memory_type() const noexcept {
    return fes_->memory_type();
  }

  std::shared_ptr<fes_t> fes() { return fes_; }

  std::shared_ptr<const fes_t> fes() const { return fes_; }

  /**
   * @brief Assemble the global finite element vector by applying
   *        the element-local operator to each element.
   *
   * @tparam Func A callable object with signature scalar_t(const scalar_t* x)
   * @param fes Finite element space.
   * @param f Function to evaluate at global coordinates.
   * @param out Output array to write the assembled vector to.
   */
  template <typename Func>
  void assemble(const utils::Lambda<Func> &f,
                memory::ArrayView<scalar_t, 1> out) const {
    // Check if the provided output vector is consistent with the FE Space
    NUMERIC_ERROR_IF(fes_->num_dofs() != out.shape(0),
                     "Output vector has wrong shape. Expected {}, but got {}.",
                     fes_->num_dofs(), out.shape(0));
    const memory::MemoryType fes_mt = fes_->memory_type();
    const memory::MemoryType out_mt = out.memory_type();
    NUMERIC_ERROR_IF(
        !((is_host_accessible(fes_mt) && is_host_accessible(out_mt)) ||
          (is_device_accessible(fes_mt) && is_device_accessible(out_mt))),
        "Output vector must be accessible from the same device that the FE "
        "space resides on, but got {} for the FES and {} for the output "
        "vector.",
        to_string(fes_mt), to_string(out_mt));

    // Initialize result vector
    out = 0;

    const auto mesh = fes_->mesh();
    const auto vertices = mesh->vertices();

    // Iterate over all element types using the for_all_element_types helper
    mesh_t::for_all_element_types([&]<typename ElementType>(
                                      meta::type_tag<ElementType>) {
      const auto elements = mesh->template get_elements<ElementType>();
      const auto dofs = fes_->template dof_map<ElementType>();
      const element_vector_t<ElementType> &element_vector =
          elem_vecs_.template get<element_vector_t<ElementType>>();
      if (is_host_accessible(memory_type())) {
        build_vector_host<ElementType>(f.f, element_vector, vertices, elements,
                                       dofs, out);
      } else if (is_device_accessible(memory_type())) {
        build_vector_device<ElementType>(f, element_vector, vertices, elements,
                                         dofs, out);
      } else {
        NUMERIC_ERROR("Unsupperted memory type: {}", to_string(memory_type()));
      }
    });
  }

private:
  std::shared_ptr<fes_t> fes_;
  /// Temporary workspace memory
  mutable memory::Array<char, 1> work_;
  /// Quadrature rule points for each element type
  utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...> qr_points_;
  /// Quadrature rule weights for each element type
  utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>
      qr_weights_;
  /// Element-local operators for each element type
  utils::Tuple<element_vector_t<ElementTypes>...> elem_vecs_;

  /// Constructor used internally to initialize all data from quadrature rules
  FiniteElementVector(
      const std::shared_ptr<fes_t> &fes,
      utils::Tuple<
          utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...>,
          utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>>
          &&qr)
      : fes_(fes),
        work_(memory::Shape<1>(apply_work_size(fes->mesh()->world_dim())),
              fes->memory_type()),
        qr_points_(std::move(qr.template get<0>())),
        qr_weights_(std::move(qr.template get<1>())),
        elem_vecs_(element_vector_t<ElementTypes>(
            qr_points_.template get<ElementTypes>(),
            qr_weights_.template get<ElementTypes>())...) {}

  /// Helper to build quadrature rules for all element types
  static utils::Tuple<
      utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...>,
      utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>>
  build_qr(dim_t order, memory::MemoryType memory_type) {
    utils::Tuple<
        utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...>,
        utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>>
        qr;
    const auto init_qr = [&]<typename Element>() {
      auto [p, w] = math::quad::quad_rule<typename Element::ref_el_t>(
          Element::order * order);
      p.to(memory_type);
      w.to(memory_type);
      qr.template get<0>().template get<Element>() = std::move(p);
      qr.template get<1>().template get<Element>() = std::move(w);
    };
    ((init_qr.template operator()<ElementTypes>()), ...);
    return qr;
  }

  /// Compute total workspace size required based on the largest element type
  static constexpr dim_t apply_work_size(dim_t world_dim) {
    dim_t size = 0;
    ((size =
          math::max(size, apply_to_element_work_size<ElementTypes>(world_dim))),
     ...);
    return size;
  }

  /// Compute work buffer size for a specific element type
  template <typename Element>
  static constexpr dim_t apply_to_element_work_size(dim_t world_dim) {
    const dim_t nodes_size = world_dim * Element::num_nodes * sizeof(scalar_t);
    const dim_t apply_size =
        element_vector_t<Element>::apply_work_size(world_dim);
    return nodes_size + apply_size;
  }

  template <typename ElementType, typename Func>
  void
  build_vector_host(Func &&f,
                    const element_vector_t<ElementType> &element_vector,
                    const memory::ArrayConstView<scalar_mesh_t, 2> &vertices,
                    const memory::ArrayConstView<dim_t, 2> &elements,
                    const memory::ArrayConstView<dim_t, 2> &dofs,
                    memory::ArrayView<scalar_t, 1> out) const {
    static constexpr dim_t num_nodes = ElementType::num_nodes;
    static constexpr dim_t num_basis_functions =
        element_vector_t<ElementType>::num_basis_functions;
    const dim_t num_elements = elements.shape(1);
    const dim_t world_dim = vertices.shape(0);

    // Local node buffer: [world_dim][num_nodes]
    scalar_t(*nodes)[num_nodes] =
        reinterpret_cast<scalar_t(*)[num_nodes]>(work_.raw());

    // Pointer to the remaining work buffer
    void *apply_work = work_.raw() + world_dim * num_nodes * sizeof(scalar_t);

    // Temporary vector for local result
    scalar_t local_vector[num_basis_functions];

    // Loop over elements of this type
    for (dim_t element = 0; element < num_elements; ++element) {
      // Extract physical coordinates of the current element's nodes
      for (dim_t node = 0; node < num_nodes; ++node) {
        for (dim_t dim = 0; dim < world_dim; ++dim) {
          nodes[dim][node] = vertices(dim, elements(node, element));
        }
      }

      // Apply the element-local operator
      element_vector.apply(f, nodes, world_dim, local_vector, apply_work);

      // Scatter local contribution to global vector
      for (dim_t bf = 0; bf < num_basis_functions; ++bf) {
        out(dofs(bf, element)) += local_vector[bf];
      }
    }
  }

  template <typename ElementType, typename Func>
  void
  build_vector_device(const utils::Lambda<Func> &f,
                      const element_vector_t<ElementType> &element_vector,
                      const memory::ArrayConstView<scalar_mesh_t, 2> &vertices,
                      const memory::ArrayConstView<dim_t, 2> &elements,
                      const memory::ArrayConstView<dim_t, 2> &dofs,
                      memory::ArrayView<scalar_t, 1> out) const {
    NUMERIC_ERROR_IF(!is_device_accessible(vertices.memory_type()),
                     "Need vertices to be accessible on the device, but got {}",
                     to_string(vertices.memory_type()));
    NUMERIC_ERROR_IF(!is_device_accessible(elements.memory_type()),
                     "Need elements to be accessible on the device, but got {}",
                     to_string(elements.memory_type()));
    NUMERIC_ERROR_IF(!is_device_accessible(dofs.memory_type()),
                     "Need dofs to be accessible on the device, but got {}",
                     to_string(dofs.memory_type()));
    NUMERIC_ERROR_IF(!is_device_accessible(out.memory_type()),
                     "Need out to be accessible on the device, but got {}",
                     to_string(out.memory_type()));
    NUMERIC_ERROR_IF(
        (!is_device_accessible(
             qr_points_.template get<ElementTypes>().memory_type()) ||
         ...),
        "Need quad rules to be accessible on the device, but at least one is "
        "not");
    NUMERIC_ERROR_IF(
        (!is_device_accessible(
             qr_weights_.template get<ElementTypes>().memory_type()) ||
         ...),
        "Need quad rules to be accessible on the device, but at least one is "
        "not");
    static hip::Kernel kernel = internal::element_vector_build_kernel(
        utils::type_name<scalar_t>(), utils::type_name<scalar_mesh_t>(),
        utils::type_name<element_vector_t<ElementType>>(), f.source);
    hip::Device device;
    static constexpr dim_t num_nodes = ElementType::num_nodes;
    const dim_t world_dim = vertices.shape(0);
    for (const auto &group :
         fes_->template independent_element_groups<ElementType>()) {
      const dim_t num_elements = group.shape(0);
      std::cout << "Launching kernel for group of " << num_elements << " "
                << ElementType::name << std::endl;
      const unsigned shared_mem_per_thread =
          world_dim * num_nodes * sizeof(scalar_t) +
          element_vector.apply_work_size(world_dim);
      const unsigned max_threads_per_block = math::min(
          device.max_threads_per_block(),
          device.max_shared_memory_per_block() / shared_mem_per_thread);
      const unsigned num_blocks =
          math::div_up(num_elements, max_threads_per_block);
      const unsigned num_threads = math::div_up(num_elements, num_blocks);
      hip::LaunchParams lp;
      lp.grid_dim_x = num_blocks;
      lp.grid_dim_y = 1;
      lp.grid_dim_z = 1;
      lp.block_dim_x = num_threads;
      lp.block_dim_y = 1;
      lp.block_dim_z = 1;
      lp.shared_mem_bytes = num_threads * shared_mem_per_thread;
      kernel(lp, hip::Stream(device), element_vector, group, vertices, elements,
             dofs, out);
    }
  }
};

} // namespace numeric::equations::fem

#endif
