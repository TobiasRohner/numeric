#ifndef NUMERIC_EQUATIONS_FEM_FINITE_ELEMENT_MATRIX_HPP_
#define NUMERIC_EQUATIONS_FEM_FINITE_ELEMENT_MATRIX_HPP_

#include <numeric/hip/program.hpp>
#include <numeric/math/fes/fe_space.hpp>
#include <numeric/math/linear_operator.hpp>
#include <numeric/math/quad/quad_rule.hpp>

namespace numeric::equations::fem {

namespace internal {

hip::Kernel element_matrix_build_kernel(std::string_view scalar,
                                        std::string_view scalar_mesh,
                                        std::string_view element_matrix);

}

/**
 * @brief Primary template for FiniteElementMatrix — not implemented.
 *
 * This serves as a static_assert placeholder for unsupported FE spaces.
 */
template <typename FES, typename ElementMatrixFactory>
class FiniteElementMatrix {
  static_assert(
      !meta::is_same_v<FES, FES>,
      "FiniteElementMatrix is not specialized for the given FE space");
};

/**
 * @brief Specialization of FiniteElementMatrix for unstructured meshes.
 *
 * This class performs matrix-vector products using element-level matrices
 * applied over the whole mesh using the provided FE space and a matrix factory.
 *
 * @tparam Basis Type of basis functions.
 * @tparam ScalarMesh Scalar type used in mesh geometry.
 * @tparam ElementTypes Variadic list of mesh element types (e.g., triangles,
 * tets).
 * @tparam ElementMatrixFactory Factory type for producing local element
 * matrices.
 */
template <typename Basis, typename ScalarMesh, typename... ElementTypes,
          typename ElementMatrixFactory>
class FiniteElementMatrix<
    math::fes::FESpace<Basis,
                       mesh::UnstructuredMesh<ScalarMesh, ElementTypes...>>,
    ElementMatrixFactory>
    : public math::LinearOperator<typename ElementMatrixFactory::scalar_t> {
  using super = math::LinearOperator<typename ElementMatrixFactory::scalar_t>;

public:
  using scalar_t =
      typename ElementMatrixFactory::scalar_t; ///< Scalar type used for
                                               ///< computation
  using scalar_mesh_t = ScalarMesh; ///< Scalar type for mesh coordinates
  using basis_t = Basis;            ///< Basis function type
  using mesh_t =
      mesh::UnstructuredMesh<ScalarMesh, ElementTypes...>; ///< Mesh type
  using fes_t = math::fes::FESpace<basis_t, mesh_t>;       ///< FE space type

  /// Element matrix type for a given element
  template <typename Element>
  using element_matrix_t =
      typename ElementMatrixFactory::template create<Element>;

  // TODO: Is this order correct?
  /**
   * @brief Constructor with default quadrature order.
   */
  FiniteElementMatrix(const std::shared_ptr<fes_t> &fes)
      : FiniteElementMatrix(fes, 2 * basis_t::order) {}

  /**
   * @brief Constructor with custom quadrature order.
   */
  FiniteElementMatrix(const std::shared_ptr<fes_t> &fes, dim_t order)
      : FiniteElementMatrix(fes, build_qr(order)) {}

  virtual ~FiniteElementMatrix() override = default;

  virtual void to(memory::MemoryType memory_type) override {
    fes_->to(memory_type);
    (qr_points_.template get<ElementTypes>().to(memory_type), ...);
    (qr_weights_.template get<ElementTypes>().to(memory_type), ...);
    ((elem_mats_.template get<element_matrix_t<ElementTypes>>() =
          std::move(element_matrix_t<ElementTypes>(
              qr_points_.template get<ElementTypes>(),
              qr_weights_.template get<ElementTypes>()))),
     ...);
  }

  virtual memory::MemoryType memory_type() const override {
    return fes_->memory_type();
  }

  std::shared_ptr<fes_t> fes() { return fes_; }

  std::shared_ptr<const fes_t> fes() const { return fes_; }

  virtual memory::Shape<2> shape() const override {
    const dim_t N = fes_->num_dofs();
    return memory::Shape<2>(N, N);
  }

  virtual dim_t shape(dim_t i) const override { return fes_->num_dofs(); }

  /**
   * @brief Applies the finite element matrix to a vector.
   *
   * Performs matrix-vector multiplication: `out = A * u`
   *
   * @param fes The finite element space.
   * @param u Input coefficient vector.
   * @param out Output vector.
   */
  virtual void operator()(const memory::ArrayConstView<scalar_t, 1> &u,
                          memory::ArrayView<scalar_t, 1> out) const override {
    // Clear out vector
    out = 0;
    // Compute matrix-vector product for each element type
    ((apply_to_element<ElementTypes>(u, out, work_.raw())), ...);
  }

  using super::operator();

private:
  std::shared_ptr<fes_t> fes_;
  mutable memory::Array<char, 1> work_; ///< Workspace buffer
  utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...>
      qr_points_; ///< Quadrature points per element type
  utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>
      qr_weights_; ///< Quadrature weights per element type
  utils::Tuple<element_matrix_t<ElementTypes>...>
      elem_mats_; ///< Element matrix functors

  /**
   * @brief Core constructor using prebuilt quadrature rules.
   */
  FiniteElementMatrix(
      const std::shared_ptr<fes_t> &fes,
      utils::Tuple<
          utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...>,
          utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>>
          &&qr)
      : fes_(fes),
        work_(memory::Shape<1>(apply_work_size(fes->mesh()->world_dim())),
              memory::MemoryType::HOST),
        qr_points_(std::move(qr.template get<0>())),
        qr_weights_(std::move(qr.template get<1>())),
        elem_mats_(element_matrix_t<ElementTypes>(
            qr_points_.template get<ElementTypes>(),
            qr_weights_.template get<ElementTypes>())...) {}

  /**
   * @brief Builds quadrature rules for all element types.
   */
  static utils::Tuple<
      utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...>,
      utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>>
  build_qr(dim_t order) {
    utils::Tuple<
        utils::TypeIndexedMap<memory::Array<scalar_t, 2>, ElementTypes...>,
        utils::TypeIndexedMap<memory::Array<scalar_t, 1>, ElementTypes...>>
        qr;
    const auto init_qr = [&]<typename Element>() {
      auto [p, w] = math::quad::quad_rule<typename Element::ref_el_t>(
          Element::order * order);
      qr.template get<0>().template get<Element>() = std::move(p);
      qr.template get<1>().template get<Element>() = std::move(w);
    };
    ((init_qr.template operator()<ElementTypes>()), ...);
    return qr;
  }

  /**
   * @brief Computes workspace size needed for apply().
   */
  static constexpr dim_t apply_work_size(dim_t world_dim) {
    dim_t size = 0;
    ((size =
          math::max(size, apply_to_element_work_size<ElementTypes>(world_dim))),
     ...);
    return size;
  }

  /**
   * @brief Computes per-element workspace size.
   */
  template <typename Element>
  static constexpr dim_t apply_to_element_work_size(dim_t world_dim) {
    constexpr dim_t num_nodes = Element::num_nodes;
    const dim_t nodes_size = sizeof(scalar_t) * world_dim * num_nodes;
    const dim_t elem_mat_size =
        element_matrix_t<Element>::apply_work_size(world_dim);
    return nodes_size + elem_mat_size;
  }

  /**
   * @brief Applies the local element matrix to a slice of the global vector.
   */
  template <typename Element>
  void apply_to_element(const memory::ArrayConstView<scalar_t, 1> &u,
                        memory::ArrayView<scalar_t, 1> out, void *work) const {
    using ref_el_t = typename Element::ref_el_t;
    static constexpr dim_t num_nodes = Element::num_nodes;
    static constexpr dim_t num_basis_functions =
        basis_t::template num_basis_functions<ref_el_t>();

    const mesh_t &mesh = *(fes_->mesh());
    const dim_t num_elements = mesh.template num_elements<Element>();
    const dim_t world_dim = mesh.world_dim();

    const memory::ArrayConstView<scalar_mesh_t, 2> vertices = mesh.vertices();
    const memory::ArrayConstView<dim_t, 2> elements =
        mesh.template get_elements<Element>();
    const memory::ArrayConstView<dim_t, 2> dofs =
        fes_->template dof_map<Element>();
    const element_matrix_t<Element> &elem_mat =
        elem_mats_.template get<element_matrix_t<Element>>();

    if (is_host_accessible(memory_type())) {
      apply_host<Element>(elem_mat, vertices, elements, dofs, u, out,
                          work_.raw());
    } else if (is_device_accessible(memory_type())) {
      apply_device<Element>(elem_mat, vertices, elements, dofs, u, out);
    } else {
      NUMERIC_ERROR("Unsupperted memory type: {}", to_string(memory_type()));
    }
  }

  template <typename Element>
  void apply_host(const element_matrix_t<Element> &elem_mat,
                  const memory::ArrayConstView<scalar_mesh_t, 2> &vertices,
                  const memory::ArrayConstView<dim_t, 2> &elements,
                  const memory::ArrayConstView<dim_t, 2> &dofs,
                  const memory::ArrayConstView<scalar_t, 1> &u,
                  memory::ArrayView<scalar_t, 1> out, void *work) const {
    static constexpr dim_t num_nodes = Element::num_nodes;
    static constexpr dim_t num_basis_functions =
        element_matrix_t<Element>::num_basis_functions;
    const dim_t num_elements = elements.shape(1);
    const dim_t world_dim = vertices.shape(0);

    // Workspace pointers
    scalar_t(*nodes)[num_nodes] = static_cast<scalar_t(*)[num_nodes]>(work);
    void *elem_mat_work = static_cast<scalar_t *>(work) + world_dim * num_nodes;

    scalar_t elem_vec_in[num_basis_functions];  // Local u vector
    scalar_t elem_vec_out[num_basis_functions]; // Output of local mat-vec

    // Compute matrix vector product
    for (dim_t element = 0; element < num_elements; ++element) {
      // Collect node positions of element
      for (dim_t node = 0; node < num_nodes; ++node) {
        const dim_t node_idx = elements(node, element);
        for (dim_t i = 0; i < world_dim; ++i) {
          nodes[i][node] = vertices(i, node_idx);
        }
      }

      // Gather the local dofs
      for (dim_t bf = 0; bf < num_basis_functions; ++bf) {
        const dim_t dof_idx = dofs(bf, element);
        elem_vec_in[bf] = u(dof_idx);
      }

      // Apply local element matrix
      elem_mat.apply(nodes, elem_vec_in, world_dim, elem_vec_out,
                     elem_mat_work);

      // Scatter onto the global coefficient vector
      for (dim_t bf = 0; bf < num_basis_functions; ++bf) {
        const dim_t dof_idx = dofs(bf, element);
        out(dof_idx) += elem_vec_out[bf];
      }
    }
  }

  template <typename Element>
  void apply_device(const element_matrix_t<Element> &elem_mat,
                    const memory::ArrayConstView<scalar_mesh_t, 2> &vertices,
                    const memory::ArrayConstView<dim_t, 2> &elements,
                    const memory::ArrayConstView<dim_t, 2> &dofs,
                    const memory::ArrayConstView<scalar_t, 1> &u,
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
    static const hip::Kernel kernel = internal::element_matrix_build_kernel(
        utils::type_name<scalar_t>(), utils::type_name<scalar_mesh_t>(),
        utils::type_name<element_matrix_t<Element>>());
    static const int kernel_max_threads_per_block =
        kernel.max_threads_per_block();
    static const int kernel_max_dynamic_memory =
        kernel.max_dynamic_shared_size_bytes();
    hip::Device device;
    static constexpr dim_t num_nodes = Element::num_nodes;
    const dim_t world_dim = vertices.shape(0);
    for (const auto &group :
         fes_->template independent_element_groups<Element>()) {
      const dim_t num_elements = group.shape(0);
      const int shared_mem_per_thread =
          world_dim * num_nodes * sizeof(scalar_t) +
          elem_mat.apply_work_size(world_dim);
      const unsigned max_threads_per_block =
          math::min(kernel_max_threads_per_block,
                    kernel_max_dynamic_memory / shared_mem_per_thread);
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
      kernel.async(lp, hip::Stream(device), elem_mat, group, vertices, elements,
                   dofs, u, out);
    }
    device.sync();
  }
};

} // namespace numeric::equations::fem

#endif
