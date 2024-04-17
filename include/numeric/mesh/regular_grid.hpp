#ifndef NUMERIC_MESH_REGULAR_GRID_HPP_
#define NUMERIC_MESH_REGULAR_GRID_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/linspace.hpp>
#include <numeric/memory/meshgrid.hpp>
#include <numeric/memory/shape.hpp>
#include <numeric/meta/integer_sequence.hpp>

namespace numeric::mesh {

template <typename Scalar, dim_t Dim> class RegularGrid {
public:
  using scalar_t = Scalar;

  RegularGrid(const memory::Shape<Dim> num_cells, dim_t num_ghost_cells = 0,
              memory::MemoryType memory_type = memory::MemoryType::HOST)
      : memory_type_(memory_type), num_ghost_cells_(num_ghost_cells) {
    for (dim_t i = 0; i < Dim; ++i) {
      num_cells_[i] = num_cells[i];
      origin_[i] = 0;
      size_[i] = 1;
    }
  }
  RegularGrid(const RegularGrid &) = default;
  RegularGrid &operator=(const RegularGrid &) = default;

  dim_t num_ghost_cells() const noexcept { return num_ghost_cells_; }

  template <typename... T> RegularGrid &set_origin(T... orig) {
    static_assert(sizeof...(orig) == Dim,
                  "Wrong number of arguments provided to set_origin");
    dim_t i = 0;
    ((origin_[i++] = orig), ...);
    return *this;
  }

  template <typename... T> RegularGrid &set_size(T... size) {
    static_assert(sizeof...(size) == Dim,
                  "Wrong number of arguments provided to set_size");
    dim_t i = 0;
    ((size_[i++] = size), ...);
    return *this;
  }

  Scalar dx(dim_t dim) const { return size_[dim] / num_cells_[dim]; }

  decltype(auto) vertex_positions() const {
    return vertex_positions_impl(meta::make_index_sequence<Dim>{});
  }

  decltype(auto) cell_positions() const {
    return cell_positions_impl(meta::make_index_sequence<Dim>{});
  }

private:
  memory::MemoryType memory_type_;
  dim_t num_ghost_cells_;
  dim_t num_cells_[Dim];
  scalar_t origin_[Dim];
  scalar_t size_[Dim];

  template <size_t... Idxs>
  decltype(auto) vertex_positions_impl(meta::index_sequence<Idxs...>) const {
    const scalar_t dx[Dim] = {(size_[Idxs] / num_cells_[Idxs])...};
    return memory::meshgrid((memory::Linspace<scalar_t>(
        origin_[Idxs] - num_ghost_cells_ * dx[Idxs],
        origin_[Idxs] + size_[Idxs] + num_ghost_cells_ * dx[Idxs],
        num_cells_[Idxs] + 1 + 2 * num_ghost_cells_, true, memory_type_))...);
  }

  template <size_t... Idxs>
  decltype(auto) cell_positions_impl(meta::index_sequence<Idxs...>) const {
    const scalar_t dx[Dim] = {(size_[Idxs] / num_cells_[Idxs])...};
    return memory::meshgrid((memory::Linspace<scalar_t>(
        origin_[Idxs] + dx[Idxs] / 2 - num_ghost_cells_ * dx[Idxs],
        origin_[Idxs] + dx[Idxs] / 2 + size_[Idxs] +
            num_ghost_cells_ * dx[Idxs],
        num_cells_[Idxs] + 2 * num_ghost_cells_, true, memory_type_))...);
  }
};

} // namespace numeric::mesh

#endif
