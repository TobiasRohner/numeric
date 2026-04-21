# numeric

**numeric** is a C++20 numerics library inspired by NumPy, with built-in GPU acceleration via HIP and support for finite element methods. It provides a numpy-style array API with device-transparent operations, along with a growing finite element toolkit for solving PDEs on unstructured meshes.

## Features

- **NumPy-style arrays** — `Array`, `ArrayView`, and `ArrayConstView` with compile-time dimensionality, strides, and C-style indexing
- **Broadcasting** — automatic broadcasting of arrays with different shapes, compatible with NumPy semantics
- **Element-wise operations** — overloaded arithmetic operators (`+`, `-`, `*`, `/`) and math functions (`sin`, `cos`, `exp`, `abs`, `pow`) work element-wise across arrays
- **Device-transparent memory** — host and GPU device memory with zero-copy views; operations work on either location automatically
- **Mesh utilities** — `linspace`, `meshgrid`, `Constant` arrays, slicing with `Slice` objects
- **Linear algebra** — conjugate gradient solver, norms (`l1`, `l2`, `lp`), reduction operations (`sum`, `min`, `max`)
- **Graph algorithms** — sparse graph data structure, DSATUR graph coloring
- **Quadrature rules** — Gaussian quadrature for segments, triangles, quadrilaterals, tetrahedra, and cubes
- **Lagrange basis functions** — H1 and L2 conforming bases for all reference element types
- **GPU compute** — HIP backend for kernel execution on compatible GPUs
- **I/O** — HDF5, NetCDF, Gmsh mesh reader, and VTK-HDF writer for visualization

## Project Structure

```
include/numeric/
├── hip/              # GPU compute (device, kernel, stream, module)
├── memory/           # NumPy-style arrays, broadcasting, operations
├── math/             # Math functions, solvers, quadrature, graph algorithms
├── io/               # HDF5, NetCDF, Gmsh reader, VTK-HDF writer
├── mesh/             # Mesh structures (work in progress)
├── equations/        # Finite element assembly (work in progress)
└── meta/             # Compile-time utilities
src/
├── hip/             # HIP runtime wrappers
├── memory/          # Copy operations, memory resources
├── math/            # Reductions, graph coloring, quadrature
├── io/              # I/O implementations
└── equations/       # FEM matrix/vector assembly
```

## Quick Start

### Arrays and Element-wise Operations

```cpp
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_op.hpp>
#include <numeric/memory/linspace.hpp>
#include <numeric/memory/meshgrid.hpp>
#include <numeric/memory/constant.hpp>

using namespace numeric;

// Create arrays with different memory locations
const memory::MemoryType mem_type = memory::MemoryType::HOST;

// linspace and meshgrid
auto x = memory::linspace(0.0, 1.0, 100, true, mem_type);
auto y = memory::linspace(0.0, 2.0, 50, true, mem_type);
auto [X, Y] = memory::meshgrid(x, y);

// Element-wise operations (NumPy-style)
auto Z = X * X + Y * Y;         // broadcasting + arithmetic
auto abs_Z = memory::abs(Z);     // element-wise abs
auto exp_Z = memory::exp(Z);     // element-wise exp

// Constant arrays
auto ones = memory::Constant<double, 2>(memory::Shape<2>(10, 10), 1.0, mem_type);

// Slicing
auto sub = Z(memory::Slice(1, -2), memory::Slice(1, -2));

// Copy between memory locations
memory::Array<double, 2> host_array(Z.shape());
memory::copy(host_array, Z);
```

### Solving Linear Systems

```cpp
#include <numeric/math/conjugate_gradient.hpp>
#include <numeric/math/custom_linear_operator.hpp>
#include <numeric/math/reduce.hpp>

using namespace numeric;

// Define a custom linear operator (e.g., Laplacian)
auto op = std::make_shared<math::CustomLinearOperator<double>>(
    memory::Shape<2>(N, N),
    [](const memory::ArrayConstView<double, 1> &x,
       memory::ArrayView<double, 1> out) {
        // ... compute matrix-vector product
    });

// Solve Ax = b with conjugate gradient
math::ConjugateGradient<double> cg(op);
cg.set_tolerance(1e-8);
cg.set_max_iterations(N);

memory::Array<double, 1> b(memory::Shape<1>(N), memory::MemoryType::HOST);
memory::Array<double, 1> x(memory::Shape<1>(N), memory::MemoryType::HOST);

cg.solve(b.const_view(), x.view());

auto [converged, num_iter, error] = cg.result();
```

### Finite Element Method (Work in Progress)

```cpp
#include <numeric/equations/fem/finite_element_matrix.hpp>
#include <numeric/equations/fem/finite_element_vector.hpp>
#include <numeric/io/gmsh_reader.hpp>
#include <numeric/math/fes/fe_space.hpp>
#include <numeric/math/fes/basis_h1.hpp>
#include <numeric/mesh/elements.hpp>

using namespace numeric;

// Load mesh from Gmsh file
using mesh_t = mesh::UnstructuredMesh<double, mesh::Tria<1>>;
auto mesh = io::GmshReader<double, mesh::Tria<1>>::load("mesh.msh", 2);

// Define finite element space (H1, order 4)
using basis_t = math::fes::BasisH1<4>;
using fes_t = math::fes::FESpace<basis_t, mesh_t>;
auto fes = std::make_shared<fes_t>(mesh);

// Build stiffness matrix (diffusion)
using element_matrix_factory_t =
    equations::fem::DiffusionElementMatrixFactory<double, basis_t>;
using stiffness_matrix_t =
    equations::fem::FiniteElementMatrix<fes_t, element_matrix_factory_t>;
auto laplacian = std::make_shared<stiffness_matrix_t>(fes);

// Build load vector
using element_vector_factory_t =
    equations::fem::LoadElementVectorFactory<double, basis_t>;
using load_vector_t =
    equations::fem::FiniteElementVector<fes_t, element_vector_factory_t>;
load_vector_t load(fes);

auto f = NUMERIC_LAMBDA([](auto *x) -> double {
    return 1.0;  // source function f(x)
});
load.assemble(f, system.rhs());

// Solve
math::LinearSystem<double> system(laplacian);
math::MeshFunction<double, fes_t> u(fes);
system.solve(u.dofs());
```

## Building

### Prerequisites

- CMake 3.24+
- C++20 compiler (GCC 11+, Clang 14+, or MSVC 19.30+)
- Optional dependencies (Automatically installed if requested but not found):
  - **HIP** — GPU acceleration (enabled by default)
  - **Eigen3** — matrix operations (enabled by default)
  - **HDF5** — HDF5 I/O (enabled by default)
  - **NetCDF** — NetCDF I/O (enabled by default)
  - **MPI** — parallelization (enabled by default)
  - **fmt** — formatting library

### Build Steps

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `NUMERIC_ENABLE_HIP` | `ON` | Enable GPU support via HIP |
| `NUMERIC_ENABLE_EIGEN` | `ON` | Enable Eigen-based matrix operations |
| `NUMERIC_ENABLE_HDF5` | `ON` | Enable HDF5 I/O |
| `NUMERIC_ENABLE_NETCDF` | `ON` | Enable NetCDF I/O |
| `NUMERIC_ENABLE_MPI` | `ON` | Enable MPI parallelization |
| `NUMERIC_BUILD_TESTS` | `ON` | Build tests |
| `NUMERIC_BUILD_EXAMPLES` | `ON` | Build examples |
| `NUMERIC_BUILD_BENCHMARKS` | `ON` | Build benchmarks |

## Examples

The `examples/` directory contains working examples:

- **`fd_poisson.cpp`** — Finite difference Poisson solver with conjugate gradient
- **`fem_poisson.cpp`** — Finite element Poisson solver on unstructured mesh
- **`stiffness_matrix.cpp`** — Building a stiffness matrix
- **`gmsh_reader.cpp`** — Reading Gmsh mesh files
- **`vtkhdf_mesh_writer.cpp`** — Writing VTK-HDF output
- **`quad.cpp`** — Quadrature rule computation

Run an example:

```bash
./examples/fd_poisson
./examples/fem_poisson unit_disk_1.msh
```

## Array API Reference

| Component | Description |
|-----------|-------------|
| `memory::Array<T, N>` | Dynamically allocated array with ownership |
| `memory::ArrayView<T, N>` | Non-owning mutable view over array data |
| `memory::ArrayConstView<T, N>` | Non-owning immutable view over array data |
| `memory::Shape<N>` | Array shape information |
| `memory::Layout<N>` | Shape + stride layout information |
| `memory::Slice` | Python-style slicing (`Slice(start, stop, step)`) |
| `memory::Constant<T, N>` | Array with a single constant value |
| `memory::linspace(...)` | Equally spaced 1D array (NumPy `linspace`) |
| `memory::meshgrid(...)` | N-D coordinate grids from 1D inputs |
| `memory::broadcast(...)` | Broadcast an array to a new shape |
| `memory::copy(dst, src)` | Copy between arrays (host/device aware) |
| `math::sum(x)`, `math::min(x)`, `math::max(x)` | Reduction operations |
| `math::norm::l1(x)`, `math::norm::l2(x)` | Norm computations |

## Mesh Types

| Type | Dimension | Nodes (order 1) |
|------|-----------|-----------------|
| `mesh::Point` | 0 | 1 |
| `mesh::Segment` | 1 | 2 |
| `mesh::Tria` | 2 | 3 |
| `mesh::Quad` | 2 | 4 |
| `mesh::Tetra` | 3 | 4 |
| `mesh::Cube` | 3 | 8 |

Higher-order elements are supported via polynomial order template parameter (e.g., `mesh::Tria<2>` for quadratic triangles).

## Status

- **`hip`** — Stable
- **`memory`** — Stable
- **`math`** — Stable
- **`io`** — Stable
- **`mesh`** — Work in progress
- **`equations`** — Work in progress
