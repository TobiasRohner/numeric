#include <numeric/hip/program.hpp>
#include <numeric/math/reduce.hpp>

namespace numeric::math {

namespace internal {

static const char kernel_includes[] = R"(
  #include <numeric/config.hpp>
  #include <numeric/memory/array_base.hpp>
  #include <numeric/memory/array_const_view.hpp>
  #include <numeric/memory/array_view.hpp>
  #include <numeric/memory/array_op.hpp>
  #include <numeric/memory/constant.hpp>
  #include <numeric/memory/linspace.hpp>
  #include <numeric/memory/meshgrid.hpp>
  #include <numeric/memory/broadcast.hpp>
)";

static const char kernel_src[] = R"(
  template <typename Src, typename... Idxs>
  static __device__ typename numeric::memory::ArrayTraits<Src>::scalar_t
  linear_access(const Src &src, numeric::dim_t i, Idxs... idxs) {
    if constexpr (sizeof...(Idxs) == Src::dim) {
      return src(idxs...);
    } else {
      static constexpr numeric::dim_t current_dim = Src::dim - sizeof...(Idxs) - 1;
      const numeric::dim_t current_size = src.shape(current_dim);
      const numeric::dim_t new_idx = i % current_size;
      return linear_access(src, i / current_size, new_idx, idxs...);
    }
  }

  template<bool f_is_atomic, typename Src>
  __global__ void reduce(
      typename numeric::memory::ArrayTraits<Src>::scalar_t *out,
      Src src,
      typename numeric::memory::ArrayTraits<Src>::scalar_t identity) {
    using Scalar = typename numeric::memory::ArrayTraits<Src>::scalar_t;

    extern __shared__ Scalar sdata[];

    const numeric::dim_t tid = hipThreadIdx_x;
    const numeric::dim_t i = hipBlockIdx_x * hipBlockDim_x * 2 + hipThreadIdx_x;
    const numeric::dim_t N = src.size();
    if (i >= N) {
      sdata[tid] = identity;
    } else {
      sdata[tid] = linear_access(src, i);
    }
    if (i + hipBlockDim_x < N) {
      f(&sdata[tid], linear_access(src, i + hipBlockDim_x));
    }
    __syncthreads();

    for (numeric::dim_t s = hipBlockDim_x / 2 ; s > 0 ; s >>= 1) {
      if (tid < s) {
	f(&sdata[tid], sdata[tid + s]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      if (f_is_atomic) {
	f(&out[0], sdata[0]);
      } else {
	out[hipBlockIdx_x] = sdata[0];
      }
    }
  }
)";

static const char kernel_1d_includes[] = R"(
  #include <numeric/config.hpp>
  #include <numeric/math/functions.hpp>
)";

static const char kernel_1d_src[] = R"(
  template<bool f_is_atomic, typename Scalar>
  __global__ void reduce(Scalar *out, const Scalar *in, size_t N, Scalar identity) {
    extern __shared__ Scalar sdata[];

    const numeric::dim_t tid = hipThreadIdx_x;
    const numeric::dim_t i = hipBlockIdx_x * hipBlockDim_x * 2 + hipThreadIdx_x;
    if (i >= N) {
      sdata[tid] = identity;
    } else {
      sdata[tid] = in[i];
    }
    if (i + hipBlockDim_x < N) {
      f(&sdata[tid], in[i + hipBlockDim_x]);
    }
    __syncthreads();

    for (numeric::dim_t s = hipBlockDim_x / 2 ; s > 0 ; s >>= 1) {
      if (tid < s) {
	f(&sdata[tid], sdata[tid + s]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      if (f_is_atomic) {
	f(&out[0], sdata[0]);
      } else {
	out[hipBlockIdx_x] = sdata[0];
      }
    }
  }
)";

hip::Kernel reduce_device_build_kernel_impl(std::string_view src,
                                            std::string_view f,
                                            bool f_is_atomic) {
  const std::string atomic_str = f_is_atomic ? "true" : "false";
  const std::string kernel_name =
      "reduce<" + atomic_str + ", " + std::string(src) + ">";
  const std::string src_f = "\ntemplate <typename Scalar> __device__ void "
                            "f(Scalar *a, Scalar b) { " +
                            std::string(f) + "(a, b); }\n";
  hip::Program program(kernel_includes + src_f + kernel_src);
  program.add_compile_option("--device-as-default-execution-space");
  program.instantiate_kernel(kernel_name);
  return program.get_kernel(kernel_name);
}

hip::Kernel reduce_device_build_kernel_1d_contiguous_impl(
    std::string_view scalar, std::string_view f, bool f_is_atomic) {
  const std::string atomic_str = f_is_atomic ? "true" : "false";
  const std::string kernel_name =
      "reduce<" + atomic_str + ", " + std::string(scalar) + ">";
  const std::string src_f = "\ntemplate <typename Scalar> __device__ void "
                            "f(Scalar *a, Scalar b) { " +
                            std::string(f) + "(a, b); }\n";
  hip::Program program(kernel_1d_includes + src_f + kernel_1d_src);
  program.add_compile_option("--device-as-default-execution-space");
  program.instantiate_kernel(kernel_name);
  return program.get_kernel(kernel_name);
}

hip::LaunchParams reduce_device_launch_params(const hip::Device &device,
                                              dim_t N, dim_t bytes_per_scalar) {
  const unsigned max_threads_per_block = device.max_threads_per_block();
  const unsigned warp_size = device.warp_size();
  N = math::div_up(N, 2);
  hip::LaunchParams lp;
  lp.block_dim_x = max_threads_per_block;
  lp.block_dim_y = 1;
  lp.block_dim_z = 1;
  while (lp.block_dim_x / 2 >= N) {
    lp.block_dim_x /= 2;
  }
  lp.grid_dim_x = math::div_up(N, lp.block_dim_x);
  lp.grid_dim_y = 1;
  lp.grid_dim_z = 1;
  lp.shared_mem_bytes = 2 * warp_size *
                        math::div_up(lp.block_dim_x, 2 * warp_size) *
                        bytes_per_scalar;
  return lp;
}

} // namespace internal

} // namespace numeric::math
