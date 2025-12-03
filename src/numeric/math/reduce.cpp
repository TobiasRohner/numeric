#include <numeric/hip/program.hpp>
#include <numeric/math/reduce.hpp>

namespace numeric::math {

namespace internal {

static const char kernel_includes[] = R"(
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
  template<typename Src>
  __global__ void reduce(typename numeric::memory::ArrayTraits<Src>::scalar_t *out, Src src) {
    // TODO: Implement
  }
)";

static const char kernel_1d_includes[] = R"(
  #include <numeric/config.hpp>
  #include <numeric/math/functions.hpp>
)";

static const char kernel_1d_src[] = R"(
  template<typename Scalar>
  static __device__ void warpReduce(volatile Scalar *sdata, numeric::dim_t tid) {
    if (warpSize > 32) { sdata[tid] = f(sdata[tid], sdata[tid + 64]);}
    sdata[tid] = f(sdata[tid], sdata[tid + 32]);
    sdata[tid] = f(sdata[tid], sdata[tid + 16]);
    sdata[tid] = f(sdata[tid], sdata[tid + 8]);
    sdata[tid] = f(sdata[tid], sdata[tid + 4]);
    sdata[tid] = f(sdata[tid], sdata[tid + 2]);
    sdata[tid] = f(sdata[tid], sdata[tid + 1]);
  }

  template<typename Scalar>
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
      sdata[tid] = f(sdata[tid], in[i + hipBlockDim_x]);
    }
    __syncthreads();

    //for (numeric::dim_t s = hipBlockDim_x / 2 ; s > warpSize ; s >>= 1) {
    for (numeric::dim_t s = hipBlockDim_x / 2 ; s > 0 ; s >>= 1) {
      if (tid < s) {
	sdata[tid] = f(sdata[tid], sdata[tid + s]);
      }
      __syncthreads();
    }
    //if (tid < warpSize) {
    //  warpReduce(sdata, tid);
    //}

    if (tid == 0) {
      out[hipBlockIdx_x] = sdata[0];
    }
  }
)";

hip::Kernel reduce_device_build_kernel(std::string_view src,
                                       std::string_view f) {
  const std::string kernel_name = "reduce<" + std::string(src) + ">";
  const std::string src_f = "\ntemplate <typename Scalar> __device__ Scalar "
                            "f(Scalar a, Scalar b) { return " +
                            std::string(f) + "(a, b); }\n";
  hip::Program program(kernel_includes + src_f + kernel_src);
  program.add_compile_option("--device-as-default-execution-space");
  program.instantiate_kernel(kernel_name);
  return program.get_kernel(kernel_name);
}

hip::Kernel
reduce_device_build_kernel_1d_contiguous_impl(std::string_view scalar,
                                              std::string_view f) {
  const std::string kernel_name = "reduce<" + std::string(scalar) + ">";
  const std::string src_f = "\ntemplate <typename Scalar> __device__ Scalar "
                            "f(Scalar a, Scalar b) { return " +
                            std::string(f) + "(a, b); }\n";
  hip::Program program(kernel_1d_includes + src_f + kernel_1d_src);
  program.add_compile_option("--device-as-default-execution-space");
  program.instantiate_kernel(kernel_name);
  return program.get_kernel(kernel_name);
}

hip::LaunchParams reduce_device_launch_params(const hip::Device &device,
                                              dim_t N, dim_t bytes_per_scalar) {
  N = math::div_up(N, 2);
  hip::LaunchParams lp;
  lp.block_dim_x = device.max_threads_per_block();
  lp.block_dim_y = 1;
  lp.block_dim_z = 1;
  while (lp.block_dim_x / 2 >= N) {
    lp.block_dim_x /= 2;
  }
  lp.grid_dim_x = math::div_up(N, lp.block_dim_x);
  lp.grid_dim_y = 1;
  lp.grid_dim_z = 1;
  lp.shared_mem_bytes = 2 * device.warp_size() *
                        math::div_up(lp.block_dim_x, 2 * device.warp_size()) *
                        bytes_per_scalar;
  return lp;
}

} // namespace internal

} // namespace numeric::math
