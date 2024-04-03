#include <numeric/hip/program.hpp>
#include <numeric/math/sum_device.hpp>

namespace numeric::math {

namespace detail {

static const char kernel_src[] = R"(
  #include <numeric/memory/array_base.hpp>
  #include <numeric/memory/array_const_view.hpp>
  #include <numeric/memory/array_view.hpp>
  #include <numeric/memory/array_op.hpp>
  #include <numeric/memory/constant.hpp>
  #include <numeric/memory/linspace.hpp>
  #include <numeric/memory/meshgrid.hpp>
  #include <numeric/memory/broadcast.hpp>

  template<typename Src>
  __global__ void sum(typename numeric::memory::ArrayTraits<Src>::scalar_t *out, Src src) {
    // TODO: Implement
  }
)";

static const char kernel_1d_src[] = R"(
  #include <numeric/config.hpp>

  template<typename Scalar>
  static __device__ void warpReduce(volatile Scalar *sdata, numeric::dim_t tid) {
    if (warpSize > 32) { sdata[tid] += sdata[tid + 32];}
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
  }

  template<typename Scalar>
  __global__ void sum(Scalar *out, const Scalar *in, size_t N) {
    extern __shared__ Scalar sdata[];

    const numeric::dim_t tid = hipThreadIdx_x;
    const numeric::dim_t i = hipBlockIdx_x * hipBlockDim_x * 2 + hipThreadIdx_x;
    if (i >= N) {
      sdata[tid] = 0;
    } else {
      sdata[tid] = in[i];
    }
    if (i + hipBlockDim_x < N) {
      sdata[tid] += in[i + hipBlockDim_x];
    }
    __syncthreads();

    for (numeric::dim_t s = hipBlockDim_x / 2 ; s > warpSize ; s >>= 1) {
      if (tid < s) {
	sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }
    if (tid < warpSize) {
      warpReduce(sdata, tid);
    }

    if (tid == 0) {
      out[hipBlockIdx_x] = sdata[0];
    }
  }
)";

hip::Kernel sum_device_build_kernel(std::string_view src) {
  const std::string kernel_name = "sum<" + std::string(src) + ">";
  hip::Program program(kernel_src);
  program.instantiate_kernel(kernel_name);
  return program.get_kernel(kernel_name);
}

hip::Kernel
sum_device_build_kernel_1d_contiguous_impl(std::string_view scalar) {
  const std::string kernel_name = "sum<" + std::string(scalar) + ">";
  hip::Program program(kernel_1d_src);
  program.instantiate_kernel(kernel_name);
  return program.get_kernel(kernel_name);
}

} // namespace detail

} // namespace numeric::math
