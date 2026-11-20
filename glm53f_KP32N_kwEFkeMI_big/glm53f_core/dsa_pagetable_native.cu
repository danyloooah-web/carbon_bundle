







#include <cstdint>

#ifndef PAGETABLE_HOST
#include <cuda_runtime.h>
#endif

namespace pagetable {

constexpr int kThreads = 512;

struct Params {
  const int32_t* src;
  int32_t* dst;
  int32_t* lens;
  int64_t src_pitch;
  int32_t width;
  int32_t padded;
};


__device__ __forceinline__ int32_t part(const Params& p, uint32_t row, int32_t t) {
  const int32_t* __restrict__ s = p.src + static_cast<int64_t>(row) * p.src_pitch;
  int32_t* __restrict__ d = p.dst + static_cast<int64_t>(row) * p.padded;
  int32_t count = 0;
#pragma unroll 4
  for (int32_t c = t; c < p.width; c += kThreads) {
    const int32_t v = s[c];
    d[c] = v;
    count += v >= 0 ? 1 : 0;
  }
  for (int32_t c = p.width + t; c < p.padded; c += kThreads) d[c] = -1;
  return count;
}


__device__ __forceinline__ void finish(const Params& p, uint32_t row, int32_t count) {
  if (count == 0) p.dst[static_cast<int64_t>(row) * p.padded] = 0;
  p.lens[row] = count > 0 ? count : 1;
}

#ifndef PAGETABLE_HOST
__global__ __launch_bounds__(kThreads) void table_kernel(const __grid_constant__ Params p) {
  __shared__ int32_t warps[kThreads / 32];
  const int32_t count = __reduce_add_sync(0xffffffffu, part(p, blockIdx.x, static_cast<int32_t>(threadIdx.x)));
  if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = count;
  __syncthreads();
  if (threadIdx.x == 0) {
    int32_t total = 0;
#pragma unroll
    for (int i = 0; i < kThreads / 32; ++i) total += warps[i];
    finish(p, blockIdx.x, total);
  }
}
#endif

}

#ifndef PAGETABLE_DEVICE_ONLY
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


void table(torch::Tensor src, torch::Tensor dst, torch::Tensor lens) {
  TORCH_CHECK(src.is_cuda() && dst.device() == src.device() && lens.device() == src.device() &&
                  src.scalar_type() == at::kInt && dst.scalar_type() == at::kInt && lens.scalar_type() == at::kInt &&
                  src.dim() == 2 && dst.dim() == 2 && lens.dim() == 1,
              "pagetable: int32 src [rows, w], dst [rows, P] and lens [rows] on one CUDA device");
  const int64_t rows = src.size(0), width = src.size(1), padded = dst.size(1);
  TORCH_CHECK(rows >= 1 && rows < (int64_t(1) << 31) && dst.size(0) == rows && lens.size(0) == rows && width >= 1 &&
                  padded >= width && padded < (int64_t(1) << 31) && src.stride(1) == 1 && src.stride(0) >= width &&
                  dst.is_contiguous() && lens.is_contiguous(),
              "pagetable: rows of unit column stride, one of dst and lens a row of src, P >= w");
  const auto s0 = reinterpret_cast<uintptr_t>(src.data_ptr()), d0 = reinterpret_cast<uintptr_t>(dst.data_ptr());
  TORCH_CHECK(s0 + uintptr_t(((rows - 1) * src.stride(0) + width) * 4) <= d0 || d0 + uintptr_t(rows * padded * 4) <= s0,
              "pagetable: src overlaps dst");
  const pagetable::Params p{src.data_ptr<int32_t>(), dst.data_ptr<int32_t>(), lens.data_ptr<int32_t>(),
                            src.stride(0), static_cast<int32_t>(width), static_cast<int32_t>(padded)};
  const c10::cuda::CUDAGuard guard(src.device());
  pagetable::table_kernel<<<static_cast<unsigned>(rows), pagetable::kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("table", &table, "owned rows of the sparse page table at its padded width and their top-k lengths");
}
#endif
