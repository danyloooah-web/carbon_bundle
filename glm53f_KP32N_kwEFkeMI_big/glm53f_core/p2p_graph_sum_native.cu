
























#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kEmpty = 0xFFFFFFFFu;
constexpr uint32_t kStandIn = 0x7FFF7FFFu;
constexpr int kThreads = 1024;
constexpr int kBatch = 4;

__device__ __forceinline__ uint32_t sent(uint32_t w) { return w == kEmpty ? kStandIn : w; }
__device__ __forceinline__ uint4 sent16(uint4 v) { return make_uint4(sent(v.x), sent(v.y), sent(v.z), sent(v.w)); }
__device__ __forceinline__ bool full16(uint4 v) {
  return v.x != kEmpty && v.y != kEmpty && v.z != kEmpty && v.w != kEmpty;
}

__device__ __forceinline__ void push16(uint4* at, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(at), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}
__device__ __forceinline__ uint4 peek16(const uint4* at) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(at));
  return v;
}
__device__ __forceinline__ void empty16(uint4* at) {
  asm volatile("st.global.v4.b32 [%0], {%1, %1, %1, %1};" ::"l"(at), "r"(kEmpty));
}

__device__ __forceinline__ uint32_t add2(uint32_t a, uint32_t b) {
  __nv_bfloat162 x, y;
  memcpy(&x, &a, 4);
  memcpy(&y, &b, 4);
  const __nv_bfloat162 r = __hadd2(x, y);
  uint32_t out;
  memcpy(&out, &r, 4);
  return out;
}

struct Params {
  uint4* x;
  uint4* to;
  uint4* from;
  int* parity;
  long long vecs, slot;
};

__global__ void __launch_bounds__(kThreads, 1) sum_kernel(const __grid_constant__ Params p) {
  const int phase = p.parity[blockIdx.x] & 1;
  uint4* const to = p.to + phase * p.slot;
  uint4* const from = p.from + phase * p.slot;
  const long long stride = static_cast<long long>(gridDim.x) * kThreads;
  const long long first = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;

#pragma unroll 1
  for (long long v0 = first; v0 < p.vecs; v0 += kBatch * stride) {
    uint4 m[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; ++b)
      if (v0 + b * stride < p.vecs) m[b] = p.x[v0 + b * stride];
#pragma unroll
    for (int b = 0; b < kBatch; ++b)
      if (v0 + b * stride < p.vecs) push16(to + v0 + b * stride, sent16(m[b]));
  }

#pragma unroll 1
  for (long long v0 = first; v0 < p.vecs; v0 += kBatch * stride) {
    uint4 got[kBatch];
    bool have[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; ++b) have[b] = v0 + b * stride >= p.vecs;
    bool missing;
    do {
#pragma unroll
      for (int b = 0; b < kBatch; ++b)
        if (!have[b]) got[b] = peek16(from + v0 + b * stride);
      missing = false;
#pragma unroll
      for (int b = 0; b < kBatch; ++b)
        if (!have[b]) {
          have[b] = full16(got[b]);
          missing |= !have[b];
        }
    } while (missing);
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const long long v = v0 + b * stride;
      if (v >= p.vecs) continue;
      const uint4 mine = p.x[v];
      p.x[v] = make_uint4(add2(mine.x, got[b].x), add2(mine.y, got[b].y), add2(mine.z, got[b].z), add2(mine.w, got[b].w));
      empty16(from + v);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) p.parity[blockIdx.x] = phase ^ 1;
}


int grid_size() {
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  TORCH_CHECK(device >= 0 && device < 64, "p2p_graph_sum: device index");
  static int known[64] = {};
  if (known[device] == 0) {
    int per_sm = 0, sms = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, sum_kernel, kThreads, 0));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    TORCH_CHECK(per_sm >= 1 && sms >= 1, "p2p_graph_sum: no CTA of ", kThreads, " threads fits an SM of this GPU");
    known[device] = sms;
  }
  return known[device];
}

}




void sum(const torch::Tensor& x, int64_t to, int64_t from, const torch::Tensor& parity, int64_t slot) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.numel() % 8 == 0 &&
                  x.numel() > 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "p2p_graph_sum: x must be a contiguous, 16-byte aligned CUDA BF16 tensor of whole 16-byte vectors");
  const int grid = grid_size();
  TORCH_CHECK(parity.is_cuda() && parity.device() == x.device() && parity.scalar_type() == at::kInt &&
                  parity.is_contiguous() && parity.numel() == grid,
              "p2p_graph_sum: parity must hold one int32 counter a CTA of the fixed grid (", grid, ")");
  TORCH_CHECK(to != 0 && from != 0 && to % 16 == 0 && from % 16 == 0 && x.numel() / 8 <= slot,
              "p2p_graph_sum: receive slots of ", slot, " vectors cannot hold ", x.numel() / 8);
  Params p{};
  p.x = reinterpret_cast<uint4*>(x.data_ptr());
  p.to = reinterpret_cast<uint4*>(to);
  p.from = reinterpret_cast<uint4*>(from);
  p.parity = parity.data_ptr<int>();
  p.vecs = x.numel() / 8;
  p.slot = slot;
  sum_kernel<<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_CHECK(cudaGetLastError());
}


std::vector<int64_t> layout() { return {grid_size(), kThreads}; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sum", &sum, "the two-rank sum of x, in place, bit for bit NCCL's (one push launch, graph-capturable)");
  m.def("layout", &layout, "(grid, threads) on the current device");
}
