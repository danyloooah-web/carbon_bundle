













#include <cuda_runtime.h>

#include <cstdint>

namespace moe_finalize {

constexpr int HIDDEN = 4096;
constexpr int TOP_K = 8;
constexpr int VECS = HIDDEN / 8;
constexpr int THREADS = 128;
constexpr int STEPS = VECS / THREADS;

#ifndef MOE_FINALIZE_HOST_PTX
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ float add_rn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ void wait_primary() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
#endif

__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

struct Params {
  const uint4* gemm2;
  const int32_t* rows;
  const uint4* weights;
  const uint4* shared;
  uint4* out;
};



template <bool kShared>
__global__ void __launch_bounds__(THREADS, 8) fuse_kernel(const Params p) {
  wait_primary();
  const int64_t t = blockIdx.x;
  const int4 r0 = reinterpret_cast<const int4*>(p.rows)[2 * t];
  const int4 r1 = reinterpret_cast<const int4*>(p.rows)[2 * t + 1];
  const uint4 wv = p.weights[t];
  const int32_t row[TOP_K] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
  const float w[TOP_K] = {lo(wv.x), hi(wv.x), lo(wv.y), hi(wv.y), lo(wv.z), hi(wv.z), lo(wv.w), hi(wv.w)};
#pragma unroll 1
  for (int step = 0; step < STEPS; ++step) {
    const int64_t v = threadIdx.x + step * THREADS;
    uint4 x[TOP_K];
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      if (row[k] != -1) x[k] = p.gemm2[int64_t(row[k]) * VECS + v];
    }
    uint4 s = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (kShared) s = p.shared[t * VECS + v];
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      if (row[k] == -1) continue;
      const uint32_t q[4] = {x[k].x, x[k].y, x[k].z, x[k].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc[2 * e] = fma_ftz(w[k], lo(q[e]), acc[2 * e]);
        acc[2 * e + 1] = fma_ftz(w[k], hi(q[e]), acc[2 * e + 1]);
      }
    }
    const uint32_t sq[4] = {s.x, s.y, s.z, s.w};
    uint32_t o[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const uint32_t routed = bf16x2(acc[2 * e], acc[2 * e + 1]);
      o[e] = kShared ? bf16x2(add_rn(lo(routed), lo(sq[e])), add_rn(hi(routed), hi(sq[e])))
                     : routed;
    }
    p.out[t * VECS + v] = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

}

#ifndef MOE_FINALIZE_DEVICE_ONLY
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>



void fuse(torch::Tensor gemm2, torch::Tensor rows, torch::Tensor weights, torch::Tensor shared, torch::Tensor out,
          bool pdl) {
  using namespace moe_finalize;
  const int64_t tokens = out.size(0);
  const auto aligned = [](const torch::Tensor& x) { return reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0; };
  const auto here = [&](const torch::Tensor& x) {
    return x.is_cuda() && x.get_device() == out.get_device() && x.is_contiguous() && aligned(x);
  };
  const auto bf16 = [&](const torch::Tensor& x, int64_t n) {
    return here(x) && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) == n && x.size(1) == HIDDEN;
  };
  TORCH_CHECK(out.dim() == 2 && bf16(out, tokens) && bf16(shared, tokens) && gemm2.dim() == 2 && gemm2.size(0) > 0 &&
                  bf16(gemm2, gemm2.size(0)),
              "moe_finalize: out, shared [tokens, 4096] and gemm2 [rows, 4096] must be contiguous 16-byte aligned BF16");
  TORCH_CHECK(here(weights) && weights.scalar_type() == at::kBFloat16 && weights.dim() == 2 &&
                  weights.size(0) == tokens && weights.size(1) == TOP_K,
              "moe_finalize: weights must be contiguous BF16 [tokens, 8]");
  TORCH_CHECK(here(rows) && rows.scalar_type() == at::kInt && rows.dim() == 1 && rows.numel() == tokens * TOP_K,
              "moe_finalize: rows must be a contiguous int32 vector of tokens * 8");
  if (tokens == 0) return;
  const c10::cuda::CUDAGuard guard(out.device());
  const Params p{reinterpret_cast<const uint4*>(gemm2.data_ptr()), rows.data_ptr<int32_t>(),
                 reinterpret_cast<const uint4*>(weights.data_ptr()), reinterpret_cast<const uint4*>(shared.data_ptr()),
                 reinterpret_cast<uint4*>(out.data_ptr())};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(unsigned(tokens));
  cfg.blockDim = dim3(THREADS);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fuse_kernel<true>, p));
}


void routed(torch::Tensor gemm2, torch::Tensor rows, torch::Tensor weights, torch::Tensor out, bool pdl) {
  using namespace moe_finalize;
  const int64_t tokens = out.size(0);
  const auto aligned = [](const torch::Tensor& x) { return reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0; };
  const auto here = [&](const torch::Tensor& x) {
    return x.is_cuda() && x.get_device() == out.get_device() && x.is_contiguous() && aligned(x);
  };
  const auto bf16 = [&](const torch::Tensor& x, int64_t n) {
    return here(x) && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) == n && x.size(1) == HIDDEN;
  };
  TORCH_CHECK(out.dim() == 2 && bf16(out, tokens) && gemm2.dim() == 2 && gemm2.size(0) > 0 &&
                  bf16(gemm2, gemm2.size(0)),
              "moe_finalize: out [tokens, 4096] and gemm2 [rows, 4096] must be contiguous 16-byte aligned BF16");
  TORCH_CHECK(here(weights) && weights.scalar_type() == at::kBFloat16 && weights.dim() == 2 &&
                  weights.size(0) == tokens && weights.size(1) == TOP_K,
              "moe_finalize: weights must be contiguous BF16 [tokens, 8]");
  TORCH_CHECK(here(rows) && rows.scalar_type() == at::kInt && rows.dim() == 1 && rows.numel() == tokens * TOP_K,
              "moe_finalize: rows must be a contiguous int32 vector of tokens * 8");
  if (tokens == 0) return;
  const c10::cuda::CUDAGuard guard(out.device());
  const Params p{reinterpret_cast<const uint4*>(gemm2.data_ptr()), rows.data_ptr<int32_t>(),
                 reinterpret_cast<const uint4*>(weights.data_ptr()), nullptr,
                 reinterpret_cast<uint4*>(out.data_ptr())};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(unsigned(tokens));
  cfg.blockDim = dim3(THREADS);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fuse_kernel<false>, p));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fuse", &fuse,
        "FlashInfer's MoE finalize and the shared expert's add, stock's two BF16 roundings, in one launch");
  m.def("routed", &routed, "FlashInfer's MoE finalize alone, its BF16 store (the shared rows added later)");
}
#endif
