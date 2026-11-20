





















#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kHidden = 4096;
constexpr int kHc = 4;
constexpr int kThreads = 128;
constexpr int kVecs = kHidden / 8;
constexpr int kPerThread = kVecs / kThreads;
constexpr int kStageThreads = 512;
constexpr int kUnroll = 4;

__device__ __forceinline__ int ld_acquire_sys(const int* p) {
  int v;
  asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release_sys(int* p, int v) {
  asm volatile("st.release.sys.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ int ld_relaxed_gpu(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint4 ld_volatile(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ uint32_t hadd2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hadd2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ float lo_f(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi_f(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

template <bool FTZ>
__device__ __forceinline__ float fmul(float a, float b) {
  float r;
  if (FTZ) asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  else asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

template <bool FTZ>
__device__ __forceinline__ float fadd(float a, float b) {
  float r;
  if (FTZ) asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  else asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

template <bool FTZ>
__device__ __forceinline__ float ffma(float a, float b, float c) {
  float r;
  if (FTZ) asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  else asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

template <int V, bool FTZ>
__device__ __forceinline__ float post_one(float c, float d, const float (&a)[kHc], const float (&b)[kHc]) {
  float acc;
  if (V == 0) {
    acc = fmul<FTZ>(c, d);
#pragma unroll
    for (int i = 0; i < kHc; ++i) acc = ffma<FTZ>(a[i], b[i], acc);
  } else if (V == 1) {
    acc = ffma<FTZ>(c, d, fmul<FTZ>(a[0], b[0]));
#pragma unroll
    for (int i = 1; i < kHc; ++i) acc = ffma<FTZ>(a[i], b[i], acc);
  } else {
    acc = fmul<FTZ>(c, d);
#pragma unroll
    for (int i = 0; i < kHc; ++i) acc = fadd<FTZ>(acc, fmul<FTZ>(a[i], b[i]));
  }
  return acc;
}

struct Params {
  const uint4* partial;
  const uint4* residual;
  const float* post;
  const float* comb;
  uint4* out;
  uint4* stage;
  const uint4* peer_stage;
  int* flags;
  int* peer_flags;
  int* state;
  int epoch;
  int64_t vecs;
};

__global__ __launch_bounds__(kStageThreads) void stage_kernel(const __grid_constant__ Params p) {
  if (threadIdx.x == 0) {
    const int need = p.state[0];
    while (ld_acquire_sys(p.flags + 1) < need) {
    }
  }
  __syncthreads();
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; v + (kUnroll - 1) * stride < p.vecs; v += kUnroll * stride) {
    uint4 r[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) r[u] = p.partial[v + u * stride];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) p.stage[v + u * stride] = r[u];
  }
  for (; v < p.vecs; v += stride) p.stage[v] = p.partial[v];
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(p.state + 1, 1) == static_cast<int>(gridDim.x) - 1) {
      p.state[1] = 0;
      __threadfence_system();
      st_release_sys(p.peer_flags, p.epoch);
      p.state[0] = p.epoch;
    }
  }
}



template <int V, bool FTZ>
__device__ __forceinline__ void post_row(const Params& p) {
  const int64_t row = blockIdx.x;
  float c[kHc], a[kHc][kHc];
#pragma unroll
  for (int o = 0; o < kHc; ++o) {
    c[o] = p.post[row * kHc + o];
#pragma unroll
    for (int i = 0; i < kHc; ++i) a[o][i] = p.comb[row * kHc * kHc + i * kHc + o];
  }
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) {
    const int64_t vid = threadIdx.x + j * kThreads;
    const uint4 mv = p.partial[row * kVecs + vid];
    const uint4 pv = ld_volatile(p.peer_stage + row * kVecs + vid);
    uint4 rv[kHc];
#pragma unroll
    for (int i = 0; i < kHc; ++i) rv[i] = p.residual[(row * kHc + i) * kVecs + vid];
    const uint32_t xs[4] = {hadd2(mv.x, pv.x), hadd2(mv.y, pv.y), hadd2(mv.z, pv.z), hadd2(mv.w, pv.w)};
    const uint32_t r[kHc][4] = {{rv[0].x, rv[0].y, rv[0].z, rv[0].w}, {rv[1].x, rv[1].y, rv[1].z, rv[1].w},
                                {rv[2].x, rv[2].y, rv[2].z, rv[2].w}, {rv[3].x, rv[3].y, rv[3].z, rv[3].w}};
#pragma unroll
    for (int o = 0; o < kHc; ++o) {
      uint32_t w[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float blo[kHc] = {lo_f(r[0][k]), lo_f(r[1][k]), lo_f(r[2][k]), lo_f(r[3][k])};
        const float bhi[kHc] = {hi_f(r[0][k]), hi_f(r[1][k]), hi_f(r[2][k]), hi_f(r[3][k])};
        w[k] = pack_bf16(post_one<V, FTZ>(c[o], lo_f(xs[k]), a[o], blo),
                         post_one<V, FTZ>(c[o], hi_f(xs[k]), a[o], bhi));
      }
      p.out[(row * kHc + o) * kVecs + vid] = make_uint4(w[0], w[1], w[2], w[3]);
    }
  }
}

template <int V, bool FTZ>
__global__ __launch_bounds__(kThreads) void post_kernel(const __grid_constant__ Params p) {
  if (threadIdx.x == 0) {
    while (ld_acquire_sys(p.flags) < p.epoch) {
    }
  }
  __syncthreads();
  post_row<V, FTZ>(p);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (atomicAdd(p.state + 2, 1) == static_cast<int>(gridDim.x) - 1) {
      p.state[2] = 0;
      __threadfence_system();
      st_release_sys(p.peer_flags + 1, p.epoch);
    }
  }
}




template <int V, bool FTZ>
__global__ __launch_bounds__(kThreads) void post_slot_kernel(const __grid_constant__ Params p) {
  if (threadIdx.x == 0) {
    if (ld_relaxed_gpu(p.state) < p.epoch && atomicMax(p.state, p.epoch) < p.epoch) {
      __threadfence_system();
      st_release_sys(p.peer_flags, p.epoch);
    }
    while (ld_acquire_sys(p.flags) < p.epoch) {
    }
  }
  __syncthreads();
  post_row<V, FTZ>(p);
}

Params make_params(const torch::Tensor& partial, const torch::Tensor& stage, int64_t peer_stage,
                   const torch::Tensor& flags, int64_t peer_flags, const torch::Tensor& state, int64_t epoch) {
  TORCH_CHECK(partial.scalar_type() == at::kBFloat16 && partial.is_contiguous() && partial.dim() == 2 &&
                  partial.size(1) == kHidden, "e31_fused: partial is contiguous bf16 [n, 4096]");
  TORCH_CHECK(stage.scalar_type() == at::kBFloat16 && stage.numel() >= partial.numel(), "e31_fused: stage");
  TORCH_CHECK(flags.scalar_type() == at::kInt && flags.numel() >= 2 && state.scalar_type() == at::kInt &&
                  state.numel() >= 3, "e31_fused: flags / state");
  Params p{};
  p.partial = reinterpret_cast<const uint4*>(partial.data_ptr());
  p.stage = reinterpret_cast<uint4*>(stage.data_ptr());
  p.peer_stage = reinterpret_cast<const uint4*>(peer_stage);
  p.flags = flags.data_ptr<int>();
  p.peer_flags = reinterpret_cast<int*>(peer_flags);
  p.state = state.data_ptr<int>();
  p.epoch = static_cast<int>(epoch);
  p.vecs = partial.numel() / 8;
  return p;
}



void set_post(Params& p, int64_t n, const torch::Tensor& residual, const torch::Tensor& post_mix,
              const torch::Tensor& comb, const torch::Tensor& out) {
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16 && residual.is_contiguous() &&
                  residual.numel() == n * kHc * kHidden, "e31_fused: residual is contiguous bf16 [n, 4 * 4096]");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == n * kHc * kHidden,
              "e31_fused: out");
  TORCH_CHECK(post_mix.scalar_type() == at::kFloat && post_mix.is_contiguous() && post_mix.numel() == n * kHc &&
                  comb.scalar_type() == at::kFloat && comb.is_contiguous() && comb.numel() == n * kHc * kHc,
              "e31_fused: post [n, 4] / comb [n, 16] contiguous f32");
  TORCH_CHECK(n >= 1 && n <= 65535 * 64, "e31_fused: rows");
  p.residual = reinterpret_cast<const uint4*>(residual.data_ptr());
  p.post = post_mix.data_ptr<float>();
  p.comb = comb.data_ptr<float>();
  p.out = reinterpret_cast<uint4*>(out.data_ptr());
}

template <bool Slot, int V, bool FTZ>
void launch(const Params& p, int64_t n) {
  const dim3 grid(static_cast<unsigned>(n));
  auto stream = at::cuda::getCurrentCUDAStream();
  if constexpr (Slot) post_slot_kernel<V, FTZ><<<grid, kThreads, 0, stream>>>(p);
  else post_kernel<V, FTZ><<<grid, kThreads, 0, stream>>>(p);
}

template <bool Slot>
void launch_post(const Params& p, int64_t n, int64_t variant, bool ftz) {
  if (variant == 0 && !ftz) launch<Slot, 0, false>(p, n);
  else if (variant == 0) launch<Slot, 0, true>(p, n);
  else if (variant == 1 && !ftz) launch<Slot, 1, false>(p, n);
  else if (variant == 1) launch<Slot, 1, true>(p, n);
  else if (variant == 2 && !ftz) launch<Slot, 2, false>(p, n);
  else if (variant == 2) launch<Slot, 2, true>(p, n);
  else TORCH_CHECK(false, "e31_fused: variant is 0, 1 or 2");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}


void stage(const torch::Tensor& partial, const torch::Tensor& stage_buf, int64_t peer_stage, const torch::Tensor& flags,
           int64_t peer_flags, const torch::Tensor& state, int64_t epoch, int64_t grid) {
  const c10::cuda::CUDAGuard guard(partial.device());
  Params p = make_params(partial, stage_buf, peer_stage, flags, peer_flags, state, epoch);
  TORCH_CHECK(grid >= 1 && grid <= 65535, "e31_fused: grid");
  stage_kernel<<<static_cast<unsigned>(grid), kStageThreads, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void post(const torch::Tensor& partial, const torch::Tensor& residual, const torch::Tensor& post_mix,
          const torch::Tensor& comb, const torch::Tensor& out, const torch::Tensor& stage_buf, int64_t peer_stage,
          const torch::Tensor& flags, int64_t peer_flags, const torch::Tensor& state, int64_t epoch, int64_t variant,
          bool ftz) {
  const c10::cuda::CUDAGuard guard(partial.device());
  Params p = make_params(partial, stage_buf, peer_stage, flags, peer_flags, state, epoch);
  const int64_t n = partial.size(0);
  set_post(p, n, residual, post_mix, comb, out);
  launch_post<false>(p, n, variant, ftz);
}




void post_slot(const torch::Tensor& partial, const torch::Tensor& residual, const torch::Tensor& post_mix,
               const torch::Tensor& comb, const torch::Tensor& out, int64_t peer_slot, const torch::Tensor& flags,
               int64_t peer_flags, const torch::Tensor& state, int64_t epoch, int64_t variant, bool ftz) {
  const c10::cuda::CUDAGuard guard(partial.device());
  TORCH_CHECK(partial.scalar_type() == at::kBFloat16 && partial.is_contiguous() && partial.dim() == 2 &&
                  partial.size(1) == kHidden, "e31_fused: partial is contiguous bf16 [n, 4096]");
  TORCH_CHECK(flags.scalar_type() == at::kInt && flags.numel() >= 1 && state.scalar_type() == at::kInt &&
                  state.numel() >= 1, "e31_fused: slot flags / state");
  TORCH_CHECK(peer_slot != 0 && peer_flags != 0 && epoch >= 1 && epoch <= INT32_MAX, "e31_fused: peer slot / epoch");
  Params p{};
  p.partial = reinterpret_cast<const uint4*>(partial.data_ptr());
  p.peer_stage = reinterpret_cast<const uint4*>(peer_slot);
  p.flags = flags.data_ptr<int>();
  p.peer_flags = reinterpret_cast<int*>(peer_flags);
  p.state = state.data_ptr<int>();
  p.epoch = static_cast<int>(epoch);
  p.vecs = partial.numel() / 8;
  const int64_t n = partial.size(0);
  set_post(p, n, residual, post_mix, comb, out);
  launch_post<true>(p, n, variant, ftz);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stage", &stage, "stage this rank's partial for the peer (E31 fused, PULL form)");
  m.def("post", &post, "the TP2 sum fused into GLM-5.3-Flash's mHC post, bitwise NCCL + stock's post");
  m.def("post_slot", &post_slot, "post's publish-only form: the partial already in this rank's stage slot");
}
