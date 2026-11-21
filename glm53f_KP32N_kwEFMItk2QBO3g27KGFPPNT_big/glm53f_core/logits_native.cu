





































#include <cstdint>

#ifndef LOGITS_HOST
#include <cuda_runtime.h>
#endif

namespace logits_pull {

constexpr int kMaxRanks = 8;
constexpr int kMaxBlocks = 1024;
constexpr int kThreads = 512;
constexpr int kUnroll = 4;
constexpr int kPackThreads = 512;
constexpr int kMaxSide = 12288;
static_assert(kMaxSide * 4 <= 48 * 1024, "the pull's side words fit the default dynamic shared memory");
constexpr uint32_t kMaxCode = 0x7Fu, kNegInf = 0xFEu, kNaN = 0xFFu;
constexpr uint32_t kMaxHalf = 0x1FC0u, kNegInfHalf = 0x9F80u, kNaNHalf = 0x9FC0u;
constexpr uint32_t kSlow = 1u;
constexpr uint32_t kFastTop = 245;

struct Params {
  const uint4* src[kMaxRanks];
  const uint32_t* side[kMaxRanks];
  uint32_t* pad[kMaxRanks];
  float* out;
  int64_t ld;
  uint32_t rows;
  uint32_t vr16;
  uint32_t world;
  uint32_t rank;
};

struct PackParams {
  const uint4* stage;
  uint2* codes;
  uint32_t* side;
  uint32_t vr8;
};

#ifndef LOGITS_HOST

__device__ __forceinline__ uint4 load16(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ uint32_t load_side(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}


__device__ __forceinline__ uint32_t cast_bf16(uint32_t h) {
  float f;
  asm("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(static_cast<uint16_t>(h)));
  return __float_as_uint(f);
}


__device__ __forceinline__ uint32_t mul_rm(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rm.f32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}


__device__ __forceinline__ uint32_t mul_rn(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.f32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}


__device__ __forceinline__ uint32_t f16_rm(uint32_t a) {
  uint16_t d;
  asm("cvt.rm.f16.f32 %0, %1;" : "=h"(d) : "r"(a));
  return d;
}


__device__ __forceinline__ uint32_t f32_of_f16(uint32_t h) {
  uint32_t d;
  asm("cvt.f32.f16 %0, %1;" : "=r"(d) : "h"(static_cast<uint16_t>(h)));
  return d;
}


__device__ __forceinline__ uint32_t max_u16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("max.u16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ uint32_t cas_release(uint32_t* slot, uint32_t expect, uint32_t value) {
  uint32_t old;
  asm volatile("atom.release.sys.global.cas.b32 %0, [%1], %2, %3;"
               : "=r"(old)
               : "l"(slot), "r"(expect), "r"(value)
               : "memory");
  return old;
}

__device__ __forceinline__ uint32_t cas_acquire(uint32_t* slot, uint32_t expect, uint32_t value) {
  uint32_t old;
  asm volatile("atom.acquire.sys.global.cas.b32 %0, [%1], %2, %3;"
               : "=r"(old)
               : "l"(slot), "r"(expect), "r"(value)
               : "memory");
  return old;
}
#endif




__device__ __forceinline__ uint32_t keys2(uint32_t w) { return w ^ (((w >> 15) & 0x00010001u) * 0x7FFFu) ^ 0x80008000u; }


__device__ __forceinline__ void pack_scan(const PackParams& q, uint32_t row, uint32_t t, uint32_t n, uint32_t& kmax,
                                          uint32_t& amax) {
  uint32_t k2 = 0, a2 = 0;
  const uint4* src = q.stage + static_cast<size_t>(row) * q.vr8;
  for (uint32_t v = t; v < q.vr8; v += n) {
    const uint4 w = src[v];
    k2 = max_u16x2(max_u16x2(k2, keys2(w.x)), max_u16x2(keys2(w.y), max_u16x2(keys2(w.z), keys2(w.w))));
    a2 = max_u16x2(max_u16x2(a2, w.x & 0x7FFF7FFFu),
                   max_u16x2(w.y & 0x7FFF7FFFu, max_u16x2(w.z & 0x7FFF7FFFu, w.w & 0x7FFF7FFFu)));
  }
  kmax = (k2 & 0xFFFFu) > (k2 >> 16) ? (k2 & 0xFFFFu) : (k2 >> 16);
  amax = (a2 & 0xFFFFu) > (a2 >> 16) ? (a2 & 0xFFFFu) : (a2 >> 16);
}


__device__ __forceinline__ void pack_rescan(const PackParams& q, uint32_t row, uint32_t t, uint32_t n, uint32_t& kmax,
                                            uint32_t& amax) {
  kmax = amax = 0;
  const uint16_t* src = reinterpret_cast<const uint16_t*>(q.stage + static_cast<size_t>(row) * q.vr8);
  for (uint32_t i = t; i < q.vr8 * 8; i += n) {
    const uint32_t h = src[i], mag = h & 0x7FFFu, key = keys2(h) & 0xFFFFu;
    if (mag <= 0x7F80u && key > kmax) kmax = key;
    if (mag < 0x7F80u && mag > amax) amax = mag;
  }
}

__device__ __forceinline__ uint32_t pack_side(uint32_t kmax, uint32_t amax, bool nonfinite) {
  const uint32_t m = (kmax & 0x8000u) ? (kmax & 0x7FFFu) : (~kmax & 0xFFFFu);
  const uint32_t e = (amax >> 7) > 1u ? (amax >> 7) : 1u;
  return (m << 16) | (e << 8) | ((nonfinite || e > kFastTop) ? kSlow : 0u);
}


template <bool kSpecial>
__device__ __forceinline__ uint32_t encode2(uint32_t w, uint32_t scale, uint32_t m) {
  uint32_t y = f16_rm(mul_rm(w << 16, scale)) | (f16_rm(mul_rm(w & 0xFFFF0000u, scale)) << 16);

  y = (y & 0xFFC0FFC0u) + (((y & 0x003F003Fu) + 0x003F003Fu) & (y >> 9) & 0x00400040u);
  uint32_t c = ((y >> 6) & 0x007F007Fu) | ((y >> 8) & 0x00800080u);
#pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    const uint32_t h = (w >> (16 * lane)) & 0xFFFFu, keep = lane ? 0x0000FFFFu : 0xFFFF0000u;
    if (h == m) {
      c = (c & keep) | (kMaxCode << (16 * lane));
    } else if (kSpecial && (h & 0x7FFFu) >= 0x7F80u) {
      c = (c & keep) | (((h & 0x7FFFu) > 0x7F80u ? kNaN : kNegInf) << (16 * lane));
    }
  }
  return c;
}


template <bool kSpecial>
__device__ __forceinline__ void pack_codes(const PackParams& q, uint32_t row, uint32_t t, uint32_t n, uint32_t side) {
  const uint32_t m = side >> 16, e = (side >> 8) & 0xFFu;
  const uint32_t scale = e <= 244u ? (245u - e) << 23 : 1u << (267u - e);
  const uint4* src = q.stage + static_cast<size_t>(row) * q.vr8;
  uint2* dst = q.codes + static_cast<size_t>(row) * q.vr8;
  for (uint32_t v = t; v < q.vr8; v += n) {
    const uint4 w = src[v];
    dst[v] = make_uint2(__byte_perm(encode2<kSpecial>(w.x, scale, m), encode2<kSpecial>(w.y, scale, m), 0x6420u),
                        __byte_perm(encode2<kSpecial>(w.z, scale, m), encode2<kSpecial>(w.w, scale, m), 0x6420u));
  }
}

#ifndef LOGITS_HOST
__device__ __forceinline__ uint32_t block_max(uint32_t v, uint32_t* red) {
  v = __reduce_max_sync(0xffffffffu, v);
  if ((threadIdx.x & 31u) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  v = __reduce_max_sync(0xffffffffu, threadIdx.x % 32u < kPackThreads / 32u ? red[threadIdx.x % 32u] : 0u);
  __syncthreads();
  return v;
}

__global__ __launch_bounds__(kPackThreads) void pack_kernel(const __grid_constant__ PackParams q) {
  __shared__ uint32_t red[kPackThreads / 32];
  const uint32_t row = blockIdx.x;
  uint32_t kmax, amax;
  pack_scan(q, row, threadIdx.x, kPackThreads, kmax, amax);
  kmax = block_max(kmax, red);
  amax = block_max(amax, red);
  const bool nonfinite = amax >= 0x7F80u;
  if (nonfinite) {
    pack_rescan(q, row, threadIdx.x, kPackThreads, kmax, amax);
    kmax = block_max(kmax, red);
    amax = block_max(amax, red);
  }
  const uint32_t side = pack_side(kmax, amax, nonfinite);
  if (threadIdx.x == 0) q.side[row] = side;
  if (nonfinite) {
    pack_codes<true>(q, row, threadIdx.x, kPackThreads, side);
  } else {
    pack_codes<false>(q, row, threadIdx.x, kPackThreads, side);
  }
}
#endif




__device__ __forceinline__ uint32_t* slot(const Params& p, uint32_t owner, uint32_t phase, uint32_t sender) {
  return p.pad[owner] + (static_cast<size_t>(phase) * kMaxBlocks + blockIdx.x) * kMaxRanks + sender;
}


__device__ __forceinline__ void arrive(const Params& p, uint32_t phase, uint32_t s) {
  while (cas_release(slot(p, s, phase, p.rank), 0u, 1u) != 0u) {
  }
}


__device__ __forceinline__ void depart(const Params& p, uint32_t phase, uint32_t s) {
  while (cas_acquire(slot(p, p.rank, phase, s), 1u, 0u) != 1u) {
  }
}


__device__ __forceinline__ void stage_side(const Params& p, uint32_t* side_s, uint32_t t, uint32_t n) {
  for (uint32_t i = t; i < p.rows * p.world; i += n) side_s[i] = load_side(p.side[i % p.world] + i / p.world);
}


__device__ __forceinline__ void locate(const Params& p, uint32_t c, uint32_t& row, uint32_t& s, uint32_t& j) {
  const uint32_t per_row = p.world * p.vr16;
  row = c / per_row;
  const uint32_t col = c - row * per_row;
  s = col / p.vr16;
  j = col - s * p.vr16;
}

__device__ __forceinline__ const uint4* source(const Params& p, uint32_t c) {
  uint32_t row, s, j;
  locate(p, c, row, s, j);
  return p.src[s] + static_cast<size_t>(row) * p.vr16 + j;
}


__device__ __forceinline__ uint32_t halves(uint32_t q) { return ((q & 0x007F007Fu) << 6) | ((q & 0x00800080u) << 8); }

__device__ __forceinline__ uint32_t value(uint32_t h, uint32_t scale, uint32_t maxf) {
  return h == kMaxHalf ? maxf : mul_rn(f32_of_f16(h), scale);
}



__device__ __forceinline__ uint32_t value_slow(uint32_t h, uint32_t e, uint32_t maxf) {
  if (h == kMaxHalf) return maxf;
  if (h == kNegInfHalf) return 0xFF800000u;
  if (h == kNaNHalf) return 0x7FFFFFFFu;
  const int x = static_cast<int>(e) - 118, a = x > 127 ? 127 : x;
  const uint32_t r = mul_rn(mul_rn(f32_of_f16(h), static_cast<uint32_t>(a + 127) << 23),
                            static_cast<uint32_t>(x - a + 127) << 23);
  return r == 0xFF800000u ? 0xFF7FFFFFu : r;
}


__device__ __forceinline__ void put(const Params& p, const uint32_t* side_s, uint32_t c, const uint4& v) {
  uint32_t row, s, j;
  locate(p, c, row, s, j);
  const uint32_t side = side_s[row * p.world + s], e = (side >> 8) & 0xFFu, maxf = cast_bf16(side >> 16);
  uint4* o = reinterpret_cast<uint4*>(p.out + static_cast<size_t>(row) * p.ld +
                                      (static_cast<size_t>(s) * p.vr16 + j) * 16);
  const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t a = halves(__byte_perm(words[i], 0u, 0x4140u)), b = halves(__byte_perm(words[i], 0u, 0x4342u));
    if (side & kSlow) {
      o[i] = make_uint4(value_slow(a & 0xFFFFu, e, maxf), value_slow(a >> 16, e, maxf), value_slow(b & 0xFFFFu, e, maxf),
                        value_slow(b >> 16, e, maxf));
    } else {
      const uint32_t scale = (e + 9u) << 23;
      o[i] = make_uint4(value(a & 0xFFFFu, scale, maxf), value(a >> 16, scale, maxf), value(b & 0xFFFFu, scale, maxf),
                        value(b >> 16, scale, maxf));
    }
  }
}


__device__ __forceinline__ void move(const Params& p, const uint32_t* side_s) {
  const uint32_t total = p.rows * p.world * p.vr16;
  const uint32_t stride = gridDim.x * kThreads;
  uint32_t c = blockIdx.x * kThreads + threadIdx.x;
  for (; c + (kUnroll - 1) * stride < total; c += kUnroll * stride) {
    uint4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) v[u] = load16(source(p, c + u * stride));
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) put(p, side_s, c + u * stride, v[u]);
  }
  for (; c < total; c += stride) put(p, side_s, c, load16(source(p, c)));
}


__device__ __forceinline__ void meet(const Params& p, uint32_t phase) {
  if (threadIdx.x < p.world) {
    arrive(p, phase, threadIdx.x);
    depart(p, phase, threadIdx.x);
  }
  __syncthreads();
}

#ifndef LOGITS_HOST
__global__ __launch_bounds__(kThreads) void gather_kernel(const __grid_constant__ Params p) {
  extern __shared__ uint32_t side_s[];
  meet(p, 0);
  stage_side(p, side_s, threadIdx.x, kThreads);
  __syncthreads();
  move(p, side_s);
  __syncthreads();
  meet(p, 1);
}
#endif

}

#ifndef LOGITS_DEVICE_ONLY
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>


int64_t pad_words() { return int64_t(2) * logits_pull::kMaxBlocks * logits_pull::kMaxRanks; }





int64_t gather(torch::Tensor out, torch::Tensor stage, std::vector<int64_t> wires, std::vector<int64_t> pads,
               int64_t rank, int64_t vr, int64_t capacity, int64_t max_blocks) {
  using namespace logits_pull;
  const int64_t world = static_cast<int64_t>(wires.size());
  TORCH_CHECK(world >= 2 && world <= kMaxRanks && static_cast<int64_t>(pads.size()) == world,
              "logits gather: 2 to 8 ranks, one wire and one pad each");
  TORCH_CHECK(rank >= 0 && rank < world, "logits gather: rank outside the group");
  TORCH_CHECK(vr > 0 && vr % 16 == 0, "logits gather: a rank's slice must be a multiple of 16 columns");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.dim() == 2 && out.size(1) == world * vr &&
                  out.stride(1) == 1 && (out.size(0) == 1 || out.stride(0) >= world * vr) && out.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "logits gather: out must be FP32 [rows, world * vr], unit column stride, 16-byte aligned rows");
  const int64_t rows = out.size(0);
  TORCH_CHECK(stage.is_cuda() && stage.device() == out.device() && stage.scalar_type() == at::kBFloat16 &&
                  stage.dim() == 2 && stage.size(0) == rows && stage.size(1) == vr && stage.is_contiguous() &&
                  reinterpret_cast<uintptr_t>(stage.data_ptr()) % 16 == 0,
              "logits gather: stage must be this rank's contiguous BF16 [rows, vr], 16-byte aligned");
  const int64_t vecs = rows * world * (vr / 16);
  TORCH_CHECK(rows >= 1 && rows <= capacity && rows * world <= kMaxSide && vecs < (int64_t(1) << 31),
              "logits gather: 1 to capacity rows, at most 12,288 rows x ranks");
  TORCH_CHECK(max_blocks >= 1 && max_blocks <= kMaxBlocks, "logits gather: max_blocks must be 1 to 1024");
  Params p{};
  for (int64_t s = 0; s < world; ++s) {
    TORCH_CHECK(wires[s] != 0 && wires[s] % 16 == 0 && pads[s] != 0 && pads[s] % 4 == 0,
                "logits gather: wires 16-byte and pads 4-byte aligned");
    p.src[s] = reinterpret_cast<const uint4*>(wires[s]);
    p.side[s] = reinterpret_cast<const uint32_t*>(wires[s] + capacity * vr);
    p.pad[s] = reinterpret_cast<uint32_t*>(pads[s]);
  }
  p.out = out.data_ptr<float>();
  p.ld = out.stride(0);
  p.rows = static_cast<uint32_t>(rows);
  p.vr16 = static_cast<uint32_t>(vr / 16);
  p.world = static_cast<uint32_t>(world);
  p.rank = static_cast<uint32_t>(rank);
  PackParams q{};
  q.stage = reinterpret_cast<const uint4*>(stage.data_ptr());
  q.codes = reinterpret_cast<uint2*>(wires[rank]);
  q.side = reinterpret_cast<uint32_t*>(wires[rank] + capacity * vr);
  q.vr8 = static_cast<uint32_t>(vr / 8);
  const int64_t grid = std::min(max_blocks, (vecs + int64_t(kThreads) * kUnroll - 1) / (int64_t(kThreads) * kUnroll));
  const c10::cuda::CUDAGuard guard(out.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  pack_kernel<<<static_cast<unsigned>(rows), kPackThreads, 0, stream>>>(q);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  gather_kernel<<<static_cast<unsigned>(grid), kThreads, static_cast<size_t>(rows * world) * 4, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grid;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pad_words", &pad_words, "int32 words of a rank's signal pad");
  m.def("gather", &gather,
        "the TP all-gather of the lm_head's BF16 vocabulary slices into [rows, vocab] FP32 logits: a pack kernel a "
        "rank (8-bit codes, each slice row's maximum exact) and one pull kernel");
}
#endif
