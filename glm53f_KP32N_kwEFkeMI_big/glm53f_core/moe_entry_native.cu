



































































#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace moe_entry {

constexpr int EXPERTS = 288;
constexpr int TOP_K = 8;
constexpr int HIDDEN = 4096;
constexpr int NW = 32;
constexpr int NT = EXPERTS / NW;
constexpr int KC = 8;
constexpr int KCH = HIDDEN / KC;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int RMAX = 64;
constexpr int GMAX = 2;
constexpr int MAX_ROWS = RMAX * GMAX;
constexpr int XS = KCH * 2 + 64;
constexpr int ES = NW + 4;
constexpr int CAND = NT * TOP_K;
constexpr int SMEM_A = 4 * RMAX * ES * 4;
constexpr int SMEM_X = RMAX * XS;
constexpr int SMEM_BYTES = SMEM_A + SMEM_X + 16;
constexpr int CNT_ARRIVE = 0;
constexpr int CNT_DONE = GMAX * KC;
constexpr int CNT_WORDS = CNT_DONE + 1;
constexpr int W_SHARE = EXPERTS * HIDDEN * 2 / 16;
constexpr unsigned FULL = 0xffffffffu;
static_assert(NW * XS <= SMEM_A && 3 * EXPERTS + 1 + 2 * MAX_ROWS * TOP_K <= SMEM_X / 4,
              "the shared regions' second uses fit");

struct Params {
  const uint16_t* x;
  const uint16_t* w;
  const float* bias;
  const float* gscale;
  const float* logits_in;
  float scale;
  int rows, group_rows, tile, local_start, local_count;
  float* logits;
  uint8_t* xq;
  uint8_t* xsf;
  int16_t* replay;
  uint16_t* weights;
  int32_t* padded;
  int32_t* e2p;
  int32_t* p2t;
  int32_t* cta_batch;
  int32_t* cta_limit;
  int32_t* ctas;
  unsigned long long* cand_key;
  float* cand_sig;
  int32_t* ent;
  int32_t* counters;
};

#ifndef M1_HOST


__device__ __forceinline__ float mul_ftz(float a, float b) {
  float d;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float add_ftz(float a, float b) {
  float d;
  asm("add.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ float tanh_approx(float a) {
  float d;
  asm("tanh.approx.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}
__device__ __forceinline__ float div_approx_ftz(float a, float b) {
  float d;
  asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ uint32_t bf16_rn(float a) {
  uint16_t d;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(a));
  return d;
}
__device__ __forceinline__ uint32_t redux_max(uint32_t v) {
  uint32_t d;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(d) : "r"(v));
  return d;
}





__device__ __forceinline__ uint32_t bmax2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ float bhmax_f32(uint32_t x) {
  float d;
  asm("{\n .reg .b32 lo, hi;\n .reg .f32 f0, f1;\n and.b32 lo, %1, 0xFFFF;\n shr.b32 hi, %1, 16;\n"
      " shl.b32 lo, lo, 16;\n shl.b32 hi, hi, 16;\n mov.b32 f0, lo;\n mov.b32 f1, hi;\n max.f32 %0, f0, f1;\n}"
      : "=f"(d) : "r"(x));
  return d;
}
__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float d;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}


__device__ __forceinline__ float dsl_mul(float a, float b) {
  float d;
  asm("mul.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ uint32_t cvt_e4m3(float a) {
  uint32_t d;
  asm("{\n .reg .b16 fp8_pair;\n .reg .f32 zero;\n mov.f32 zero, 0f00000000;\n"
      " cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, %1;\n cvt.u32.u16 %0, fp8_pair;\n}"
      : "=r"(d) : "f"(a));
  return d;
}
__device__ __forceinline__ float output_scale(uint32_t fp8, float gs) {
  float d;
  asm("{\n .reg .pred p_zero;\n .reg .b16 fp8_pair;\n .reg .b32 h2_32;\n .reg .b16 h_lo, h_hi;\n"
      " .reg .f32 scale_f32, rcp_gs, product, result;\n cvt.u16.u32 fp8_pair, %1;\n"
      " cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;\n mov.b32 {h_lo, h_hi}, h2_32;\n cvt.f32.f16 scale_f32, h_lo;\n"
      " rcp.approx.ftz.f32 rcp_gs, %2;\n mul.f32 product, scale_f32, rcp_gs;\n rcp.approx.ftz.f32 result, product;\n"
      " setp.eq.f32 p_zero, scale_f32, 0f00000000;\n selp.f32 %0, 0f00000000, result, p_zero;\n}"
      : "=f"(d) : "r"(fp8), "f"(gs));
  return d;
}
__device__ __forceinline__ void bf2_scaled(uint32_t h, float s, float& f0, float& f1) {
  asm("{\n .reg .b32 lo, hi;\n .reg .f32 f0, f1;\n and.b32 lo, %2, 0xFFFF;\n shr.b32 hi, %2, 16;\n"
      " shl.b32 lo, lo, 16;\n shl.b32 hi, hi, 16;\n mov.b32 f0, lo;\n mov.b32 f1, hi;\n mul.f32 %0, f0, %3;\n"
      " mul.f32 %1, f1, %3;\n}"
      : "=f"(f0), "=f"(f1) : "r"(h), "f"(s));
}
__device__ __forceinline__ uint32_t e2m1x8(const float (&v)[8]) {
  uint32_t d;
  asm("{\n .reg .b8 byte0, byte1, byte2, byte3;\n cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
      " cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
      " cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n mov.b32 %0, {byte0, byte1, byte2, byte3};\n}"
      : "=r"(d) : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return d;
}


__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ unsigned lane_ballot(bool v) { return __ballot_sync(FULL, v); }
__device__ __forceinline__ float lane_shfl(float v, int src) { return __shfl_sync(FULL, v, src); }
__device__ __forceinline__ float lane_xor(float v, int mask) { return __shfl_xor_sync(FULL, v, mask); }
__device__ __forceinline__ int lane_up(int v, int d) { return __shfl_up_sync(FULL, v, d); }
__device__ __forceinline__ int lane_get(int v, int src) { return __shfl_sync(FULL, v, src); }
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 lds128(const uint8_t* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ void sts128(uint8_t* p, uint4 v) { *reinterpret_cast<uint4*>(p) = v; }
__device__ __forceinline__ void cp_async16(uint8_t* dst, const uint8_t* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ void prefetch_l2(const uint8_t* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ float dsmem_ld(const float* local, uint32_t rank) {
  uint32_t remote;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(smem_u32(local)), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");
  return v;
}
__device__ __forceinline__ int atomic_add(int* p, int v) { return atomicAdd(p, v); }
__device__ __forceinline__ void fence_gpu() { __threadfence(); }
__device__ __forceinline__ unsigned long long ldcg_u64(const unsigned long long* p) { return __ldcg(p); }
__device__ __forceinline__ float ldcg_f32(const float* p) { return __ldcg(p); }
__device__ __forceinline__ int ldcg_i32(const int* p) { return __ldcg(p); }
__device__ __forceinline__ float bits_f32(uint32_t v) { return __uint_as_float(v); }
__device__ __forceinline__ uint32_t f32_bits(float v) { return __float_as_uint(v); }
#endif


__device__ __forceinline__ float sigmoid(float x) { return fma_ftz(tanh_approx(mul_ftz(x, 0.5f)), 0.5f, 0.5f); }

__device__ __forceinline__ uint64_t order_key(float key, int e) {
  uint32_t bits = f32_bits(key);
  bits ^= (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
  return (uint64_t(bits) << 32) | uint64_t(0xFFFF & (65535 - e));
}

__device__ __forceinline__ uint64_t warp_max(uint64_t v) {
  const uint32_t hi = uint32_t(v >> 32), top = redux_max(hi);
  return (uint64_t(top) << 32) | redux_max(hi == top ? uint32_t(v) : 0u);
}



__device__ __forceinline__ void quantize_block(uint4 lo, uint4 hi, float gs, uint8_t& sf, uint64_t& packed) {
  const uint32_t h[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  constexpr uint32_t ABS = 0x7FFF7FFFu;
  const uint32_t m01 = bmax2(h[0] & ABS, h[1] & ABS), m23 = bmax2(h[2] & ABS, h[3] & ABS);
  const uint32_t m45 = bmax2(h[4] & ABS, h[5] & ABS), m67 = bmax2(h[6] & ABS, h[7] & ABS);
  const float amax = bhmax_f32(bmax2(bmax2(m01, m23), bmax2(m45, m67)));
  const float scale = dsl_mul(gs, dsl_mul(amax, rcp_approx_ftz(6.0f)));
  const uint32_t e4 = cvt_e4m3(scale);
  sf = uint8_t(e4 & 0xFFu);
  const float out = output_scale(e4, gs);
  float v[8];
  bf2_scaled(h[0], out, v[0], v[1]);
  bf2_scaled(h[1], out, v[2], v[3]);
  bf2_scaled(h[2], out, v[4], v[5]);
  bf2_scaled(h[3], out, v[6], v[7]);
  const uint32_t q_lo = e2m1x8(v);
  bf2_scaled(h[4], out, v[0], v[1]);
  bf2_scaled(h[5], out, v[2], v[3]);
  bf2_scaled(h[6], out, v[4], v[5]);
  bf2_scaled(h[7], out, v[6], v[7]);
  packed = (uint64_t(e2m1x8(v)) << 32) | q_lo;
}



__device__ __forceinline__ void merge_row(const Params& p, int row, int lane) {
  const unsigned long long* ck = p.cand_key + size_t(row) * CAND;
  const float* cs = p.cand_sig + size_t(row) * CAND;
  uint64_t key[3];
  float sig[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int idx = lane + 32 * i;
    key[i] = idx < CAND ? uint64_t(ldcg_u64(ck + idx)) : 0ull;
    sig[i] = idx < CAND ? ldcg_f32(cs + idx) : 0.f;
  }
  int mine_e = 0;
  float mine_s = 0.f;
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    uint64_t m = key[0] > key[1] ? key[0] : key[1];
    m = m > key[2] ? m : key[2];
    const uint64_t best = warp_max(m);
    bool holds = false;
    float held = 0.f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      if (key[i] == best) {
        holds = true;
        held = sig[i];
        key[i] = 0;
      }
    }
    const float s = lane_shfl(held, __ffs(lane_ballot(holds)) - 1);
    if (lane == k) {
      mine_e = 65535 - int(best & 0xFFFFu);
      mine_s = s;
    }
  }
  float sum = lane < TOP_K ? mine_s : 0.f;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) sum = add_ftz(sum, lane_xor(sum, mask));
  if (lane < TOP_K) {
    const uint32_t w = bf16_rn(div_approx_ftz(mul_ftz(mine_s, p.scale), add_ftz(sum, 1e-20f)));
    const int x = row * TOP_K + lane;
    p.replay[x] = int16_t(mine_e);
    p.weights[x] = uint16_t(w);
    const int local = mine_e - p.local_start;
    p.ent[x] = (local >= 0 && local < p.local_count) ? mine_e : -1;
  }
}



__device__ __forceinline__ void permute(const Params& p, int* scratch, int tid, int lane, int warp) {
  int* count = scratch;
  int* cta_first = count + EXPERTS;
  int* first = cta_first + EXPERTS + 1;
  int* pick = first + EXPERTS;
  int* place = pick + MAX_ROWS * TOP_K;
  const int expanded = p.rows * TOP_K;
  for (int e = tid; e < EXPERTS; e += THREADS) count[e] = 0;
  __syncthreads();
  for (int x = tid; x < expanded; x += THREADS) {
    const int e = ldcg_i32(p.ent + x);
    pick[x] = e;
    place[x] = e < 0 ? -1 : atomic_add(&count[e], 1);
  }
  __syncthreads();
  if (warp == 0) {
    int carry = 0;
    for (int base = 0; base < EXPERTS; base += 32) {
      const int e = base + lane;
      const int n = (count[e] + p.tile - 1) / p.tile;
      int incl = n;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int up = lane_up(incl, d);
        if (lane >= d) incl += up;
      }
      cta_first[e] = carry + incl - n;
      carry += lane_get(incl, 31);
    }
    if (lane == 0) {
      cta_first[EXPERTS] = carry;
      p.padded[0] = carry * p.tile;
      p.ctas[0] = carry;
    }
  }
  __syncthreads();
  for (int e = tid; e < EXPERTS; e += THREADS) {
    const int n = count[e], c0 = cta_first[e], ncta = cta_first[e + 1] - c0;
    first[e] = c0 * p.tile;
    for (int q = 0; q < ncta; ++q) {
      const int end = (c0 + q + 1) * p.tile, limit = min(end, c0 * p.tile + n);
      p.cta_batch[c0 + q] = e - p.local_start;
      p.cta_limit[c0 + q] = limit;
      for (int r = limit; r < end; ++r) p.p2t[r] = -1;
    }
  }
  __syncthreads();
  for (int x = tid; x < expanded; x += THREADS) {
    const int perm = pick[x] < 0 ? -1 : first[pick[x]] + place[x];
    p.e2p[x] = perm;
    if (perm >= 0) p.p2t[perm] = x / TOP_K;
  }
  for (int i = tid; i < CNT_WORDS; i += THREADS) p.counters[i] = 0;
}

__global__ void __launch_bounds__(THREADS, 2) entry_kernel(const Params p) {
#ifndef M1_HOST
  extern __shared__ __align__(128) uint8_t smem[];
#else
  uint8_t* const smem = m1h_smem();
#endif
  uint8_t* const sa = smem;
  uint8_t* const sx = smem + SMEM_A;
  int* const flags = reinterpret_cast<int*>(smem + SMEM_A + SMEM_X);
  float* const part = reinterpret_cast<float*>(sa);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = blockIdx.x, t = blockIdx.y, grp = blockIdx.z;
  const int r0 = grp * p.group_rows;
  const int nr = min(p.group_rows, p.rows - r0);
  const int nj = (nr + 7) >> 3;


  {
    const uint8_t* w = reinterpret_cast<const uint8_t*>(p.w) + (size_t(NW) * t * HIDDEN + size_t(KCH) * c) * 2;
    for (int i = tid; i < NW * (KCH / 8); i += THREADS) {
      const int row = i >> 6, ch = i & 63;
      cp_async16(sa + row * XS + ch * 16, w + size_t(row) * HIDDEN * 2 + ch * 16);
    }
    cp_async_commit();
    const int linear = c + KC * (t + NT * grp);
    if (linear < 16 && tid < 9)
      prefetch_l2(reinterpret_cast<const uint8_t*>(p.w) + size_t(linear) * W_SHARE + size_t(tid) * (W_SHARE / 9),
                  W_SHARE / 9);
    for (int i = tid; i < (8 * nj - nr) * (KCH / 8); i += THREADS)
      sts128(sx + (nr + (i >> 6)) * XS + (i & 63) * 16, make_uint4(0u, 0u, 0u, 0u));
  }
  const float bias_lane = p.bias[NW * t + lane];
  const float gs = p.gscale[0];
  griddep_wait();


  {
    const uint8_t* x = reinterpret_cast<const uint8_t*>(p.x) + (size_t(r0) * HIDDEN + size_t(KCH) * c) * 2;
    for (int i = tid; i < nr * (KCH / 8); i += THREADS) {
      const int row = i >> 6, ch = i & 63;
      cp_async16(sx + row * XS + ch * 16, x + size_t(row) * HIDDEN * 2 + ch * 16);
    }
    cp_async_commit();
  }
  cp_async_wait_all();
  __syncthreads();


  {
    const int mt = warp & 1, kq = warp >> 1, g = lane >> 2, tq = lane & 3;
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    const uint8_t* a_lo = sa + (mt * 16 + g) * XS + (kq * 128 + 8 * tq) * 2;
    const uint8_t* a_hi = a_lo + 8 * XS;
    const uint8_t* b_row = sx + g * XS + (kq * 128 + 8 * tq) * 2;
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      const uint4 alo = lds128(a_lo + kb * 64), ahi = lds128(a_hi + kb * 64);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j < nj) {
          const uint4 b = lds128(b_row + j * 8 * XS + kb * 64);
          mma16816(acc[j], alo.x, ahi.x, alo.y, ahi.y, b.x, b.y);
          mma16816(acc[j], alo.z, ahi.z, alo.w, ahi.w, b.z, b.w);
        }
      }
    }
    __syncthreads();
    float* red = part + kq * (RMAX * ES);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j < nj) {
        const int r = 8 * j + 2 * tq, e = mt * 16 + g;
        red[r * ES + e] = acc[j][0];
        red[(r + 1) * ES + e] = acc[j][1];
        red[r * ES + e + 8] = acc[j][2];
        red[(r + 1) * ES + e + 8] = acc[j][3];
      }
    }
  }
  __syncthreads();

  for (int i = tid; i < nj * 8 * NW; i += THREADS) {
    const int o = (i >> 5) * ES + (i & 31);
    part[o] = add_ftz(add_ftz(add_ftz(part[o], part[RMAX * ES + o]), part[2 * RMAX * ES + o]),
                      part[3 * RMAX * ES + o]);
  }
  cluster_arrive();


  {
    const int nq = (nr - t + NT - 1) / NT;
    for (int i = tid; i < nq * (KCH / 16); i += THREADS) {
      const int lr = t + NT * (i >> 5), blk = i & 31;
      uint8_t sf;
      uint64_t q;
      quantize_block(lds128(sx + lr * XS + blk * 32), lds128(sx + lr * XS + blk * 32 + 16), gs, sf, q);
      const size_t row = size_t(r0 + lr);
      const int col = (KCH / 16) * c + blk;
      *reinterpret_cast<uint64_t*>(p.xq + row * (HIDDEN / 2) + size_t(col) * 8) = q;
      p.xsf[row * (HIDDEN / 16) + col] = sf;
    }
  }
  cluster_wait();


  const int lr = c + KC * warp;
  if (lr < nr) {
    const int row = r0 + lr, e = NW * t + lane;
    const float* mine = part + lr * ES + lane;
    float pv[KC];
#pragma unroll
    for (int q = 0; q < KC; ++q) pv[q] = dsmem_ld(mine, q);
    float v = pv[0];
#pragma unroll
    for (int q = 1; q < KC; ++q) v = add_ftz(v, pv[q]);
    p.logits[size_t(row) * EXPERTS + e] = v;
    if (p.logits_in != nullptr) v = p.logits_in[size_t(row) * EXPERTS + e];
    const float s = sigmoid(v);
    uint64_t key = order_key(add_ftz(s, bias_lane), e);
    unsigned long long* ck = p.cand_key + size_t(row) * CAND + TOP_K * t;
    float* cs = p.cand_sig + size_t(row) * CAND + TOP_K * t;
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      const uint64_t best = warp_max(key);
      if (key == best) {
        ck[k] = best;
        cs[k] = s;
        key = 0;
      }
    }
  }
  cluster_arrive();


  fence_gpu();
  __syncthreads();
  if (tid == 0) flags[0] = atomic_add(p.counters + CNT_ARRIVE + KC * grp + c, 1) == NT - 1;
  __syncthreads();
  if (flags[0]) {
    fence_gpu();
    if (lr < nr) merge_row(p, r0 + lr, lane);
    fence_gpu();
    __syncthreads();
    if (tid == 0) {
      const int n = nr > c ? (nr - c + KC - 1) / KC : 0;
      flags[1] = n > 0 && atomic_add(p.counters + CNT_DONE, n) + n == p.rows;
    }
    __syncthreads();
    if (flags[1]) {
      fence_gpu();
      permute(p, reinterpret_cast<int*>(sx), tid, lane, warp);
      fence_gpu();
      __syncthreads();
    }
  }
  griddep_launch();
  cluster_wait();
}

}

#ifndef MOE_ENTRY_DEVICE_ONLY
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>


void prepare() {
  C10_CUDA_CHECK(cudaFuncSetAttribute(moe_entry::entry_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      moe_entry::SMEM_BYTES));
}







void entry(torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor gscale, double scale, int64_t tile,
           int64_t local_start, int64_t local_count, std::optional<torch::Tensor> logits_in, torch::Tensor logits,
           torch::Tensor xq, torch::Tensor xsf, std::vector<torch::Tensor> record, std::vector<torch::Tensor> scratch,
           bool pdl) {
  using namespace moe_entry;
  const int64_t rows = x.size(0);
  const auto here = [&](const torch::Tensor& t, at::ScalarType s) {
    return t.is_cuda() && t.get_device() == x.get_device() && t.is_contiguous() && t.scalar_type() == s;
  };
  TORCH_CHECK(x.dim() == 2 && x.size(1) == HIDDEN && rows >= 1 && rows <= MAX_ROWS && here(x, at::kBFloat16) &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "moe_entry: x must be BF16 [1..128, 4096], contiguous, 16-byte aligned");
  TORCH_CHECK(w.dim() == 2 && w.size(0) == EXPERTS && w.size(1) == HIDDEN && here(w, at::kBFloat16) &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "moe_entry: the gate weight must be BF16 [288, 4096], contiguous, 16-byte aligned");
  TORCH_CHECK(bias.numel() == EXPERTS && here(bias, at::kFloat) && gscale.numel() == 1 && here(gscale, at::kFloat),
              "moe_entry: an FP32 bias [288] and an FP32 global scale [1]");
  TORCH_CHECK(tile >= 1 && local_start >= 0 && local_count >= 1 && local_start + local_count <= EXPERTS,
              "moe_entry: a positive tile and a local expert range inside the 288");
  TORCH_CHECK(logits.numel() == rows * EXPERTS && here(logits, at::kFloat) && xq.numel() == rows * HIDDEN / 2 &&
                  here(xq, at::kByte) && xsf.numel() == rows * HIDDEN / 16 && here(xsf, at::kByte) &&
                  reinterpret_cast<uintptr_t>(xq.data_ptr()) % 8 == 0,
              "moe_entry: logits [rows, 288] FP32, codes [rows, 2048] and scales [rows, 256] uint8");
  if (logits_in.has_value())
    TORCH_CHECK(logits_in->numel() == rows * EXPERTS && here(*logits_in, at::kFloat), "moe_entry: logits_in");
  TORCH_CHECK(record.size() == 11, "moe_entry: the canonical record has eleven tensors");
  const at::ScalarType kinds[11] = {at::kShort, at::kBFloat16, at::kInt, at::kInt, at::kInt, at::kInt,
                                    at::kInt,   at::kInt,      at::kInt, at::kInt, at::kInt};
  for (int i = 0; i < 11; ++i) TORCH_CHECK(here(record[i], kinds[i]), "moe_entry: record tensor ", i);
  TORCH_CHECK(record[0].numel() == rows * TOP_K && record[1].numel() == rows * TOP_K && record[4].numel() == 1 &&
                  record[5].numel() == rows * TOP_K && record[6].numel() >= 2 &&
                  record[8].numel() == record[9].numel() && record[10].numel() == 1,
              "moe_entry: the record's sizes are not those of ", rows, " rows");
  TORCH_CHECK(scratch.size() == 4 && scratch[0].numel() >= MAX_ROWS * CAND && here(scratch[0], at::kLong) &&
                  scratch[1].numel() >= MAX_ROWS * CAND && here(scratch[1], at::kFloat) &&
                  scratch[2].numel() >= MAX_ROWS * TOP_K && here(scratch[2], at::kInt) &&
                  scratch[3].numel() >= CNT_WORDS && here(scratch[3], at::kInt),
              "moe_entry: the scratch (candidate keys, sigmoids, entries, counters)");
  const int groups = int((rows + RMAX - 1) / RMAX);
  const int group_rows = int(((rows + groups - 1) / groups + 7) / 8 * 8);
  const c10::cuda::CUDAGuard guard(x.device());
  const Params p{reinterpret_cast<const uint16_t*>(x.data_ptr()), reinterpret_cast<const uint16_t*>(w.data_ptr()),
                 bias.data_ptr<float>(), gscale.data_ptr<float>(),
                 logits_in.has_value() ? logits_in->data_ptr<float>() : nullptr, float(scale), int(rows), group_rows,
                 int(tile), int(local_start), int(local_count), logits.data_ptr<float>(), xq.data_ptr<uint8_t>(),
                 xsf.data_ptr<uint8_t>(), reinterpret_cast<int16_t*>(record[0].data_ptr()),
                 reinterpret_cast<uint16_t*>(record[1].data_ptr()), record[4].data_ptr<int32_t>(),
                 record[5].data_ptr<int32_t>(), record[6].data_ptr<int32_t>(), record[8].data_ptr<int32_t>(),
                 record[9].data_ptr<int32_t>(), record[10].data_ptr<int32_t>(),
                 reinterpret_cast<unsigned long long*>(scratch[0].data_ptr()), scratch[1].data_ptr<float>(),
                 scratch[2].data_ptr<int32_t>(), scratch[3].data_ptr<int32_t>()};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(KC, NT, groups);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2] = {};
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = KC;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 2 : 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, entry_kernel, p));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &prepare, "the entry kernel's shared-memory attribute (once, outside capture)");
  m.def("entry", &entry,
        "router logits, FlashInfer's DeepSeekV3 routing (288 experts, top 8) into the canonical record and the "
        "NVFP4 quantize of x, in one launch");
}
#endif
