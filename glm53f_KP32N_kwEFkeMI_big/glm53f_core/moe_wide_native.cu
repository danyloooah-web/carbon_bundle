








































































#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace moe_entry {
constexpr unsigned FULL = 0xffffffffu;


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

}

namespace moe_wide {
using namespace moe_entry;

constexpr int EXPERTS = 288;
constexpr int TOP_K = 8;
constexpr int HIDDEN = 4096;
constexpr int CS = 8;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int ROUTE_WARPS = 4;
constexpr int MAX_ROWS = 768;
constexpr int MAX_CLUSTERS = 12;
constexpr int EPL = EXPERTS / 32;
constexpr int QBLK = HIDDEN / 16;
constexpr int RT = ROUTE_WARPS * 32;
constexpr int XPT = MAX_ROWS * TOP_K / CS / RT;
constexpr int EPT = (EXPERTS + RT - 1) / RT;
constexpr int QB = 4;
constexpr int OFF_HIST = 0;
constexpr int OFF_LENT = OFF_HIST + EXPERTS * 4;
constexpr int OFF_LPLACE = OFF_LENT + WARPS * TOP_K * 4;
constexpr int OFF_PRE = OFF_LPLACE + WARPS * TOP_K * 4;
constexpr int OFF_FIRST = OFF_PRE + MAX_CLUSTERS * EXPERTS * 4;
constexpr int OFF_FLAGS = OFF_FIRST + (EXPERTS + 4) * 4;

constexpr int SMEM_BYTES = OFF_FLAGS + 16;
static_assert(SMEM_BYTES <= 48 * 1024 && OFF_FLAGS % 16 == 0 && XPT * RT * CS == MAX_ROWS * TOP_K,
              "the shared layout; the final step's entries a thread");

struct Params {
  const float* logits;
  const uint16_t* x;
  const float* bias;
  const float* gscale;
  float scale;
  int rows, rpc, nclusters, tile, local_start, local_count;
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
  int32_t* counts;
  int32_t* ent;
  int32_t* place;
  int32_t* counter;
};

#ifndef M1_HOST
__device__ __forceinline__ void route_sync() { asm volatile("bar.sync 1, 128;" ::: "memory"); }
__device__ __forceinline__ int dsmem_atomic_add(int* local, uint32_t rank, int v) {
  uint32_t remote;
  int old;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(smem_u32(local)), "r"(rank));
  asm volatile("atom.shared::cluster.add.u32 %0, [%1], %2;" : "=r"(old) : "r"(remote), "r"(v) : "memory");
  return old;
}
#endif
__device__ __forceinline__ uint4 ld128(const uint16_t* p) { return lds128(reinterpret_cast<const uint8_t*>(p)); }
__device__ __forceinline__ int dsmem_ld_i32(const int* local, uint32_t rank) {
  return int(f32_bits(dsmem_ld(reinterpret_cast<const float*>(local), rank)));
}



__device__ __forceinline__ void route_row(const Params& p, int row, int lane, const float (&bias)[EPL], int* hist,
                                          int* lent, int* lplace) {
  const float* lg = p.logits + size_t(row) * EXPERTS;
  float sig[EPL];
  uint64_t key[EPL];
#pragma unroll
  for (int i = 0; i < EPL; ++i) sig[i] = ldcg_f32(lg + lane + 32 * i);
#pragma unroll
  for (int i = 0; i < EPL; ++i) {
    sig[i] = sigmoid(sig[i]);
    key[i] = order_key(add_ftz(sig[i], bias[i]), lane + 32 * i);
  }
  int mine_e = 0;
  float mine_s = 0.f;
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    uint64_t m = key[0];
#pragma unroll
    for (int i = 1; i < EPL; ++i) m = key[i] > m ? key[i] : m;
    const uint64_t best = warp_max(m);
    bool holds = false;
    float held = 0.f;
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
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
    const bool on = local >= 0 && local < p.local_count;
    lent[lane] = on ? mine_e : -1;
    lplace[lane] = on ? atomic_add(hist + mine_e, 1) : -1;
  }
}





__device__ __forceinline__ void final_step(const Params& p, int* tot, int* pre, int* first, int* wsum, int tid,
                                           int lane, int warp, int rank) {
  int n[EPT][MAX_CLUSTERS];
#pragma unroll
  for (int k = 0; k < EPT; ++k) {
    const int e = tid + k * RT;
#pragma unroll
    for (int c = 0; c < MAX_CLUSTERS; ++c)
      n[k][c] = e < EXPERTS && c < p.nclusters ? ldcg_i32(p.counts + c * EXPERTS + e) : 0;
  }
  const int expanded = p.rows * TOP_K, share = (expanded + CS - 1) / CS, span = CS * p.rpc * TOP_K;
  const int x0 = rank * share + tid, x1 = min(expanded, (rank + 1) * share);
  int ent[XPT], place[XPT];
#pragma unroll
  for (int i = 0; i < XPT; ++i) {
    const int x = x0 + i * RT;
    ent[i] = x < x1 ? ldcg_i32(p.ent + x) : -1;
    place[i] = x < x1 ? ldcg_i32(p.place + x) : 0;
  }
#pragma unroll
  for (int k = 0; k < EPT; ++k) {
    const int e = tid + k * RT;
    if (e < EXPERTS) {
      int run = 0;
#pragma unroll
      for (int c = 0; c < MAX_CLUSTERS; ++c) {
        if (c < p.nclusters) pre[c * EXPERTS + e] = run;
        run += n[k][c];
      }
      tot[e] = run;
    }
  }
  route_sync();
  int cnt[3], s = 0;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int e = 3 * tid + j;
    cnt[j] = e < EXPERTS ? (tot[e] + p.tile - 1) / p.tile : 0;
    s += cnt[j];
  }
  int incl = s;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int up = lane_up(incl, d);
    if (lane >= d) incl += up;
  }
  if (lane == 31) wsum[warp] = incl;
  route_sync();
  int base = incl - s;
#pragma unroll
  for (int w = 0; w < ROUTE_WARPS - 1; ++w)
    if (w < warp) base += wsum[w];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int e = 3 * tid + j;
    if (e < EXPERTS) first[e] = base;
    base += cnt[j];
  }
  if (tid == RT - 1) {
    first[EXPERTS] = base;
    if (rank == 0) {
      p.padded[0] = base * p.tile;
      p.ctas[0] = base;
    }
  }
  route_sync();
  for (int t = tid; t * CS + rank < EXPERTS; t += RT) {
    const int e = t * CS + rank, n = tot[e], c0 = first[e], ncta = first[e + 1] - c0;
    for (int q = 0; q < ncta; ++q) {
      const int end = (c0 + q + 1) * p.tile, limit = min(end, c0 * p.tile + n);
      p.cta_batch[c0 + q] = e - p.local_start;
      p.cta_limit[c0 + q] = limit;
      for (int r = limit; r < end; ++r) p.p2t[r] = -1;
    }
  }
#pragma unroll
  for (int i = 0; i < XPT; ++i) {
    const int x = x0 + i * RT, e = ent[i];
    if (x < x1) {
      int perm = -1;
      if (e >= 0) {
        perm = first[e] * p.tile + pre[(x / span) * EXPERTS + e] + place[i];
        p.p2t[perm] = x / TOP_K;
      }
      p.e2p[x] = perm;
    }
  }
  if (rank == 0 && tid == 0) p.counter[0] = 0;
}

__global__ void __launch_bounds__(THREADS, 1) wide_kernel(const Params p) {
#ifndef M1_HOST
  extern __shared__ __align__(16) uint8_t smem[];
#else
  uint8_t* const smem = m1h_smem();
#endif
  int* const hist = reinterpret_cast<int*>(smem + OFF_HIST);
  int* const lent = reinterpret_cast<int*>(smem + OFF_LENT);
  int* const lplace = reinterpret_cast<int*>(smem + OFF_LPLACE);
  int* const pre = reinterpret_cast<int*>(smem + OFF_PRE);
  int* const first = reinterpret_cast<int*>(smem + OFF_FIRST);
  int* const flags = reinterpret_cast<int*>(smem + OFF_FLAGS);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rank = blockIdx.x, cl = blockIdx.y;
  const int r0 = (cl * CS + rank) * p.rpc;
  const int nr = max(0, min(p.rpc, p.rows - r0));


  for (int e = tid; e < EXPERTS; e += THREADS) hist[e] = 0;
  if (tid == 0) flags[1] = 0;
  float bias_l[EPL];
#pragma unroll
  for (int i = 0; i < EPL; ++i) bias_l[i] = p.bias[lane + 32 * i];
  const float gs = p.gscale[0];
  __syncthreads();
  griddep_wait();


  if (warp < nr) route_row(p, r0 + warp, lane, bias_l, hist, lent + TOP_K * warp, lplace + TOP_K * warp);
  cluster_arrive();
  cluster_wait();

  if (warp >= ROUTE_WARPS) {
    cluster_arrive();
    for (int i0 = tid - RT; i0 < nr * QBLK; i0 += QB * RT) {
      uint4 v[QB][2];
#pragma unroll
      for (int b = 0; b < QB; ++b) {
        const int i = i0 + b * RT;
        if (i < nr * QBLK) {
          const uint16_t* src = p.x + size_t(r0 + (i >> 8)) * HIDDEN + size_t(i & (QBLK - 1)) * 16;
          v[b][0] = ld128(src);
          v[b][1] = ld128(src + 8);
        }
      }
#pragma unroll
      for (int b = 0; b < QB; ++b) {
        const int i = i0 + b * RT;
        if (i < nr * QBLK) {
          const int row = r0 + (i >> 8), blk = i & (QBLK - 1);
          uint8_t sf;
          uint64_t q;
          quantize_block(v[b][0], v[b][1], gs, sf, q);
          *reinterpret_cast<uint64_t*>(p.xq + size_t(row) * (HIDDEN / 2) + size_t(blk) * 8) = q;
          p.xsf[size_t(row) * QBLK + blk] = sf;
        }
      }
    }
    cluster_wait();
  } else {

    for (int j = tid; j < nr * TOP_K; j += RT) {
      const int e = lent[j];
      int place = -1;
      if (e >= 0) {
        place = lplace[j];
#pragma unroll
        for (int q = 0; q < CS - 1; ++q)
          if (q < rank) place += dsmem_ld_i32(hist + e, q);
      }
      p.ent[r0 * TOP_K + j] = e;
      p.place[r0 * TOP_K + j] = place;
    }
    for (int t = tid; t * CS + rank < EXPERTS; t += RT) {
      const int e = t * CS + rank;
      int n = 0;
#pragma unroll
      for (int q = 0; q < CS; ++q) n += dsmem_ld_i32(hist + e, q);
      p.counts[cl * EXPERTS + e] = n;
    }
    fence_gpu();
    route_sync();
    if (tid == 0) {
      fence_gpu();
      int last = 0;
      if (dsmem_atomic_add(flags + 1, 0, 1) == CS - 1) {
        fence_gpu();
        last = atomic_add(p.counter, 1) == p.nclusters - 1;
      }
      flags[0] = last;
    }
    cluster_arrive();
    cluster_wait();


    bool last = false;
#pragma unroll
    for (int q = 0; q < CS; ++q) last |= dsmem_ld_i32(flags, q) != 0;
    if (last) {
      fence_gpu();
      final_step(p, hist, pre, first, lent, tid, lane, warp, rank);
    }
  }


  fence_gpu();
  __syncthreads();
  griddep_launch();
  cluster_arrive();
  cluster_wait();
}

}

#ifndef MOE_WIDE_DEVICE_ONLY
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>







void route(torch::Tensor logits, torch::Tensor x, torch::Tensor bias, torch::Tensor gscale, double scale, int64_t tile,
           int64_t local_start, int64_t local_count, torch::Tensor xq, torch::Tensor xsf,
           std::vector<torch::Tensor> record, std::vector<torch::Tensor> scratch, bool pdl) {
  using namespace moe_wide;
  const int64_t rows = x.size(0);
  const auto here = [&](const torch::Tensor& t, at::ScalarType s) {
    return t.is_cuda() && t.get_device() == x.get_device() && t.is_contiguous() && t.scalar_type() == s;
  };
  TORCH_CHECK(x.dim() == 2 && x.size(1) == HIDDEN && rows >= 1 && rows <= MAX_ROWS && here(x, at::kBFloat16) &&
                  reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "moe_wide: x must be BF16 [1..768, 4096], contiguous, 16-byte aligned");
  TORCH_CHECK(logits.dim() == 2 && logits.size(0) == rows && logits.size(1) == EXPERTS && here(logits, at::kFloat),
              "moe_wide: the logits must be FP32 [rows, 288], contiguous");
  TORCH_CHECK(bias.numel() == EXPERTS && here(bias, at::kFloat) && gscale.numel() == 1 && here(gscale, at::kFloat),
              "moe_wide: an FP32 bias [288] and an FP32 global scale [1]");
  TORCH_CHECK(tile >= 1 && local_start >= 0 && local_count >= 1 && local_start + local_count <= EXPERTS,
              "moe_wide: a positive tile and a local expert range inside the 288");
  TORCH_CHECK(xq.numel() == rows * HIDDEN / 2 && here(xq, at::kByte) && xsf.numel() == rows * HIDDEN / 16 &&
                  here(xsf, at::kByte) && reinterpret_cast<uintptr_t>(xq.data_ptr()) % 8 == 0,
              "moe_wide: codes [rows, 2048] and scales [rows, 256] uint8");
  TORCH_CHECK(record.size() == 11, "moe_wide: the canonical record has eleven tensors");
  const at::ScalarType kinds[11] = {at::kShort, at::kBFloat16, at::kInt, at::kInt, at::kInt, at::kInt,
                                    at::kInt,   at::kInt,      at::kInt, at::kInt, at::kInt};
  for (int i = 0; i < 11; ++i) TORCH_CHECK(here(record[i], kinds[i]), "moe_wide: record tensor ", i);
  TORCH_CHECK(record[0].numel() == rows * TOP_K && record[1].numel() == rows * TOP_K && record[4].numel() == 1 &&
                  record[5].numel() == rows * TOP_K && record[6].numel() >= 2 &&
                  record[8].numel() == record[9].numel() && record[10].numel() == 1,
              "moe_wide: the record's sizes are not those of ", rows, " rows");
  TORCH_CHECK(scratch.size() == 4 && scratch[0].numel() >= MAX_CLUSTERS * EXPERTS && here(scratch[0], at::kInt) &&
                  scratch[1].numel() >= MAX_ROWS * TOP_K && here(scratch[1], at::kInt) &&
                  scratch[2].numel() >= MAX_ROWS * TOP_K && here(scratch[2], at::kInt) && scratch[3].numel() >= 1 &&
                  here(scratch[3], at::kInt),
              "moe_wide: the scratch (cluster counts, entries, places, counter)");
  const int rpc = int((rows + CS * MAX_CLUSTERS - 1) / (CS * MAX_CLUSTERS));
  const int nclusters = int((rows + CS * rpc - 1) / (CS * rpc));
  TORCH_CHECK(rpc <= WARPS && nclusters <= MAX_CLUSTERS, "moe_wide: ", rows, " rows do not deal");
  const c10::cuda::CUDAGuard guard(x.device());
  const Params p{logits.data_ptr<float>(), reinterpret_cast<const uint16_t*>(x.data_ptr()), bias.data_ptr<float>(),
                 gscale.data_ptr<float>(), float(scale), int(rows), rpc, nclusters, int(tile), int(local_start),
                 int(local_count), xq.data_ptr<uint8_t>(), xsf.data_ptr<uint8_t>(),
                 reinterpret_cast<int16_t*>(record[0].data_ptr()), reinterpret_cast<uint16_t*>(record[1].data_ptr()),
                 record[4].data_ptr<int32_t>(), record[5].data_ptr<int32_t>(), record[6].data_ptr<int32_t>(),
                 record[8].data_ptr<int32_t>(), record[9].data_ptr<int32_t>(), record[10].data_ptr<int32_t>(),
                 scratch[0].data_ptr<int32_t>(), scratch[1].data_ptr<int32_t>(), scratch[2].data_ptr<int32_t>(),
                 scratch[3].data_ptr<int32_t>()};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CS, nclusters, 1);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2] = {};
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CS;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 2 : 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, wide_kernel, p));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("route", &route,
        "FlashInfer's DeepSeekV3 routing (288 experts, top 8) of stock's logits into the canonical record and the "
        "NVFP4 quantize of x, in one launch (129-768 rows in the route)");
}
#endif
