





















#pragma once


























#ifndef MHC_AR_HOST
#include "mhc_fi_kernels.cuh"
#include "mhc_fi_pdl.cuh"
#endif

namespace flashinfer {
namespace mhc_fpp {
namespace fpp_ar_decode {

#ifdef MHC_AR_HOST
constexpr int kArCtas = MHC_AR_HOST_CTAS, kArWarps = MHC_AR_HOST_WARPS, kArLanes = MHC_AR_HOST_LANES;
#else
constexpr int kArCtas = CL, kArWarps = NCW, kArLanes = 32;
#endif
constexpr int kArRanks = 4;
constexpr int kArSliceVecs = kArLanes;
constexpr int kArRowVecs = kArCtas * kArSliceVecs;
constexpr int kArMaxRows = 80;
constexpr size_t kArSlotVecs = size_t(kArMaxRows) * kArRowVecs;
constexpr uint32_t kArMark = 0x00008000u;
#ifndef MHC_AR_HOST
static_assert(kArRowVecs * 8 == H && kArSliceVecs * 8 == W, "one vector a lane covers a CTA's slice of a row");
#endif

struct ArSync {
  const uint4* partial;
  uint4* slots[kArRanks];
  uint32_t* epoch;
  uint32_t* done;
  uint32_t rank;
  uint4* x_out;
};

#ifndef MHC_AR_HOST
__device__ __forceinline__ void ar_st_relaxed(uint4* p, const uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(p)
               : "memory");
}
__device__ __forceinline__ uint4 ar_ld_relaxed(const uint4* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void ar_st_plain(uint4* p, const uint4 v) {
  asm volatile("st.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(p) : "memory");
}
__device__ __forceinline__ float ar_add_rn(const float a, const float b) {
  float c;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}
__device__ __forceinline__ uint32_t ar_cvt_bf16x2(const float lo, const float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float ar_as_float(const uint32_t bits) { return __uint_as_float(bits); }
__device__ __forceinline__ uint32_t ar_count(uint32_t* p) { return atomicAdd(p, 1u); }
__device__ __forceinline__ uint64_t ar_global_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#endif
#ifndef MHC_AR_SPIN_GUARD
#define MHC_AR_SPIN_GUARD(waiting)
#endif

__device__ __forceinline__ uint32_t ar_mark(const uint32_t w) { return w ? w : kArMark; }
__device__ __forceinline__ bool ar_unarrived(const uint4 v) { return !v.x || !v.y || !v.z || !v.w; }
__device__ __forceinline__ uint32_t ar_word(const uint4& v, const int j) {
  return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
}


__device__ __forceinline__ uint4 ar_reduce(const uint4 (&got)[kArRanks]) {
  uint32_t sum[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float lo = ar_as_float(ar_word(got[0], j) << 16), hi = ar_as_float(ar_word(got[0], j) & 0xffff0000u);
#pragma unroll
    for (int s = 1; s < kArRanks; ++s) {
      lo = ar_add_rn(lo, ar_as_float(ar_word(got[s], j) << 16));
      hi = ar_add_rn(hi, ar_as_float(ar_word(got[s], j) & 0xffff0000u));
    }
    sum[j] = ar_cvt_bf16x2(lo, hi);
  }
  return make_uint4(sum[0], sum[1], sum[2], sum[3]);
}


__device__ __forceinline__ uint32_t ar_half(const ArSync& a) { return *a.epoch & 1u; }


__device__ __forceinline__ size_t ar_vec(const int row, const int r, const int lane) {
  return size_t(row) * kArRowVecs + size_t(r) * kArSliceVecs + size_t(lane);
}


__device__ __forceinline__ void ar_push(const ArSync& a, const uint32_t half, const int t0, const int nv, const int r,
                                        const int warp, const int lane) {
  const size_t mine = (size_t(half) * kArRanks + a.rank) * kArSlotVecs;
  for (int t = warp; t < nv; t += kArWarps) {
    const size_t v = ar_vec(t0 + t, r, lane);
    const uint4 s = a.partial[v];
    const uint4 x = make_uint4(ar_mark(s.x), ar_mark(s.y), ar_mark(s.z), ar_mark(s.w));
#pragma unroll
    for (int d = 0; d < kArRanks; ++d) ar_st_relaxed(a.slots[d] + mine + v, x);
  }
}



__device__ __forceinline__ void ar_poll(const ArSync& a, const uint32_t half, const int t0, const int nv, const int tm,
                                        const int r, const int warp, const int lane, uint4* x, const int x_stride) {
  uint4* const own = a.slots[a.rank] + size_t(half) * kArRanks * kArSlotVecs;
  for (int t = warp; t < tm; t += kArWarps) {
    if (t >= nv) {
      x[size_t(t) * x_stride + lane] = make_uint4(0u, 0u, 0u, 0u);
      continue;
    }
    const size_t v = ar_vec(t0 + t, r, lane);
    uint4 got[kArRanks];
    bool waiting;
    do {
      waiting = false;
#pragma unroll
      for (int s = 0; s < kArRanks; ++s) {
        got[s] = ar_ld_relaxed(own + size_t(s) * kArSlotVecs + v);
        waiting |= ar_unarrived(got[s]);
      }
      MHC_AR_SPIN_GUARD(waiting);
    } while (waiting);
    const uint4 sum = ar_reduce(got);
    x[size_t(t) * x_stride + lane] = sum;
    if (a.x_out != nullptr) a.x_out[v] = sum;
#pragma unroll
    for (int s = 0; s < kArRanks; ++s) ar_st_plain(own + size_t(s) * kArSlotVecs + v, make_uint4(0u, 0u, 0u, 0u));
  }
}



__device__ __forceinline__ void ar_end(const ArSync& a, const uint32_t half, const uint32_t ctas) {
  if (ar_count(a.done) == ctas - 1u) {
    *a.done = 0u;
    *a.epoch = half ^ 1u;
  }
}

#ifndef MHC_AR_HOST
struct ArParams {
  Params base;
  ArSync ar;
  uint64_t* stamps;
};




template <bool kNarrow, bool kFused, bool kStamp>
__global__ void __launch_bounds__(NTHREADS, 1)
    mhc_ar_post_pre_decode_kernel(const __grid_constant__ ArParams ap, const __grid_constant__ CUtensorMap tm_x,
                                     const __grid_constant__ CUtensorMap tm_res,
                                     const __grid_constant__ CUtensorMap tm_rc) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const Params& p = ap.base;
  const uint32_t sbase = smem_u32(smem);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t r = cluster_ctarank();
  const int cid = blockIdx.x / CL;
  const int ncl = gridDim.x / CL;
  const int TM = p.TM;
  const int M = p.M;



  constexpr int nlocal = 1;

#define BAR(i) (sbase + OFF_BAR + (i) * 8)

  if (threadIdx.x == 0) {

    mbar_init(BAR(B_FULL), 1);
    mbar_init(BAR(B_G1), 1);
    mbar_init(BAR(B_PM), 1);
    mbar_init(BAR(B_G2), 1);
    mbar_init(BAR(B_QF), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }




  __syncthreads();
  cluster_arrive();

  if (warp < NCW) {

    const int tid = threadIdx.x;
    const int g = lane >> 2, q = lane & 3;
    const int js = warp >> 1;
    const int hs = (warp & 1) * 128;


    uint32_t fnr[4][3][8];
    {
      const float* fb = p.fn + (size_t)js * H + (size_t)r * W + hs + 8 * q;
#pragma unroll
      for (int kg = 0; kg < 4; ++kg) {
#pragma unroll
        for (int nt = 0; nt < 3; ++nt) {
          const float* src = fb + (size_t)(g + 8 * nt) * (4 * H) + 32 * kg;

          ldg_v8(src, fnr[kg][nt]);
        }
      }
    }

    uint2 nwr[2];
#pragma unroll
    for (int ch = 0; ch < 2; ++ch)
      nwr[ch] = __ldg(reinterpret_cast<const uint2*>(p.nw + (size_t)r * W + ch * 128 + 4 * lane));

    pdl_wait();
    pdl_trigger();


    uint32_t ar_h = 0;
    if constexpr (kFused) {
      const int ar_t0 = cid * TM;
      const int ar_nv = min(TM, M - ar_t0);
      ar_h = ar_half(ap.ar);
      ar_push(ap.ar, ar_h, ar_t0, ar_nv, (int)r, warp, lane);
      ar_poll(ap.ar, ar_h, ar_t0, ar_nv, TM, (int)r, warp, lane, reinterpret_cast<uint4*>(smem + X_OFF), kArSliceVecs);
      __syncwarp();
    }
    uint32_t lpb[2][2][2];
    float* ssq_base = reinterpret_cast<float*>(smem + OFF_SSQ);
    float* red = reinterpret_cast<float*>(smem + OFF_RED);

    for (int it = 0; it < nlocal + 2; ++it) {

      if (it < nlocal) {
        const int k = it;
        const int s = k % S;
        const int t0 = (cid + k * ncl) * TM;
        const int nv = min(TM, M - t0);
        if (tid == 0) {
          mbar_arrive_expect_tx(BAR(B_PM + s), TM * 16);
          mbar_arrive_expect_tx(BAR(B_G2 + s), CL * TM * 4);
        }
        mbar_wait(BAR(B_FULL + s), (k / S) & 1);
        if constexpr (kStamp) {
          if (tid == 0) ap.stamps[blockIdx.x] = ar_global_ns();
        }
        uint8_t* st = smem + s * STAGEB;
        float* ssq = ssq_base + (k & 1) * TMAX;
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          const int t = warp + tt * NCW;
          if (t < TM) {
            const float4 pl = reinterpret_cast<const float4*>(st + PLM_OFF)[t];
            const float4* cb = reinterpret_cast<const float4*>(st + COMB_OFF) + 4 * t;
            const float4 c0 = cb[0], c1 = cb[1], c2 = cb[2], c3 = cb[3];

            const float cm[4][4] = {{c0.x, c0.y, c0.z, c0.w},
                                    {c1.x, c1.y, c1.z, c1.w},
                                    {c2.x, c2.y, c2.z, c2.w},
                                    {c3.x, c3.y, c3.z, c3.w}};
            const float plv[4] = {pl.x, pl.y, pl.z, pl.w};
            float2 ss2 = make_float2(0.0f, 0.0f);
#pragma unroll
            for (int ch = 0; ch < 2; ++ch) {
              const int hb = 2 * (ch * 128 + 4 * lane);
              uint8_t* base = st + R_OFF + t * 4 * ROWB + hb;
              const uint2 xv = *reinterpret_cast<const uint2*>(st + X_OFF + t * ROWB + hb);
              uint2 rv[4];
#pragma unroll
              for (int i = 0; i < 4; ++i) rv[i] = *reinterpret_cast<const uint2*>(base + i * ROWB);
              const float2 xf[2] = {bf2_to_f2(xv.x), bf2_to_f2(xv.y)};
              float2 rf[4][2];
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                rf[i][0] = bf2_to_f2(rv[i].x);
                rf[i][1] = bf2_to_f2(rv[i].y);
              }
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                uint32_t ow[2];
#pragma unroll
                for (int h2 = 0; h2 < 2; ++h2) {
                  float2 o = post_map_first(xf[h2], plv[j], rf[0][h2], cm[0][j]);
#pragma unroll
                  for (int i = 1; i < 4; ++i) o = __ffma2_rn(rf[i][h2], f2s(cm[i][j]), o);
                  ow[h2] = pack_bf2(o.x, o.y);
                  const float2 b = bf2_to_f2(ow[h2]);
                  ss2 = __ffma2_rn(b, b, ss2);
                }
                *reinterpret_cast<uint2*>(base + j * ROWB) = make_uint2(ow[0], ow[1]);
              }
            }
            float ss = ss2.x + ss2.y;
            ss = warp_sum(ss);
            if (lane == 0) ssq[t] = ss;
          }
        }
        fence_proxy_async();
        compute_bar();
        if (tid == 0) {
          tma_store_3d(&tm_rc, (int)r * W, 0, t0, sbase + s * STAGEB + R_OFF);
          bulk_commit();

          if constexpr (kFused) ar_end(ap.ar, ar_h, gridDim.x);
        }
        if (k == 0) {

#pragma unroll
          for (int kg = 0; kg < 4; ++kg)
#pragma unroll
            for (int nt = 0; nt < 3; ++nt)
#pragma unroll
              for (int e = 0; e < 8; ++e) fnr[kg][nt][e] = f2tf32(__uint_as_float(fnr[kg][nt][e]));
        }

        float* rw = red + warp * TMAX * NMIX;
        if constexpr (kNarrow) {



          float acc[2][2][4];
#pragma unroll
          for (int c2 = 0; c2 < 2; ++c2)
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
              for (int e = 0; e < 4; ++e) acc[c2][mt][e] = 0.0f;
#pragma unroll
          for (int kg = 0; kg < 4; ++kg) {
            const int aoff = R_OFF + js * ROWB + 2 * (hs + 32 * kg + 8 * q);
            const uint4 B0 = *reinterpret_cast<const uint4*>(st + g * 4 * ROWB + aoff);
            const uint32_t w0[4] = {B0.x, B0.y, B0.z, B0.w};
#pragma unroll
            for (int ss_ = 0; ss_ < 4; ++ss_) {
              const uint32_t b0 = w0[ss_] << 16, b1 = w0[ss_] & 0xffff0000u;
              mma_tf32(acc[kg & 1][0], fnr[kg][0][2 * ss_], fnr[kg][1][2 * ss_],
                       fnr[kg][0][2 * ss_ + 1], fnr[kg][1][2 * ss_ + 1], b0, b1);
              mma_tf32(acc[kg & 1][1], fnr[kg][2][2 * ss_], 0u, fnr[kg][2][2 * ss_ + 1], 0u, b0,
                       b1);
            }
          }

#pragma unroll
          for (int h = 0; h < 2; ++h) {
            float* rt = rw + (2 * q + h) * NMIX;
            rt[g] = acc[0][0][h] + acc[1][0][h];
            rt[g + 8] = acc[0][0][2 + h] + acc[1][0][2 + h];
            rt[g + 16] = acc[0][1][h] + acc[1][1][h];
          }
        } else {
          float acc[3][4];
#pragma unroll
          for (int nt = 0; nt < 3; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[nt][e] = 0.0f;
#pragma unroll
          for (int kg = 0; kg < 4; ++kg) {
            const int aoff = R_OFF + js * ROWB + 2 * (hs + 32 * kg + 8 * q);
            const uint4 A0 = *reinterpret_cast<const uint4*>(st + g * 4 * ROWB + aoff);
            const uint4 A1 = *reinterpret_cast<const uint4*>(st + (g + 8) * 4 * ROWB + aoff);
            const uint32_t w0[4] = {A0.x, A0.y, A0.z, A0.w};
            const uint32_t w1[4] = {A1.x, A1.y, A1.z, A1.w};
#pragma unroll
            for (int ss_ = 0; ss_ < 4; ++ss_) {
              const uint32_t a0 = w0[ss_] << 16, a2 = w0[ss_] & 0xffff0000u;
              const uint32_t a1 = w1[ss_] << 16, a3 = w1[ss_] & 0xffff0000u;
#pragma unroll
              for (int nt = 0; nt < 3; ++nt)
                mma_tf32(acc[nt], a0, a1, a2, a3, fnr[kg][nt][2 * ss_], fnr[kg][nt][2 * ss_ + 1]);
            }
          }
#pragma unroll
          for (int nt = 0; nt < 3; ++nt) {
            *reinterpret_cast<float2*>(rw + g * NMIX + 8 * nt + 2 * q) =
                make_float2(acc[nt][0], acc[nt][1]);
            *reinterpret_cast<float2*>(rw + (g + 8) * NMIX + 8 * nt + 2 * q) =
                make_float2(acc[nt][2], acc[nt][3]);
          }
        }
        compute_bar();
        cluster_wait();
        const uint32_t g1rec = sbase + OFF_G1 + ((s * CL + r) * G1_REC) * 4;
        if (tid < TM * 6) {
          const int t = tid / 6, n4 = tid % 6;
          float4 sum = *reinterpret_cast<const float4*>(red + t * NMIX + 4 * n4);
#pragma unroll
          for (int w = 1; w < NCW; ++w) {
            const float4 v =
                *reinterpret_cast<const float4*>(red + w * TMAX * NMIX + t * NMIX + 4 * n4);
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
          }
          st_async_v4(mapa(g1rec + 16 * n4, t), sum.x, sum.y, sum.z, sum.w, mapa(BAR(B_G1 + s), t));
        } else if (tid >= 128 && tid < 128 + TM) {
          const int t = tid - 128;
          st_async_b32(mapa(g1rec + 4 * 24, t), ssq[t], mapa(BAR(B_G1 + s), t));
        }
      }

      if (it >= 2) {
        const int k = it - 2;
        const int s = k % S;
        const int t0 = (cid + k * ncl) * TM;
        const int nv = min(TM, M - t0);
        mbar_wait(BAR(B_G2 + s), (k / S) & 1);
        const float* g2 = reinterpret_cast<const float*>(smem + OFF_G2) + s * TMAX * CL;
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          const int t = warp + tt * NCW;
          if (t < nv) {
            const float4* gp = reinterpret_cast<const float4*>(g2 + t * CL);
            float tot = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const float4 v = gp[i];
              tot += v.x;
              tot += v.y;
              tot += v.z;
              tot += v.w;
            }
            const float rs = rsqrtf(tot * (1.0f / H) + NORM_EPS);
#pragma unroll
            for (int ch = 0; ch < 2; ++ch) {
              uint2 ov;
              {
                const float2 l0 = bf2_to_f2(lpb[tt][ch][0]), l1 = bf2_to_f2(lpb[tt][ch][1]);
                const float2 w0 = bf2_to_f2(nwr[ch].x), w1 = bf2_to_f2(nwr[ch].y);
                const float2 o0 = __fmul2_rn(__fmul2_rn(l0, f2s(rs)), w0);
                const float2 o1 = __fmul2_rn(__fmul2_rn(l1, f2s(rs)), w1);
                ov.x = pack_bf2(o0.x, o0.y);
                ov.y = pack_bf2(o1.x, o1.y);
              }
              *reinterpret_cast<uint2*>(p.li_out + (size_t)(t0 + t) * H + (size_t)r * W + ch * 128 +
                                        4 * lane) = ov;
            }
          }
        }
      }

      if (it >= 1 && it - 1 < nlocal) {
        const int k = it - 1;
        const int s = k % S;
        mbar_wait(BAR(B_PM + s), (k / S) & 1);
        const float4* pm = reinterpret_cast<const float4*>(smem + OFF_PM) + s * TMAX;
        const uint8_t* st = smem + s * STAGEB;
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
          const int t = warp + tt * NCW;
          if (t < TM) {
            const float4 pr = pm[t];
            float2 ss2 = make_float2(0.0f, 0.0f);
#pragma unroll
            for (int ch = 0; ch < 2; ++ch) {
              const uint8_t* base = st + R_OFF + t * 4 * ROWB + 2 * (ch * 128 + 4 * lane);
              uint2 rv[4];
#pragma unroll
              for (int i = 0; i < 4; ++i) rv[i] = *reinterpret_cast<const uint2*>(base + i * ROWB);
              float lv[4];
#pragma unroll
              for (int h2 = 0; h2 < 2; ++h2) {
                const uint32_t sel[4] = {h2 ? rv[0].y : rv[0].x, h2 ? rv[1].y : rv[1].x,
                                         h2 ? rv[2].y : rv[2].x, h2 ? rv[3].y : rv[3].x};
                float2 v = __fmul2_rn(bf2_to_f2(sel[0]), f2s(pr.x));
                v = __ffma2_rn(bf2_to_f2(sel[1]), f2s(pr.y), v);
                v = __ffma2_rn(bf2_to_f2(sel[2]), f2s(pr.z), v);
                v = __ffma2_rn(bf2_to_f2(sel[3]), f2s(pr.w), v);
                ss2 = __ffma2_rn(v, v, ss2);
                lv[2 * h2] = v.x;
                lv[2 * h2 + 1] = v.y;
              }
              lpb[tt][ch][0] = pack_bf2(lv[0], lv[1]);
              lpb[tt][ch][1] = pack_bf2(lv[2], lv[3]);
            }
            float ss = warp_sum(ss2.x + ss2.y);
            if (lane < CL) {
              const uint32_t dst = sbase + OFF_G2 + ((s * TMAX + t) * CL + r) * 4;
              st_async_b32(mapa(dst, lane), ss, mapa(BAR(B_G2 + s), lane));
            }
          }
        }


      }
    }
    if (tid == 0) bulk_wait<0>();
  } else if (warp == WARP_PROD) {

    pdl_wait();
    pdl_trigger();
    const uint64_t pol = policy_evict_first();
    for (int k = 0; k < nlocal; ++k) {
      const int s = k % S;
      const int t0 = (cid + k * ncl) * TM;
      const int nv = min(TM, M - t0);
      if (k >= S) mbar_wait(BAR(B_EMPTY + s), ((k / S) - 1) & 1);
      if (lane == 0) {
        const uint32_t stg = sbase + s * STAGEB;

        mbar_arrive_expect_tx(BAR(B_FULL + s), TM * (kFused ? 4 : 5) * ROWB + nv * (16 + 64));
        if constexpr (!kFused) tma_load_2d(stg + X_OFF, &tm_x, (int)r * W, t0, BAR(B_FULL + s), pol);
        tma_load_3d(stg + R_OFF, &tm_res, (int)r * W, 0, t0, BAR(B_FULL + s), pol);
        bulk_g2s(stg + PLM_OFF, p.plm + (size_t)t0 * 4, nv * 16, BAR(B_FULL + s), pol);
        bulk_g2s(stg + COMB_OFF, p.comb + (size_t)t0 * 16, nv * 64, BAR(B_FULL + s), pol);
      }
      __syncwarp();
    }
    cluster_wait();
  } else if (warp == WARP_F0) {

    cluster_wait();
    pdl_wait();
    pdl_trigger();
    if ((int)r < TM) {
      const int scale_idx = (lane < 4) ? 0 : ((lane < 8) ? 1 : 2);
      const float scale = __ldg(p.hc_scale + scale_idx);
      const float base = (lane < NMIX) ? __ldg(p.hc_base + lane) : 0.0f;
      const float* g1 = reinterpret_cast<const float*>(smem + OFF_G1);
      for (int k = 0; k < nlocal; ++k) {
        const int s = k % S;
        const int t0 = (cid + k * ncl) * TM;
        if (lane == 0) mbar_arrive_expect_tx(BAR(B_G1 + s), CL * 100);
        mbar_wait(BAR(B_G1 + s), (k / S) & 1);
        float v = 0.0f;
        if (lane < 25) {
          const float* gs = g1 + s * CL * G1_REC + lane;
          float a[CL];
#pragma unroll
          for (int src = 0; src < CL; ++src) a[src] = gs[src * G1_REC];
          v = (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))) +
              (((a[8] + a[9]) + (a[10] + a[11])) + ((a[12] + a[13]) + (a[14] + a[15])));
        }
        const float sq = __shfl_sync(0xffffffffu, v, 24);
        const float rsq = rsqrtf(sq * (1.0f / (4 * H)) + RMS_EPS);
        const float m = __fmul_rn(v, rsq);
        const float logit = __fadd_rn(__fmul_rn(m, scale), base);
        const int tok = t0 + (int)r;
        float sig = 0.0f;
        if (lane < 8) sig = sigmoid_exact(logit);
        const float pre = (lane < 4) ? __fadd_rn(sig, HC_PRE_EPS) : 0.0f;
        const float p0 = __shfl_sync(0xffffffffu, pre, 0);
        const float p1 = __shfl_sync(0xffffffffu, pre, 1);
        const float p2 = __shfl_sync(0xffffffffu, pre, 2);
        const float p3 = __shfl_sync(0xffffffffu, pre, 3);
        if (lane < CL) {
          const uint32_t dst = sbase + OFF_PM + ((s * TMAX + r) * 4) * 4;
          st_async_v4(mapa(dst, lane), p0, p1, p2, p3, mapa(BAR(B_PM + s), lane));
        }
        if (tok < M) {
          if (lane >= 4 && lane < 8) p.post_out[(size_t)tok * 4 + (lane - 4)] = sig * HC_POST_MULT;
        }
      }
    }
  } else {





    const float s2 = __ldg(p.hc_scale + 2);
    const float base = (lane < 16) ? __ldg(p.hc_base + 8 + lane) : 0.0f;
    cluster_wait();
    pdl_wait();
    pdl_trigger();
    const int tok = cid * TM + (int)r;
    if ((int)r >= TM || tok >= M) return;
    mbar_wait(BAR(B_G1), 0);
    float v = 0.0f;
    if (lane <= 16) {
      const float* gs =
          reinterpret_cast<const float*>(smem + OFF_G1) + ((lane < 16) ? 8 + lane : 24);
      float a[CL];
#pragma unroll
      for (int src = 0; src < CL; ++src) a[src] = gs[src * G1_REC];
      v = (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))) +
          (((a[8] + a[9]) + (a[10] + a[11])) + ((a[12] + a[13]) + (a[14] + a[15])));
    }
    const float sq = __shfl_sync(0xffffffffu, v, 16);
    if (lane < 16) {
      constexpr unsigned MASK16 = 0x0000ffffu;
      const float rsq = rsqrtf(sq * (1.0f / (4 * H)) + RMS_EPS);
      const float in = __fadd_rn(__fmul_rn(__fmul_rn(v, rsq), s2), base);
      bool bad = false;
      float c = in;
      if (!p.split_sinkhorn) {
        c = sinkhorn_element<false>(in, bad);
        if (__any_sync(MASK16, bad)) {
          bad = false;
          c = sinkhorn_element<true>(in, bad);
        }
      }
      p.comb_out[(size_t)tok * 16 + lane] = c;
    }
  }
#undef BAR
}


template <bool kNarrow, bool kFused, bool kStamp>
cudaError_t prepare_ar_decode() {
  return configure_cluster_kernel(reinterpret_cast<const void*>(mhc_ar_post_pre_decode_kernel<kNarrow, kFused, kStamp>),
                                  NTHREADS, SMEM_BYTES, nullptr);
}



template <bool kFused, bool kStamp>
cudaError_t launch_ar_decode(const ArParams& ap, const int num_clusters, const TensorMaps& maps, cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CL * num_clusters, 1, 1);
  cfg.blockDim = dim3(NTHREADS, 1, 1);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CL;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 2;
  if (ap.base.TM <= 8)
    return cudaLaunchKernelEx(&cfg, mhc_ar_post_pre_decode_kernel<true, kFused, kStamp>, ap, maps.x, maps.residual,
                              maps.residual_cur);
  return cudaLaunchKernelEx(&cfg, mhc_ar_post_pre_decode_kernel<false, kFused, kStamp>, ap, maps.x, maps.residual,
                            maps.residual_cur);
}
#endif

}
}
}
