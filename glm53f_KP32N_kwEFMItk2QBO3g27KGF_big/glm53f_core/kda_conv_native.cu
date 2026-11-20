#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define KC_HD __host__ __device__ __forceinline__
#else
#define KC_HD inline
#endif

#ifndef KDA_CONV_HOST_TEST
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace kdaconv {

constexpr int kThreads = 128;
constexpr int kVec = 4;
constexpr int kBlockC = kThreads * kVec;
constexpr float kLog2e = 1.44269502162933349609375f;

enum : int { kNegFold = 1, kEx2Ftz = 2, kDivShift = 2, kDivMask = 12, kTap0Mul = 16, kExpLibdevice = 32 };
enum : int { kDivFull = 0, kDivFullPos = 1, kDivRn = 2, kDivApprox = 3 };

struct Params {
  const uint16_t* x;
  const float* w;
  uint16_t* state;
  const int32_t* qsl;
  const void* cidx;
  const uint8_t* init;
  uint16_t* out;
  int64_t sx, sw0, sw1, ss0, ss2, total_tokens, pad;
  int32_t dim, split, ncl, cidx64, use_pad;
  uint32_t rt_mask;
  int32_t rt_form;
};

struct F2 {
  float x, y;
};

KC_HD float bits_f(uint32_t u) {
#ifdef __CUDA_ARCH__
  return __uint_as_float(u);
#else
  float f;
  std::memcpy(&f, &u, 4);
  return f;
#endif
}

KC_HD float s_fma(float a, float b, float c) {
#ifdef __CUDA_ARCH__
  float d;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
#else
  return std::fma(a, b, c);
#endif
}

KC_HD float s_mul(float a, float b) {
#ifdef __CUDA_ARCH__
  float d;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
#else
  return a * b;
#endif
}

KC_HD float s_add(float a, float b) {
#ifdef __CUDA_ARCH__
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
#else
  return a + b;
#endif
}

KC_HD float s_sub(float a, float b) {
#ifdef __CUDA_ARCH__
  float d;
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
#else
  return a - b;
#endif
}

KC_HD float s_ex2(float t, bool ftz) {
#ifdef __CUDA_ARCH__
  float e;
  if (ftz) {
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(t));
  } else {
    asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  }
  return e;
#else
  (void)ftz;
  return std::exp2(t);
#endif
}

#if defined(__CUDACC__)
extern "C" __device__ float __nv_expf(float);
#endif

KC_HD float s_exp_libdevice(float s) {
#ifdef __CUDA_ARCH__
  return __nv_expf(s);
#else
  return std::exp(s);
#endif
}

KC_HD float s_div(float a, float b, int kind) {
#ifdef __CUDA_ARCH__
  float d;
  if (kind == kDivFull) {
    asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  } else if (kind == kDivFullPos) {

    asm("{\n\t.reg .pred big;\n\t.reg .f32 aa, dd, rr;\n\t"
        "setp.gt.f32 big, %2, 0f7E800000;\n\t"
        "mov.f32 aa, %1;\n\tmov.f32 dd, %2;\n\t"
        "@big mul.rn.f32 aa, aa, 0f3E800000;\n\t"
        "@big mul.rn.f32 dd, dd, 0f3E800000;\n\t"
        "rcp.approx.ftz.f32 rr, dd;\n\t"
        "mul.rn.f32 %0, rr, aa;\n\t}"
        : "=f"(d) : "f"(a), "f"(b));
  } else if (kind == kDivRn) {
    asm("div.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  } else {
    asm("div.approx.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  }
  return d;
#else
  (void)kind;
  return a / b;
#endif
}

KC_HD F2 p_fma(F2 a, F2 b, F2 c) {
#ifdef __CUDA_ARCH__
  F2 d;
  asm("{\n\t.reg .b64 pa, pb, pc, pd;\n\t"
      "mov.b64 pa, {%2, %3};\n\tmov.b64 pb, {%4, %5};\n\tmov.b64 pc, {%6, %7};\n\t"
      "fma.rn.f32x2 pd, pa, pb, pc;\n\tmov.b64 {%0, %1}, pd;\n\t}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
  return d;
#else
  return {s_fma(a.x, b.x, c.x), s_fma(a.y, b.y, c.y)};
#endif
}

KC_HD F2 p_mul(F2 a, F2 b) {
#ifdef __CUDA_ARCH__
  F2 d;
  asm("{\n\t.reg .b64 pa, pb, pd;\n\t"
      "mov.b64 pa, {%2, %3};\n\tmov.b64 pb, {%4, %5};\n\t"
      "mul.rn.f32x2 pd, pa, pb;\n\tmov.b64 {%0, %1}, pd;\n\t}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
#else
  return {s_mul(a.x, b.x), s_mul(a.y, b.y)};
#endif
}

KC_HD F2 p_add(F2 a, F2 b) {
#ifdef __CUDA_ARCH__
  F2 d;
  asm("{\n\t.reg .b64 pa, pb, pd;\n\t"
      "mov.b64 pa, {%2, %3};\n\tmov.b64 pb, {%4, %5};\n\t"
      "add.rn.f32x2 pd, pa, pb;\n\tmov.b64 {%0, %1}, pd;\n\t}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
#else
  return {s_add(a.x, b.x), s_add(a.y, b.y)};
#endif
}

KC_HD F2 p_sub(F2 a, F2 b) {
#ifdef __CUDA_ARCH__
  F2 d;
  asm("{\n\t.reg .b64 pa, pb, pd;\n\t"
      "mov.b64 pa, {%2, %3};\n\tmov.b64 pb, {%4, %5};\n\t"
      "sub.rn.f32x2 pd, pa, pb;\n\tmov.b64 {%0, %1}, pd;\n\t}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
#else
  return {s_sub(a.x, b.x), s_sub(a.y, b.y)};
#endif
}

KC_HD uint32_t pack_bf16x2(float hi, float lo) {
#ifdef __CUDA_ARCH__
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
#else
  auto rne = [](float f) -> uint32_t {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fffu;
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
  };
  return (rne(hi) << 16) | rne(lo);
#endif
}

struct Raw {
  uint32_t lo, hi;
};

KC_HD Raw ld_x(const uint16_t* p) {
#ifdef __CUDA_ARCH__
  Raw r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.b32 {%0, %1}, [%2];" : "=r"(r.lo), "=r"(r.hi) : "l"(p));
  return r;
#else
  Raw r;
  std::memcpy(&r, p, 8);
  return r;
#endif
}

KC_HD Raw ld_state(const uint16_t* p) {
#ifdef __CUDA_ARCH__
  Raw r;
  asm volatile("ld.global.v2.b32 {%0, %1}, [%2];" : "=r"(r.lo), "=r"(r.hi) : "l"(p) : "memory");
  return r;
#else
  Raw r;
  std::memcpy(&r, p, 8);
  return r;
#endif
}

KC_HD void st_8(uint16_t* p, Raw v) {
#ifdef __CUDA_ARCH__
  asm volatile("st.global.v2.b32 [%0], {%1, %2};" ::"l"(p), "r"(v.lo), "r"(v.hi) : "memory");
#else
  std::memcpy(p, &v, 8);
#endif
}

template <class T>
KC_HD T ld_ro(const T* p) {
#ifdef __CUDA_ARCH__
  return __ldg(p);
#else
  return *p;
#endif
}

struct Win {
  float v[4];
};

KC_HD Win unpack(Raw r) {
  Win f;
  f.v[0] = bits_f(r.lo << 16);
  f.v[1] = bits_f(r.lo & 0xffff0000u);
  f.v[2] = bits_f(r.hi << 16);
  f.v[3] = bits_f(r.hi & 0xffff0000u);
  return f;
}

KC_HD bool unfused(uint32_t mask, int e, int j) { return (mask >> (e * 3 + j - 1)) & 1u; }

KC_HD float tap_in(const Win& h1, const Win& h2, const Win& xc, int j, int e) {
  return j == 1 ? h1.v[e] : (j == 2 ? h2.v[e] : xc.v[e]);
}

template <uint32_t MASK, int FORM, bool PACKED, bool RT>
KC_HD Raw conv_token(const Win& h0, const Win& h1, const Win& h2, const Win& xc, const float (&w)[4][4],
                     uint32_t rt_mask, int rt_form) {
  const uint32_t mask = RT ? rt_mask : MASK;
  const int form = RT ? rt_form : FORM;
  constexpr bool kPacked = PACKED && !RT && !(FORM & (kTap0Mul | kExpLibdevice));
  float acc[4];
  if (kPacked) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int e0 = q, e1 = q + 2;
      F2 a = p_fma(F2{w[0][e0], w[0][e1]}, F2{h0.v[e0], h0.v[e1]}, F2{0.0f, 0.0f});
#pragma unroll
      for (int j = 1; j < 4; ++j) {
        const float x0 = tap_in(h1, h2, xc, j, e0), x1 = tap_in(h1, h2, xc, j, e1);
        if (!unfused(mask, e0, j) && !unfused(mask, e1, j)) {
          a = p_fma(F2{w[j][e0], w[j][e1]}, F2{x0, x1}, a);
        } else {
          a.x = unfused(mask, e0, j) ? s_add(a.x, s_mul(w[j][e0], x0)) : s_fma(w[j][e0], x0, a.x);
          a.y = unfused(mask, e1, j) ? s_add(a.y, s_mul(w[j][e1], x1)) : s_fma(w[j][e1], x1, a.y);
        }
      }
      acc[e0] = a.x;
      acc[e1] = a.y;
    }
  } else {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float a = (form & kTap0Mul) ? s_mul(w[0][e], h0.v[e]) : s_fma(w[0][e], h0.v[e], 0.0f);
#pragma unroll
      for (int j = 1; j < 4; ++j) {
        const float xv = tap_in(h1, h2, xc, j, e);
        a = unfused(mask, e, j) ? s_add(a, s_mul(w[j][e], xv)) : s_fma(w[j][e], xv, a);
      }
      acc[e] = a;
    }
  }
  const bool fold = form & kNegFold, ftz = form & kEx2Ftz;
  const int div = (form & kDivMask) >> kDivShift;
  float y[4];
  if (kPacked) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const F2 a{acc[q], acc[q + 2]};
      const F2 t = fold ? p_mul(a, F2{-kLog2e, -kLog2e}) : p_mul(p_sub(F2{0.0f, 0.0f}, a), F2{kLog2e, kLog2e});
      const F2 d = p_add(F2{s_ex2(t.x, ftz), s_ex2(t.y, ftz)}, F2{1.0f, 1.0f});
      y[q] = s_div(a.x, d.x, div);
      y[q + 2] = s_div(a.y, d.y, div);
    }
  } else {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float ex;
      if (form & kExpLibdevice) {
        ex = s_exp_libdevice(s_sub(0.0f, acc[e]));
      } else {
        ex = s_ex2(fold ? s_mul(acc[e], -kLog2e) : s_mul(s_sub(0.0f, acc[e]), kLog2e), ftz);
      }
      y[e] = s_div(acc[e], s_add(ex, 1.0f), div);
    }
  }
  return Raw{pack_bf16x2(y[1], y[0]), pack_bf16x2(y[3], y[2])};
}

struct Lane {
  Win h0, h1, h2;
  float w[4][4];
  uint32_t rt_mask;
  int rt_form;
};

template <uint32_t MASK, int FORM, bool PACKED, bool RT>
KC_HD void step(Lane& l, Raw cur, uint16_t* o) {
  const Win xc = unpack(cur);
  st_8(o, conv_token<MASK, FORM, PACKED, RT>(l.h0, l.h1, l.h2, xc, l.w, l.rt_mask, l.rt_form));
  l.h0 = l.h1;
  l.h1 = l.h2;
  l.h2 = xc;
}

KC_HD const uint16_t* at(const uint16_t* base, int32_t bytes) {
  return reinterpret_cast<const uint16_t*>(reinterpret_cast<const char*>(base) + bytes);
}

KC_HD uint16_t* at(uint16_t* base, int32_t bytes) {
  return reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(base) + bytes);
}

template <int TT, int D, uint32_t MASK, int FORM, bool PACKED, bool RT, bool FULL>
KC_HD void conv_chunk(const Params& p, int s, int c0, int a, int L, int t0, int64_t k) {
  static_assert(TT % D == 0 && D % 3 == 0 && TT >= 3, "TT a multiple of D, D of 3 (window rotation), TT >= 3");
  const int n = FULL ? TT : (L - t0 < TT ? L - t0 : TT);

  const int32_t sxb = static_cast<int32_t>(p.sx * 2), sob = p.split * 2;
  const uint16_t* xb = p.x + (int64_t)(a + t0) * p.sx + c0;
  Raw ring[D];
#pragma unroll
  for (int u = 0; u < D; ++u) {
    if (FULL || u < n) ring[u] = ld_x(at(xb, u * sxb));
  }
  Lane l;
  float(&w)[4][4] = l.w;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
#pragma unroll
    for (int j = 0; j < 4; ++j) w[j][e] = ld_ro(p.w + (int64_t)(c0 + e) * p.sw0 + (int64_t)j * p.sw1);
  }
  Raw r0{0u, 0u}, r1{0u, 0u}, r2{0u, 0u};
  if (t0 == 0) {
    const bool init = p.init != nullptr && ld_ro(p.init + s) != 0;
    uint16_t* sb = p.state + k * p.ss0 + c0;
    if (init) {
      r0 = ld_state(sb);
      r1 = ld_state(sb + p.ss2);
      r2 = ld_state(sb + 2 * p.ss2);
    }
    Raw nw[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int src = i + L;
      if (src >= 3) {
        nw[i] = ld_x(p.x + (int64_t)(a + src - 3) * p.sx + c0);
      } else {
        const Raw old = src == 0 ? r0 : (src == 1 ? r1 : r2);
        nw[i] = (init && k < p.ncl) ? old : Raw{0u, 0u};
      }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) st_8(sb + i * p.ss2, nw[i]);
  } else {
    r0 = ld_x(xb - 3 * p.sx);
    r1 = ld_x(xb - 2 * p.sx);
    r2 = ld_x(xb - p.sx);
  }
  l.h0 = unpack(r0);
  l.h1 = unpack(r1);
  l.h2 = unpack(r2);
  l.rt_mask = p.rt_mask;
  l.rt_form = p.rt_form;
  uint16_t* ob = p.out + (int64_t)(c0 / p.split) * p.total_tokens * p.split + (int64_t)(a + t0) * p.split +
                 (c0 % p.split);
  const uint16_t* gl = at(xb, D * sxb);
  if (FULL) {
#pragma unroll 1
    for (int g = 0; g < TT - D; g += D) {
#pragma unroll
      for (int u = 0; u < D; ++u) {
        const Raw cur = ring[u];
        ring[u] = ld_x(at(gl, u * sxb));
        step<MASK, FORM, PACKED, RT>(l, cur, at(ob, u * sob));
      }
      gl = at(gl, D * sxb);
      ob = at(ob, D * sob);
    }
#pragma unroll
    for (int u = 0; u < D; ++u) step<MASK, FORM, PACKED, RT>(l, ring[u], at(ob, u * sob));
  } else {
#pragma unroll 1
    for (int g = 0; g < n; g += D) {
#pragma unroll
      for (int u = 0; u < D; ++u) {
        if (g + u >= n) break;
        const Raw cur = ring[u];
        if (g + u + D < n) ring[u] = ld_x(at(gl, u * sxb));
        step<MASK, FORM, PACKED, RT>(l, cur, at(ob, u * sob));
      }
      gl = at(gl, D * sxb);
      ob = at(ob, D * sob);
    }
  }
}

template <int TT, int D, uint32_t MASK, int FORM, bool PACKED, bool RT>
KC_HD void conv_thread(const Params& p, int bx, int by, int bz, int tx) {
  const int c0 = (bx * kThreads + tx) * kVec;
  if (c0 >= p.dim) return;
  const int s = bz;
  const int a = ld_ro(p.qsl + s);
  const int L = ld_ro(p.qsl + s + 1) - a;
  const int t0 = by * TT;
  if (t0 >= L) return;
  int64_t k = s;
  if (p.cidx != nullptr) {
    k = p.cidx64 ? ld_ro(static_cast<const int64_t*>(p.cidx) + s)
                 : (int64_t)ld_ro(static_cast<const int32_t*>(p.cidx) + s);
  }
  if (p.use_pad && k == p.pad) return;
  if (L - t0 >= TT) {
    conv_chunk<TT, D, MASK, FORM, PACKED, RT, true>(p, s, c0, a, L, t0, k);
  } else {
    conv_chunk<TT, D, MASK, FORM, PACKED, RT, false>(p, s, c0, a, L, t0, k);
  }
}

#if defined(__CUDACC__)
template <int TT, int D, uint32_t MASK, int FORM, bool PACKED, bool RT>
__global__ void __launch_bounds__(kThreads) conv_kernel(const Params p) {
  conv_thread<TT, D, MASK, FORM, PACKED, RT>(p, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);
}
#endif

constexpr uint32_t kMaskFused = 0x000u, kMaskT371 = (3u << 4) | (3u << 10), kMaskUnfused = 0xfffu;
constexpr int kF0 = 0, kF1 = kNegFold | kEx2Ftz | (kDivFullPos << kDivShift);

struct Variant {
  const char* name;
  int tt, d;
  uint32_t mask;
  int form, packed, runtime;
};

#define KDA_CONV_VARIANTS(X)                                       \
  X(0, "fused_f1_packed_tt48_d6", 48, 6, kMaskFused, kF1, true, false)      \
  X(1, "fused_f1_packed_tt96_d6", 96, 6, kMaskFused, kF1, true, false)      \
  X(2, "fused_f1_packed_tt96_d12", 96, 12, kMaskFused, kF1, true, false)    \
  X(3, "fused_f1_scalar_tt96_d6", 96, 6, kMaskFused, kF1, false, false)     \
  X(4, "fused_f0_packed_tt96_d6", 96, 6, kMaskFused, kF0, true, false)      \
  X(5, "t371_f1_packed_tt96_d6", 96, 6, kMaskT371, kF1, true, false)        \
  X(6, "t371_f0_scalar_tt96_d6", 96, 6, kMaskT371, kF0, false, false)       \
  X(7, "unfused_f1_packed_tt96_d6", 96, 6, kMaskUnfused, kF1, true, false)  \
  X(8, "fused_f1_packed_tt192_d6", 192, 6, kMaskFused, kF1, true, false)    \
  X(9, "probe_runtime_tt96_d6", 96, 6, 0u, 0, false, true)

#ifdef KDA_CONV_EXTRA_MASK
#ifndef KDA_CONV_EXTRA_FORM
#define KDA_CONV_EXTRA_FORM kF1
#endif
#define KDA_CONV_EXTRA(X) X(10, "extra_packed_tt96_d6", 96, 6, (uint32_t)(KDA_CONV_EXTRA_MASK), KDA_CONV_EXTRA_FORM, true, false)
#else
#define KDA_CONV_EXTRA(X)
#endif

#define KDA_CONV_ROW(id, nm, tt, d, mask, form, packed, rt) Variant{nm, tt, d, mask, form, packed, rt},
inline const Variant kVariants[] = {KDA_CONV_VARIANTS(KDA_CONV_ROW) KDA_CONV_EXTRA(KDA_CONV_ROW)};
#undef KDA_CONV_ROW
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

#ifndef KDA_CONV_HOST_TEST

void conv(int64_t variant, const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& conv_states,
          const torch::Tensor& query_start_loc, const c10::optional<torch::Tensor>& cache_indices,
          const c10::optional<torch::Tensor>& has_initial_state, const torch::Tensor& out, int64_t batch,
          int64_t max_seqlen, c10::optional<int64_t> pad_slot_id, int64_t rt_mask, int64_t rt_form) {
  TORCH_CHECK(variant >= 0 && variant < kNumVariants, "kda_conv: unknown variant ", variant);
  const Variant& v = kVariants[variant];
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2, "kda_conv: x must be a 2-D bf16 CUDA tensor");
  const int64_t dim = x.size(0), total = x.size(1);
  TORCH_CHECK(x.stride(0) == 1 && x.stride(1) % kVec == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0,
              "kda_conv: x needs channel stride 1, token stride a multiple of 4 and an 8-byte aligned base (got strides ",
              x.strides(), ")");
  TORCH_CHECK(dim % kVec == 0 && dim > 0, "kda_conv: dim must be a positive multiple of 4, got ", dim);
  TORCH_CHECK(x.stride(1) * 2 * (v.tt + 2 * v.d) < (int64_t(1) << 31), "kda_conv: token stride too large for the 32-bit "
              "in-chunk byte offsets: ", x.stride(1));
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kFloat32 && weight.dim() == 2 && weight.size(0) == dim &&
                  weight.size(1) == 4,
              "kda_conv: weight must be (dim, 4) fp32 (kernel width 4 only), got ", weight.sizes(), " ", weight.scalar_type());
  TORCH_CHECK(conv_states.is_cuda() && conv_states.scalar_type() == torch::kBFloat16 && conv_states.dim() == 3 &&
                  conv_states.size(1) == dim && conv_states.size(2) >= 3,
              "kda_conv: conv_states must be (lines, dim, >= 3) bf16, got ", conv_states.sizes(), " ", conv_states.scalar_type());
  TORCH_CHECK(conv_states.stride(1) == 1 && conv_states.stride(0) % kVec == 0 && conv_states.stride(2) % kVec == 0 &&
                  reinterpret_cast<uintptr_t>(conv_states.data_ptr()) % 8 == 0,
              "kda_conv: conv_states needs channel stride 1 and other strides multiples of 4 (got ", conv_states.strides(), ")");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16 && out.dim() == 3 && out.is_contiguous() &&
                  out.size(1) == total && out.size(0) * out.size(2) == dim && out.size(2) % kVec == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
              "kda_conv: out must be contiguous, 8-byte aligned bf16 (dim / split, T, split) with split a multiple of 4, "
              "got ", out.sizes());
  TORCH_CHECK(query_start_loc.is_cuda() && query_start_loc.scalar_type() == torch::kInt32 && query_start_loc.is_contiguous() &&
                  query_start_loc.numel() >= batch + 1,
              "kda_conv: query_start_loc must be contiguous int32 with batch + 1 entries");
  TORCH_CHECK(batch >= 0 && batch <= 65535, "kda_conv: batch out of range: ", batch);
  const int64_t chunks = (max_seqlen + v.tt - 1) / v.tt;
  TORCH_CHECK(max_seqlen >= 0 && chunks <= 65535, "kda_conv: max_seqlen out of range: ", max_seqlen);
  Params p{};
  p.cidx = nullptr;
  p.cidx64 = 0;
  if (cache_indices.has_value()) {
    const auto& ci = *cache_indices;
    TORCH_CHECK(ci.is_cuda() && ci.is_contiguous() && ci.numel() >= batch &&
                    (ci.scalar_type() == torch::kInt32 || ci.scalar_type() == torch::kInt64),
                "kda_conv: cache_indices must be contiguous int32 / int64 with batch entries");
    p.cidx = ci.data_ptr();
    p.cidx64 = ci.scalar_type() == torch::kInt64;
  }
  p.init = nullptr;
  if (has_initial_state.has_value()) {
    const auto& hi = *has_initial_state;
    TORCH_CHECK(hi.is_cuda() && hi.is_contiguous() && hi.numel() >= batch && hi.scalar_type() == torch::kBool,
                "kda_conv: has_initial_state must be contiguous bool with batch entries");
    p.init = reinterpret_cast<const uint8_t*>(hi.data_ptr());
  }
  for (const torch::Tensor* t : {&weight, &conv_states, &query_start_loc, &out}) {
    TORCH_CHECK(t->device() == x.device(), "kda_conv: all tensors must be on x's device");
  }
  if (batch == 0 || max_seqlen == 0) return;
  p.x = reinterpret_cast<const uint16_t*>(x.data_ptr());
  p.w = weight.data_ptr<float>();
  p.state = reinterpret_cast<uint16_t*>(conv_states.data_ptr());
  p.qsl = query_start_loc.data_ptr<int32_t>();
  p.out = reinterpret_cast<uint16_t*>(out.data_ptr());
  p.sx = x.stride(1);
  p.sw0 = weight.stride(0);
  p.sw1 = weight.stride(1);
  p.ss0 = conv_states.stride(0);
  p.ss2 = conv_states.stride(2);
  p.total_tokens = total;
  p.dim = static_cast<int32_t>(dim);
  p.split = static_cast<int32_t>(out.size(2));
  p.ncl = static_cast<int32_t>(conv_states.size(0));
  p.use_pad = pad_slot_id.has_value();
  p.pad = pad_slot_id.has_value() ? *pad_slot_id : 0;
  p.rt_mask = static_cast<uint32_t>(rt_mask);
  p.rt_form = static_cast<int32_t>(rt_form);
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(static_cast<unsigned>((dim + kBlockC - 1) / kBlockC), static_cast<unsigned>(chunks),
                  static_cast<unsigned>(batch));
  switch (variant) {
#define KDA_CONV_CASE(id, nm, tt, d, mask, form, packed, rt)          \
  case id:                                                         \
    conv_kernel<tt, d, mask, form, packed, rt><<<grid, kThreads, 0, stream>>>(p); \
    break;
    KDA_CONV_VARIANTS(KDA_CONV_CASE)
    KDA_CONV_EXTRA(KDA_CONV_CASE)
#undef KDA_CONV_CASE
    default:
      TORCH_CHECK(false, "kda_conv: unknown variant ", variant);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<std::tuple<int64_t, std::string, int64_t, int64_t, int64_t, int64_t, bool, bool>> variants() {
  std::vector<std::tuple<int64_t, std::string, int64_t, int64_t, int64_t, int64_t, bool, bool>> rows;
  for (int i = 0; i < kNumVariants; ++i) {
    const Variant& v = kVariants[i];
    rows.emplace_back(i, v.name, v.tt, v.d, v.mask, v.form, v.packed != 0, v.runtime != 0);
  }
  return rows;
}
#endif

}

#ifndef KDA_CONV_HOST_TEST
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv", &kdaconv::conv, "KDA short conv (bit-exact to SGLang 314bce7c's Triton kernel)", pybind11::arg("variant"),
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("conv_states"), pybind11::arg("query_start_loc"),
        pybind11::arg("cache_indices"), pybind11::arg("has_initial_state"), pybind11::arg("out"), pybind11::arg("batch"),
        pybind11::arg("max_seqlen"), pybind11::arg("pad_slot_id"), pybind11::arg("rt_mask") = 0,
        pybind11::arg("rt_form") = 0);
  m.def("variants", &kdaconv::variants, "compiled variants: (id, name, tt, d, mask, form, packed, runtime)");
}
#endif
