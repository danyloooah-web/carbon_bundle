


































#include <cuda_runtime.h>

#include <cstdint>

namespace rotq {

constexpr int D = 128;
constexpr int THREADS = 256;
constexpr int ROWS_A_WARP = 4;

struct Params {
  const uint16_t* x;
  uint16_t* out;
  uint8_t* y;
  float* s;
  int64_t rows;
  float scale;

  const float* raw;
  float* gate;
  float c1;
  float ss;
};

__device__ __forceinline__ float add_rn(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float sub_rn(float a, float b) {
  float r;
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float max_f(float a, float b) {
  float r;
  asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float min_f(float a, float b) {
  float r;
  asm("min.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float abs_f(float a) {
  float r;
  asm("abs.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ uint32_t bf16_rn(float a) {
  uint16_t h;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(a));
  return h;
}
__device__ __forceinline__ uint32_t fp8x2(float hi, float lo) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}



__device__ __forceinline__ float head_gate(float raw, float c1, float scale, float ss) {
  return mul_rn(mul_rn(mul_rn(raw, c1), scale), ss);
}




__device__ __forceinline__ float act_quant_scale(float amax) {
  float r;
  asm("{\n\t"
      ".reg .pred tiny, finite, zero;\n\t"
      ".reg .f32 a, x, e0, fe, e, f, p, q, c, big;\n\t"
      ".reg .b32 xi, i1, i2, mi;\n\t"
      "mul.f32 a, %1, 0f3B124925;\n\t"
      "setp.lt.f32 tiny, a, 0f00800000;\n\t"
      "mul.f32 x, a, 0f4B000000;\n\t"
      "selp.f32 x, x, a, tiny;\n\t"
      "selp.f32 e0, 0fC1B80000, 0f00000000, tiny;\n\t"
      "mov.b32 xi, x;\n\t"
      "add.s32 i1, xi, -1060439283;\n\t"
      "and.b32 i2, i1, -8388608;\n\t"
      "sub.s32 mi, xi, i2;\n\t"
      "cvt.rn.f32.s32 fe, i2;\n\t"
      "mov.b32 c, 0f34000000;\n\t"
      "fma.rn.ftz.f32 e, fe, c, e0;\n\t"
      "mov.b32 f, mi;\n\t"
      "add.f32 f, f, 0fBF800000;\n\t"
      "mov.b32 q, 0fBE2C7F30;\n\t"
      "mov.b32 c, 0f3DC6B27F;\n\t"
      "fma.rn.ftz.f32 p, c, f, q;\n\t"
      "mov.b32 c, 0f3E2FCF2A;\n\t"
      "fma.rn.ftz.f32 p, p, f, c;\n\t"
      "mov.b32 c, 0fBE374E43;\n\t"
      "fma.rn.ftz.f32 p, p, f, c;\n\t"
      "mov.b32 c, 0f3E520BF4;\n\t"
      "fma.rn.ftz.f32 p, p, f, c;\n\t"
      "mov.b32 c, 0fBE763C8B;\n\t"
      "fma.rn.ftz.f32 p, p, f, c;\n\t"
      "mov.b32 c, 0f3E93BF99;\n\t"
      "fma.rn.ftz.f32 p, p, f, c;\n\t"
      "mov.b32 c, 0fBEB8AA49;\n\t"
      "fma.rn.ftz.f32 p, p, f, c;\n\t"
      "mov.b32 c, 0f3EF6384A;\n\t"
      "fma.rn.ftz.f32 p, p, f, c;\n\t"
      "mov.b32 c, 0fBF38AA3B;\n\t"
      "fma.rn.ftz.f32 p, p, f, c;\n\t"
      "mul.f32 q, f, p;\n\t"
      "mul.f32 q, f, q;\n\t"
      "mov.b32 c, 0f3FB8AA3B;\n\t"
      "fma.rn.ftz.f32 p, f, c, q;\n\t"
      "add.f32 p, e, p;\n\t"
      "setp.lt.u32 finite, xi, 2139095040;\n\t"
      "mov.b32 big, 0f7F800000;\n\t"
      "@!finite fma.rn.ftz.f32 p, x, big, big;\n\t"
      "setp.eq.f32 zero, x, 0f00000000;\n\t"
      "selp.f32 p, 0fFF800000, p, zero;\n\t"
      "cvt.rpi.ftz.f32.f32 p, p;\n\t"
      "ex2.approx.ftz.f32 %0, p;\n\t"
      "}"
      : "=f"(r)
      : "f"(amax));
  return r;
}

struct Row {
  uint32_t ob[8];
  uint32_t oy[4];
  float scale;
};



__device__ __forceinline__ Row compute(const uint4 v0, const uint4 v1, const uint32_t t, const float rot_scale) {
  Row out;
  const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
  float x[16];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[2 * j] = __uint_as_float(w[j] << 16);
    x[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
  }

#pragma unroll
  for (int s = 1; s < 16; s <<= 1) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (j & s) continue;
      const float a = x[j], b = x[j + s];
      x[j] = add_rn(a, b);
      x[j + s] = sub_rn(a, b);
    }
  }



#pragma unroll
  for (uint32_t m = 1; m < 8; m <<= 1) {
    const uint32_t flip = (t & m) != 0 ? 0x80000000u : 0u;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float o = __shfl_xor_sync(0xffffffffu, x[j], m);
      x[j] = add_rn(o, __uint_as_float(__float_as_uint(x[j]) ^ flip));
    }
  }

  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t lo = bf16_rn(mul_rn(x[2 * j], rot_scale)), hi = bf16_rn(mul_rn(x[2 * j + 1], rot_scale));
    out.ob[j] = lo | (hi << 16);
    x[2 * j] = __uint_as_float(lo << 16);
    x[2 * j + 1] = __uint_as_float(hi << 16);
    amax = max_f(amax, abs_f(x[2 * j]));
    amax = max_f(amax, abs_f(x[2 * j + 1]));
  }
#pragma unroll
  for (uint32_t m = 1; m < 8; m <<= 1) amax = max_f(amax, __shfl_xor_sync(0xffffffffu, amax, m));
  out.scale = act_quant_scale(max_f(amax, __uint_as_float(0x38D1B717u)));
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float q[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float d;
      asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(x[4 * j + k]), "f"(out.scale));
      q[k] = min_f(max_f(d, -448.0f), 448.0f);
    }
    out.oy[j] = fp8x2(q[1], q[0]) | (fp8x2(q[3], q[2]) << 16);
  }
  return out;
}

__device__ __forceinline__ void store(const Row& o, const int64_t r, const uint32_t t, const Params& p) {
  uint4* b = reinterpret_cast<uint4*>(p.out + r * D + 16 * t);
  b[0] = make_uint4(o.ob[0], o.ob[1], o.ob[2], o.ob[3]);
  b[1] = make_uint4(o.ob[4], o.ob[5], o.ob[6], o.ob[7]);
  *reinterpret_cast<uint4*>(p.y + r * D + 16 * t) = make_uint4(o.oy[0], o.oy[1], o.oy[2], o.oy[3]);
  if (t == 0) p.s[r] = o.scale;
}

__global__ void __launch_bounds__(THREADS) rotq_kernel(Params p) {
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const uint32_t t = lane & 7;
  const int64_t mine = int64_t(warp) * ROWS_A_WARP + (lane >> 3);
  const int64_t a_cta = int64_t(THREADS / 32) * ROWS_A_WARP;
  const int64_t step = int64_t(gridDim.x) * a_cta;




  for (int64_t base = int64_t(blockIdx.x) * a_cta; base < p.rows; base += 2 * step) {
    const int64_t r = base + mine, r2 = r + step;
    const bool in = r < p.rows, in2 = r2 < p.rows;
    const uint4* src = reinterpret_cast<const uint4*>(p.x + (in ? r : p.rows - 1) * D + 16 * t);
    const uint4* src2 = reinterpret_cast<const uint4*>(p.x + (in2 ? r2 : p.rows - 1) * D + 16 * t);
    const uint4 a0 = src[0], a1 = src[1], b0 = src2[0], b1 = src2[1];
    const Row o = compute(a0, a1, t, p.scale);
    const Row o2 = compute(b0, b1, t, p.scale);
    if (in) store(o, r, t, p);
    if (in2) store(o2, r2, t, p);
  }
}




__global__ void __launch_bounds__(THREADS) rotq_gates_kernel(Params p) {
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const uint32_t t = lane & 7;
  const int64_t mine = int64_t(warp) * ROWS_A_WARP + (lane >> 3);
  const int64_t a_cta = int64_t(THREADS / 32) * ROWS_A_WARP;
  const int64_t step = int64_t(gridDim.x) * a_cta;
  for (int64_t base = int64_t(blockIdx.x) * a_cta; base < p.rows; base += 2 * step) {
    const int64_t r = base + mine, r2 = r + step;
    const bool in = r < p.rows, in2 = r2 < p.rows;
    const uint4* src = reinterpret_cast<const uint4*>(p.x + (in ? r : p.rows - 1) * D + 16 * t);
    const uint4* src2 = reinterpret_cast<const uint4*>(p.x + (in2 ? r2 : p.rows - 1) * D + 16 * t);
    const uint4 a0 = src[0], a1 = src[1], b0 = src2[0], b1 = src2[1];
    float g = 0.0f, g2 = 0.0f;
    if (t == 0 && in) g = p.raw[r];
    if (t == 0 && in2) g2 = p.raw[r2];
    const Row o = compute(a0, a1, t, p.scale);
    const Row o2 = compute(b0, b1, t, p.scale);
    if (in) {
      store(o, r, t, p);
      if (t == 0) p.gate[r] = head_gate(g, p.c1, o.scale, p.ss);
    }
    if (in2) {
      store(o2, r2, t, p);
      if (t == 0) p.gate[r2] = head_gate(g2, p.c1, o2.scale, p.ss);
    }
  }
}

}

#ifndef ROTQ_DEVICE_ONLY
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {


rotq::Params checked(const torch::Tensor& x, const torch::Tensor& out, const torch::Tensor& y, const torch::Tensor& s,
                     double scale, int64_t grid_per_sm) {
  using namespace rotq;
  const auto aligned = [](const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; };
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() >= 1 && x.size(-1) == D &&
                  x.is_contiguous() && aligned(x),
              "rotq: x must be contiguous 16-byte aligned BF16 [..., 128]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.sizes() == x.sizes() &&
                  out.is_contiguous() && aligned(out),
              "rotq: out must be a contiguous BF16 tensor of x's shape");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::ScalarType::Float8_e4m3fn && y.sizes() == x.sizes() &&
                  y.is_contiguous() && aligned(y),
              "rotq: y must be a contiguous FP8 tensor of x's shape");
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(s.is_cuda() && s.scalar_type() == at::kFloat && s.is_contiguous() && s.numel() == rows,
              "rotq: s must be contiguous fp32 with one scale a row");
  TORCH_CHECK(grid_per_sm >= 1 && grid_per_sm <= 16, "rotq: grid_per_sm must be 1 to 16");
  return Params{reinterpret_cast<const uint16_t*>(x.data_ptr()), reinterpret_cast<uint16_t*>(out.data_ptr()),
                reinterpret_cast<uint8_t*>(y.data_ptr()), s.data_ptr<float>(), rows, static_cast<float>(scale)};
}


unsigned grid(int64_t rows, int64_t grid_per_sm) {
  const int64_t a_cta = int64_t(rotq::THREADS / 32) * rotq::ROWS_A_WARP;
  const int64_t full = (rows + a_cta - 1) / a_cta;
  const int64_t cap = int64_t(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * grid_per_sm;
  return unsigned(full < cap ? full : cap);
}

}



void rotate_quant(torch::Tensor x, torch::Tensor out, torch::Tensor y, torch::Tensor s, double scale,
                  int64_t grid_per_sm) {
  const rotq::Params p = checked(x, out, y, s, scale, grid_per_sm);
  if (p.rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  rotq::rotq_kernel<<<grid(p.rows, grid_per_sm), rotq::THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}



void rotate_quant_gated(torch::Tensor x, torch::Tensor out, torch::Tensor y, torch::Tensor s, torch::Tensor raw,
                        torch::Tensor gate, double scale, double c1, double ss, int64_t grid_per_sm) {
  rotq::Params p = checked(x, out, y, s, scale, grid_per_sm);
  TORCH_CHECK(raw.is_cuda() && raw.device() == x.device() && raw.scalar_type() == at::kFloat && raw.is_contiguous() &&
                  raw.numel() == p.rows,
              "rotq: raw must be contiguous fp32 with one head gate a row");
  TORCH_CHECK(gate.is_cuda() && gate.device() == x.device() && gate.scalar_type() == at::kFloat &&
                  gate.is_contiguous() && gate.numel() == p.rows,
              "rotq: gate must be contiguous fp32 with one head gate a row");
  if (p.rows == 0) return;
  p.raw = raw.data_ptr<float>();
  p.gate = gate.data_ptr<float>();
  p.c1 = static_cast<float>(c1);
  p.ss = static_cast<float>(ss);
  const c10::cuda::CUDAGuard guard(x.device());
  rotq::rotq_gates_kernel<<<grid(p.rows, grid_per_sm), rotq::THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rotate_quant", &rotate_quant,
        "rotate_activation + act_quant (ue8m0) of the sparse-attention indexer query [..., 128] in one launch");
  m.def("rotate_quant_gated", &rotate_quant_gated,
        "rotate_quant and the indexer's final head gates ((raw * c1) * scale) * ss in the same launch");
}
#endif
