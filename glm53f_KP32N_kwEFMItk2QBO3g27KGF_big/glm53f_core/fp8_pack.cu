





















#include <cuda_runtime.h>

#include <cstdint>

namespace fp8_pack {

constexpr int D = 512;
constexpr int THREADS = 256;


__device__ __forceinline__ uint32_t fp8_from_bf16(uint32_t h) {
  const uint32_t v = h & 0x7fffu;


  const uint32_t t = v - 0x3c00u;
  uint32_t r = (t + 7u + ((t >> 4) & 1u)) >> 4;
  r = r < 0x7eu ? r : 0x7eu;

  const uint32_t tiny =
      __float_as_uint(__fadd_rn(__uint_as_float(v << 16), __uint_as_float(141u << 23))) - (141u << 23);
  r = v < 0x3c80u ? tiny : r;
  r |= (h >> 8) & 0x80u;
  return v > 0x7f80u ? 0x7fu : r;
}

#if defined(__CUDA_ARCH__)






__device__ __forceinline__ uint32_t pack2(uint32_t w) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
      : "=h"(r)
      : "f"(__uint_as_float(w & 0xffff0000u)), "f"(__uint_as_float(w << 16)));
  uint32_t out = r;
  out = (w & 0x7fffu) > 0x7f80u ? (out & 0xff00u) | 0x7fu : out;
  out = ((w >> 16) & 0x7fffu) > 0x7f80u ? (out & 0x00ffu) | 0x7f00u : out;
  return out;
}

__device__ __forceinline__ uint32_t pack4(uint32_t w0, uint32_t w1) {
  return pack2(w0) | (pack2(w1) << 16);
}
#else
__device__ __forceinline__ uint32_t pack4(uint32_t w0, uint32_t w1) {
  return fp8_from_bf16(w0 & 0xffffu) | (fp8_from_bf16(w0 >> 16) << 8) | (fp8_from_bf16(w1 & 0xffffu) << 16) |
         (fp8_from_bf16(w1 >> 16) << 24);
}
#endif

struct Params {
  const uint16_t* q;
  int64_t stride_t, stride_h;
  uint8_t* q_out;
  int64_t tokens, heads;
  const uint16_t* k;
  uint8_t* k_out;
  int64_t k_rows;
};

__global__ void __launch_bounds__(THREADS) pack_kernel(Params p) {
  const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int64_t q_rows = p.tokens * p.heads, rows = q_rows + p.k_rows;
  const uint32_t step = gridDim.x * (THREADS / 32);
  const uint32_t first = blockIdx.x * (THREADS / 32) + warp;


  const uint32_t heads = p.heads > 0 ? uint32_t(p.heads) : 1u;
  const uint32_t step_h = step % heads;
  uint32_t h = first % heads;
  int64_t off = int64_t(first / heads) * p.stride_t + int64_t(h) * p.stride_h;
  const int64_t step_off = int64_t(step / heads) * p.stride_t + int64_t(step_h) * p.stride_h;
  const int64_t wrap_off = p.stride_t - int64_t(heads) * p.stride_h;


  for (int64_t r = first; r < rows; r += 2 * int64_t(step)) {
    const int64_t r2 = r + step;
    uint32_t h2 = h + step_h;
    int64_t off2 = off + step_off;
    if (h2 >= heads) {
      h2 -= heads;
      off2 += wrap_off;
    }
    const bool two = r2 < rows;
    const uint16_t* src = r < q_rows ? p.q + off : p.k + (r - q_rows) * D;



    const int64_t rr = two ? r2 : r;
    const int64_t qo = two ? off2 : off;
    const uint16_t* src2 = rr < q_rows ? p.q + qo : p.k + (rr - q_rows) * D;
    const uint4 a = *reinterpret_cast<const uint4*>(src + lane * 16);
    const uint4 b = *reinterpret_cast<const uint4*>(src + lane * 16 + 8);
    const uint4 a2 = *reinterpret_cast<const uint4*>(src2 + lane * 16);
    const uint4 b2 = *reinterpret_cast<const uint4*>(src2 + lane * 16 + 8);
    uint8_t* dst = r < q_rows ? p.q_out + r * D : p.k_out + (r - q_rows) * D;
    *reinterpret_cast<uint4*>(dst + lane * 16) =
        make_uint4(pack4(a.x, a.y), pack4(a.z, a.w), pack4(b.x, b.y), pack4(b.z, b.w));
    if (two) {
      uint8_t* dst2 = r2 < q_rows ? p.q_out + r2 * D : p.k_out + (r2 - q_rows) * D;
      *reinterpret_cast<uint4*>(dst2 + lane * 16) =
          make_uint4(pack4(a2.x, a2.y), pack4(a2.z, a2.w), pack4(b2.x, b2.y), pack4(b2.z, b2.w));
    }
    h = h2 + step_h;
    off = off2 + step_off;
    if (h >= heads) {
      h -= heads;
      off += wrap_off;
    }
  }
}

}

#ifndef FP8_PACK_DEVICE_ONLY
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>



void pack(torch::Tensor q_nope, torch::Tensor q_out, torch::Tensor k_nope, torch::Tensor k_out, int64_t grid_per_sm) {
  using namespace fp8_pack;
  const auto aligned = [](const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; };
  TORCH_CHECK(q_nope.is_cuda() && q_nope.dim() == 3 && q_nope.size(2) == D && q_nope.scalar_type() == at::kBFloat16 &&
                  q_nope.stride(2) == 1 && q_nope.stride(0) % 8 == 0 && q_nope.stride(1) % 8 == 0 &&
                  q_nope.stride(0) >= 0 && q_nope.stride(1) >= 0 && aligned(q_nope),
              "fp8_pack: q_nope must be BF16 [T, H, 512] with a unit last stride and 16-byte aligned rows");
  TORCH_CHECK(q_out.is_cuda() && q_out.is_contiguous() && q_out.sizes() == q_nope.sizes() &&
                  q_out.scalar_type() == at::ScalarType::Float8_e4m3fn && aligned(q_out),
              "fp8_pack: q_out must be a contiguous FP8 tensor of q_nope's shape");
  TORCH_CHECK(k_nope.is_cuda() && k_nope.dim() == 2 && k_nope.size(1) == D && k_nope.is_contiguous() &&
                  k_nope.scalar_type() == at::kBFloat16 && (k_nope.numel() == 0 || aligned(k_nope)),
              "fp8_pack: k_nope must be contiguous BF16 [N, 512]");
  TORCH_CHECK(k_out.is_cuda() && k_out.is_contiguous() && k_out.sizes() == k_nope.sizes() &&
                  k_out.scalar_type() == at::ScalarType::Float8_e4m3fn && (k_out.numel() == 0 || aligned(k_out)),
              "fp8_pack: k_out must be a contiguous FP8 tensor of k_nope's shape");
  TORCH_CHECK(grid_per_sm >= 1 && grid_per_sm <= 16, "fp8_pack: grid_per_sm must be 1 to 16");
  const int64_t rows = q_nope.size(0) * q_nope.size(1) + k_nope.size(0);
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(q_nope.device());
  const Params p{reinterpret_cast<const uint16_t*>(q_nope.data_ptr()),
                 q_nope.stride(0),
                 q_nope.stride(1),
                 reinterpret_cast<uint8_t*>(q_out.data_ptr()),
                 q_nope.size(0),
                 q_nope.size(1),
                 reinterpret_cast<const uint16_t*>(k_nope.data_ptr()),
                 reinterpret_cast<uint8_t*>(k_out.data_ptr()),
                 k_nope.size(0)};
  const int64_t full = (rows + THREADS / 32 - 1) / (THREADS / 32);
  const int64_t cap = int64_t(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * grid_per_sm;
  pack_kernel<<<unsigned(full < cap ? full : cap), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack", &pack, "BF16 -> FP8 E4M3 of the absorbed MLA query [T, H, 512] and of k [N, 512] in one launch");
}
#endif
