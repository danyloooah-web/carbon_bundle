// finar_ar.cuh -- r60 FINAR (ours, 2026-11-20): the TP4 decode engine's MoE exit = the king's A4 finalize (FlashInfer's
// unfinalized FP4 MoE pieces: GEMM2 rows, expanded_idx_to_permuted_idx, bf16 expert weights; + the shared expert's rows)
// folded into the PUSH READ of a two-shot Lamport all-reduce of the [T, 4096] bf16 partial, so the finalized partial is
// never written and re-read. A declared header of the king's dsa_pagetable_native.cu unit (registered there as its
// submodule "finar"; no new compiled unit: the validator's 16 MiB native-metadata cap).
//
// Transport = OUR r44 / r57 ARPF fork of SGLang 314bce7c's CustomAllReduceV2 2shot_push (kernels/jit/csrc/distributed/
// custom_all_reduce.cuh, read in r44 lane S; GPU-proved on this image by kit apS44vg4b: B0-B5 bitwise on every rank,
// transport parity 0.999), on a bundle-owned symmetric plane (never stock's push plane or counters):
//   partition   flat 16-byte vector index v in [0, 512 T); owner of v = v / (128 T) (W = 4 equal flat chunks, exact at
//               every T; at T = 2 mod 4 the chunk boundaries fall at half rows, as stock's)
//   RS push     every rank pushes its word of each foreign chunk to the owner's (parity, RS, src = rank) slot; a pushed
//               32-bit word that is 0 becomes 0x00008000 (the Lamport empty marker is an all-zero word)
//   reduce      the owner sums in RANK order 0..3, fp32, starting FROM the first word (no +0), its own word RAW (not
//               remapped), one cvt.rn.bf16x2 an element pair; a reduced word that is 0 becomes 0x00008000
//   AG push     the owner pushes the reduced chunk to every peer's (parity, AG, src = owner) slot and stores it locally
//   receive     spin until all four 32-bit words of a vector are non-zero, consume, clear to 0
//   parity      counter[blockIdx.x] & 1 (a device counter a CTA, +1 a call: CUDA-graph replay safe; every rank makes
//               the same calls with the same grid, so the counters stay equal)
// (stock's knobs, B0 on the image: acc from the first word, own raw, markers 0x8000 / 0x8000 = r57 AP_SEM (0, 1, ..)).
//
// Mode 0: the partial is a bf16 [T, 4096] tensor x (plain all-reduce: the kit's / boot proof's transport parity check).
// Mode 4 (FINAR): this rank's word of flat vector v (row t = v / 512, 16-byte column j = v % 512) is the king's A4
//   moe_finalize_native.cu fuse_kernel<true> chain, element by element, verbatim:
//     acc = +0; for k = 0..7 in slot order, slot idx[t, k] == -1 skipped: acc = fma.rn.ftz.f32(w[t, k], g[idx, 8 j + e], acc)
//     routed = cvt.rn.bf16x2(acc pair); word = cvt.rn.bf16x2(add.rn.f32(lo(routed), lo(s)), add.rn.f32(hi(routed), hi(s)))
//   (w[t, k] and g as bf16 widened by a 16-bit shift, as A4's lo() / hi()). Every element's bits are independent of the
//   thread mapping, so the pushed words are exactly the bytes A4 writes; the transport then equals stock's.
// Every fp operation is inline PTX (the bits do not depend on --use_fast_math). Every load of a kernel input is issued
// after griddepcontrol.wait (plain loads behind the wait's memory clobber: A4's form; PDL transitivity as A4's).
// Hang safety: every poll loop reads %globaltimer every 1024 spins and traps past timeout_ns (a loud device fault).
// Host API (pybind11, torch tensors; no allocation, no host sync, one launch on the current stream: capturable):
//   version() -> kVersion; info(mode, block, trig) -> [SMs, CTAs an SM, regs, local bytes, cc x10, device, err, max
//   threads]; run(mode, block, trig, out, x, g, idx, w, s, peers, counter, slot_bytes, timeout_ns, rank, pdl).
#ifndef R60_FINAR_AR_CUH
#define R60_FINAR_AR_CUH

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace finar {

constexpr int64_t kVersion = 6001;  // the route's literal FN_VERSION (a stale build of another source is refused)
constexpr int kW = 4;               // TP4
constexpr int kHidden = 4096;
constexpr int kRowVecs = kHidden / 8;  // 512 16-byte vectors a row
constexpr int kTopK = 8;
constexpr unsigned int kMark = 0x00008000u;  // stock's marker for an all-zero 32-bit word (pushed and reduced)

struct Params {
  const uint4* x;                 // mode 0: the partial [T, 4096] bf16
  const uint4* g;                 // mode 4: FlashInfer's GEMM2 rows [S, 4096] bf16
  const int4* idx;                // mode 4: expanded_idx_to_permuted_idx [T * 8] int32 (two int4 a row)
  const uint4* w;                 // mode 4: expert weights [T, 8] bf16 (one uint4 a row)
  const uint4* s;                 // mode 4: the shared expert's rows [T, 4096] bf16
  uint4* out;                     // [T, 4096] bf16: the reduced rows (a fresh tensor, as stock's)
  long long peer[kW];             // plane base of every rank mapped here (peer[rank] = this rank's own)
  int* counter;                   // [gridDim.x] calls a CTA (parity = counter & 1)
  long long slot_bytes;           // bytes of one (parity, kind, src) slot (>= 2048 T)
  unsigned long long timeout_ns;  // poll trap bound
  int rank;                       // TP rank (= the plane's symmetric rank)
  int T;                          // rows
};

__device__ __forceinline__ float add_rn(float a, float b) {
  float d;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {  // A4's fma_ftz
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ unsigned int pack_rn(float lo, float hi) {  // A4's bf16x2: a -> upper half, b -> lower
  unsigned int d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ float lo_f(unsigned int w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi_f(unsigned int w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ unsigned int mark(unsigned int w) { return w == 0u ? kMark : w; }
__device__ __forceinline__ uint4 mark4(uint4 v) { return make_uint4(mark(v.x), mark(v.y), mark(v.z), mark(v.w)); }
__device__ __forceinline__ unsigned int red_word(unsigned int w0, unsigned int w1, unsigned int w2, unsigned int w3) {
  float l = lo_f(w0), h = hi_f(w0);
  l = add_rn(l, lo_f(w1)); h = add_rn(h, hi_f(w1));
  l = add_rn(l, lo_f(w2)); h = add_rn(h, hi_f(w2));
  l = add_rn(l, lo_f(w3)); h = add_rn(h, hi_f(w3));
  return pack_rn(l, h);
}

__device__ __forceinline__ uint4 ld_poll(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_vol(uint4* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %4};"
               :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ bool full4(const uint4& v) { return v.x != 0u && v.y != 0u && v.z != 0u && v.w != 0u; }
__device__ __forceinline__ uint4 poll(const uint4* p, unsigned long long timeout_ns) {
  uint4 v = ld_poll(p);
  if (full4(v)) return v;
  const unsigned long long t0 = now_ns();
  unsigned int spins = 0;
  do {
    v = ld_poll(p);
    if (((++spins) & 1023u) == 0u && now_ns() - t0 > timeout_ns) __trap();
  } while (!full4(v));
  return v;
}

// One A4 slot on a 16-byte vector: acc[2e] / acc[2e + 1] = fma_ftz(wk, lo / hi(q[e]), acc) (A4 fuse_kernel's inner loop)
__device__ __forceinline__ void fin_slot(float (&acc)[8], float wk, const uint4 q) {
  acc[0] = fma_ftz(wk, lo_f(q.x), acc[0]); acc[1] = fma_ftz(wk, hi_f(q.x), acc[1]);
  acc[2] = fma_ftz(wk, lo_f(q.y), acc[2]); acc[3] = fma_ftz(wk, hi_f(q.y), acc[3]);
  acc[4] = fma_ftz(wk, lo_f(q.z), acc[4]); acc[5] = fma_ftz(wk, hi_f(q.z), acc[5]);
  acc[6] = fma_ftz(wk, lo_f(q.w), acc[6]); acc[7] = fma_ftz(wk, hi_f(q.w), acc[7]);
}

// Mode 4: the A4 word of flat vector v (finalize in slot order, -1 skipped, one RN; then the shared add, one RN).
// Slots gathered four at a time (16 data registers live, not 32: no spill at 1024 threads a CTA).
__device__ __forceinline__ uint4 fin_word(const Params& p, int v) {
  const int t = v >> 9, j = v & (kRowVecs - 1);
  const int4 r0 = p.idx[2 * t], r1 = p.idx[2 * t + 1];
  const uint4 wv = p.w[t];
  const int row[kTopK] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
  const float wk[kTopK] = {lo_f(wv.x), hi_f(wv.x), lo_f(wv.y), hi_f(wv.y),
                           lo_f(wv.z), hi_f(wv.z), lo_f(wv.w), hi_f(wv.w)};
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const uint4* gj = p.g + j;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint4 q[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int k = 4 * h + c;
      q[c] = row[k] == -1 ? make_uint4(0u, 0u, 0u, 0u) : gj[static_cast<long long>(row[k]) * kRowVecs];
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      if (row[4 * h + c] != -1) fin_slot(acc, wk[4 * h + c], q[c]);
    }
  }
  const uint4 s = p.s[v];
  uint4 o;
  unsigned int rt;
  rt = pack_rn(acc[0], acc[1]); o.x = pack_rn(add_rn(lo_f(rt), lo_f(s.x)), add_rn(hi_f(rt), hi_f(s.x)));
  rt = pack_rn(acc[2], acc[3]); o.y = pack_rn(add_rn(lo_f(rt), lo_f(s.y)), add_rn(hi_f(rt), hi_f(s.y)));
  rt = pack_rn(acc[4], acc[5]); o.z = pack_rn(add_rn(lo_f(rt), lo_f(s.z)), add_rn(hi_f(rt), hi_f(s.z)));
  rt = pack_rn(acc[6], acc[7]); o.w = pack_rn(add_rn(lo_f(rt), lo_f(s.w)), add_rn(hi_f(rt), hi_f(s.w)));
  return o;
}

template <int kMode>
__device__ __forceinline__ uint4 word(const Params& p, int v) {
  if constexpr (kMode == 4) {
    return fin_word(p, v);
  } else {
    return p.x[v];
  }
}

template <int kMode, int kBlock, int kTrig>
__global__ void __launch_bounds__(kBlock, 1) finar_kernel(const __grid_constant__ Params p) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int chunk = 128 * p.T;  // vectors an owner: 512 T / 4
  const int rank = p.rank;
  const int nthr = gridDim.x * kBlock;
  const int gtid = blockIdx.x * kBlock + threadIdx.x;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  const int c0 = p.counter[blockIdx.x];
  const long long par = static_cast<long long>(c0 & 1) * 2 * kW * p.slot_bytes;
  char* const mine = reinterpret_cast<char*>(p.peer[rank]) + par;

  // 1. RS push: this rank's word of every foreign chunk into its owner's (parity, RS, rank) slot
  for (int i = gtid; i < 3 * chunk; i += nthr) {
    const int k = i / chunk, off = i - k * chunk, o = (rank + 1 + k) & (kW - 1);
    const uint4 wv = mark4(word<kMode>(p, o * chunk + off));
    st_vol(reinterpret_cast<uint4*>(reinterpret_cast<char*>(p.peer[o]) + par + rank * p.slot_bytes) + off, wv);
  }
  // 2. owner reduce of this rank's chunk in rank order (own word raw), push d' to every peer's (parity, AG, rank)
  //    slot, store d' here
  for (int i = gtid; i < chunk; i += nthr) {
    const int v = rank * chunk + i;
    uint4 w[kW];
#pragma unroll
    for (int r = 0; r < kW; ++r) {
      if (r == rank) {
        w[r] = word<kMode>(p, v);
      } else {
        uint4* q = reinterpret_cast<uint4*>(mine + r * p.slot_bytes) + i;
        w[r] = poll(q, p.timeout_ns);
        st_vol(q, zero);
      }
    }
    uint4 d;
    d.x = red_word(w[0].x, w[1].x, w[2].x, w[3].x);
    d.y = red_word(w[0].y, w[1].y, w[2].y, w[3].y);
    d.z = red_word(w[0].z, w[1].z, w[2].z, w[3].z);
    d.w = red_word(w[0].w, w[1].w, w[2].w, w[3].w);
    d = mark4(d);
#pragma unroll
    for (int q = 1; q < kW; ++q) {
      const int dst = (rank + q) & (kW - 1);
      st_vol(reinterpret_cast<uint4*>(reinterpret_cast<char*>(p.peer[dst]) + par + (kW + rank) * p.slot_bytes) + i, d);
    }
    p.out[v] = d;
  }
  // 3. PDL trigger (a speed choice: every dependent still waits at griddepcontrol.wait for this grid's end)
  if constexpr (kTrig == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  // 4. AG receive: the three foreign chunks from their owners, consumed, cleared, stored
  for (int i = gtid; i < 3 * chunk; i += nthr) {
    const int k = i / chunk, off = i - k * chunk, o = (rank + 1 + k) & (kW - 1);
    uint4* q = reinterpret_cast<uint4*>(mine + (kW + o) * p.slot_bytes) + off;
    const uint4 d = poll(q, p.timeout_ns);
    st_vol(q, zero);
    p.out[o * chunk + off] = d;
  }
  if constexpr (kTrig == 1) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  // 5. the CTA's call counter (every thread read c0 above)
  __syncthreads();
  if (threadIdx.x == 0) p.counter[blockIdx.x] = c0 + 1;
}

using KernelFn = void (*)(const Params);

#define FINAR_CASE(M, B, G) \
  if (mode == M && block == B && trig == G) return &finar_kernel<M, B, G>;

inline KernelFn pick(int64_t mode, int64_t block, int64_t trig) {
  FINAR_CASE(0, 1024, 0) FINAR_CASE(0, 1024, 1) FINAR_CASE(0, 512, 0) FINAR_CASE(0, 512, 1)
  FINAR_CASE(4, 1024, 0) FINAR_CASE(4, 1024, 1) FINAR_CASE(4, 512, 0) FINAR_CASE(4, 512, 1)
  return nullptr;
}
#undef FINAR_CASE

inline int sm_count(int device) {
  static int known[64] = {};
  TORCH_CHECK(device >= 0 && device < 64, "finar: device index");
  if (known[device] == 0) {
    int sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    TORCH_CHECK(sms >= 1, "finar: no SM count");
    known[device] = sms;
  }
  return known[device];
}

inline bool aligned16(const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; }

}  // namespace finar

static int64_t finar_version() { return finar::kVersion; }

// One instance's facts on the current device: [SMs, CTAs an SM at `block` threads, registers a thread, local bytes a
// thread, compute capability x10, device, CUDA error of the queries, max threads a CTA]; an unknown instance: err -1.
static std::vector<int64_t> finar_info(int64_t mode, int64_t block, int64_t trig) {
  int dev = -1, sms = 0, maj = 0, mnr = 0, nb = 0;
  cudaFuncAttributes fa{};
  cudaError_t e = cudaGetDevice(&dev);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev);
  finar::KernelFn fn = finar::pick(mode, block, trig);
  if (fn == nullptr) return {sms, 0, 0, 0, maj * 10 + mnr, dev, -1, 0};
  if (e == cudaSuccess) e = cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(fn));
  if (e == cudaSuccess)
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, reinterpret_cast<const void*>(fn), static_cast<int>(block), 0);
  if (e != cudaSuccess) {
    cudaGetLastError();
    return {sms, 0, 0, 0, maj * 10 + mnr, dev, static_cast<int64_t>(e), 0};
  }
  return {sms, nb, fa.numRegs, static_cast<int64_t>(fa.localSizeBytes), maj * 10 + mnr, dev, 0,
          fa.maxThreadsPerBlock};
}

// One FINAR launch on the current stream, one CTA an SM. mode 0: x = the partial (g / idx / w / s unused); mode 4: g /
// idx / w / s = A4's operands (x unused). out: bf16 [T, 4096], never aliasing an input.
static void finar_run(int64_t mode, int64_t block, int64_t trig, const torch::Tensor& out, const torch::Tensor& x,
                      const torch::Tensor& g, const torch::Tensor& idx, const torch::Tensor& w, const torch::Tensor& s,
                      const std::vector<int64_t>& peers, const torch::Tensor& counter, int64_t slot_bytes,
                      int64_t timeout_ns, int64_t rank, bool pdl) {
  using namespace finar;
  const KernelFn fn = pick(mode, block, trig);
  TORCH_CHECK(fn != nullptr, "finar: no instance mode ", mode, " block ", block, " trig ", trig);
  const auto here = [&](const torch::Tensor& t) {
    return t.is_cuda() && t.get_device() == out.get_device() && t.is_contiguous() && aligned16(t);
  };
  const auto bf16 = [&](const torch::Tensor& t, int64_t n) {
    return here(t) && t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.size(0) == n && t.size(1) == kHidden;
  };
  TORCH_CHECK(out.dim() == 2 && out.size(0) >= 1 && out.size(0) <= (int64_t(1) << 20) && bf16(out, out.size(0)),
              "finar: out must be a contiguous 16-byte aligned BF16 [T, 4096] CUDA tensor");
  const int64_t T = out.size(0);
  const char* ob = static_cast<const char*>(out.data_ptr());
  const auto apart = [&](const torch::Tensor& t) {
    const char* b = static_cast<const char*>(t.data_ptr());
    const int64_t n = t.numel() * t.element_size();
    return b + n <= ob || ob + T * kHidden * 2 <= b;
  };
  if (mode == 0) {
    TORCH_CHECK(bf16(x, T) && apart(x), "finar: mode 0 needs x BF16 [T, 4096], not aliasing out");
  } else {
    TORCH_CHECK(g.dim() == 2 && g.size(0) >= 1 && bf16(g, g.size(0)) && apart(g),
                "finar: mode 4 needs g = GEMM2 rows BF16 [S, 4096] (contiguous, aligned, not aliasing out)");
    TORCH_CHECK(here(idx) && idx.scalar_type() == at::kInt && idx.dim() == 1 && idx.numel() == T * kTopK,
                "finar: mode 4 needs idx = int32 [T * 8]");
    TORCH_CHECK(here(w) && w.scalar_type() == at::kBFloat16 && w.dim() == 2 && w.size(0) == T && w.size(1) == kTopK,
                "finar: mode 4 needs w = BF16 [T, 8]");
    TORCH_CHECK(bf16(s, T) && apart(s), "finar: mode 4 needs s = the shared rows BF16 [T, 4096], not aliasing out");
  }
  TORCH_CHECK(peers.size() == size_t(kW) && rank >= 0 && rank < kW, "finar: four plane bases and a rank in [0, 4)");
  for (int64_t b : peers) TORCH_CHECK(b != 0 && b % 16 == 0, "finar: a plane base is null or not 16-byte aligned");
  TORCH_CHECK(slot_bytes >= 2048 * T && slot_bytes % 16 == 0, "finar: slot of ", slot_bytes, " bytes < 2048 x ", T);
  TORCH_CHECK(timeout_ns > 0, "finar: timeout");
  const c10::cuda::CUDAGuard guard(out.device());
  const int grid = sm_count(out.get_device());
  TORCH_CHECK(counter.is_cuda() && counter.get_device() == out.get_device() && counter.scalar_type() == at::kInt &&
                  counter.is_contiguous() && counter.numel() == grid,
              "finar: counter must hold one int32 a CTA of the fixed grid (", grid, ")");
  Params p{};
  p.x = mode == 0 ? reinterpret_cast<const uint4*>(x.data_ptr()) : nullptr;
  p.g = mode == 4 ? reinterpret_cast<const uint4*>(g.data_ptr()) : nullptr;
  p.idx = mode == 4 ? reinterpret_cast<const int4*>(idx.data_ptr()) : nullptr;
  p.w = mode == 4 ? reinterpret_cast<const uint4*>(w.data_ptr()) : nullptr;
  p.s = mode == 4 ? reinterpret_cast<const uint4*>(s.data_ptr()) : nullptr;
  p.out = reinterpret_cast<uint4*>(out.data_ptr());
  for (int r = 0; r < kW; ++r) p.peer[r] = static_cast<long long>(peers[r]);
  p.counter = counter.data_ptr<int>();
  p.slot_bytes = slot_bytes;
  p.timeout_ns = static_cast<unsigned long long>(timeout_ns);
  p.rank = static_cast<int>(rank);
  p.T = static_cast<int>(T);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(grid));
  cfg.blockDim = dim3(static_cast<unsigned>(block));
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fn, p));
}

// Registered by the king's dsa_pagetable_native.cu PYBIND11_MODULE on m.def_submodule("finar") (r60 apply_finar.py).
static void r60_finar(pybind11::module_ m) {
  m.def("version", &finar_version, "r60 FINAR finar_ar.cuh version");
  m.def("info", &finar_info, "[SMs, CTAs an SM, regs, local bytes, cc x10, device, err, max threads] of an instance");
  m.def("run", &finar_run,
        "the TP4 two-shot push all-reduce of a [T, 4096] bf16 partial (mode 0) or of the king's A4 MoE finalize + "
        "shared add computed in the push read (mode 4), into out");
}

#endif  // R60_FINAR_AR_CUH
