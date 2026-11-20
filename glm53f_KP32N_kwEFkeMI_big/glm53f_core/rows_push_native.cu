








































#include <cstdint>

#ifndef ROWS_HOST
#include <cuda_runtime.h>
#endif
#ifndef ROWS_SPIN_GUARD
#define ROWS_SPIN_GUARD
#endif

namespace rows_push {

constexpr int kMaxRanks = 8;
constexpr int kMaxBlocks = 1024;
constexpr int kThreads = 512;
constexpr int kUnroll = 4;
constexpr int kSignalWords = 2 * kMaxBlocks * kMaxRanks;

struct Params {
  const uint4* src;
  uint4* dst[kMaxRanks];
  uint32_t* pad[kMaxRanks];
  uint32_t rows;
  uint32_t share;
  uint32_t chunk;
  uint32_t world;
  uint32_t rank;
  uint32_t mode;
};

#ifndef ROWS_HOST

__device__ __forceinline__ void store_release(uint32_t* slot, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(slot), "r"(value) : "memory");
}


__device__ __forceinline__ uint32_t load_acquire(const uint32_t* slot) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(slot) : "memory");
  return value;
}
#endif


__device__ __forceinline__ uint32_t* slot(const Params& p, uint32_t owner, uint32_t sender) {
  return p.pad[owner] + (static_cast<size_t>(p.mode) * kMaxBlocks + blockIdx.x) * kMaxRanks + sender;
}


__device__ __forceinline__ uint32_t* epochs(const Params& p) {
  return p.pad[p.rank] + kSignalWords + static_cast<size_t>(p.mode) * kMaxBlocks + blockIdx.x;
}


__device__ __forceinline__ uint32_t begin(const Params& p) { return *epochs(p) + 1; }
__device__ __forceinline__ void finish(const Params& p, uint32_t epoch) { *epochs(p) = epoch; }


__device__ __forceinline__ void arrive(const Params& p, uint32_t s, uint32_t epoch) {
  store_release(slot(p, s, p.rank), epoch);
}


__device__ __forceinline__ void depart(const Params& p, uint32_t s, uint32_t epoch) {
  while (static_cast<int32_t>(load_acquire(slot(p, p.rank, s)) - epoch) < 0) {
    ROWS_SPIN_GUARD
  }
}


__device__ __forceinline__ uint4* destination(const Params& p, uint32_t c) {
  const uint32_t t = c / p.chunk;
  const uint32_t v = c - t * p.chunk;
  if (p.mode == 0) {
    const uint32_t owner = t / p.share;
    return p.dst[owner] + (static_cast<size_t>(t - owner * p.share) * p.world + p.rank) * p.chunk + v;
  }
  const uint32_t i = t / p.world;
  const uint32_t g = t - i * p.world;
  return p.dst[g] + (static_cast<size_t>(p.rank) * p.share + i) * p.chunk + v;
}


__device__ __forceinline__ void move(const Params& p) {
  const uint32_t total = p.rows * p.chunk;
  const uint32_t stride = gridDim.x * kThreads;
  uint32_t c = blockIdx.x * kThreads + threadIdx.x;
  for (; c + (kUnroll - 1) * stride < total; c += kUnroll * stride) {
    uint4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) v[u] = p.src[c + u * stride];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) *destination(p, c + u * stride) = v[u];
  }
  for (; c < total; c += stride) *destination(p, c) = p.src[c];
}

__global__ __launch_bounds__(kThreads) void push_kernel(const __grid_constant__ Params p) {
  const uint32_t epoch = begin(p);
  move(p);
  __syncthreads();
  if (threadIdx.x < p.world) {
    arrive(p, threadIdx.x, epoch);
    depart(p, threadIdx.x, epoch);
  }
  if (threadIdx.x == 0) finish(p, epoch);
}

}

#ifndef ROWS_DEVICE_ONLY
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>


int64_t pad_words() { return int64_t(rows_push::kSignalWords) + 2 * rows_push::kMaxBlocks; }





int64_t push(torch::Tensor src, std::vector<int64_t> dsts, std::vector<int64_t> pads, int64_t rank, int64_t mode,
             int64_t rows, int64_t chunk_bytes, int64_t dst_rows, int64_t max_blocks) {
  using namespace rows_push;
  const int64_t world = static_cast<int64_t>(dsts.size());
  TORCH_CHECK(world >= 2 && world <= kMaxRanks && static_cast<int64_t>(pads.size()) == world,
              "rows push: 2 to 8 ranks, one destination and one pad each");
  TORCH_CHECK(rank >= 0 && rank < world, "rows push: rank outside the group");
  TORCH_CHECK(mode == 0 || mode == 1, "rows push: mode 0 (queries) or 1 (outputs)");
  TORCH_CHECK(rows >= world && rows % world == 0, "rows push: the rows must be world equal shares");
  TORCH_CHECK(chunk_bytes >= 16 && chunk_bytes % 16 == 0, "rows push: an item must be a multiple of 16 bytes");
  const int64_t share = rows / world;
  TORCH_CHECK(dst_rows >= (mode == 0 ? share : rows), "rows push: the destination buffers hold too few rows");
  const int64_t chunk = chunk_bytes / 16;
  const int64_t total = rows * chunk;
  TORCH_CHECK(total < (int64_t(1) << 31), "rows push: under 2^31 vectors a call");
  TORCH_CHECK(src.is_cuda() && src.is_contiguous() && reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0 &&
                  static_cast<int64_t>(src.numel()) * src.element_size() >= total * 16,
              "rows push: src must be a contiguous, 16-byte aligned CUDA tensor of rows x chunk_bytes");
  TORCH_CHECK(max_blocks >= 1 && max_blocks <= kMaxBlocks, "rows push: max_blocks must be 1 to 1024");
  Params p{};
  for (int64_t s = 0; s < world; ++s) {
    TORCH_CHECK(dsts[s] != 0 && dsts[s] % 16 == 0 && pads[s] != 0 && pads[s] % 4 == 0,
                "rows push: destinations 16-byte and pads 4-byte aligned");
    p.dst[s] = reinterpret_cast<uint4*>(dsts[s]);
    p.pad[s] = reinterpret_cast<uint32_t*>(pads[s]);
  }
  p.src = reinterpret_cast<const uint4*>(src.data_ptr());
  p.rows = static_cast<uint32_t>(rows);
  p.share = static_cast<uint32_t>(share);
  p.chunk = static_cast<uint32_t>(chunk);
  p.world = static_cast<uint32_t>(world);
  p.rank = static_cast<uint32_t>(rank);
  p.mode = static_cast<uint32_t>(mode);
  const int64_t grid = std::min(max_blocks, (total + int64_t(kThreads) * kUnroll - 1) / (int64_t(kThreads) * kUnroll));
  const c10::cuda::CUDAGuard guard(src.device());
  push_kernel<<<static_cast<unsigned>(grid), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grid;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pad_words", &pad_words, "int32 words of a rank's pad");
  m.def("push", &push, "one push of row-owned attention's exchange: queries to their rows' owners (mode 0) or outputs "
                       "to their heads' ranks (mode 1), then the group's device-side meeting");
}
#endif
