

























#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <vector>

#include "mhc_fi_crossover.cuh"
#include "mhc_fi_decode_stream.cuh"
#include "mhc_fi_kernels.cuh"
#include "mhc_fi_long_stream.cuh"
#include "mhc_fi_short_stream.cuh"
#include "mhc_ar_decode.cuh"

namespace flashinfer {
namespace mhc_fpp {
namespace {

constexpr int kMaxDevices = 64;

struct DeviceState {
  std::atomic<bool> ready{false};
  ClusterLimits limits{0, 0, 0, 0, 0};
};
std::array<DeviceState, kMaxDevices> g_devices;
std::mutex g_prepare_mutex;

using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                   const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                   const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
std::atomic<EncodeTiledFn> g_encode_tiled{nullptr};

std::atomic<int64_t> g_tensor_map_hits{0};
std::atomic<int64_t> g_tensor_map_misses{0};
std::array<std::atomic<int64_t>, kNumVariants> g_launches{};



void encode_tensor_map(CUtensorMap* map, const void* ptr, int rank, int num_tokens, int box_rows) {
  cuuint64_t dims[3];
  cuuint64_t strides[2];
  cuuint32_t box[3];
  cuuint32_t elem_strides[3] = {1, 1, 1};
  if (rank == 2) {
    dims[0] = kHidden;
    dims[1] = static_cast<cuuint64_t>(num_tokens);
    strides[0] = static_cast<cuuint64_t>(kHidden) * 2;
    box[0] = kSliceWidth;
    box[1] = static_cast<cuuint32_t>(box_rows);
  } else {
    dims[0] = kHidden;
    dims[1] = kStreams;
    dims[2] = static_cast<cuuint64_t>(num_tokens);
    strides[0] = static_cast<cuuint64_t>(kHidden) * 2;
    strides[1] = static_cast<cuuint64_t>(kStreams) * kHidden * 2;
    box[0] = kSliceWidth;
    box[1] = kStreams;
    box[2] = static_cast<cuuint32_t>(box_rows);
  }
  const CUresult res = g_encode_tiled.load(std::memory_order_acquire)(
      map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<void*>(ptr), dims, strides, box,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "mhc_fi: cuTensorMapEncodeTiled failed with CUresult ",
              static_cast<int>(res));
}







struct TensorMapSlot {
  CUtensorMap map;
  uint64_t address;
  int32_t num_tokens;
  int16_t box_rows;
  int8_t rank;
  int8_t device;
  bool valid;
};
constexpr int kTensorMapSlots = 64;
struct TensorMapCache {
  TensorMapSlot slots[kTensorMapSlots];
};
thread_local std::unique_ptr<TensorMapCache> tls_tensor_maps;

CUtensorMap cached_tensor_map(const void* ptr, int rank, int num_tokens, int box_rows, int device) {
  if (!tls_tensor_maps) tls_tensor_maps = std::make_unique<TensorMapCache>();
  const uint64_t address = reinterpret_cast<uint64_t>(ptr);
  uint64_t h = address ^ (static_cast<uint64_t>(num_tokens) * 0x9E3779B97F4A7C15ull) ^
               (static_cast<uint64_t>(box_rows) << 3) ^ (static_cast<uint64_t>(rank) << 9) ^
               (static_cast<uint64_t>(device) << 11);
  h ^= h >> 29;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 32;
  TensorMapSlot& slot = tls_tensor_maps->slots[h & (kTensorMapSlots - 1)];
  if (slot.valid && slot.address == address && slot.num_tokens == num_tokens &&
      slot.box_rows == box_rows && slot.rank == rank && slot.device == device) {
    g_tensor_map_hits.fetch_add(1, std::memory_order_relaxed);
    return slot.map;
  }
  slot.valid = false;
  encode_tensor_map(&slot.map, ptr, rank, num_tokens, box_rows);
  slot.address = address;
  slot.num_tokens = num_tokens;
  slot.box_rows = static_cast<int16_t>(box_rows);
  slot.rank = static_cast<int8_t>(rank);
  slot.device = static_cast<int8_t>(device);
  slot.valid = true;
  g_tensor_map_misses.fetch_add(1, std::memory_order_relaxed);
  return slot.map;
}










constexpr int BAND_ROWS[2] = {385, 768};
constexpr Variant BAND_VARIANT = Variant::kCrossover;

LaunchPlan band_plan(int num_tokens, const ClusterLimits& limits) {
  if (num_tokens < BAND_ROWS[0] || num_tokens > BAND_ROWS[1]) return plan_launch(num_tokens, limits);
  const int capacity = BAND_VARIANT == Variant::kShortStream ? limits.short_stream
                       : BAND_VARIANT == Variant::kCrossover ? limits.crossover
                       : BAND_VARIANT == Variant::kLongStream ? limits.long_stream
                                                              : limits.stream;
  const int tiles = (num_tokens + kTileTokens - 1) / kTileTokens;

  if (capacity <= 0 || tiles <= limits.decode || tiles <= capacity) return {Variant::kUnsupported, 0, 0, 0};
  return {BAND_VARIANT, kTileTokens, tiles, capacity};
}

void check_cuda(cudaError_t err, const char* what) {
  TORCH_CHECK(err == cudaSuccess, "mhc_fi: ", what, " failed: ", cudaGetErrorString(err));
}

bool same_shape(const at::Tensor& t, std::initializer_list<int64_t> shape) {
  if (t.dim() != static_cast<int64_t>(shape.size())) return false;
  int64_t i = 0;
  for (int64_t s : shape) {
    if (t.size(i++) != s) return false;
  }
  return true;
}

uintptr_t address_of(const at::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()); }

bool overlaps(const at::Tensor& a, const at::Tensor& b) {
  const uintptr_t a0 = address_of(a), b0 = address_of(b);
  return a0 < b0 + static_cast<uintptr_t>(b.nbytes()) && b0 < a0 + static_cast<uintptr_t>(a.nbytes());
}



void check_operands(const std::array<const at::Tensor*, 8>& in, const std::array<const at::Tensor*, 4>& out) {
  const at::Tensor &x = *in[0], &residual = *in[1], &post_layer_mix = *in[2], &comb_res_mix = *in[3],
                   &fn = *in[4], &hc_scale = *in[5], &hc_base = *in[6], &norm_weight = *in[7];
  for (const at::Tensor* t : {&x, &residual, &norm_weight, out[0], out[3]})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16, "mhc_fi: x, residual, norm_weight, residual_cur and layer_input_cur are bf16");
  for (const at::Tensor* t : {&post_layer_mix, &comb_res_mix, &fn, &hc_scale, &hc_base, out[1], out[2]})
    TORCH_CHECK(t->scalar_type() == at::kFloat, "mhc_fi: the mixes, fn, hc_scale and hc_base are fp32");
  TORCH_CHECK(x.dim() == 2, "mhc_fi: x is [M, 4096]");
  const int64_t m = x.size(0);
  TORCH_CHECK(m > 0 && m <= (int64_t{1} << 24), "mhc_fi: token count ", m);
  TORCH_CHECK(same_shape(x, {m, kHidden}) && same_shape(residual, {m, kStreams, kHidden}) &&
                  same_shape(post_layer_mix, {m, kStreams}) &&
                  same_shape(comb_res_mix, {m, kStreams, kStreams}) &&
                  same_shape(fn, {kMixWidth, kStreams * kHidden}) && same_shape(hc_scale, {3}) &&
                  same_shape(hc_base, {kMixWidth}) && same_shape(norm_weight, {kHidden}) &&
                  same_shape(*out[0], {m, kStreams, kHidden}) && same_shape(*out[1], {m, kStreams}) &&
                  same_shape(*out[2], {m, kStreams, kStreams}) && same_shape(*out[3], {m, kHidden}),
              "mhc_fi: operand shapes (H 4096, four streams, 24 mixes)");
  for (const at::Tensor* t : in) {
    TORCH_CHECK(t->is_contiguous() && address_of(*t) % 16 == 0, "mhc_fi: inputs dense and 16-byte aligned");
  }
  for (const at::Tensor* t : out) {
    TORCH_CHECK(t->is_contiguous() && address_of(*t) % 16 == 0, "mhc_fi: outputs dense and 16-byte aligned");
  }
  TORCH_CHECK(address_of(fn) % 32 == 0, "mhc_fi: fn 32-byte aligned");

  for (int i = 0; i < 4; ++i) {
    for (const at::Tensor* t : in) TORCH_CHECK(!overlaps(*out[i], *t), "mhc_fi: an output aliases an input");
    for (int j = i + 1; j < 4; ++j) TORCH_CHECK(!overlaps(*out[i], *out[j]), "mhc_fi: outputs alias");
  }
  const c10::Device device = x.device();
  TORCH_CHECK(device.is_cuda() && device.index() >= 0 && device.index() < kMaxDevices, "mhc_fi: x on a CUDA device");
  for (const at::Tensor* t : in) TORCH_CHECK(t->device() == device, "mhc_fi: operands on one device");
  for (const at::Tensor* t : out) TORCH_CHECK(t->device() == device, "mhc_fi: operands on one device");
}

}




void prepare(int64_t device_id) {
  TORCH_CHECK(device_id >= 0 && device_id < kMaxDevices, "mhc_fi: device index out of range: ", device_id);
  DeviceState& state = g_devices[device_id];
  if (state.ready.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(g_prepare_mutex);
  if (state.ready.load(std::memory_order_acquire)) return;
  const c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device_id));
  if (g_encode_tiled.load(std::memory_order_acquire) == nullptr) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult query;
    check_cuda(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000,
                                                cudaEnableDefault, &query),
               "cudaGetDriverEntryPointByVersion(cuTensorMapEncodeTiled)");
    TORCH_CHECK(query == cudaDriverEntryPointSuccess && fn != nullptr,
                "mhc_fi: cuTensorMapEncodeTiled is not available");
    g_encode_tiled.store(reinterpret_cast<EncodeTiledFn>(fn), std::memory_order_release);
  }
  ClusterLimits limits{0, 0, 0, 0, 0};
  check_cuda(fpp_decode_stream::prepare_decode_stream(&limits.decode, &limits.stream), "decode/stream prepare");
  check_cuda(fpp_short_stream::prepare_short_stream(&limits.short_stream), "short-stream prepare");
  check_cuda(fpp_crossover::prepare_crossover(&limits.crossover), "crossover prepare");
  check_cuda(fpp_long_stream::prepare_long_stream(&limits.long_stream), "long-stream prepare");
  state.limits = limits;
  state.ready.store(true, std::memory_order_release);
}


std::vector<int64_t> limits(int64_t device_id) {
  TORCH_CHECK(device_id >= 0 && device_id < kMaxDevices, "mhc_fi: device index out of range: ", device_id);
  const DeviceState& state = g_devices[device_id];
  if (!state.ready.load(std::memory_order_acquire)) return {0, 0, 0, 0, 0, 0};
  const ClusterLimits& l = state.limits;
  return {1, l.decode, l.stream, l.short_stream, l.crossover, l.long_stream};
}


std::vector<int64_t> plan(int64_t num_tokens, int64_t decode, int64_t stream, int64_t short_stream,
                          int64_t crossover, int64_t long_stream) {
  const ClusterLimits limits{static_cast<int>(decode), static_cast<int>(stream),
                             static_cast<int>(short_stream), static_cast<int>(crossover),
                             static_cast<int>(long_stream)};
  const LaunchPlan p = band_plan(static_cast<int>(num_tokens), limits);
  return {static_cast<int64_t>(p.variant), p.tile_tokens, p.num_tiles, p.num_clusters};
}


std::vector<int64_t> counters() {
  std::vector<int64_t> out{g_tensor_map_hits.load(std::memory_order_relaxed),
                           g_tensor_map_misses.load(std::memory_order_relaxed)};
  for (const auto& n : g_launches) out.push_back(n.load(std::memory_order_relaxed));
  return out;
}




void run(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& post_layer_mix,
         const at::Tensor& comb_res_mix, const at::Tensor& fn, const at::Tensor& hc_scale,
         const at::Tensor& hc_base, const at::Tensor& norm_weight, const at::Tensor& residual_cur,
         const at::Tensor& post_mix_cur, const at::Tensor& comb_mix_cur, const at::Tensor& layer_input_cur,
         int64_t split_sinkhorn) {
  check_operands({&x, &residual, &post_layer_mix, &comb_res_mix, &fn, &hc_scale, &hc_base, &norm_weight},
                 {&residual_cur, &post_mix_cur, &comb_mix_cur, &layer_input_cur});
  const int64_t m = x.size(0);
  const int device = x.device().index();
  const DeviceState& state = g_devices[device];
  TORCH_CHECK(state.ready.load(std::memory_order_acquire), "mhc_fi: cuda:", device, " is not prepared");
  const LaunchPlan plan = band_plan(static_cast<int>(m), state.limits);
  TORCH_CHECK(plan.variant != Variant::kUnsupported, "mhc_fi: no kernel variant for ", m, " tokens on cuda:", device);
  TORCH_CHECK(split_sinkhorn == 0 || split_sinkhorn == 1, "mhc_fi: split_sinkhorn is 0 or 1");
  TORCH_CHECK(!split_sinkhorn || (plan.variant != Variant::kShortStream && plan.variant != Variant::kLongStream),
              "mhc_fi: the split Sinkhorn runs on the decode, stream and crossover plans, not variant ",
              static_cast<int>(plan.variant), " (", m, " rows)");


  const c10::cuda::CUDAGuard device_guard(x.device());
  TensorMaps maps;
  maps.x = cached_tensor_map(x.data_ptr(), 2, static_cast<int>(m), plan.tile_tokens, device);
  maps.residual = cached_tensor_map(residual.data_ptr(), 3, static_cast<int>(m), plan.tile_tokens, device);
  maps.residual_cur = cached_tensor_map(residual_cur.data_ptr(), 3, static_cast<int>(m), plan.tile_tokens, device);

  LaunchArgs args;
  args.x = x.data_ptr();
  args.residual = residual.data_ptr();
  args.post_layer_mix = post_layer_mix.data_ptr<float>();
  args.comb_res_mix = comb_res_mix.data_ptr<float>();
  args.fn = fn.data_ptr<float>();
  args.hc_scale = hc_scale.data_ptr<float>();
  args.hc_base = hc_base.data_ptr<float>();
  args.norm_weight = norm_weight.data_ptr();
  args.residual_cur = residual_cur.data_ptr();
  args.post_mix_cur = post_mix_cur.data_ptr<float>();
  args.comb_mix_cur = comb_mix_cur.data_ptr<float>();
  args.layer_input_cur = layer_input_cur.data_ptr();
  args.num_tokens = static_cast<int>(m);
  args.tile_tokens = plan.tile_tokens;
  args.num_tiles = plan.num_tiles;
  args.split_sinkhorn = static_cast<int>(split_sinkhorn);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  cudaError_t err = cudaSuccess;
  switch (plan.variant) {
    case Variant::kDecodeNarrow:
    case Variant::kDecodeWide:
      err = fpp_decode_stream::launch_decode(args, plan.num_clusters, maps, stream);
      break;
    case Variant::kStream:
    case Variant::kStreamLongRegs:
      err = fpp_decode_stream::launch_stream(args, plan.num_clusters,
                                             plan.variant == Variant::kStreamLongRegs, maps, stream);
      break;
    case Variant::kShortStream:
      err = fpp_short_stream::launch_short_stream(args, plan.num_clusters, maps, stream);
      break;
    case Variant::kCrossover:
      err = fpp_crossover::launch_crossover(args, plan.num_clusters, maps, stream);
      break;
    case Variant::kLongStream:
      err = fpp_long_stream::launch_long_stream(args, plan.num_clusters, maps, stream);
      break;
    default:
      TORCH_CHECK(false, "mhc_fi: no kernel variant for ", m, " tokens");
  }
  check_cuda(err, "kernel launch");
  g_launches[static_cast<int>(plan.variant)].fetch_add(1, std::memory_order_relaxed);
}





namespace {
constexpr int AR_ROWS[2] = {6, 80};
std::array<std::atomic<bool>, kMaxDevices> g_ar_ready{};
}


void prepare_ar(int64_t device_id) {
  prepare(device_id);
  if (g_ar_ready[device_id].load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(g_prepare_mutex);
  if (g_ar_ready[device_id].load(std::memory_order_acquire)) return;
  const c10::cuda::CUDAGuard device_guard(static_cast<c10::DeviceIndex>(device_id));
  check_cuda(fpp_ar_decode::prepare_ar_decode<true, true, false>(), "fused all-reduce prepare (narrow)");
  check_cuda(fpp_ar_decode::prepare_ar_decode<false, true, false>(), "fused all-reduce prepare (wide)");
  g_ar_ready[device_id].store(true, std::memory_order_release);
}





void run_ar(const at::Tensor& partial, const at::Tensor& residual, const at::Tensor& post_layer_mix,
            const at::Tensor& comb_res_mix, const at::Tensor& fn, const at::Tensor& hc_scale, const at::Tensor& hc_base,
            const at::Tensor& norm_weight, const at::Tensor& residual_cur, const at::Tensor& post_mix_cur,
            const at::Tensor& comb_mix_cur, const at::Tensor& layer_input_cur, const std::vector<int64_t>& slots,
            const at::Tensor& sync, int64_t rank, const at::Tensor& x_out, int64_t split_sinkhorn) {
  check_operands({&partial, &residual, &post_layer_mix, &comb_res_mix, &fn, &hc_scale, &hc_base, &norm_weight},
                 {&residual_cur, &post_mix_cur, &comb_mix_cur, &layer_input_cur});
  const int64_t m = partial.size(0);
  TORCH_CHECK(m >= AR_ROWS[0] && m <= AR_ROWS[1], "mhc_fi: the fused all-reduce serves ", AR_ROWS[0], " to ",
              AR_ROWS[1], " rows, not ", m);
  const int device = partial.device().index();
  TORCH_CHECK(g_ar_ready[device].load(std::memory_order_acquire), "mhc_fi: cuda:", device,
              " is not prepared for the fused all-reduce");
  const LaunchPlan plan = band_plan(static_cast<int>(m), g_devices[device].limits);
  TORCH_CHECK(plan.variant == Variant::kDecodeNarrow || plan.variant == Variant::kDecodeWide,
              "mhc_fi: the fused all-reduce runs on the decode plan, not variant ", static_cast<int>(plan.variant),
              " (", m, " rows)");
  TORCH_CHECK(static_cast<int64_t>(slots.size()) == fpp_ar_decode::kArRanks, "mhc_fi: one slot area a rank (4)");
  for (const int64_t s : slots) TORCH_CHECK(s != 0 && s % 16 == 0, "mhc_fi: slot areas 16-byte aligned");
  TORCH_CHECK(rank >= 0 && rank < fpp_ar_decode::kArRanks, "mhc_fi: rank ", rank, " outside the group of 4");
  TORCH_CHECK(sync.device() == partial.device() && sync.scalar_type() == at::kInt && sync.is_contiguous() &&
                  sync.numel() == 2,
              "mhc_fi: sync an int32 [parity, done] on the partial's device");
  const bool keep_x = x_out.numel() != 0;
  TORCH_CHECK(!keep_x || (x_out.device() == partial.device() && x_out.scalar_type() == at::kBFloat16 &&
                          x_out.is_contiguous() && same_shape(x_out, {m, kHidden}) && address_of(x_out) % 16 == 0 &&
                          !overlaps(x_out, partial)),
              "mhc_fi: x_out empty or a dense [M, 4096] BF16 apart from the partial");
  TORCH_CHECK(split_sinkhorn == 0 || split_sinkhorn == 1, "mhc_fi: split_sinkhorn is 0 or 1");

  const c10::cuda::CUDAGuard device_guard(partial.device());
  TensorMaps maps;
  maps.x = cached_tensor_map(partial.data_ptr(), 2, static_cast<int>(m), plan.tile_tokens, device);
  maps.residual = cached_tensor_map(residual.data_ptr(), 3, static_cast<int>(m), plan.tile_tokens, device);
  maps.residual_cur = cached_tensor_map(residual_cur.data_ptr(), 3, static_cast<int>(m), plan.tile_tokens, device);
  LaunchArgs args;
  args.x = partial.data_ptr();
  args.residual = residual.data_ptr();
  args.post_layer_mix = post_layer_mix.data_ptr<float>();
  args.comb_res_mix = comb_res_mix.data_ptr<float>();
  args.fn = fn.data_ptr<float>();
  args.hc_scale = hc_scale.data_ptr<float>();
  args.hc_base = hc_base.data_ptr<float>();
  args.norm_weight = norm_weight.data_ptr();
  args.residual_cur = residual_cur.data_ptr();
  args.post_mix_cur = post_mix_cur.data_ptr<float>();
  args.comb_mix_cur = comb_mix_cur.data_ptr<float>();
  args.layer_input_cur = layer_input_cur.data_ptr();
  args.num_tokens = static_cast<int>(m);
  args.tile_tokens = plan.tile_tokens;
  args.num_tiles = plan.num_tiles;
  args.split_sinkhorn = static_cast<int>(split_sinkhorn);
  fpp_ar_decode::ArParams ap{};
  ap.base = make_params(args);
  ap.ar.partial = reinterpret_cast<const uint4*>(partial.data_ptr());
  for (int s = 0; s < fpp_ar_decode::kArRanks; ++s) ap.ar.slots[s] = reinterpret_cast<uint4*>(slots[s]);
  ap.ar.epoch = reinterpret_cast<uint32_t*>(sync.data_ptr<int>());
  ap.ar.done = ap.ar.epoch + 1;
  ap.ar.rank = static_cast<uint32_t>(rank);
  ap.ar.x_out = keep_x ? reinterpret_cast<uint4*>(x_out.data_ptr()) : nullptr;
  ap.stamps = nullptr;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  check_cuda(fpp_ar_decode::launch_ar_decode<true, false>(ap, plan.num_clusters, maps, stream),
             "fused all-reduce launch");
}







namespace {
__global__ void __launch_bounds__(256) comb_sinkhorn_kernel(float* comb, int m) {
  const int gl = blockIdx.x * blockDim.x + threadIdx.x;
  const int tok = gl >> 2, i = gl & 3;
  const bool act = tok < m;
  float in[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (act) {
    const float4 v = *reinterpret_cast<const float4*>(comb + (size_t)tok * 16 + 4 * i);
    in[0] = v.x;
    in[1] = v.y;
    in[2] = v.z;
    in[3] = v.w;
  }
  float c[4] = {in[0], in[1], in[2], in[3]};
  bool bad = false;
  sinkhorn_rows<false, true>(c, bad);
  if (__any_sync(0xffffffffu, bad)) {
    c[0] = in[0];
    c[1] = in[1];
    c[2] = in[2];
    c[3] = in[3];
    sinkhorn_rows<true, true>(c, bad);
  }
  if (act) *reinterpret_cast<float4*>(comb + (size_t)tok * 16 + 4 * i) = make_float4(c[0], c[1], c[2], c[3]);
}
}

void sinkhorn(const at::Tensor& comb) {
  TORCH_CHECK(comb.is_cuda() && comb.scalar_type() == at::kFloat && comb.is_contiguous() && comb.numel() % 16 == 0 &&
                  address_of(comb) % 16 == 0,
              "mhc_fi: comb a dense fp32 [M, 4, 4] at a 16-byte address");
  const int64_t m = comb.numel() / 16;
  TORCH_CHECK(m >= 1 && m <= (int64_t{1} << 28), "mhc_fi: ", m, " comb rows");
  const c10::cuda::CUDAGuard device_guard(comb.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(comb.device().index()).stream();
  comb_sinkhorn_kernel<<<static_cast<unsigned>((4 * m + 255) / 256), 256, 0, stream>>>(comb.data_ptr<float>(),
                                                                                      static_cast<int>(m));
  check_cuda(cudaGetLastError(), "comb Sinkhorn launch");
}

}
}

#ifndef MHC_FI_NO_MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &flashinfer::mhc_fpp::prepare, "set the kernels' attributes and measure cluster capacity (outside capture)");
  m.def("limits", &flashinfer::mhc_fpp::limits, "[prepared, decode, stream, short, crossover, long] clusters on a device");
  m.def("plan", &flashinfer::mhc_fpp::plan, "regime selection: [variant, tile tokens, tiles, clusters]");
  m.def("counters", &flashinfer::mhc_fpp::counters, "[tensor-map hits, misses, launches per variant]");
  m.def("run", &flashinfer::mhc_fpp::run, "FlashInfer's fused mHC post -> pre + input RMSNorm, one launch "
        "(split_sinkhorn 1: comb gets the Sinkhorn's input logits, sinkhorn() finishes it)");
  m.def("prepare_ar", &flashinfer::mhc_fpp::prepare_ar, "prepare() and the fused all-reduce kernel's attributes");
  m.def("run_ar", &flashinfer::mhc_fpp::run_ar, "run() of the TP4 one-shot sum of a partial, summed in the kernel");
  m.def("sinkhorn", &flashinfer::mhc_fpp::sinkhorn, "the comb Sinkhorn in place on a split run's logits");
}
#endif
