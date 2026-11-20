




























#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <vector>

#include "mhc_fi_kernels.cuh"
#include "mhc_fp_long_stream.cuh"

namespace flashinfer {
namespace mhc_fpp {
namespace {

constexpr int kMaxDevices = 64;


constexpr int kLongMinTokens = 6245;
constexpr int kLongTilesPerCluster = 20;

struct DeviceState {
  std::atomic<bool> ready{false};
  int clusters = 0;
};
std::array<DeviceState, kMaxDevices> g_devices;
std::mutex g_prepare_mutex;

using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                   const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                   const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
std::atomic<EncodeTiledFn> g_encode_tiled{nullptr};



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
  TORCH_CHECK(res == CUDA_SUCCESS, "mhc_fp: cuTensorMapEncodeTiled failed with CUresult ",
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
  return slot.map;
}

void check_cuda(cudaError_t err, const char* what) {
  TORCH_CHECK(err == cudaSuccess, "mhc_fp: ", what, " failed: ", cudaGetErrorString(err));
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
    TORCH_CHECK(t->scalar_type() == at::kBFloat16, "mhc_fp: x, residual, norm_weight, residual_cur and layer_input_cur are bf16");
  for (const at::Tensor* t : {&post_layer_mix, &comb_res_mix, &fn, &hc_scale, &hc_base, out[1], out[2]})
    TORCH_CHECK(t->scalar_type() == at::kFloat, "mhc_fp: the mixes, fn, hc_scale and hc_base are fp32");
  TORCH_CHECK(x.dim() == 2, "mhc_fp: x is [M, 4096]");
  const int64_t m = x.size(0);
  TORCH_CHECK(m > 0 && m <= (int64_t{1} << 24), "mhc_fp: token count ", m);
  TORCH_CHECK(same_shape(x, {m, kHidden}) && same_shape(residual, {m, kStreams, kHidden}) &&
                  same_shape(post_layer_mix, {m, kStreams}) &&
                  same_shape(comb_res_mix, {m, kStreams, kStreams}) &&
                  same_shape(fn, {kMixWidth, kStreams * kHidden}) && same_shape(hc_scale, {3}) &&
                  same_shape(hc_base, {kMixWidth}) && same_shape(norm_weight, {kHidden}) &&
                  same_shape(*out[0], {m, kStreams, kHidden}) && same_shape(*out[1], {m, kStreams}) &&
                  same_shape(*out[2], {m, kStreams, kStreams}) && same_shape(*out[3], {m, kHidden}),
              "mhc_fp: operand shapes (H 4096, four streams, 24 mixes)");
  for (const at::Tensor* t : in) {
    TORCH_CHECK(t->is_contiguous() && address_of(*t) % 16 == 0, "mhc_fp: inputs dense and 16-byte aligned");
  }
  for (const at::Tensor* t : out) {
    TORCH_CHECK(t->is_contiguous() && address_of(*t) % 16 == 0, "mhc_fp: outputs dense and 16-byte aligned");
  }
  TORCH_CHECK(address_of(fn) % 32 == 0, "mhc_fp: fn 32-byte aligned");

  for (int i = 0; i < 4; ++i) {
    for (const at::Tensor* t : in) TORCH_CHECK(!overlaps(*out[i], *t), "mhc_fp: an output aliases an input");
    for (int j = i + 1; j < 4; ++j) TORCH_CHECK(!overlaps(*out[i], *out[j]), "mhc_fp: outputs alias");
  }
  const c10::Device device = x.device();
  TORCH_CHECK(device.is_cuda() && device.index() >= 0 && device.index() < kMaxDevices, "mhc_fp: x on a CUDA device");
  for (const at::Tensor* t : in) TORCH_CHECK(t->device() == device, "mhc_fp: operands on one device");
  for (const at::Tensor* t : out) TORCH_CHECK(t->device() == device, "mhc_fp: operands on one device");
}

}




void prepare(int64_t device_id) {
  TORCH_CHECK(device_id >= 0 && device_id < kMaxDevices, "mhc_fp: device index out of range: ", device_id);
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
                "mhc_fp: cuTensorMapEncodeTiled is not available");
    g_encode_tiled.store(reinterpret_cast<EncodeTiledFn>(fn), std::memory_order_release);
  }
  int clusters = 0;
  check_cuda(fpp_long_peer::prepare_long_peer(&clusters), "fused-peer long-stream prepare");
  state.clusters = clusters;
  state.ready.store(true, std::memory_order_release);
}


std::vector<int64_t> limits(int64_t device_id) {
  TORCH_CHECK(device_id >= 0 && device_id < kMaxDevices, "mhc_fp: device index out of range: ", device_id);
  const DeviceState& state = g_devices[device_id];
  if (!state.ready.load(std::memory_order_acquire)) return {0, 0, fpp_long_peer::BG_SMEM_BYTES};
  return {1, state.clusters, fpp_long_peer::BG_SMEM_BYTES};
}






void run_peer(const at::Tensor& mine, int64_t peer, const at::Tensor& residual, const at::Tensor& post_layer_mix,
              const at::Tensor& comb_res_mix, const at::Tensor& fn, const at::Tensor& hc_scale,
              const at::Tensor& hc_base, const at::Tensor& norm_weight, const at::Tensor& residual_cur,
              const at::Tensor& post_mix_cur, const at::Tensor& comb_mix_cur, const at::Tensor& layer_input_cur,
              const at::Tensor& flags, int64_t peer_flags, const at::Tensor& published, int64_t epoch) {
  check_operands({&mine, &residual, &post_layer_mix, &comb_res_mix, &fn, &hc_scale, &hc_base, &norm_weight},
                 {&residual_cur, &post_mix_cur, &comb_mix_cur, &layer_input_cur});
  TORCH_CHECK(flags.scalar_type() == at::kInt && flags.numel() >= 1 && flags.device() == mine.device() &&
                  published.scalar_type() == at::kInt && published.numel() >= 1 &&
                  published.device() == mine.device(),
              "mhc_fp: the slot flags and the published epoch are int32 on the partial's device");
  TORCH_CHECK(peer != 0 && peer % 16 == 0 && peer_flags != 0 && peer_flags % 4 == 0 && epoch >= 1 &&
                  epoch <= INT32_MAX,
              "mhc_fp: the peer's slot (16-byte aligned), its flags and the epoch");
  const int64_t m = mine.size(0);
  const int device = mine.device().index();
  const DeviceState& state = g_devices[device];
  TORCH_CHECK(state.ready.load(std::memory_order_acquire), "mhc_fp: cuda:", device, " is not prepared");
  const int tiles = static_cast<int>((m + kTileTokens - 1) / kTileTokens);
  const int clusters = std::min(tiles, state.clusters);
  TORCH_CHECK(m >= kLongMinTokens && clusters >= 1 && tiles >= kLongTilesPerCluster * clusters,
              "mhc_fp: ", m, " tokens on ", state.clusters, " clusters are not FlashInfer's long-stream regime");


  const c10::cuda::CUDAGuard device_guard(mine.device());
  TensorMaps maps;
  maps.x = cached_tensor_map(mine.data_ptr(), 2, static_cast<int>(m), kTileTokens, device);
  maps.residual = cached_tensor_map(residual.data_ptr(), 3, static_cast<int>(m), kTileTokens, device);
  maps.residual_cur = cached_tensor_map(residual_cur.data_ptr(), 3, static_cast<int>(m), kTileTokens, device);
  const CUtensorMap peer_map =
      cached_tensor_map(reinterpret_cast<const void*>(peer), 2, static_cast<int>(m), kTileTokens, device);

  LaunchArgs args;
  args.x = mine.data_ptr();
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
  args.tile_tokens = kTileTokens;
  args.num_tiles = tiles;

  fpp_long_peer::PeerSync sync;
  sync.flags = flags.data_ptr<int>();
  sync.peer_flags = reinterpret_cast<int*>(peer_flags);
  sync.published = published.data_ptr<int>();
  sync.epoch = static_cast<int>(epoch);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
  check_cuda(fpp_long_peer::launch_long_peer(args, clusters, maps, peer_map, sync, stream), "kernel launch");
}

}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &flashinfer::mhc_fpp::prepare, "set the kernel's attributes and measure its cluster capacity (outside capture)");
  m.def("limits", &flashinfer::mhc_fpp::limits, "[prepared, co-resident 16-CTA clusters, shared memory bytes] on a device");
  m.def("run_peer", &flashinfer::mhc_fpp::run_peer,
        "FlashInfer's fused mHC post -> pre + input RMSNorm on this rank's partial + the peer's, under E31's slot "
        "handshake, one launch (glm53flash)");
}
