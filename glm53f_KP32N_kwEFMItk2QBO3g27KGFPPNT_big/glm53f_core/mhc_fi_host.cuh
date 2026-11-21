

















#pragma once




#include <cuda.h>
#include <cuda_runtime.h>

namespace flashinfer {
namespace mhc_fpp {

constexpr int kHidden = 4096;
constexpr int kStreams = 4;
constexpr int kMixWidth = 24;
constexpr int kClusterSize = 16;
constexpr int kSliceWidth = kHidden / kClusterSize;
constexpr int kTileTokens = 16;


struct LaunchArgs {
  const void* x;
  const void* residual;
  const float* post_layer_mix;
  const float* comb_res_mix;
  const float* fn;
  const float* hc_scale;
  const float* hc_base;
  const void* norm_weight;
  void* residual_cur;
  float* post_mix_cur;
  float* comb_mix_cur;
  void* layer_input_cur;
  int num_tokens;
  int tile_tokens;
  int num_tiles;

  int split_sinkhorn = 0;
};


struct TensorMaps {
  CUtensorMap x;
  CUtensorMap residual;
  CUtensorMap residual_cur;
};




enum class Variant : int {
  kUnsupported = -1,
  kDecodeNarrow = 0,
  kDecodeWide = 1,
  kStream = 2,
  kStreamLongRegs = 3,
  kShortStream = 4,
  kCrossover = 5,
  kLongStream = 6,
};
constexpr int kNumVariants = 7;



struct ClusterLimits {
  int decode;
  int stream;
  int short_stream;
  int crossover;
  int long_stream;
};

struct LaunchPlan {
  Variant variant;
  int tile_tokens;
  int num_tiles;
  int num_clusters;
};

inline LaunchPlan plan_launch(int num_tokens, const ClusterLimits& limits) {
  const LaunchPlan unsupported{Variant::kUnsupported, 0, 0, 0};
  if (num_tokens <= 0 || limits.decode <= 0) return unsupported;
  const int m = num_tokens;
  const int tiles16 = (m + kTileTokens - 1) / kTileTokens;
  const bool long_regime = m >= 6245;
  const bool crossover_regime = m >= 1800 && m <= 2048;
  const bool short_regime = m >= 1024 && m < 1800;
  if (!long_regime && !crossover_regime && !short_regime) {
    if (tiles16 <= limits.decode) {

      int tm = tiles16 >= limits.decode ? kTileTokens : (m + limits.decode - 1) / limits.decode;
      if (tm < 1) tm = 1;
      if (tm > kTileTokens) tm = kTileTokens;
      const int tiles = (m + tm - 1) / tm;
      const int clusters = tiles < limits.decode ? tiles : limits.decode;
      return {tm <= 8 ? Variant::kDecodeNarrow : Variant::kDecodeWide, tm, tiles, clusters};
    }
    if (limits.stream <= 0) return unsupported;
    const int clusters = tiles16 < limits.stream ? tiles16 : limits.stream;
    const bool long_regs = tiles16 >= 20 * clusters;
    return {long_regs ? Variant::kStreamLongRegs : Variant::kStream, kTileTokens, tiles16,
            clusters};
  }



  if (tiles16 <= limits.decode) return unsupported;
  const int capacity = long_regime ? limits.long_stream
                                   : (crossover_regime ? limits.crossover : limits.short_stream);
  if (capacity <= 0) return unsupported;
  const int clusters = tiles16 < capacity ? tiles16 : capacity;
  const bool long_regs = tiles16 >= (crossover_regime ? 32 : 20) * clusters;
  if (long_regs != long_regime) return unsupported;
  return {long_regime ? Variant::kLongStream
                      : (crossover_regime ? Variant::kCrossover : Variant::kShortStream),
          kTileTokens, tiles16, clusters};
}

}
}
