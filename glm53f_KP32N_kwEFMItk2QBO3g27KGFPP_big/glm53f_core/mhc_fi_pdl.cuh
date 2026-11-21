


















#pragma once















#include "mhc_fi_kernels.cuh"

namespace flashinfer {
namespace mhc_fpp {

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }


template <typename Kernel>
inline cudaError_t launch_cluster_kernel_pdl(Kernel kernel, int threads, int smem_bytes, int num_clusters,
                                             const Params& p, const TensorMaps& maps, cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CL * num_clusters, 1, 1);
  cfg.blockDim = dim3(threads, 1, 1);
  cfg.dynamicSmemBytes = smem_bytes;
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
  return cudaLaunchKernelEx(&cfg, kernel, p, maps.x, maps.residual, maps.residual_cur);
}

}
}
