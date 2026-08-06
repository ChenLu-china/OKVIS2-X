/*********************************************************************************
 * gpu_brisk_matcher.cu  -- see gpu_brisk_matcher.hpp for the contract.
 *
 * One CUDA thread per query keypoint. Each thread scans all candidate landmarks
 * (ascending-LandmarkId order), gates by squared reprojection distance, and for
 * the surviving landmarks compares the 48-byte BRISK descriptor by Hamming
 * distance (popcount of XOR over 12 uint32 words), keeping the single smallest.
 * Strict "<" reproduces the CPU's first-seen tie-breaking.
 *
 * All inputs are ordinary host pointers: on Thor (integrated GPU,
 * pageableMemoryAccess == 1) the device dereferences them directly, so there is
 * no staging copy. Outputs are caller-owned host buffers written by the kernel.
 *********************************************************************************/
#include "okvis/gpu_brisk_matcher.hpp"

#include <cuda_runtime.h>

#include <cstdio>

namespace okvis {
namespace gpu {

namespace {

// Hamming distance over a 48-byte (384-bit) BRISK descriptor = 12 uint32 words.
__device__ __forceinline__ int hamming48(const uint8_t* a, const uint8_t* b) {
  const uint32_t* ua = reinterpret_cast<const uint32_t*>(a);
  const uint32_t* ub = reinterpret_cast<const uint32_t*>(b);
  int d = 0;
#pragma unroll
  for (int i = 0; i < 12; ++i) {
    d += __popc(ua[i] ^ ub[i]);
  }
  return d;
}

__global__ void matchToMapKernel(GpuMatchInput in,
                                 int* __restrict__ bestLmIdx,
                                 float* __restrict__ bestHamming,
                                 float* __restrict__ bestReprPix) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= in.numKeypoints) return;

  if (!in.useKp[k]) {
    bestLmIdx[k] = -1;
    bestHamming[k] = in.matchThresh;
    bestReprPix[k] = 0.0f;
    return;
  }

  const float kx = in.curKpXY[2 * k + 0];
  const float ky = in.curKpXY[2 * k + 1];
  const uint8_t* dk = in.curDesc + static_cast<size_t>(k) * 48;

  float best = in.matchThresh;  // strict "<" below => needs to beat the threshold
  int bestLm = -1;
  float bestRepr = 0.0f;

  for (int j = 0; j < in.numLandmarks; ++j) {
    const float dx = in.lmProjXY[2 * j + 0] - kx;
    const float dy = in.lmProjXY[2 * j + 1] - ky;
    const float r2 = dx * dx + dy * dy;
    if (r2 > in.reprThreshSq) continue;  // geometric gate

    const int off = in.lmDescOffRow[j];
    const int rows = in.lmDescRows[j];
    for (int d = 0; d < rows; ++d) {
      const int dist =
          hamming48(dk, in.poolBase + static_cast<size_t>(off + d) * 48);
      if (static_cast<float>(dist) < best) {
        best = static_cast<float>(dist);
        bestLm = j;
        bestRepr = sqrtf(r2);
      }
    }
  }

  bestLmIdx[k] = bestLm;
  bestHamming[k] = best;
  bestReprPix[k] = bestRepr;
}

// Probe once whether we have a usable integrated-GPU with pageable host access.
int probeAvailability() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 0;
  int pageable = 0;
  if (cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0) !=
      cudaSuccess) {
    return 0;
  }
  // Pageable host access is what lets the kernel read cv::Mat/std::vector
  // buffers directly. Without it this design would need explicit staging.
  return pageable ? 1 : 0;
}

}  // namespace

bool gpuMatcherAvailable() {
  static const int available = probeAvailability();
  return available != 0;
}

bool matchToMapGPU(const GpuMatchInput& in,
                   int* bestLmIdx,
                   float* bestHamming,
                   float* bestReprPix) {
  if (!gpuMatcherAvailable()) return false;
  if (in.numKeypoints <= 0) return true;  // nothing to do, trivially "done"

  const int block = 128;
  const int grid = (in.numKeypoints + block - 1) / block;

  cudaEvent_t evStart, evStop;
  cudaEventCreate(&evStart);
  cudaEventCreate(&evStop);
  cudaEventRecord(evStart);
  matchToMapKernel<<<grid, block>>>(in, bestLmIdx, bestHamming, bestReprPix);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[gpu_brisk_matcher] launch failed: %s\n",
                 cudaGetErrorString(err));
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    return false;
  }
  cudaEventRecord(evStop);
  err = cudaEventSynchronize(evStop);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[gpu_brisk_matcher] sync failed: %s\n",
                 cudaGetErrorString(err));
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    return false;
  }
  float kernelMs = 0.f;
  cudaEventElapsedTime(&kernelMs, evStart, evStop);
  cudaEventDestroy(evStart);
  cudaEventDestroy(evStop);

  // Lightweight diagnostics: confirm the GPU path is live and report kernel-only
  // time vs the CPU-side flatten (the caller measures the whole 2.01 stage).
  static double accMs = 0.0;
  static long nCalls = 0;
  accMs += kernelMs;
  ++nCalls;
  if (nCalls == 1 || (nCalls % 200) == 0) {
    std::fprintf(stderr,
                 "[gpu_brisk_matcher] call#%ld N=%d M=%d kernel=%.3fms avg=%.3fms\n",
                 nCalls, in.numKeypoints, in.numLandmarks, kernelMs,
                 accMs / double(nCalls));
  }
  return true;
}

}  // namespace gpu
}  // namespace okvis
