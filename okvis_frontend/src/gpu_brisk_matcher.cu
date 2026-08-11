/*********************************************************************************
 * gpu_brisk_matcher.cu  -- see gpu_brisk_matcher.hpp for the contract.
 *
 * One CUDA thread per query keypoint. Each thread scans all candidate landmarks
 * (ascending-LandmarkId order), gates by squared reprojection distance, and for
 * the surviving landmarks compares the 48-byte BRISK descriptor by Hamming
 * distance (popcount of XOR over 12 uint32 words), keeping the single smallest.
 * Strict "<" reproduces the CPU's first-seen tie-breaking.
 *
 * Each thread also accumulates the reprojection distance of every improvement of
 * its running best, in the same order the CPU visits them, because that -- not
 * the final best -- is what feeds reprErr and hence the RANSAC decision.
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
                                 double* __restrict__ bestHamming,
                                 double* __restrict__ reprSum,
                                 int* __restrict__ reprCount) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= in.numKeypoints) return;

  if (!in.useKp[k]) {
    bestLmIdx[k] = -1;
    bestHamming[k] = in.matchThresh;
    reprSum[k] = 0.0;
    reprCount[k] = 0;
    return;
  }

  const double kx = in.curKpXY[2 * k + 0];
  const double ky = in.curKpXY[2 * k + 1];
  const uint8_t* dk = in.curDesc + static_cast<size_t>(k) * 48;

  double best = in.matchThresh;  // strict "<" below => needs to beat the threshold
  int bestLm = -1;
  double sum = 0.0;
  int cnt = 0;

  for (int j = 0; j < in.numLandmarks; ++j) {
    const double dx = in.lmProjXY[2 * j + 0] - kx;
    const double dy = in.lmProjXY[2 * j + 1] - ky;
    const double r2 = dx * dx + dy * dy;
    if (r2 > in.reprThreshSq) continue;  // geometric gate

    const int off = in.lmDescOffRow[j];
    const int rows = in.lmDescRows[j];
    for (int d = 0; d < rows; ++d) {
      const int dist =
          hamming48(dk, in.poolBase + static_cast<size_t>(off + d) * 48);
      if (static_cast<double>(dist) < best) {
        best = static_cast<double>(dist);
        bestLm = j;
        // One term per improvement, exactly as the CPU inner loop does.
        sum += sqrt(r2);
        ++cnt;
      }
    }
  }

  bestLmIdx[k] = bestLm;
  bestHamming[k] = best;
  reprSum[k] = sum;
  reprCount[k] = cnt;
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

// Created once and reused: creating and destroying a pair of events per call is
// several microseconds of pure measurement overhead on a ~0.2 ms kernel.
struct TimingEvents {
  cudaEvent_t start{}, stop{};
  bool ok{false};
  TimingEvents() {
    ok = cudaEventCreate(&start) == cudaSuccess &&
         cudaEventCreate(&stop) == cudaSuccess;
  }
};

}  // namespace

bool gpuMatcherAvailable() {
  static const int available = probeAvailability();
  return available != 0;
}

bool matchToMapGPU(const GpuMatchInput& in,
                   int* bestLmIdx,
                   double* bestHamming,
                   double* reprSum,
                   int* reprCount,
                   double* kernelMsOut) {
  if (kernelMsOut) *kernelMsOut = 0.0;
  if (!gpuMatcherAvailable()) return false;
  if (in.numKeypoints <= 0) return true;  // nothing to do, trivially "done"

  static TimingEvents ev;

  const int block = 128;
  const int grid = (in.numKeypoints + block - 1) / block;

  if (ev.ok) cudaEventRecord(ev.start);
  matchToMapKernel<<<grid, block>>>(in, bestLmIdx, bestHamming, reprSum,
                                    reprCount);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[gpu_brisk_matcher] launch failed: %s\n",
                 cudaGetErrorString(err));
    return false;
  }
  if (ev.ok) {
    cudaEventRecord(ev.stop);
    err = cudaEventSynchronize(ev.stop);
  } else {
    err = cudaDeviceSynchronize();
  }
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[gpu_brisk_matcher] sync failed: %s\n",
                 cudaGetErrorString(err));
    return false;
  }
  float kernelMs = 0.f;
  if (ev.ok) cudaEventElapsedTime(&kernelMs, ev.start, ev.stop);
  if (kernelMsOut) *kernelMsOut = double(kernelMs);

  // Lightweight diagnostics: confirm the GPU path is live and report kernel-only
  // time vs the CPU-side flatten (the caller times each stage separately).
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
