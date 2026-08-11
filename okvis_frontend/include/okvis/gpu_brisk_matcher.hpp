/*********************************************************************************
 * gpu_brisk_matcher.hpp
 *
 * GPU-accelerated "match to map" for the OKVIS frontend, tailored to Jetson
 * Thor's integrated GPU (unified memory, cudaDevAttrPageableMemoryAccess == 1).
 *
 * The CPU path (Frontend::matchToMapByThread) is a geometrically-gated sparse
 * double loop: for every visible map landmark, gate keypoints by reprojection
 * distance, then compare BRISK descriptors by Hamming distance and keep, per
 * keypoint, the single best (smallest-distance) landmark. This kernel computes
 * exactly that, one keypoint per thread.
 *
 * Because the iGPU can dereference ordinary pageable host pointers directly,
 * every input below is a plain CPU pointer (the descriptor pool, keypoint
 * coordinates, projections, ...) - no cudaMemcpy, no pinned/managed staging.
 * Outputs are plain host buffers written by the kernel and read back after a
 * device sync.
 *
 * Determinism vs the CPU: landmarks must be laid out in the SAME order the CPU
 * iterates them (ascending LandmarkId, i.e. std::map order). Combined with a
 * strict "<" update this reproduces the CPU's tie-breaking (first-seen wins),
 * so lmIdx per keypoint matches the CPU bit-for-bit.
 *
 * Coordinates and thresholds are double, not float. The geometric gate
 * (reprojection distance vs threshold) decides which descriptor comparisons
 * happen at all, so a float gate flips matches for keypoints sitting within
 * ~1e-5 px of the ~24 px threshold -- a silent estimator change rather than an
 * acceleration (risk R1 in docs/VIO_GPU_FRONTEND_PLAN.md). Doubles make the
 * gate, and therefore the match set, bit-identical to the CPU path. Hamming
 * distances are integers, so those comparisons are exact either way.
 *********************************************************************************/
#ifndef INCLUDE_OKVIS_GPU_BRISK_MATCHER_HPP_
#define INCLUDE_OKVIS_GPU_BRISK_MATCHER_HPP_

#include <cstdint>

namespace okvis {
namespace gpu {

/// \brief Flattened, device-readable inputs for one camera's match-to-map.
/// All pointers are ordinary host pointers (Thor unified memory) and must point
/// into buffers the frontend owns: the back-end BA thread writes the graph's
/// parameter blocks without any lock, so a kernel reading those directly would
/// be an untraceable data race (risk R5).
struct GpuMatchInput {
  // ---- current frame (query) ----
  const uint8_t* curDesc;    ///< N*48 BRISK descriptors, row k at curDesc + k*48.
  const double*  curKpXY;    ///< N*2 keypoint pixel coords (x,y interleaved).
  const uint8_t* useKp;      ///< N flags: 1 = try to match this keypoint, 0 = skip.
  int            numKeypoints;

  // ---- map landmarks (train), laid out in ascending LandmarkId order ----
  const double*  lmProjXY;   ///< M*2 landmark projections into the current frame.
  const uint8_t* poolBase;   ///< Contiguous descriptor pool base (rows of 48 B).
  const int*     lmDescOffRow;  ///< M: first descriptor row (in pool) of landmark j.
  const int*     lmDescRows;    ///< M: number of descriptor rows for landmark j (1..3).
  int            numLandmarks;

  // ---- gating / thresholds ----
  double reprThreshSq;       ///< Squared reprojection gate (pixels^2).
  double matchThresh;        ///< Max/initial Hamming distance (briskMatchingThreshold_).
};

/// \brief Run match-to-map on the GPU.
/// Outputs (host buffers, length numKeypoints, allocated by the caller):
///   bestLmIdx[k]   : index into the landmark arrays of the best match, or -1.
///   bestHamming[k] : Hamming distance of that best match (matchThresh if none).
///   reprSum[k]     : sum of the reprojection distances of every *improvement*
///                    of the running best for this keypoint.
///   reprCount[k]   : number of such improvements.
/// The last two exist because the CPU does not average the final best
/// reprojection error: it accumulates one term per improvement of the running
/// minimum (Frontend.cpp, `if (dist < distances[k])` -> `reprErrors[threadIdx]
/// += sqrt(...)`, `ctrs[threadIdx]++`) and divides by that count. reprErr in
/// turn decides numInitIter and whether RANSAC runs, so anything other than
/// this exact quantity changes estimator behaviour instead of accelerating it
/// (risk R2). Returning (sum, count) per keypoint lets the caller rebuild the
/// per-thread means the CPU would have produced.
///
/// kernelMs (optional) receives the CUDA-event measured kernel-only time, so
/// the caller can separate it from launch + synchronisation overhead.
/// Returns false if no CUDA device is usable (caller should fall back to CPU).
bool matchToMapGPU(const GpuMatchInput& in,
                   int* bestLmIdx,
                   double* bestHamming,
                   double* reprSum,
                   int* reprCount,
                   double* kernelMs = nullptr);

/// \brief Whether a usable CUDA device with pageable host access is present.
/// Cached after first call. When false, matchToMapGPU() is a no-op returning false.
bool gpuMatcherAvailable();

}  // namespace gpu
}  // namespace okvis

#endif  // INCLUDE_OKVIS_GPU_BRISK_MATCHER_HPP_
