/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
* @file VioHealth.hpp
* @brief Process-wide counters for recurring estimator events.
*
* Events such as "RANSAC was triggered" or "tracking is weak" happen at frame
* rate. Logging each one costs more than it tells: on a headless field box the
* per-line detail is never read, while the thing an operator actually needs
* after the fact -- how often it happened and when it started -- is what gets
* buried. Counting them here lets the caller publish rates on its own cadence
* and keeps the per-event log lines behind VLOG.
*
* Producers and consumers sit on different threads: the frontend increments
* from the main processing thread, ViSlamBackend from the optimisation thread,
* and the application reads from whichever thread prints its status line. The
* counters are therefore atomic. They are monotonic for the life of the
* process; a reader that wants a rate keeps its own previous Snapshot and
* subtracts, which is also what makes several independent readers safe.
*/

#ifndef OKVIS_VIOHEALTH_HPP_
#define OKVIS_VIOHEALTH_HPP_

#include <atomic>
#include <cstdint>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// \brief A plain, non-atomic copy of the counters taken at one instant.
struct VioHealthSnapshot {
  uint64_t ransacTriggered = 0;  ///< Reprojection error over threshold, RANSAC run.
  uint64_t ransacFailed = 0;     ///< 3d2d RANSAC did not reach enough inliers.
  uint64_t ransacUninit = 0;     ///< Retry pass over uninitialised landmarks: first
                                 ///< RANSAC failed, or match-to-map found <=3 landmarks.
  uint64_t trackingWeak = 0;     ///< Frontend: quality under threshold, still >=3 matches.
  uint64_t trackingLost = 0;     ///< Frontend: quality under threshold, under 3 matches.
  uint64_t backendWeak = 0;      ///< Backend: tracking quality weak at marginalisation.
  uint64_t backendFailure = 0;   ///< Backend: tracking quality effectively zero.

  /// \brief Element-wise this - earlier, i.e. what happened in between.
  VioHealthSnapshot since(const VioHealthSnapshot& earlier) const {
    VioHealthSnapshot d;
    d.ransacTriggered = ransacTriggered - earlier.ransacTriggered;
    d.ransacFailed = ransacFailed - earlier.ransacFailed;
    d.ransacUninit = ransacUninit - earlier.ransacUninit;
    d.trackingWeak = trackingWeak - earlier.trackingWeak;
    d.trackingLost = trackingLost - earlier.trackingLost;
    d.backendWeak = backendWeak - earlier.backendWeak;
    d.backendFailure = backendFailure - earlier.backendFailure;
    return d;
  }

  /// \brief Whether anything at all was counted (used to keep quiet runs quiet).
  bool any() const {
    return (ransacTriggered | ransacFailed | ransacUninit | trackingWeak
            | trackingLost | backendWeak | backendFailure) != 0;
  }

  /// \brief Whether tracking was degraded, in the frontend or the backend.
  bool anyTrackingTrouble() const {
    return (trackingWeak | trackingLost | backendWeak | backendFailure) != 0;
  }
};

/// \brief The counters themselves. Increment with bump(), read with snapshot().
struct VioHealth {
  std::atomic<uint64_t> ransacTriggered{0};
  std::atomic<uint64_t> ransacFailed{0};
  std::atomic<uint64_t> ransacUninit{0};
  std::atomic<uint64_t> trackingWeak{0};
  std::atomic<uint64_t> trackingLost{0};
  std::atomic<uint64_t> backendWeak{0};
  std::atomic<uint64_t> backendFailure{0};

  /// \brief Read all counters. Not a consistent cut across counters -- an
  /// increment can land between two loads -- which only shifts an event
  /// between adjacent reporting windows and never loses it.
  VioHealthSnapshot snapshot() const {
    VioHealthSnapshot s;
    s.ransacTriggered = ransacTriggered.load(std::memory_order_relaxed);
    s.ransacFailed = ransacFailed.load(std::memory_order_relaxed);
    s.ransacUninit = ransacUninit.load(std::memory_order_relaxed);
    s.trackingWeak = trackingWeak.load(std::memory_order_relaxed);
    s.trackingLost = trackingLost.load(std::memory_order_relaxed);
    s.backendWeak = backendWeak.load(std::memory_order_relaxed);
    s.backendFailure = backendFailure.load(std::memory_order_relaxed);
    return s;
  }
};

/// \brief The one instance. All okvis libraries are static archives linked into
/// a single executable, so the function-local static is unique process-wide.
inline VioHealth& vioHealth() {
  static VioHealth instance;
  return instance;
}

/// \brief Increment one counter. Relaxed: the counters carry no other memory
/// with them, and readers only need eventual visibility.
inline void bump(std::atomic<uint64_t>& counter) {
  counter.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace okvis

#endif  // OKVIS_VIOHEALTH_HPP_
