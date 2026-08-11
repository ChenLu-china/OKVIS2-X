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
 * @file Frontend.hpp
 * @brief Header file for the Frontend class.
 * @author Andreas Forster
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_FRONTEND_HPP_
#define INCLUDE_OKVIS_FRONTEND_HPP_

#include <algorithm>
#include <cstdint>
#include <mutex>

#include <okvis/Component.hpp>
#include <okvis/ViFrontendInterface.hpp>
#include <okvis/ViSlamBackend.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/timing/Timer.hpp>
#include <thread>

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * @brief A frontend using BRISK features
 */
class Frontend : public ViFrontendInterface {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Constructor.
   * @param numCameras Number of cameras in the sensor configuration.
   * @param dBowVocDir The directory containing the DBoW vocabulary.
   */
  Frontend(size_t numCameras, std::string dBowVocDir);
  virtual ~Frontend() override;

  /**
   * @brief Load another, previously saved VI-SLAM component.
   * @param filename Filename.
   * @param imuParameters Imu parameters of the loaded component.
   * @param nCameraSystem Multi-camera configuration of the loaded component.
   * @param componentFixed Should the component be treated as fixed?
   * @return True on success.
   */
  bool loadComponent(std::string filename,
                     const ImuParameters &imuParameters,
                     const cameras::NCameraSystem &nCameraSystem,
                     bool componentFixed = true);

  ///@{
  /**
   * @brief Detection and descriptor extraction on a per image basis.
   * @remark This method is threadsafe.
   * @param cameraIndex Index of camera to do detection and description.
   * @param frameOut    Multiframe containing the frames.
   *                    Resulting keypoints and descriptors are saved in here.
   * @param T_WC        Pose of camera with index cameraIndex at image capture time.
   * @param[in] keypoints If the keypoints are already available from a different source, provide
   *                      them here in order to skip detection.
   * @warning Using keypoints from a different source is not yet implemented.
   * @return True if successful.
   */
  virtual bool detectAndDescribe(size_t cameraIndex,
                                 std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint> * keypoints) override final;

  /**
   * @brief Matching as well as initialization of landmarks and state.
   * @warning This method is not threadsafe.
   * @warning This method uses the estimator. Make sure to not access it in another thread.
   * @param estimator       Estimator.
   * @param params          Configuration parameters.
   * @param framesInOut     Multiframe including the descriptors of all the keypoints.
   * @param kfPrior         A prior to trigger new keyframe from other criteria (e.g. LiDAR overlap)
   * @param[out] asKeyframe Should the frame be a keyframe?
   * @return True if successful.
   */
  virtual bool dataAssociationAndInitialization(
      Estimator& estimator,
      const okvis::ViParameters & params,
      std::shared_ptr<okvis::MultiFrame> framesInOut, bool kfPrior, bool* asKeyframe) override final;

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @see okvis::ceres::ImuError::propagation()
   * @remark This method is threadsafe.
   * @param[in] imuMeasurements All the IMU measurements.
   * @param[in] imuParams The parameters to be used.
   * @param[inout] T_WS_propagated Start pose.
   * @param[inout] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @param[out] covariance Covariance for GIVEN start states.
   * @param[out] jacobian Jacobian w.r.t. start states.
   * @return True on success.
   */
  virtual bool propagation(const okvis::ImuMeasurementDeque & imuMeasurements,
                           const okvis::ImuParameters & imuParams,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBias & speedAndBiases,
                           const okvis::Time& t_start, const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const override final;

  ///@}
  /// @name Getters related to the BRISK detector
  /// @{

  /// @brief Get the number of octaves of the BRISK detector.
  size_t getBriskDetectionOctaves() const {
    return briskDetectionOctaves_;
  }

  /// @brief Get the detection threshold of the BRISK detector.
  double getBriskDetectionThreshold() const {
    return briskDetectionThreshold_;
  }

  /// @brief Get the absolute threshold of the BRISK detector.
  double getBriskDetectionAbsoluteThreshold() const {
    return briskDetectionAbsoluteThreshold_;
  }

  /// @brief Get the maximum amount of keypoints of the BRISK detector.
  size_t getBriskDetectionMaximumKeypoints() const {
    return briskDetectionMaximumKeypoints_;
  }

  ///@}
  /// @name Getters related to the BRISK descriptor
  /// @{

  /// @brief Get the rotation invariance setting of the BRISK descriptor.
  bool getBriskDescriptionRotationInvariance() const {
    return briskDescriptionRotationInvariance_;
  }

  /// @brief Get the scale invariance setting of the BRISK descriptor.
  bool getBriskDescriptionScaleInvariance() const {
    return briskDescriptionScaleInvariance_;
  }

  ///@}
  /// @name Other getters
  /// @{

  /// @brief Get the matching threshold.
  double getBriskMatchingThreshold() const {
    return briskMatchingThreshold_;
  }

  /// @brief Get the area overlap threshold under which a new keyframe is inserted.
  float getKeyframeInsertionOverlapThershold() const {
    return keyframeInsertionOverlapThreshold_;
  }

  /// @brief Returns true if the initialization has been completed (RANSAC with actual translation)
  bool isInitialized() {
    return isInitialized_;
  }

  /// @}
  /// @name Setters related to the BRISK detector
  /// @{

  /// @brief Set the number of octaves of the BRISK detector.
  void setBriskDetectionOctaves(size_t octaves) {
    briskDetectionOctaves_ = octaves;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the detection threshold of the BRISK detector.
  void setBriskDetectionThreshold(double threshold) {
    briskDetectionThreshold_ = threshold;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the absolute threshold of the BRISK detector.
  void setBriskDetectionAbsoluteThreshold(double threshold) {
    briskDetectionAbsoluteThreshold_ = threshold;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the maximum number of keypoints of the BRISK detector.
  void setBriskDetectionMaximumKeypoints(size_t maxKeypoints) {
    briskDetectionMaximumKeypoints_ = maxKeypoints;
    initialiseBriskFeatureDetectors();
  }

  /// @}
  /// @name Setters related to the BRISK descriptor
  /// @{

  /// @brief Set the rotation invariance setting of the BRISK descriptor.
  void setBriskDescriptionRotationInvariance(bool invariance) {
    briskDescriptionRotationInvariance_ = invariance;
    initialiseBriskFeatureDetectors();
  }

  /// @brief Set the scale invariance setting of the BRISK descriptor.
  void setBriskDescriptionScaleInvariance(bool invariance) {
    briskDescriptionScaleInvariance_ = invariance;
    initialiseBriskFeatureDetectors();
  }

  ///@}
  /// @name Other setters
  /// @{

  /// @brief Set the matching threshold.
  void setBriskMatchingThreshold(double threshold) {
    briskMatchingThreshold_ = threshold;
  }

  /// @brief Set the area overlap threshold under which a new keyframe is inserted.
  void setKeyframeInsertionOverlapThreshold(float threshold) {
    keyframeInsertionOverlapThreshold_ = threshold;
  }

  /// @}

  /// \brief Stop all CNN background threads.
  void endCnnThreads();

  /// \brief Clears and resets everything (so you can re-start).
  void clear();

private:

  /**
   * @brief   feature detectors with the current settings.
   *          The vector contains one for each camera to ensure that there are no problems with
   *          parallel detection.
   * @warning Lock with featureDetectorMutexes_[cameraIndex] when using the detector.
   */
  std::vector<std::shared_ptr<cv::FeatureDetector> > featureDetectors_;
  /**
   * @brief   feature descriptors with the current settings.
   *          The vector contains one for each camera to ensure that there are no problems with
   *          parallel detection.
   * @warning Lock with featureDetectorMutexes_[cameraIndex] when using the descriptor.
   */
  std::vector<std::shared_ptr<cv::DescriptorExtractor> > descriptorExtractors_;
  /// Mutexes for feature detectors and descriptors.
  std::vector<std::unique_ptr<std::mutex> > featureDetectorMutexes_;

  bool isInitialized_;        ///< Is the pose initialised?
  const size_t numCameras_;   ///< Number of cameras in the configuration.

  /// @name BRISK detection parameters
  /// @{

  size_t briskDetectionOctaves_;            ///< The set number of brisk octaves.
  double briskDetectionThreshold_;          ///< The set BRISK detection threshold.
  double briskDetectionAbsoluteThreshold_;  ///< The set BRISK absolute detection threshold.
  size_t briskDetectionMaximumKeypoints_;   ///< The set maximum number of keypoints.

  /// @}
  /// @name BRISK descriptor extractor parameters
  /// @{

  bool briskDescriptionRotationInvariance_; ///< The set rotation invariance setting.
  bool briskDescriptionScaleInvariance_;    ///< The set scale invariance setting.

  ///@}
  /// @name BRISK matching parameters
  ///@{

  double briskMatchingThreshold_; ///< The set BRISK matching threshold.

  /// \brief Landmark snapshot reused across matchToMap calls (OKVIS_MTM_LMSYNC!=0, the default).
  MapPoints cachedPointMap_;

  /// \brief Flat landmark snapshot reused across matchToMap calls (OKVIS_SOA!=0).
  MapPointsSoA soaPointMap_;

  ///@}

  /**
   * @brief If the hull-area around all matched keypoints of the current frame (with existing
   *        landmarks)
   *        divided by the hull-area around all keypoints in the current frame is lower than
   *        this threshold it should be a new keyframe.
   * @see   doWeNeedANewKeyframe()
   */
  float keyframeInsertionOverlapThreshold_;  //0.55

  /**
   * @brief Decision whether a new frame should be keyframe or not, based on overlap heuristic.
   * @param estimator     const reference to the estimator.
   * @param currentFrame  Keyframe candidate.
   * @return True if it should be a new keyframe.
   */
  bool doWeNeedANewKeyframe(const Estimator& estimator,
                            std::shared_ptr<okvis::MultiFrame> currentFrame);

  /**
   * @brief Match a new multiframe to existing keyframes
   * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing landmarks
   * @warning As this function uses the estimator it is not threadsafe
   * @param      estimator              Estimator.
   * @param[in]  params                 Parameter struct.
   * @param[in]  currentFrameId         ID of the current frame that should be matched against
   *                                    keyframes.
   * @param[in]  loopClosureLandmarksToUseExclusively Use these landmarks exclusively, if supplied.
   * @return The number of matches in total.
   */
  template<class CAMERA_GEOMETRY>
  int matchToMap(Estimator& estimator,
                 const okvis::ViParameters& params,
                 const uint64_t currentFrameId,
                 const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively = nullptr);

  /**
   * @brief Match the frames inside the multiframe to each other to initialise new landmarks.
   * @tparam MATCHING_ALGORITHM Algorithm to match new keypoints to existing landmarks.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator Estimator.
   * @param multiFrame Multiframe containing the frames to match.
   * @param params The VI parameters.
   * @param asKeyframe Whether this is a keyframe.
   */
  template<class CAMERA_GEOMETRY>
  void matchStereo(Estimator& estimator,
                   std::shared_ptr<okvis::MultiFrame> multiFrame,
                   const okvis::ViParameters& params,
                   bool asKeyframe);

  /// \brief DBoW for loop closure
  /// https://en.cppreference.com/w/cpp/language/pimpl
  class DBoW;
  std::unique_ptr<DBoW> dBow_; ///< DBoW object (PIMPL).

  /**
   * @brief Get filtered DBoW query.
   * @param[in] dBow DBoW database to use.
   * @param[in] features Features to match against DBoW.
   * @param[out] stateIds Resulting matching (keyframe) pose IDs with corresponding scores.
   * @return Number of matching keyframes.
   */
  int getFilteredDBoWResult(const std::unique_ptr<DBoW> &dBow,
                            const std::vector<std::vector<uchar>> &features,
                            std::vector<std::pair<StateId, double>> &stateIds) const;

  /**
   * @brief Verifies a recognised place with 3D2D matching, ransac, and nonlinear pose refinement.
   * @param[in] estimator Estimator.
   * @param[in] params The VI parameters.
   * @param[in] framesInOut Current multiframe.
   * @param[in] oldFrame Old multiframe to match against.
   * @param[out] T_Sold_Snew The relative pose found.
   * @param[out] H Information matrix corresponding to T_Sold_Snew.
   * @param[in] minInliers Minimum number of inliers required.
   */
  bool verifyRecognisedPlace(const Estimator &estimator,
                             const okvis::ViParameters &params,
                             const std::shared_ptr<const MultiFrame> framesInOut,
                             const std::shared_ptr<const MultiFrame> oldFrame,
                             kinematics::Transformation &T_Sold_Snew,
                             Eigen::Matrix<double, 6, 6>& H,
                             int minInliers = 10);

  /**
   * @brief Perform 3D/2D RANSAC.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator       Estimator.
   * @param nCameraSystem   Camera configuration and parameters.
   * @param currentFrame    Frame with the new potential matches.
   * @param initializePose  Initialize the pose from RANSAC?
   * @param removeOutliers  Remove observation of outliers in estimator.
   * @return True on success.
   */
  bool runRansac3d2d(Estimator &estimator,
                    const okvis::cameras::NCameraSystem &nCameraSystem,
                    std::shared_ptr<okvis::MultiFrame> currentFrame,
                    bool initializePose,
                    bool removeOutliers);
  /**
   * @brief Remove outliers on current frame.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator       Estimator.
   * @param nCameraSystem   Camera configuration and parameters.
   * @param currentFrame    Frame with the new potential matches.
   * @return Number of inliers.
   */
  template <class CAMERA_GEOMETRY>
  int removeOutliers(Estimator &estimator,
                     const okvis::cameras::NCameraSystem &nCameraSystem,
                     std::shared_ptr<okvis::MultiFrame> currentFrame);

  /**
   * @brief Perform 2D/2D RANSAC.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator         Estimator.
   * @param params            Parameter struct.
   * @param currentFrameId    ID of the new multiframe containing matches with the frame with ID
   *                          olderFrameId.
   * @param olderFrameId      ID of the multiframe to which the current frame has been matched
   *                          against.
   * @param initializePose    If the pose has not yet been initialised should the function try to
   *                          initialise it.
   * @param removeOutliers    Remove observation of outliers in estimator.
   * @param[out] rotationOnly Was the rotation only RANSAC model enough to explain the matches.
   * @return Number of inliers.
   */
  int runRansac2d2d(Estimator& estimator,
                    const okvis::ViParameters& params, uint64_t currentFrameId,
                    uint64_t olderFrameId, bool initializePose,
                    bool removeOutliers, bool &rotationOnly);

  /// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or
  /// at startup.
  void initialiseBriskFeatureDetectors();

  /// \brief Classification network for keypoints (if enabled).
  std::vector<std::shared_ptr<Network>> networks_;

  /**
   * @brief Match the frames to older, co-visible frames and triangulate.
   * @tparam CAMERA_GEOMETRY The camera geometry type to use.
   * @warning As this function uses the estimator it is not threadsafe.
   * @param estimator Estimator.
   * @param params The VI parameters.
   * @param currentFrameId Current frame ID.
   * @param[out] rotationOnly Result obtained matches with rotation-only RANSAC.
   */
  template<class CAMERA_GEOMETRY>
  int matchMotionStereo(Estimator &estimator, const okvis::ViParameters &params,
                        const uint64_t currentFrameId, bool &rotationOnly);

  /// \brief Helper struct (internally) for landmarks to match.
  struct LandmarkToMatch {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d p_W; ///< 3D point in World coordinates.
    cv::Mat descriptors; ///< All its descriptors.
    std::vector<KeypointIdentifier> kids; ///< All its observations.
    Eigen::Matrix3Xd e_W; ///< All directions in W-coords of the observations.
    Eigen::Matrix3Xd r_W; ///< Image centres of all the observations (W-coords).
    bool is3d = false; ///< Determine whether treated as 3D initialised.
    Eigen::Vector2d projection; ///< 2D projection location in pixels.
    bool ignore = false; ///< Ignore if classified as sky / person.
  };

  /// \brief Number of descriptors kept per candidate landmark.
  static constexpr int kNumDescriptorsToKeep = 3;
  /// \brief BRISK descriptor length in bytes.
  static constexpr int kDescriptorBytes = 48;

  /**
   * @brief The candidate table of one camera pass, as parallel arrays.
   *
   * This is the SoA form of AlignedMap<LandmarkId, LandmarkToMatch>. The stock
   * container costs about six heap blocks per candidate -- the map node, the
   * kids vector, e_W and r_W, plus the two buffers conservativeResize()
   * reallocates when the unused columns are cropped -- and is then deep-copied
   * a second time into landmarksToMatchVec so the uninitialised pass can still
   * see it. At roughly 1150 candidates per camera pass that is ~14k allocations
   * per frame, which is what VIO_PROFILING.md 3.12 identified as the dominant
   * cost of "2 Match".
   *
   * Here every field lives in a vector whose capacity only grows, so a warm
   * frame allocates nothing, the whole table is contiguous, and no copy is
   * needed at all: the same table serves the 3D pass, the uninitialised pass and
   * the observation insertion.
   *
   * Entries are appended in ascending landmark-id order, which the stock
   * std::map also produced. That is load-bearing rather than cosmetic: the
   * matchers keep the *first* candidate that improves the running best distance
   * with a strict "<", so any reordering silently changes which landmark a
   * keypoint matches (risk R3 in VIO_GPU_FRONTEND_PLAN.md).
   */
  struct LandmarkMatchSoA {
    std::vector<uint64_t> id;        ///< Landmark id, ascending.
    std::vector<double> projXY;      ///< Projection in pixels, 2 per entry.
    std::vector<int32_t> descRow;    ///< First descriptor row in `desc`.
    std::vector<int32_t> descRows;   ///< Descriptor count, 1..kNumDescriptorsToKeep.
    std::vector<uint8_t> is3d;       ///< Treat as an initialised 3D point?
    std::vector<uint8_t> ignore;     ///< Classified as sky / person?
    /// \brief Observation directions in W, kNumDescriptorsToKeep columns per entry.
    std::vector<double> eW;
    /// \brief Observing camera centres in W, kNumDescriptorsToKeep columns per entry.
    std::vector<double> rW;
    /// \brief Descriptor arena: gathered kDescriptorBytes rows, packed as written.
    std::vector<uint8_t> desc;
    size_t rowsUsed = 0;             ///< Rows of `desc` written this pass.

    /// \brief Number of candidates.
    size_t size() const { return id.size(); }

    /// \brief Drop all entries, keeping every allocation for the next frame.
    void clear() {
      id.clear(); projXY.clear(); descRow.clear(); descRows.clear();
      is3d.clear(); ignore.clear(); eW.clear(); rW.clear();
      rowsUsed = 0;
    }

    /**
     * @brief Index of landmark \p lm, or -1 if it is not a candidate.
     *
     * Ascending ids, so this is a binary search over a contiguous array. The
     * stock code did the same lookup with std::map::operator[], which also
     * *inserted* a default entry on a miss -- see the call sites for why that
     * mattered.
     */
    int find(uint64_t lm) const {
      const auto it = std::lower_bound(id.begin(), id.end(), lm);
      if (it == id.end() || *it != lm) return -1;
      return int(it - id.begin());
    }
  };

  /**
   * @brief Read-only cursor interface over the stock candidate map.
   *
   * The two matcher inner loops are the only place where correctness depends on
   * subtle details (iteration order, strict "<" tie-breaking, the exact Eigen
   * expressions), so they exist exactly once and are templated on one of these
   * views instead of being copy-pasted per layout. Everything is returned as a
   * raw pointer to contiguous doubles, which is what both layouts already hold:
   * Eigen matrices are column-major, so a column is contiguous.
   */
  struct CandidateMapView {
    using Cursor = AlignedMap<LandmarkId, LandmarkToMatch>::const_iterator;
    const AlignedMap<LandmarkId, LandmarkToMatch>* map;

    Cursor begin() const { return map->begin(); }
    Cursor end() const { return map->end(); }
    static LandmarkId id(Cursor c) { return c->first; }
    static bool is3d(Cursor c) { return c->second.is3d; }
    static const double* projXY(Cursor c) { return c->second.projection.data(); }
    static const uchar* desc(Cursor c) { return c->second.descriptors.data; }
    static int descRows(Cursor c) { return c->second.descriptors.rows; }
    static const double* eW(Cursor c, int d) { return c->second.e_W.col(d).data(); }
    static const double* rW(Cursor c, int d) { return c->second.r_W.col(d).data(); }
  };

  /// \brief Read-only cursor interface over LandmarkMatchSoA. See CandidateMapView.
  struct CandidateSoaView {
    using Cursor = size_t;
    const LandmarkMatchSoA* soa;

    Cursor begin() const { return 0; }
    Cursor end() const { return soa->id.size(); }
    LandmarkId id(Cursor c) const { return LandmarkId(soa->id[c]); }
    bool is3d(Cursor c) const { return soa->is3d[c] != 0; }
    const double* projXY(Cursor c) const { return &soa->projXY[2 * c]; }
    const uchar* desc(Cursor c) const {
      return &soa->desc[size_t(soa->descRow[c]) * kDescriptorBytes];
    }
    int descRows(Cursor c) const { return soa->descRows[c]; }
    const double* eW(Cursor c, int d) const {
      return &soa->eW[c * 3 * kNumDescriptorsToKeep + 3 * size_t(d)];
    }
    const double* rW(Cursor c, int d) const {
      return &soa->rW[c * 3 * kNumDescriptorsToKeep + 3 * size_t(d)];
    }
  };

  /// \brief Candidate table per camera, reused across frames (OKVIS_SOA!=0).
  std::vector<LandmarkMatchSoA> soaCandidates_;

  /**
   * @brief Parallelisable sub-part of matchToMap -- proper 3D points..
   * @tparam CAMERA_GEOMETRY The camera geometry type to use.
   * @param threadIdx Thread index.
   * @param numThreads Total number of threads to use.
   * @param estimator Estimator.
   * @param params The VI parameters.
   * @param currentFrameId Current frame ID.
   * @param loopClosureLandmarksToUseExclusively Only use these, if not nullptr.
   * @param T_WS1 Pose guess.
   * @param landmarksToMatch Landmarks to be matched against.
   * @param numKeypoints Number of keypoints (in image im)
   * @param pointMap Current map.
   * @param im The image idx.
   * @param multiFrame current multiFrame.
   * @param[out] distances Distances of landmarks.
   * @param[out] lmIds matched landmark IDs.
   * @param[out] hps_W matched landmarks (homogeneous) positions.
   * @param[out] ctrs Number of matches (by im).
   * @param[out] reprErrs Reprojection errors (by im).
   */
  /**
   * @brief Fill the flat candidate table for one camera pass (OKVIS_SOA!=0).
   *
   * The SoA counterpart of the "2.01a build landmarksToMatch" loop: same
   * projection, same field-of-view rejection, same descriptor selection, same
   * ordering, but reading the flat snapshot instead of MapPoints and writing
   * parallel arrays instead of a map of per-landmark heap blocks.
   *
   * It is deliberately a second loop rather than a refactor of the original: the
   * original then stays byte-for-byte what it was, so OKVIS_SOA=0 is provably
   * stock, and OKVIS_SOA=2 compares the two outputs field by field on every
   * frame instead of relying on a "should be equivalent" argument.
   * @param estimator Estimator (read only).
   * @param multiFrame The current multiframe.
   * @param im Camera index.
   * @param loopClosureLandmarksToUseExclusively Only use these, if not nullptr.
   * @param T_WC1 Current camera pose.
   * @param T_CW1 Its inverse.
   * @param reprThreshold Reprojection threshold in pixels.
   * @param maxU Image width plus the threshold.
   * @param maxV Image height plus the threshold.
   * @param focalLength Sum of both focal lengths, as the stock loop uses it.
   * @param[out] out The candidate table (cleared first, capacity retained).
   */
  template<class CAMERA_GEOMETRY>
  void buildCandidatesSoA(
      const Estimator& estimator, const MultiFramePtr& multiFrame, size_t im,
      const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
      const kinematics::Transformation& T_WC1,
      const kinematics::Transformation& T_CW1,
      double reprThreshold, double maxU, double maxV, double focalLength,
      LandmarkMatchSoA& out) const;

  template<class CAMERA_GEOMETRY, class CANDIDATES>
  void matchToMapByThread(
      size_t threadIdx, size_t numThreads, const Estimator &estimator,
      const okvis::ViParameters& params, const uint64_t currentFrameId,
      const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
      const kinematics::Transformation& T_WS1,
      const CANDIDATES& landmarksToMatch,
      size_t numKeypoints,
      size_t im, const MultiFramePtr&  multiFrame,
      std::vector<double>& distances, std::vector<LandmarkId>& lmIds,
      AlignedVector<Eigen::Vector4d>& hps_W, std::vector<size_t>& ctrs,
      std::vector<double>& reprErrs) const;

  /**
   * @brief Parallelisable sub-part of matchToMap -- unitialised points.
   * @tparam CAMERA_GEOMETRY The camera geometry type to use.
   * @param threadIdx Thread index.
   * @param numThreads Total number of threads to use.
   * @param estimator Estimator.
   * @param params The VI parameters.
   * @param currentFrameId Current frame ID.
   * @param loopClosureLandmarksToUseExclusively Only use these, if not nullptr.
   * @param T_WS1 Pose guess.
   * @param landmarksToMatch Landmarks to be matched against.
   * @param numKeypoints Number of keypoints (in image im)
   * @param pointMap Current map.
   * @param im The image idx.
   * @param multiFrame current multiFrame.
   * @param[out] distances Distances of landmarks.
   * @param[out] lmIds matched landmark IDs.
   * @param[out] hps_W matched landmarks (homogeneous) positions.
   * @param[out] ctrs Number of matches (by im).
   */
  template<class CAMERA_GEOMETRY, class CANDIDATES>
  void matchToMapByThreadUnitialised(
      size_t threadIdx, size_t numThreads, const Estimator &estimator,
      const okvis::ViParameters& params, const uint64_t currentFrameId,
      const std::set<LandmarkId>* loopClosureLandmarksToUseExclusively,
      const kinematics::Transformation& T_WS1,
      const CANDIDATES& landmarksToMatch,
      size_t numKeypoints,
      size_t im, const MultiFramePtr&  multiFrame,
      std::vector<double>& distances, std::vector<LandmarkId>& lmIds,
      AlignedVector<Eigen::Vector4d>& hps_W, std::vector<size_t>& ctrs) const;

  std::atomic_bool trackingLost_; ///< Is the tracking currently lost?

  /// \brief Hacky: remember which CNNs are running on what frame.
  std::map<StateId,std::vector<std::thread*>> cnnThreads_;

  AlignedVector<Component> components_; ///< Loaded other components.
  std::vector<std::unique_ptr<DBoW>> componentDBows_; ///< Corresponding DBoWs for place recogn.
  std::vector<bool> componentsFixed_; ///< Which ones of the other comp's are to be treated fixed.
};

}  // namespace okvis

#endif // INCLUDE_OKVIS_FRONTEND_HPP_
