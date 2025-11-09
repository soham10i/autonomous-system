
import numpy as np
import cv2
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time

import logging
import time
from slam_logging_config import get_slam_logger, performance_monitor, SLAMLogger


@dataclass
class KeyFrame:
    """Represents a keyframe for loop closure detection"""
    id: int = 0
    pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, theta)
    timestamp: float = 0.0
    image: np.ndarray = None  # Camera image
    lidar_point_cloud: Optional[np.ndarray] = None  # LiDAR scan points
    features: List[cv2.KeyPoint] = None  # Visual features
    descriptors: Optional[np.ndarray] = None  # Feature descriptors
    bow_vector: Optional[np.ndarray] = None  # Bag-of-words representation

@dataclass
class LoopClosure:
    """Represents a detected loop closure"""
    current_keyframe_id: int = 0
    matched_keyframe_id: int = 0
    relative_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Transformation between frames
    confidence: float = 0.0
    feature_matches: int = 0
    geometric_consistency: float = 0.0

class VisualPlaceRecognition:
    """
    Visual place recognition using ORB features and bag-of-words.
    Detects when the robot returns to previously visited locations.
    """
    
    def __init__(self, vocab_size=1000, max_features=500):
        self.vocab_size = vocab_size
        self.max_features = max_features
        
        # ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=max_features)
        
        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Bag-of-words vocabulary (simplified clustering)
        self.vocabulary = None
        self.vocab_built = False
        
        # Place recognition parameters
        self.min_feature_matches = 20
        self.min_geometric_inliers = 15
        self.max_reproj_error = 3.0  # pixels
        self.similarity_threshold = 0.6
        
    def extract_features(self, image):
        """Extract ORB features from image."""
        if image is None or image.size == 0:
            return [], None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect and compute features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def build_vocabulary(self, all_descriptors):
        """Build bag-of-words vocabulary from training descriptors."""
        if len(all_descriptors) == 0:
            return
        
        # Combine all descriptors
        combined_descriptors = np.vstack(all_descriptors)
        
        # Use K-means clustering to build vocabulary
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        
        # Simple clustering using OpenCV
        _, labels, centers = cv2.kmeans(
            combined_descriptors.astype(np.float32), 
            self.vocab_size, 
            None, 
            criteria, 
            10, 
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        self.vocabulary = centers.astype(np.uint8)
        self.vocab_built = True
    
    def compute_bow_vector(self, descriptors):
        """Compute bag-of-words vector for descriptors."""
        if descriptors is None or not self.vocab_built:
            return np.zeros(self.vocab_size)
        
        bow_vector = np.zeros(self.vocab_size)
        
        # Find closest vocabulary word for each descriptor
        for desc in descriptors:
            distances = np.linalg.norm(self.vocabulary - desc, axis=1)
            closest_word = np.argmin(distances)
            bow_vector[closest_word] += 1
        
        # Normalize
        if np.sum(bow_vector) > 0:
            bow_vector = bow_vector / np.sum(bow_vector)
        
        return bow_vector
    
    def calculate_similarity(self, bow1, bow2):
        """Calculate similarity between two bag-of-words vectors."""
        if bow1 is None or bow2 is None:
            return 0.0
        
        # Use cosine similarity
        norm1 = np.linalg.norm(bow1)
        norm2 = np.linalg.norm(bow2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(bow1, bow2) / (norm1 * norm2)
        return similarity
    
    def find_loop_candidates(self, current_keyframe, keyframe_database, min_temporal_distance=30):
        """
        Find potential loop closure candidates based on visual similarity.
        """
        candidates = []
        
        if not self.vocab_built or current_keyframe.bow_vector is None:
            return candidates
        
        current_bow = current_keyframe.bow_vector
        
        for keyframe in keyframe_database:
            # Skip recent keyframes to avoid trivial matches
            if abs(keyframe.id - current_keyframe.id) < min_temporal_distance:
                continue
            
            if keyframe.bow_vector is None:
                continue
            
            # Calculate visual similarity
            similarity = self.calculate_similarity(current_bow, keyframe.bow_vector)
            
            if similarity > self.similarity_threshold:
                candidates.append((keyframe, similarity))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:5]  # Return top 5 candidates

class GeometricVerification:
    """
    Geometric verification of loop closure candidates using feature matching
    and pose estimation.
    """
    
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = 20
        self.ransac_threshold = 3.0
        self.max_iterations = 1000
    
    def verify_loop_closure(self, current_keyframe, candidate_keyframe):
        """
        Verify loop closure between two keyframes using geometric constraints.
        Returns LoopClosure object if verified, None otherwise.
        """
        if current_keyframe.descriptors is None or \
            candidate_keyframe.descriptors is None or \
            len(current_keyframe.descriptors) == 0 or \
            len(candidate_keyframe.descriptors) == 0:
            return None
        
        # Match features
        matches = self.matcher.match(current_keyframe.descriptors, candidate_keyframe.descriptors)
        
        if len(matches) < self.min_matches:
            return None
        
        # Extract matched points
        current_points = []
        candidate_points = []
        
        for match in matches:
            current_pt = current_keyframe.features[match.queryIdx].pt
            candidate_pt = candidate_keyframe.features[match.trainIdx].pt
            current_points.append(current_pt)
            candidate_points.append(candidate_pt)
        
        current_points = np.array(current_points)
        candidate_points = np.array(candidate_points)
        
        # Estimate fundamental matrix with RANSAC
        if len(current_points) >= 8:
            F, inlier_mask = cv2.findFundamentalMat(
                current_points, candidate_points,
                cv2.FM_RANSAC, self.ransac_threshold, 0.99
            )
            
            if F is not None and inlier_mask is not None:
                num_inliers = np.sum(inlier_mask)
                inlier_ratio = num_inliers / len(matches)
                
                if num_inliers >= self.min_matches and inlier_ratio > 0.3:
                    # Estimate relative pose (simplified 2D case)
                    relative_pose = self._estimate_relative_pose_2d(
                        current_keyframe.pose, candidate_keyframe.pose
                    )
                    
                    # Calculate confidence
                    confidence = min(1.0, inlier_ratio * (num_inliers / 50.0))
                    
                    return LoopClosure(
                        current_keyframe_id=current_keyframe.id,
                        matched_keyframe_id=candidate_keyframe.id,
                        relative_pose=relative_pose,
                        confidence=confidence,
                        feature_matches=num_inliers,
                        geometric_consistency=inlier_ratio
                    )
        
        return None
    
    def _estimate_relative_pose_2d(self, pose1, pose2):
        """Estimate 2D relative pose between two robot poses."""
        x1, y1, theta1 = pose1
        x2, y2, theta2 = pose2
        
        # Transform pose2 relative to pose1
        cos_theta1 = math.cos(theta1)
        sin_theta1 = math.sin(theta1)
        
        dx = x2 - x1
        dy = y2 - y1
        
        # Relative position in pose1's frame
        rel_x = dx * cos_theta1 + dy * sin_theta1
        rel_y = -dx * sin_theta1 + dy * cos_theta1
        rel_theta = theta2 - theta1
        
        # Normalize angle
        while rel_theta > math.pi:
            rel_theta -= 2 * math.pi
        while rel_theta < -math.pi:
            rel_theta += 2 * math.pi
        
        return (rel_x, rel_y, rel_theta)

class LoopClosureDetector:
    """
    Main loop closure detection system that integrates visual place recognition
    and geometric verification.
    """
    
    def __init__(self, keyframe_distance_threshold=1.0, keyframe_angle_threshold=0.5):
        self.visual_recognition = VisualPlaceRecognition()
        self.geometric_verification = GeometricVerification()
        
        # Keyframe selection parameters
        self.keyframe_distance_threshold = keyframe_distance_threshold  # meters
        self.keyframe_angle_threshold = keyframe_angle_threshold  # radians
        
        # Database
        self.keyframes = []
        self.loop_closures = []
        self.next_keyframe_id = 0
        
        # State
        self.last_keyframe_pose = None
        self.vocabulary_update_counter = 0
        self.vocabulary_update_interval = 10  # Update vocabulary every N keyframes
        
    def should_create_keyframe(self, current_pose):
        """Determine if a new keyframe should be created."""
        if self.last_keyframe_pose is None:
            return True
        
        # Calculate distance and angle change
        last_x, last_y, last_theta = self.last_keyframe_pose
        curr_x, curr_y, curr_theta = current_pose
        
        distance = math.sqrt((curr_x - last_x)**2 + (curr_y - last_y)**2)
        angle_diff = abs(curr_theta - last_theta)
        
        # Normalize angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        angle_diff = abs(angle_diff)
        
        return (distance > self.keyframe_distance_threshold or 
                angle_diff > self.keyframe_angle_threshold)
    
    def create_keyframe(self, pose, image, lidar_scan):
        """Create a new keyframe and add it to the database."""
        # Extract visual features
        features, descriptors = self.visual_recognition.extract_features(image)
        
        if descriptors is None:
            return None
        
        # Create keyframe
        keyframe = KeyFrame(
            id=self.next_keyframe_id,
            pose=pose,
            timestamp=time.time(),
            image=image.copy(),
            lidar_scan=lidar_scan.copy() if lidar_scan is not None else None,
            features=features,
            descriptors=descriptors
        )
        
        # Update vocabulary periodically
        if (self.vocabulary_update_counter % self.vocabulary_update_interval == 0 and
            len(self.keyframes) > 0):
            self._update_vocabulary()
        
        # Compute bag-of-words vector
        if self.visual_recognition.vocab_built:
            keyframe.bow_vector = self.visual_recognition.compute_bow_vector(descriptors)
        
        # Add to database
        self.keyframes.append(keyframe)
        self.next_keyframe_id += 1
        self.last_keyframe_pose = pose
        self.vocabulary_update_counter += 1
        
        return keyframe
    
    def detect_loop_closure(self, current_pose: List[float], image, lidar_point_cloud):
        """
        Main loop closure detection function.
        Returns detected loop closure or None.
        """
        # Check if we should create a new keyframe
        if not self.should_create_keyframe(current_pose):
            return None
        
        # Create new keyframe
        current_keyframe = self.create_keyframe(current_pose, image, lidar_point_cloud)        
        if current_keyframe is None:
            return None
        
        # Skip loop detection for first few keyframes
        if len(self.keyframes) < 10:
            return None
        
        # Find loop closure candidates
        candidates = self.visual_recognition.find_loop_candidates(
            current_keyframe, self.keyframes[:-1]  # Exclude current keyframe
        )
        
        # Verify candidates geometrically
        for candidate_keyframe, similarity in candidates:
            loop_closure = self.geometric_verification.verify_loop_closure(
                current_keyframe, candidate_keyframe
            )
            
            if loop_closure is not None:
                # Valid loop closure detected
                self.loop_closures.append(loop_closure)
                print(f"Loop closure detected! Current: {current_keyframe.id}, "
                      f"Matched: {candidate_keyframe.id}, "
                      f"Confidence: {loop_closure.confidence:.3f}")
                return loop_closure
        
        return None
    
    def _update_vocabulary(self):
        """Update the bag-of-words vocabulary with new keyframes."""
        if len(self.keyframes) < 5:
            return
        
        # Collect descriptors from recent keyframes
        all_descriptors = []
        for keyframe in self.keyframes[-20:]:  # Use last 20 keyframes
            if keyframe.descriptors is not None:
                all_descriptors.append(keyframe.descriptors)
        
        if all_descriptors:
            self.visual_recognition.build_vocabulary(all_descriptors)
            
            # Update bow vectors for all keyframes
            for keyframe in self.keyframes:
                if keyframe.descriptors is not None:
                    keyframe.bow_vector = self.visual_recognition.compute_bow_vector(
                        keyframe.descriptors
                    )
    
    def get_loop_closures(self):
        """Get all detected loop closures."""
        return self.loop_closures
    
    def get_keyframes(self):
        """Get all keyframes."""
        return self.keyframes
    
    def get_statistics(self):
        """Get loop closure detection statistics."""
        return {
            'total_keyframes': len(self.keyframes),
            'total_loop_closures': len(self.loop_closures),
            'vocabulary_built': self.visual_recognition.vocab_built,
            'vocabulary_size': self.visual_recognition.vocab_size,
            'last_keyframe_id': self.next_keyframe_id - 1 if self.keyframes else -1
        }
    
    def visualize_loop_closure(self, loop_closure):
        """
        Create visualization of a detected loop closure.
        Returns images showing the matched keyframes.
        """
        if not loop_closure:
            return None, None
        
        # Find keyframes
        current_kf = None
        matched_kf = None
        
        for kf in self.keyframes:
            if kf.id == loop_closure.current_keyframe_id:
                current_kf = kf
            elif kf.id == loop_closure.matched_keyframe_id:
                matched_kf = kf
        
        if current_kf is None or matched_kf is None:
            return None, None
        
        # Create visualization
        current_vis = current_kf.image.copy()
        matched_vis = matched_kf.image.copy()
        
        # Draw feature points
        for kp in current_kf.features:
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(current_vis, pt, 3, (0, 255, 0), -1)
        
        for kp in matched_kf.features:
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(matched_vis, pt, 3, (0, 255, 0), -1)
        
        # Add text information
        info_text = f"Loop: {loop_closure.current_keyframe_id} -> {loop_closure.matched_keyframe_id}"
        confidence_text = f"Confidence: {loop_closure.confidence:.3f}"
        
        cv2.putText(current_vis, "Current", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(matched_vis, "Matched", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(current_vis, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(matched_vis, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return current_vis, matched_vis
