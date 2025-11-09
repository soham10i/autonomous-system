
import numpy as np
import math
from typing import Tuple, Optional, Dict, List
import time
from dataclasses import dataclass, field

import logging
import time
from slam_logging_config import get_slam_logger, performance_monitor, SLAMLogger

@dataclass
class PoseEstimate:
    """Represents a pose estimate with uncertainty"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # orientation in radians
    covariance: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 covariance matrix
    timestamp: float = 0.0
    source: str = ""  # 'visual', 'lidar', 'wheel', 'fused'
    confidence: float = 1.0  # [0.0, 1.0]

@dataclass
class SensorReading:
    """Container for sensor measurements"""
    sensor_type: str
    data: any
    timestamp: float = 0.0
    quality: float = 1.0  # measurement quality [0.0, 1.0]

class KalmanFilter:
    """
    Extended Kalman Filter for robot pose estimation.
    State vector: [x, y, theta, vx, vy, vtheta]
    """
    
    def __init__(self, initial_pose=(0.0, 0.0, 0.0), process_noise=0.1, measurement_noise=0.5):
        # State: [x, y, theta, vx, vy, vtheta]
        self.state = np.array([initial_pose[0], initial_pose[1], initial_pose[2], 0.0, 0.0, 0.0])
        
        # Covariance matrix (6x6)
        self.P = np.eye(6) * 0.1
        
        # Process noise
        self.Q = np.eye(6) * process_noise
        self.Q[3:, 3:] *= 2.0  # Higher noise for velocities
        
        # Measurement noise
        self.R_pose = np.eye(3) * measurement_noise  # For pose measurements
        self.R_velocity = np.eye(2) * (measurement_noise * 0.5)  # For velocity measurements
        
        self.last_update_time = time.time()
    
    @performance_monitor
    def predict(self, dt, control_input=None):
        """
        Prediction step with motion model.
        """
        # Simple motion model: x_{k+1} = x_k + v_k * dt
        F = np.eye(6)
        F[0, 3] = dt  # x position
        F[1, 4] = dt  # y position  
        F[2, 5] = dt  # orientation
        
        # Apply control input if provided (wheel odometry)
        if control_input is not None:
            vx, vy, vtheta = control_input
            self.state[3] = vx
            self.state[4] = vy
            self.state[5] = vtheta
        
        # Predict state
        self.state = F @ self.state
        
        # Normalize orientation
        self.state[2] = self._normalize_angle(self.state[2])
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
    
    @performance_monitor
    def update_pose(self, measurement, measurement_covariance=None):
        """
        Update step with pose measurement (x, y, theta).
        """
        z = np.array(measurement[:3])  # [x, y, theta]
        
        # Measurement model (observe position and orientation)
        H = np.zeros((3, 6))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # theta
        
        # Predicted measurement
        h = H @ self.state
        h[2] = self._normalize_angle(h[2])
        
        # Innovation
        y = z - h
        y[2] = self._normalize_angle(y[2])  # Handle angle wraparound
        
        # Use provided covariance or default
        R = measurement_covariance if measurement_covariance is not None else self.R_pose
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y
        self.state[2] = self._normalize_angle(self.state[2])
        
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P
    
    def get_pose_estimate(self):
        """Get current pose estimate with covariance."""
        return PoseEstimate(
            x=self.state[0],
            y=self.state[1], 
            theta=self.state[2],
            covariance=self.P[:3, :3].copy(),
            timestamp=time.time(),
            source='fused',
            confidence=self._calculate_confidence()
        )
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _calculate_confidence(self):
        """Calculate confidence based on covariance trace."""
        trace = np.trace(self.P[:3, :3])
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + trace)))
        return confidence

class LidarScanMatcher:
    """
    LiDAR scan matching for pose correction and loop closure detection.
    Uses Iterative Closest Point (ICP) algorithm.
    """
    
    def __init__(self, max_iterations=20, convergence_threshold=0.001, max_correspondence_distance=0.5):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.max_correspondence_distance = max_correspondence_distance
        
        # Store reference scans for matching
        self.reference_scans = []
        self.reference_poses = []
        self.scan_history_size = 50
        
    def add_reference_scan(self, scan_points, pose):
        """Add a scan as reference for future matching."""
        self.reference_scans.append(scan_points.copy())
        self.reference_poses.append(pose)
        
        # Maintain history size
        if len(self.reference_scans) > self.scan_history_size:
            self.reference_scans.pop(0)
            self.reference_poses.pop(0)
    
    def match_scan(self, current_scan, initial_pose_guess):
        """
        Match current scan against reference scans to get pose correction.
        Returns corrected pose and match quality.
        """
        if not self.reference_scans:
            return initial_pose_guess, 0.0
        
        best_pose = initial_pose_guess
        best_quality = 0.0
        
        # Try matching against recent reference scans
        for i, ref_scan in enumerate(self.reference_scans[-5:]):  # Last 5 scans
            corrected_pose, quality = self._icp_match(current_scan, ref_scan, initial_pose_guess)
            
            if quality > best_quality:
                best_quality = quality
                best_pose = corrected_pose
        
        return best_pose, best_quality
    
    def _icp_match(self, source_points, target_points, initial_pose):
        """
        Iterative Closest Point algorithm for scan matching.
        """
        if len(source_points) == 0 or len(target_points) == 0:
            return initial_pose, 0.0
        
        # Convert to numpy arrays
        source = np.array(source_points)
        target = np.array(target_points)
        
        # Initialize transformation
        x, y, theta = initial_pose
        T = np.array([[math.cos(theta), -math.sin(theta), x],
                     [math.sin(theta), math.cos(theta), y],
                     [0, 0, 1]])
        
        prev_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # Transform source points
            source_homogeneous = np.hstack([source, np.ones((source.shape[0], 1))])
            transformed_source = (T @ source_homogeneous.T).T[:, :2]
            
            # Find correspondences
            correspondences = []
            total_error = 0.0
            
            for src_point in transformed_source:
                # Find closest point in target
                distances = np.linalg.norm(target - src_point, axis=1)
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                
                if min_distance < self.max_correspondence_distance:
                    correspondences.append((src_point, target[min_idx]))
                    total_error += min_distance
            
            if len(correspondences) == 0:
                break
            
            # Calculate average error
            avg_error = total_error / len(correspondences)
            
            # Check convergence
            if abs(prev_error - avg_error) < self.convergence_threshold:
                break
            
            prev_error = avg_error
            
            # Estimate transformation from correspondences
            if len(correspondences) >= 3:
                src_points = np.array([c[0] for c in correspondences])
                tgt_points = np.array([c[1] for c in correspondences])
                
                # Simple least squares solution
                delta_T = self._estimate_transformation(src_points, tgt_points)
                T = delta_T @ T
        
        # Extract final pose
        final_x = T[0, 2]
        final_y = T[1, 2]
        final_theta = math.atan2(T[1, 0], T[0, 0])
        
        # Calculate match quality
        quality = len(correspondences) / max(len(source), len(target))
        quality *= max(0.0, 1.0 - avg_error)  # Penalize large errors
        
        return (final_x, final_y, final_theta), quality
    
    def _estimate_transformation(self, source_points, target_points):
        """Estimate 2D transformation from point correspondences."""
        # Simple centroid-based estimation
        src_centroid = np.mean(source_points, axis=0)
        tgt_centroid = np.mean(target_points, axis=0)
        
        # Translation
        translation = tgt_centroid - src_centroid
        
        # For simplicity, assume no rotation change in this step
        T = np.array([[1, 0, translation[0]],
                     [0, 1, translation[1]],
                     [0, 0, 1]])
        
        return T

class SensorFusion:
    """
    Main sensor fusion system combining visual odometry, wheel odometry, and LiDAR.
    """
    
    def __init__(self, initial_pose=(0.0, 0.0, 0.0)):
        self.kalman_filter = KalmanFilter(initial_pose)
        self.scan_matcher = LidarScanMatcher()
        
        # Sensor confidence weights
        self.sensor_weights = {
            'visual': 0.6,
            'wheel': 0.3,
            'lidar': 0.8
        }
        
        # Fusion parameters
        self.min_lidar_quality = 0.3  # Minimum quality for LiDAR correction
        self.max_pose_uncertainty = 1.0  # Maximum allowed position uncertainty
        
        # State tracking
        self.last_pose_estimates = {}
        self.fusion_history = []
        self.current_fused_pose = None
        
    @performance_monitor
    def update_visual_odometry(self, pose_estimate: PoseEstimate):
        """Update with visual odometry measurement."""
        self.last_pose_estimates['visual'] = pose_estimate
        self._perform_fusion()
    
    @performance_monitor
    def update_wheel_odometry(self, pose_estimate: PoseEstimate):
        """Update with wheel odometry measurement."""
        self.last_pose_estimates["wheel"] = pose_estimate
        
        # The Kalman filter's prediction step should be handled by the main loop
        # or by a dedicated motion model update, not directly by sensor updates.
        # This method only updates the measurement for the fusion step.
        
        self._perform_fusion()
    
    @performance_monitor
    def update_lidar_scan(self, lidar_point_cloud, estimated_pose):
        """Update with LiDAR scan for scan matching."""
        # Extract 2D points (x, y) from the 3D point cloud for scan matching
        scan_points_2d = []
        if lidar_point_cloud:
            for point in lidar_point_cloud:
                if hasattr(point, 'x') and hasattr(point, 'y'):
                    scan_points_2d.append([point.x, point.y])
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    scan_points_2d.append([point[0], point[1]])

        if not scan_points_2d:
            return # No valid 2D points to process

        # Perform scan matching
        corrected_pose, match_quality = self.scan_matcher.match_scan(scan_points_2d, estimated_pose)        
        if match_quality > self.min_lidar_quality:
            # Create high-confidence pose estimate from LiDAR
            lidar_covariance = np.eye(3) * (0.1 / match_quality)  # Inverse quality as uncertainty
            
            lidar_estimate = PoseEstimate(
                x=corrected_pose[0],
                y=corrected_pose[1],
                theta=corrected_pose[2],
                covariance=lidar_covariance,
                timestamp=time.time(),
                source='lidar',
                confidence=match_quality
            )
            
            self.last_pose_estimates['lidar'] = lidar_estimate
            self._perform_fusion()
        
        # Add current scan as reference for future matching
        self.scan_matcher.add_reference_scan(scan_points_2d, estimated_pose)
    
    def _perform_fusion(self):
        """
        Perform sensor fusion using Kalman filter and weighted averaging.
        """
        if not self.last_pose_estimates:
            return
        
        # Get the most recent estimates
        recent_estimates = []
        current_time = time.time()
        
        for source, estimate in self.last_pose_estimates.items():
            # Only use recent estimates (within 1 second)
            if current_time - estimate.timestamp < 1.0:
                recent_estimates.append(estimate)
        
        if not recent_estimates:
            return
        
        # Weighted fusion based on confidence and sensor weights
        fused_pose = self._weighted_pose_fusion(recent_estimates)
        
        # Update Kalman filter with fused measurement
        measurement = [fused_pose.x, fused_pose.y, fused_pose.theta]
        self.kalman_filter.update_pose(measurement, fused_pose.covariance)
        
        # Get final filtered estimate
        self.current_fused_pose = self.kalman_filter.get_pose_estimate()
        
        # Store in history
        self.fusion_history.append(self.current_fused_pose)
        if len(self.fusion_history) > 100:  # Keep last 100 estimates
            self.fusion_history.pop(0)
    
    def _weighted_pose_fusion(self, estimates):
        """
        Fuse multiple pose estimates using weighted averaging.
        """
        total_weight = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        weighted_cos_theta = 0.0
        weighted_sin_theta = 0.0
        
        # Combined covariance (simplified as average)
        combined_covariance = np.zeros((3, 3))
        
        for estimate in estimates:
            # Calculate weight based on source type and confidence
            source_weight = self.sensor_weights.get(estimate.source, 0.5)
            confidence_weight = estimate.confidence
            total_weight_i = source_weight * confidence_weight
            
            # Weighted position
            weighted_x += estimate.x * total_weight_i
            weighted_y += estimate.y * total_weight_i
            
            # Weighted orientation (use complex representation to handle wraparound)
            weighted_cos_theta += math.cos(estimate.theta) * total_weight_i
            weighted_sin_theta += math.sin(estimate.theta) * total_weight_i
            
            # Accumulate covariance
            combined_covariance += estimate.covariance * total_weight_i
            
            total_weight += total_weight_i
        
        if total_weight == 0:
            # Fallback to first estimate
            return estimates[0]
        
        # Normalize
        fused_x = weighted_x / total_weight
        fused_y = weighted_y / total_weight
        fused_theta = math.atan2(weighted_sin_theta / total_weight, weighted_cos_theta / total_weight)
        fused_covariance = combined_covariance / total_weight
        
        return PoseEstimate(
            x=fused_x,
            y=fused_y,
            theta=fused_theta,
            covariance=fused_covariance,
            timestamp=time.time(),
            source='fused',
            confidence=min(1.0, total_weight / len(estimates))
        )
    
    def get_current_pose(self):
        """Get the current best pose estimate."""
        return self.current_fused_pose
    
    def get_pose_uncertainty(self):
        """Get current pose uncertainty (standard deviation)."""
        if self.current_fused_pose is None:
            return float('inf')
        
        # Calculate position uncertainty as trace of covariance
        position_variance = self.current_fused_pose.covariance[0, 0] + self.current_fused_pose.covariance[1, 1]
        return math.sqrt(position_variance)
    
    def is_localization_reliable(self):
        """Check if current localization is reliable."""
        uncertainty = self.get_pose_uncertainty()
        return uncertainty < self.max_pose_uncertainty
    
    def detect_loop_closure(self, current_pose, threshold_distance=1.0):
        """
        Detect potential loop closures based on pose history.
        Returns True if robot has returned to a previously visited area.
        """
        if len(self.fusion_history) < 10:  # Need some history
            return False
        
        current_pos = (current_pose.x, current_pose.y)
        
        # Check against poses from at least 30 steps ago to avoid trivial matches
        for old_pose in self.fusion_history[:-30]:
            old_pos = (old_pose.x, old_pose.y)
            distance = math.sqrt((current_pos[0] - old_pos[0])**2 + (current_pos[1] - old_pos[1])**2)
            
            if distance < threshold_distance:
                return True
        
        return False
    
    def get_fusion_statistics(self):
        """Get statistics about sensor fusion performance."""
        if not self.last_pose_estimates:
            return {}
        
        stats = {
            'active_sensors': list(self.last_pose_estimates.keys()),
            'pose_uncertainty': self.get_pose_uncertainty(),
            'localization_reliable': self.is_localization_reliable(),
            'fusion_history_length': len(self.fusion_history)
        }
        
        # Add per-sensor confidence
        for source, estimate in self.last_pose_estimates.items():
            stats[f'{source}_confidence'] = estimate.confidence
        
        return stats
