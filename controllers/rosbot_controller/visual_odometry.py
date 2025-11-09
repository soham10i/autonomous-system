import cv2
import numpy as np
import time
from sensor_fusion import PoseEstimate # Assuming PoseEstimate is in sensor_fusion.py

class VisualOdometry:
    def __init__(self):
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.current_pose = PoseEstimate() # Initialize current pose
        
        # Initialize feature detector (ORB is a good choice for real-time)
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Initialize brute-force matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_pose_estimate(self, current_frame, timestamp):
        # Convert to grayscale
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2GRAY)

        # Detect ORB features and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray_frame, None)

        if self.prev_frame is None:
            self.prev_frame = gray_frame
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.current_pose = PoseEstimate(x=0.0, y=0.0, theta=0.0, timestamp=timestamp, source='visual', confidence=1.0)
            return self.current_pose

        # Match descriptors
        matches = self.bf.match(self.prev_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate affine transformation (translation, rotation, scale) between the two sets of points
        if len(src_pts) > 4:
            M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
            if M is None:
                # If estimation fails, return previous pose with reduced confidence
                self.current_pose.confidence *= 0.9
                self.current_pose.timestamp = timestamp
                return self.current_pose
            
            # Extract rotation and translation
            rotation_matrix = M[:, :2]
            translation_vector = M[:, 2]
            
            # Calculate yaw from rotation matrix
            delta_theta = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            
            # Simple scaling for translation (pixels to meters) - needs calibration
            # This value is highly dependent on the camera's intrinsic parameters and the robot's movement scale.
            # A proper calibration would involve moving the robot a known distance and observing the pixel displacement.
            # For now, a placeholder value is used.
            pixel_to_meter_scale = 0.001 # Example: 1 pixel = 1 mm
            delta_x = translation_vector[0] * pixel_to_meter_scale
            delta_y = translation_vector[1] * pixel_to_meter_scale

            # Update current pose based on estimated deltas
            # Assuming motion in robot's local frame, then transform to global frame
            current_theta = self.current_pose.theta
            global_delta_x = delta_x * np.cos(current_theta) - delta_y * np.sin(current_theta)
            global_delta_y = delta_x * np.sin(current_theta) + delta_y * np.cos(current_theta)

            self.current_pose.x += global_delta_x
            self.current_pose.y += global_delta_y
            self.current_pose.theta += delta_theta
            self.current_pose.timestamp = timestamp
            self.current_pose.source = 'visual'
            self.current_pose.confidence = 1.0 # Reset confidence on successful update
            
        else:
            # Not enough points, reduce confidence
            self.current_pose.confidence *= 0.9
            self.current_pose.timestamp = timestamp

        self.prev_frame = gray_frame
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        return self.current_pose
    
    