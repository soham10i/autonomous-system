import cv2
import numpy as np

class VisualOdometry:
    def __init__(self):
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Initialize feature detector (ORB is a good choice for real-time)
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Initialize brute-force matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def process_frame(self, current_frame):
        # Convert to grayscale
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2GRAY)

        # Detect ORB features and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray_frame, None)

        if self.prev_frame is None:
            self.prev_frame = gray_frame
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return np.eye(4) # Return identity transformation for the first frame

        # Match descriptors
        matches = self.bf.match(self.prev_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find essential matrix or fundamental matrix (for monocular VO)
        # For simplicity, we'll use findHomography for now, which is 2D. 
        # For 3D pose estimation, we'd need calibrateCamera and solvePnP.
        # Given we have an RGB-D camera (Astra), we can use depth information for 3D points.
        # For now, let's just estimate a 2D transformation (translation and rotation in the image plane)
        # This is a simplified approach for visual odometry without full 3D reconstruction.
        
        # Estimate affine transformation (translation, rotation, scale) between the two sets of points
        # This is a 2D transformation, not a full 3D pose.
        if len(src_pts) > 4:
            M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
            if M is None:
                return np.eye(4)
            
            # M is a 2x3 matrix: [[cos(theta) -sin(theta) tx], [sin(theta) cos(theta) ty]]
            # We need to convert this to a 4x4 transformation matrix (homogeneous coordinates)
            # Assuming motion is in X-Y plane and rotation around Z-axis
            
            # Extract rotation and translation
            rotation_matrix = M[:, :2]
            translation_vector = M[:, 2]
            
            # Calculate yaw from rotation matrix
            theta = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            
            # Create 4x4 transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[0, 0] = np.cos(theta)
            transformation_matrix[0, 1] = -np.sin(theta)
            transformation_matrix[1, 0] = np.sin(theta)
            transformation_matrix[1, 1] = np.cos(theta)
            transformation_matrix[0, 3] = translation_vector[0] # tx
            transformation_matrix[1, 3] = translation_vector[1] # ty
            
        else:
            transformation_matrix = np.eye(4)

        self.prev_frame = gray_frame
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        return transformation_matrix