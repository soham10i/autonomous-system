import numpy as np
import cv2
from PIL import Image
import logging
import time
from slam_logging_config import get_slam_logger, performance_monitor, SLAMLogger


# Constants defined locally
GRID_RESOLUTION = 0.05
MAZE_WIDTH = 200
MAZE_HEIGHT = 200
MAZE_ORIGIN_X = -5.0
MAZE_ORIGIN_Y = -5.0

class EnhancedOccupancyGrid:
    @property
    def grid(self):
        """
        Returns the occupancy probability grid (values in [0,1]).
        This allows visualization code to use occupancy_grid.grid.
        """
        return self.get_probability_grid()

    @property
    def shape(self):
        return self.log_odds_grid.shape
    def __init__(self, resolution=GRID_RESOLUTION, width=MAZE_WIDTH, height=MAZE_HEIGHT, origin_x=MAZE_ORIGIN_X, origin_y=MAZE_ORIGIN_Y):
        self.resolution = resolution  # meters per cell
        self.width = width            # meters
        self.height = height          # meters
        self.origin_x = origin_x      # x-coordinate of the grid's origin in world frame
        self.origin_y = origin_y      # y-coordinate of the grid's origin in world frame

        self.grid_width = int(self.width / self.resolution)
        self.grid_height = int(self.height / self.resolution)

        # Initialize grid with unknown values (0.5 in probability, 0 in log-odds)
        self.log_odds_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

        # Log-odds update values
        self.L_occ = 0.4  # log-odds increase for occupied
        self.L_free = -0.4 # log-odds decrease for free
        self.L_prior = 0.0 # initial log-odds (0.5 probability)

        # Additional grid for visual features (for loop closure detection)
        self.visual_features_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

    def world_to_grid(self, x_world, y_world):
        grid_x = int((x_world - self.origin_x) / self.resolution)
        grid_y = int((y_world - self.origin_y) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        x_world = grid_x * self.resolution + self.origin_x + self.resolution / 2
        y_world = grid_y * self.resolution + self.origin_y + self.resolution / 2
        return x_world, y_world

    def is_valid_grid_coord(self, grid_x, grid_y):
        return 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height

    def is_occupied(self, x_world, y_world, threshold=0.7):
        grid_x, grid_y = self.world_to_grid(x_world, y_world)
        return self.is_occupied_grid(grid_x, grid_y, threshold)

    def is_occupied_grid(self, grid_x, grid_y, threshold=0.7):
        if self.is_valid_grid_coord(grid_x, grid_y):
            probability = 1 - (1 / (1 + np.exp(self.log_odds_grid[grid_y, grid_x])))
            return probability > threshold
        return False

    @performance_monitor
    def update_grid(self, robot_pose, lidar_point_cloud, camera_image=None):
        robot_x, robot_y, robot_theta = robot_pose
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)

        # Process LiDAR point cloud
        if lidar_point_cloud is not None and len(lidar_point_cloud) > 0:
            for point in lidar_point_cloud:
                # Assuming point is an object with x, y, z attributes or a tuple (x, y, z)
                # These points are typically relative to the LiDAR sensor.
                # We need to transform them to world coordinates using the robot's pose.
                
                # Convert point from sensor frame to robot base frame (assuming LiDAR is at robot origin for simplicity)
                # If LiDAR has an offset from robot base, that needs to be added here.
                point_x_robot_frame = point.x if hasattr(point, 'x') else point[0]
                point_y_robot_frame = point.y if hasattr(point, 'y') else point[1]
                
                # Rotate and translate point from robot frame to world frame
                cos_theta = np.cos(robot_theta)
                sin_theta = np.sin(robot_theta)
                
                hit_x_world = robot_x + (point_x_robot_frame * cos_theta - point_y_robot_frame * sin_theta)
                hit_y_world = robot_y + (point_x_robot_frame * sin_theta + point_y_robot_frame * cos_theta)
                
                hit_grid_x, hit_grid_y = self.world_to_grid(hit_x_world, hit_y_world)

                if self.is_valid_grid_coord(hit_grid_x, hit_grid_y):
                    self.log_odds_grid[hit_grid_y, hit_grid_x] += self.L_occ

                # Ray tracing for free space
                x0, y0 = robot_grid_x, robot_grid_y
                x1, y1 = hit_grid_x, hit_grid_y

                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                sx = 1 if x0 < x1 else -1
                sy = 1 if y0 < y1 else -1
                err = dx - dy

                while True:
                    if self.is_valid_grid_coord(x0, y0):
                        if (x0, y0) != (hit_grid_x, hit_grid_y):
                            self.log_odds_grid[y0, x0] += self.L_free

                    if x0 == x1 and y0 == y1:
                        break
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x0 += sx
                    if e2 < dx:
                        err += dx
                        y0 += sy

        # Update visual features grid using camera data
        if camera_image is not None:
            self.update_visual_features(robot_pose, camera_image)

    @performance_monitor
    def update_visual_features(self, robot_pose, camera_image):
        # Extract visual features from the camera image and project them onto the grid
        # This is a simplified approach - in practice, you'd use depth information
        # from the Astra camera to get 3D coordinates of features
        
        robot_x, robot_y, robot_theta = robot_pose
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)

        # Convert camera image to grayscale for feature detection
        gray_image = cv2.cvtColor(camera_image, cv2.COLOR_BGRA2GRAY)
        
        # Detect corners using Harris corner detector
        corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        if corners is not None:
            # For each detected corner, project it to the world coordinate system
            # This is a simplified projection assuming features are on the ground plane
            # In practice, you'd use camera calibration and depth information
            
            camera_height = camera_image.shape[0]
            camera_width = camera_image.shape[1]
            
            for corner in corners:
                x_pixel, y_pixel = corner.ravel()
                
                # Simple projection: assume features are at a fixed distance in front of the robot
                # This is a very rough approximation
                feature_distance = 1.0  # meters (this should come from depth sensor)
                
                # Convert pixel coordinates to angle relative to robot's heading
                # Assuming camera FOV is about 60 degrees (this should be calibrated)
                camera_fov = np.pi / 3  # 60 degrees
                angle_per_pixel = camera_fov / camera_width
                feature_angle = (x_pixel - camera_width / 2) * angle_per_pixel
                
                # Calculate world coordinates of the feature
                feature_x_world = robot_x + feature_distance * np.cos(robot_theta + feature_angle)
                feature_y_world = robot_y + feature_distance * np.sin(robot_theta + feature_angle)
                
                # Update visual features grid
                feature_grid_x, feature_grid_y = self.world_to_grid(feature_x_world, feature_y_world)
                if self.is_valid_grid_coord(feature_grid_x, feature_grid_y):
                    self.visual_features_grid[feature_grid_y, feature_grid_x] = 255

    def get_probability_grid(self):
        return 1 - (1 / (1 + np.exp(self.log_odds_grid)))

    def save_map_as_image(self, filename="enhanced_occupancy_grid.png", lidar_points_to_plot=None):
        prob_grid = self.get_probability_grid()
        image_data = (1 - prob_grid) * 255
        image_data = image_data.astype(np.uint8)
        
        # Create RGB image to overlay visual features and LiDAR points
        rgb_image = np.stack([image_data, image_data, image_data], axis=-1)
        
        # Overlay visual features in red
        feature_mask = self.visual_features_grid > 0
        rgb_image[feature_mask, 0] = 255  # Red channel
        rgb_image[feature_mask, 1] = 0    # Green channel
        rgb_image[feature_mask, 2] = 0    # Blue channel
        
        # Overlay LiDAR points in blue
        if lidar_points_to_plot is not None:
            for point in lidar_points_to_plot:
                if hasattr(point, 'x') and hasattr(point, 'y'):
                    px, py = self.world_to_grid(point.x, point.y)
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    px, py = self.world_to_grid(point[0], point[1])
                else:
                    continue
                
                if self.is_valid_grid_coord(px, py):
                    # Draw a small blue circle for each LiDAR point
                    cv2.circle(rgb_image, (py, px), 1, (255, 0, 0), -1) # OpenCV uses (col, row) for (x,y)

        img = Image.fromarray(rgb_image)
        img.save(filename)
        print(f"Enhanced map saved to {filename}.")

    @performance_monitor
    def detect_loop_closure(self, current_pose, threshold=0.5):
        # Simple loop closure detection based on visual features
        # This is a very basic implementation
        
        robot_x, robot_y, robot_theta = current_pose
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
        
        # Check if there are visual features in the vicinity that we've seen before
        search_radius = int(2.0 / self.resolution)  # 2 meter search radius
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                check_x = robot_grid_x + dx
                check_y = robot_grid_y + dy
                
                if self.is_valid_grid_coord(check_x, check_y):
                    if self.visual_features_grid[check_y, check_x] > 0:
                        # Found a previously seen visual feature nearby
                        distance = np.sqrt(dx**2 + dy**2) * self.resolution
                        if distance < threshold:
                            return True
        
        return False