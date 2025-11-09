
import numpy as np
import cv2
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import logging
import time
from slam_logging_config import get_slam_logger, performance_monitor, SLAMLogger

@dataclass
class DetectedPillar:
    """Represents a detected pillar with its properties"""
    color: str = ""  # 'blue' or 'yellow'
    world_position: Tuple[float, float] = (0.0, 0.0)  # (x, y) in world coordinates
    confidence: float = 1.0  # Detection confidence [0.0, 1.0]
    pixel_position: Tuple[int, int] = (0, 0)  # (u, v) in image coordinates
    area: int = 0  # Pixel area of detection
    timestamp: float = 0.0  # When detected
    
class PillarDetector:
    """
    Computer vision system for detecting colored pillars using camera and LiDAR data.
    Performs color segmentation in HSV space and calculates 3D world positions.
    """
    
    def __init__(self, camera_width=640, camera_height=480, camera_fov=1.047):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fov = camera_fov  # Field of view in radians
        
        # Color ranges in HSV space (tuned for typical lighting)
        self.color_ranges = {
            'blue': {
                'lower': np.array([100, 50, 50]),   # Lower HSV bound for blue
                'upper': np.array([130, 255, 255]) # Upper HSV bound for blue
            },
            'yellow': {
                'lower': np.array([20, 50, 50]),    # Lower HSV bound for yellow
                'upper': np.array([30, 255, 255])  # Upper HSV bound for yellow
            }
        }
        
        # Detection parameters
        self.min_contour_area = 100      # Minimum pixel area for valid detection
        self.max_contour_area = 10000    # Maximum pixel area to avoid false positives
        self.confidence_threshold = 0.5  # Minimum confidence for valid detection
        self.max_detection_distance = 3.0  # Maximum distance to detect pillars (meters)
        
        # Filtering parameters
        self.gaussian_blur_kernel = (5, 5)
        self.morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Detection history for stability
        self.detection_history = []
        self.max_history_size = 10
        
    @performance_monitor
    def detect_pillars(self, camera_image, robot_pose, lidar_ranges=None):
        """
        Main pillar detection function.
        
        # Args:
            camera_image: BGR image from camera
            robot_pose: (x, y, theta) robot pose in world coordinates
            lidar_ranges: Optional LiDAR data for distance estimation
            
        # Returns:
            List of DetectedPillar objects
        """
        if camera_image is None or camera_image.size == 0:
            return []
        
        # Convert to HSV for better color segmentation
        hsv_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        blurred_hsv = cv2.GaussianBlur(hsv_image, self.gaussian_blur_kernel, 0)
        
        detected_pillars = []
        
        # Detect each color
        for color_name, color_range in self.color_ranges.items():
            pillars = self._detect_color_pillars(
                blurred_hsv, camera_image, color_name, color_range,
                robot_pose, lidar_ranges
            )
            detected_pillars.extend(pillars)
        
        # Filter and validate detections
        valid_pillars = self._filter_detections(detected_pillars)
        
        # Update detection history
        self._update_detection_history(valid_pillars)
        
        return valid_pillars
    
    def _detect_color_pillars(self, hsv_image, original_image, color_name, color_range, 
                            robot_pose, lidar_ranges):
        """Detect pillars of a specific color."""
        # Create color mask
        mask = cv2.inRange(hsv_image, color_range['lower'], color_range['upper'])
        
        # Apply morphological operations to clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morphology_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morphology_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_pillars = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
                
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Calculate confidence based on color consistency and shape
            confidence = self._calculate_detection_confidence(
                mask, contour, area, w, h
            )
            
            if confidence < self.confidence_threshold:
                continue
            
            # Estimate 3D world position
            world_pos = self._estimate_world_position(
                cx, cy, robot_pose, lidar_ranges
            )
            if world_pos is None:
                continue
            
            # Create pillar detection
            pillar = DetectedPillar(
                color=color_name,
                world_position=world_pos,
                confidence=confidence,
                pixel_position=(cx, cy),
                area=area,
                timestamp=0.0  # Should be set by caller with actual timestamp
            )
            
            detected_pillars.append(pillar)
        
        return detected_pillars
    
    def _calculate_detection_confidence(self, mask, contour, area, width, height):
        """
        Calculate confidence score for a detection based on multiple factors.
        """
        # Shape factor: pillars should be roughly circular or rectangular
        aspect_ratio = width / height if height > 0 else 0
        shape_score = 1.0 - abs(aspect_ratio - 1.0)  # Prefer square-ish shapes
        shape_score = max(0.0, min(1.0, shape_score))
        
        # Area factor: normalize based on expected pillar size
        area_score = min(area / 1000.0, 1.0)  # Prefer larger detections up to a point
        
        # Solidity: ratio of contour area to its convex hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Extent: ratio of contour area to bounding rectangle area
        rect_area = width * height
        extent = area / rect_area if rect_area > 0 else 0
        
        # Combined confidence score
        confidence = (0.3 * shape_score + 
                     0.2 * area_score + 
                     0.3 * solidity + 
                     0.2 * extent)
        
        return confidence
    
    def _estimate_world_position(self, pixel_x, pixel_y, robot_pose, lidar_ranges):
        """
        Estimate 3D world position of detected pillar using camera projection and LiDAR data.
        """
        robot_x, robot_y, robot_theta = robot_pose

        # Calculate bearing angle from camera center
        # Camera FOV spans the full image width
        pixel_offset = pixel_x - (self.camera_width / 2)
        bearing_offset = (pixel_offset / (self.camera_width / 2)) * (self.camera_fov / 2)
        
        # Absolute bearing in world coordinates
        absolute_bearing = robot_theta + bearing_offset
        
        # Estimate distance using LiDAR data if available
        distance = self._estimate_distance_from_lidar(bearing_offset, lidar_ranges)
        
        if distance is None or distance > self.max_detection_distance:
            # Fallback: estimate based on pixel size (less accurate)
            distance = self._estimate_distance_from_pixel_size(pixel_y)
            
        if distance is None:
            return None
        
        # Calculate world coordinates
        world_x = robot_x + distance * math.cos(absolute_bearing)
        world_y = robot_y + distance * math.sin(absolute_bearing)
        
        return (world_x, world_y)
    
    def _estimate_distance_from_lidar(self, bearing_offset, lidar_ranges):
        """
        Estimate distance to pillar using LiDAR data.
        Maps camera bearing to corresponding LiDAR ray.
        """
        if lidar_ranges is None or len(lidar_ranges) == 0:
            return None
        
        # Map bearing offset to LiDAR ray index
        # Assume LiDAR has 360-degree coverage with uniform angular resolution
        num_rays = len(lidar_ranges)
        lidar_angular_resolution = 2 * math.pi / num_rays
        
        # Convert bearing offset to LiDAR ray index
        ray_index = int((bearing_offset + math.pi) / lidar_angular_resolution) % num_rays
        
        # Get distance from corresponding LiDAR ray
        distance = lidar_ranges[ray_index]
        
        # Validate distance
        if distance <= 0 or distance > self.max_detection_distance:
            return None
        
        return distance
    
    def _estimate_distance_from_pixel_size(self, pixel_y):
        """
        Rough distance estimation based on pillar's vertical position in image.
        This is less accurate but serves as a fallback.
        """
        # Assume pillars are on the ground and camera height is known
        camera_height = 0.1  # Approximate camera height above ground (meters)
        
        # Calculate depression angle based on pixel position
        pixel_offset_y = pixel_y - (self.camera_height / 2)
        depression_angle = (pixel_offset_y / (self.camera_height / 2)) * (self.camera_fov / 2)
        
        # Estimate distance using trigonometry
        if abs(depression_angle) < 0.01:  # Avoid division by small numbers
            return 2.0  # Default distance
        
        distance = camera_height / math.tan(abs(depression_angle))
        
        # Clamp to reasonable range
        return max(0.5, min(distance, self.max_detection_distance))
    
    def _filter_detections(self, detections):
        """
        Filter and validate detections to remove false positives.
        """
        if not detections:
            return []
        
        # Remove duplicates (multiple detections of same pillar)
        filtered = []
        for detection in detections:
            is_duplicate = False
            for existing in filtered:
                # Check if positions are very close
                distance = math.sqrt(
                    (detection.world_position[0] - existing.world_position[0])**2 +
                    (detection.world_position[1] - existing.world_position[1])**2
                )
                if distance < 0.3:  # 30cm threshold for same pillar
                    # Keep the one with higher confidence
                    if detection.confidence > existing.confidence:
                        filtered.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def _update_detection_history(self, current_detections):
        """
        Update detection history for temporal consistency.
        """
        self.detection_history.append(current_detections)
        
        # Maintain history size
        if len(self.detection_history) > self.max_history_size:
            self.detection_history.pop(0)
    
    def get_stable_detections(self, min_detections=3):
        """
        Get pillars that have been consistently detected over multiple frames.
        This provides more reliable pillar locations.
        """
        if len(self.detection_history) < min_detections:
            return []
        
        stable_pillars = []
        
        # For each color, find consistently detected pillars
        for color in ['blue', 'yellow']:
            color_detections = []
            
            # Collect all detections of this color from history
            for frame_detections in self.detection_history[-min_detections:]:
                frame_color_detections = [d for d in frame_detections if d.color == color]
                color_detections.append(frame_color_detections)
            
            # Find consistently detected positions
            for reference_frame in color_detections[0]:
                consistent_detections = [reference_frame]
                
                for other_frame in color_detections[1:]:
                    best_match = None
                    best_distance = float('inf')
                    
                    for detection in other_frame:
                        distance = math.sqrt(
                            (reference_frame.world_position[0] - detection.world_position[0])**2 +
                            (reference_frame.world_position[1] - detection.world_position[1])**2
                        )
                        if distance < best_distance and distance < 0.5:  # 50cm threshold
                            best_distance = distance
                            best_match = detection
                    
                    if best_match:
                        consistent_detections.append(best_match)
                
                # If detected in enough frames, consider it stable
                if len(consistent_detections) >= min_detections:
                    # Calculate average position and confidence
                    avg_x = sum(d.world_position[0] for d in consistent_detections) / len(consistent_detections)
                    avg_y = sum(d.world_position[1] for d in consistent_detections) / len(consistent_detections)
                    avg_confidence = sum(d.confidence for d in consistent_detections) / len(consistent_detections)
                    
                    stable_pillar = DetectedPillar(
                        color=color,
                        world_position=(avg_x, avg_y),
                        confidence=avg_confidence,
                        pixel_position=reference_frame.pixel_position,
                        area=reference_frame.area,
                        timestamp=reference_frame.timestamp
                    )
                    
                    stable_pillars.append(stable_pillar)
        
        return stable_pillars
    
    def visualize_detections(self, image, detections, robot_position=None):
        """
        Create a visualization of detected pillars on the camera image.
        """
        vis_image = image.copy()
        
        for pillar in detections:
            # Color for visualization
            if pillar.color == 'blue':
                draw_color = (255, 0, 0)  # Blue in BGR
            else:  # yellow
                draw_color = (0, 255, 255)  # Yellow in BGR
            
            # Draw detection circle
            center = pillar.pixel_position
            radius = int(math.sqrt(pillar.area / math.pi))
            cv2.circle(vis_image, center, radius, draw_color, 2)
            
            # Draw confidence text
            confidence_text = f"{pillar.color}: {pillar.confidence:.2f}"
            cv2.putText(vis_image, confidence_text, 
                       (center[0] - 30, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)
            
            # Draw world position if robot position is known
            if robot_position:
                distance = math.sqrt(
                    (pillar.world_position[0] - robot_position[0])**2 +
                    (pillar.world_position[1] - robot_position[1])**2
                )
                distance_text = f"{distance:.1f}m"
                cv2.putText(vis_image, distance_text,
                           (center[0] - 20, center[1] + radius + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)
        
        return vis_image


class PillarMapper:
    """
    Manages detected pillars and integrates them with the occupancy grid.
    Maintains a map of known pillar locations.
    """
    
    def __init__(self, occupancy_grid):
        self.occupancy_grid = occupancy_grid
        self.known_pillars = {}  # Dict[color, List[DetectedPillar]]
        self.pillar_merge_distance = 0.5  # Merge pillars within 50cm
        
    def update_pillar_map(self, new_detections):
        """
        Update the pillar map with new detections.
        Merges nearby detections and maintains the best estimates.
        """
        for detection in new_detections:
            color = detection.color
            
            if color not in self.known_pillars:
                self.known_pillars[color] = []
            
            # Check if this is a new pillar or update to existing one
            merged = False
            for i, existing_pillar in enumerate(self.known_pillars[color]):
                distance = math.sqrt(
                    (detection.world_position[0] - existing_pillar.world_position[0])**2 +
                    (detection.world_position[1] - existing_pillar.world_position[1])**2
                )
                
                if distance < self.pillar_merge_distance:
                    # Update existing pillar with weighted average
                    if detection.confidence > existing_pillar.confidence:
                        # Use new detection if it's more confident
                        self.known_pillars[color][i] = detection
                    else:
                        # Weighted average based on confidence
                        w1 = existing_pillar.confidence
                        w2 = detection.confidence
                        total_weight = w1 + w2
                        
                        avg_x = (existing_pillar.world_position[0] * w1 + detection.world_position[0] * w2) / total_weight
                        avg_y = (existing_pillar.world_position[1] * w1 + detection.world_position[1] * w2) / total_weight
                        avg_confidence = (w1 + w2) / 2  # Average confidence
                        
                        self.known_pillars[color][i] = DetectedPillar(
                            color=color,
                            world_position=(avg_x, avg_y),
                            confidence=avg_confidence,
                            pixel_position=detection.pixel_position,
                            area=detection.area,
                            timestamp=detection.timestamp
                        )
                    
                    merged = True
                    break
            
            if not merged:
                # New pillar
                self.known_pillars[color].append(detection)
    
    def get_all_pillars(self):
        """Get all known pillars."""
        all_pillars = []
        for color_pillars in self.known_pillars.values():
            all_pillars.extend(color_pillars)
        return all_pillars
    
    def get_pillars_by_color(self, color):
        """Get pillars of a specific color."""
        return self.known_pillars.get(color, [])
    
    def find_nearest_pillars(self, robot_position, max_count=5, max_distance=2.0):
        """Find pillars within a certain distance of the robot."""
        nearby_pillars = []
        
        for pillar in self.get_all_pillars():
            distance = math.sqrt(
                (pillar.world_position[0] - robot_position[0])**2 +
                (pillar.world_position[1] - robot_position[1])**2
            )
            
            if distance <= max_distance:
                pillar_dict = {
                    'x': pillar.world_position[0],
                    'y': pillar.world_position[1],
                    'color': pillar.color,
                    'confidence': pillar.confidence,
                    'distance': distance
                }
                nearby_pillars.append(pillar_dict)
        
        # Sort by distance and limit count
        nearby_pillars.sort(key=lambda x: x['distance'])
        return nearby_pillars[:max_count]
    
    def mark_pillars_in_grid(self):
        """
        Mark known pillars as occupied cells in the occupancy grid.
        This helps with path planning around pillars.
        """
        for pillar in self.get_all_pillars():
            # Mark pillar location as occupied
            grid_x, grid_y = self.occupancy_grid.world_to_grid(
                pillar.world_position[0], pillar.world_position[1]
            )
            
            if self.occupancy_grid.is_valid_grid_coord(grid_x, grid_y):
                # Mark a small area around the pillar as occupied
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        neighbor_x, neighbor_y = grid_x + dx, grid_y + dy
                        if self.occupancy_grid.is_valid_grid_coord(neighbor_x, neighbor_y):
                            self.occupancy_grid.update_cell(neighbor_x, neighbor_y, occupied=True)
    
    def get_pillar_pair_for_navigation(self):
        """
        Get blue and yellow pillar pair for A* navigation.
        Returns (blue_pillar, yellow_pillar) or (None, None) if not found.
        """
        blue_pillars = self.get_pillars_by_color('blue')
        yellow_pillars = self.get_pillars_by_color('yellow')
        
        if not blue_pillars or not yellow_pillars:
            return None, None
        
        # Return the most confident pillars of each color
        best_blue = max(blue_pillars, key=lambda p: p.confidence)
        best_yellow = max(yellow_pillars, key=lambda p: p.confidence)
        
        return best_blue, best_yellow
