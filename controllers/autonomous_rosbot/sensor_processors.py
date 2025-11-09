#!/usr/bin/env python3
"""
Professional Sensor Processors with Performance Optimization
Memory-efficient implementations ready for C++ migration
Author: Sensor Systems Engineer - October 2025
"""

import time
import numpy as np
import cv2
import math
from typing import Dict, List, Optional, Any, Tuple
import psutil
import os

from core_types import (
    Point2D, Point3D, SensorData, Pillar, SensorType, 
    PerformanceProfiler, MemoryOptimizer, SystemConfig
)
from interfaces import ILidarProcessor, ICameraProcessor, ISensorProcessor

class OptimizedLidarProcessor(ILidarProcessor):
    """High-performance LIDAR processor with memory optimization

    Designed for easy C++ migration with minimal memory footprint.
    """

    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        super().__init__(profiler)
        self.sensor_type = SensorType.LIDAR

        # Performance optimization parameters
        self.max_range = SystemConfig.LIDAR_MAX_RANGE
        self.min_range = SystemConfig.LIDAR_MIN_RANGE
        self.angular_resolution = 0.5  # degrees
        self.passage_detection_threshold = 0.3  # radians for gap detection
        self.wall_fitting_tolerance = 0.05  # meters

        # Memory optimization
        self.point_buffer_size = 5000  # Limit point cloud size
        self.enable_voxel_filtering = True
        self.voxel_size = 0.02  # 2cm voxels

        # Performance tracking
        self.processing_times = []
        self.memory_usage = []

    def process_data(self, raw_data: Dict[str, Any]) -> SensorData:
        """Process LIDAR data with performance monitoring"""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            ranges = raw_data.get('ranges', [])
            angles = raw_data.get('angles', [])

            if not self.is_data_valid(raw_data):
                return SensorData(
                    timestamp=time.time(),
                    sensor_type=self.sensor_type,
                    data={'error': 'Invalid LIDAR data'},
                    processing_time=time.time() - start_time
                )

            # Convert to numpy for efficient processing
            ranges_array = np.array(ranges, dtype=np.float32)
            angles_array = np.array(angles, dtype=np.float32)

            # Filter valid measurements
            valid_mask = self._create_validity_mask(ranges_array)

            # Extract features
            passages = self.detect_passages(ranges_array[valid_mask], angles_array[valid_mask])
            walls = self.detect_walls(ranges_array[valid_mask], angles_array[valid_mask])
            front_clearance = self.get_front_clearance(ranges_array, angles_array)

            # Generate optimized point cloud
            points_3d = self._generate_point_cloud(ranges_array[valid_mask], angles_array[valid_mask])

            # Memory optimization
            if self.enable_voxel_filtering and len(points_3d) > self.point_buffer_size:
                points_3d = MemoryOptimizer.optimize_point_cloud(points_3d, self.voxel_size)

            processing_time = time.time() - start_time
            memory_used = self._get_memory_usage() - start_memory

            # Record performance metrics
            if self.profiler:
                self.profiler.record_timing("lidar_processing", processing_time)
                self.profiler.record_memory("lidar_processing", memory_used)

            self.processing_times.append(processing_time)
            self.memory_usage.append(memory_used)

            return SensorData(
                timestamp=time.time(),
                sensor_type=self.sensor_type,
                data={
                    'points_3d': points_3d,
                    'passages': passages,
                    'walls': walls,
                    'front_clearance': front_clearance,
                    'valid_points_count': np.sum(valid_mask),
                    'total_points_count': len(ranges)
                },
                processing_time=processing_time
            )

        except Exception as e:
            return SensorData(
                timestamp=time.time(),
                sensor_type=self.sensor_type,
                data={'error': str(e)},
                processing_time=time.time() - start_time
            )

    def is_data_valid(self, raw_data: Dict[str, Any]) -> bool:
        """Validate LIDAR data with performance checks"""
        ranges = raw_data.get('ranges', [])
        angles = raw_data.get('angles', [])

        if not ranges or not angles:
            return False

        if len(ranges) != len(angles):
            return False

        # Check for reasonable data size (performance consideration)
        if len(ranges) > 2000:  # Typical LIDAR has ~720 points
            return False

        return True

    def detect_passages(self, ranges: np.ndarray, angles: np.ndarray) -> List[Dict[str, Any]]:
        """Optimized passage detection with vectorized operations"""
        if len(ranges) < 10:
            return []

        passages = []

        # Sort by angle for efficient processing
        sort_indices = np.argsort(angles)
        sorted_angles = angles[sort_indices]
        sorted_ranges = ranges[sort_indices]

        # Vectorized gap detection
        angle_diffs = np.diff(sorted_angles)
        gap_indices = np.where(angle_diffs > self.passage_detection_threshold)[0]

        for gap_idx in gap_indices:
            if gap_idx < len(sorted_ranges) - 1:
                # Calculate passage properties
                left_angle = sorted_angles[gap_idx]
                right_angle = sorted_angles[gap_idx + 1]
                left_range = sorted_ranges[gap_idx]
                right_range = sorted_ranges[gap_idx + 1]

                # Convert to Cartesian coordinates
                left_x = left_range * np.cos(left_angle)
                left_y = left_range * np.sin(left_angle)
                right_x = right_range * np.cos(right_angle)
                right_y = right_range * np.sin(right_angle)

                passage_width = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
                passage_direction = np.mean([left_angle, right_angle])

                # Filter valid passages
                if 0.5 < passage_width < 3.0:
                    passages.append({
                        'width': float(passage_width),
                        'direction': float(passage_direction),
                        'left_point': Point2D(float(left_x), float(left_y)),
                        'right_point': Point2D(float(right_x), float(right_y)),
                        'confidence': min(1.0, passage_width / 2.0)
                    })

        return passages

    def detect_walls(self, ranges: np.ndarray, angles: np.ndarray) -> List[Dict[str, Any]]:
        """Optimized wall detection using RANSAC-like approach"""
        if len(ranges) < 5:
            return []

        walls = []

        # Convert to Cartesian coordinates
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)
        points = np.column_stack([x_coords, y_coords])

        # Cluster nearby points efficiently
        clustered_points = self._fast_clustering(points, distance_threshold=0.15)

        for cluster in clustered_points:
            if len(cluster) >= 3:
                wall_params = self._fit_line_ransac(cluster)
                if wall_params:
                    walls.append(wall_params)

        return walls

    def get_front_clearance(self, ranges: np.ndarray, angles: np.ndarray) -> float:
        """Get front clearance distance with vectorized operations"""
        # Front sector mask (±15 degrees)
        front_mask = np.abs(angles) < np.radians(15)

        if not np.any(front_mask):
            return 0.0

        front_ranges = ranges[front_mask]
        valid_front = front_ranges[(front_ranges > self.min_range) & (front_ranges < self.max_range)]

        return float(np.min(valid_front)) if len(valid_front) > 0 else 0.0

    def get_processing_stats(self) -> Dict[str, float]:
        """Get detailed processing statistics"""
        if not self.processing_times:
            return {}

        return {
            'avg_processing_time_ms': np.mean(self.processing_times) * 1000,
            'max_processing_time_ms': np.max(self.processing_times) * 1000,
            'avg_memory_usage_mb': np.mean(self.memory_usage),
            'max_memory_usage_mb': np.max(self.memory_usage),
            'total_calls': len(self.processing_times)
        }

    def _create_validity_mask(self, ranges: np.ndarray) -> np.ndarray:
        """Create validity mask for range data"""
        return ((ranges > self.min_range) & 
                (ranges < self.max_range) & 
                np.isfinite(ranges))

    def _generate_point_cloud(self, ranges: np.ndarray, angles: np.ndarray) -> List[Point3D]:
        """Generate optimized 3D point cloud"""
        points = []

        # Vectorized coordinate transformation
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)
        z_coords = np.zeros_like(ranges)  # 2D LIDAR

        # Convert to Point3D objects (could be optimized to use numpy arrays in C++)
        for x, y, z in zip(x_coords, y_coords, z_coords):
            points.append(Point3D(float(x), float(y), float(z)))

        return points

    def _fast_clustering(self, points: np.ndarray, distance_threshold: float) -> List[np.ndarray]:
        """Fast clustering using vectorized operations"""
        if len(points) == 0:
            return []

        clusters = []
        unassigned = np.arange(len(points))

        while len(unassigned) > 0:
            # Start new cluster with first unassigned point
            seed_idx = unassigned[0]
            seed_point = points[seed_idx]

            # Find all points within distance threshold
            distances = np.linalg.norm(points[unassigned] - seed_point, axis=1)
            cluster_mask = distances < distance_threshold
            cluster_indices = unassigned[cluster_mask]

            if len(cluster_indices) >= 3:
                clusters.append(points[cluster_indices])

            # Remove assigned points
            unassigned = unassigned[~cluster_mask]

        return clusters

    def _fit_line_ransac(self, points: np.ndarray) -> Optional[Dict[str, Any]]:
        """Fit line using simplified RANSAC"""
        if len(points) < 3:
            return None

        try:
            # Simple linear regression (could be upgraded to RANSAC in C++)
            x_coords = points[:, 0]
            y_coords = points[:, 1]

            # Fit line y = mx + b
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, b = np.linalg.lstsq(A, y_coords, rcond=None)[0]

            # Calculate line endpoints
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            start_point = Point2D(float(x_min), float(m * x_min + b))
            end_point = Point2D(float(x_max), float(m * x_max + b))

            line_length = start_point.distance_to(end_point)

            return {
                'start': start_point,
                'end': end_point,
                'slope': float(m),
                'intercept': float(b),
                'length': float(line_length),
                'points_count': len(points)
            }

        except (np.linalg.LinAlgError, ValueError):
            return None

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

class OptimizedCameraProcessor(ICameraProcessor):
    """High-performance camera processor for pillar detection

    Optimized for real-time performance with memory-efficient operations.
    """

    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        super().__init__(profiler)
        self.sensor_type = SensorType.RGB_CAMERA

        # Color ranges for pillar detection (HSV)
        self.color_ranges = {
            'blue': [(np.array([100, 80, 80]), np.array([130, 255, 255]))],
            'yellow': [(np.array([20, 80, 80]), np.array([35, 255, 255]))],
            'red': [(np.array([0, 80, 80]), np.array([10, 255, 255])),
                   (np.array([170, 80, 80]), np.array([180, 255, 255]))]
        }

        # Performance parameters
        self.min_pillar_area = 400
        self.max_pillar_area = 5000
        self.circularity_threshold = 0.3
        self.aspect_ratio_range = (0.5, 2.0)

        # Camera calibration (approximate)
        self.camera_fov_degrees = SystemConfig.CAMERA_FOV_DEGREES
        self.known_pillar_diameter = 0.2  # 20cm

        # Performance tracking
        self.processing_times = []
        self.detection_counts = []

    def process_data(self, raw_data: Dict[str, Any]) -> SensorData:
        """Process camera data with performance monitoring"""
        start_time = time.time()

        try:
            image = raw_data.get('image')

            if not self.is_data_valid(raw_data):
                return SensorData(
                    timestamp=time.time(),
                    sensor_type=self.sensor_type,
                    data={'error': 'Invalid camera data'},
                    processing_time=time.time() - start_time
                )

            # Detect pillars
            pillars = self.detect_pillars(image)

            # Classify walls
            wall_classification = self.classify_walls(image)

            processing_time = time.time() - start_time

            # Record performance
            if self.profiler:
                self.profiler.record_timing("camera_processing", processing_time)

            self.processing_times.append(processing_time)
            self.detection_counts.append(len(pillars))

            return SensorData(
                timestamp=time.time(),
                sensor_type=self.sensor_type,
                data={
                    'pillars': pillars,
                    'wall_classification': wall_classification,
                    'detections_count': len(pillars)
                },
                processing_time=processing_time
            )

        except Exception as e:
            return SensorData(
                timestamp=time.time(),
                sensor_type=self.sensor_type,
                data={'error': str(e)},
                processing_time=time.time() - start_time
            )

    def is_data_valid(self, raw_data: Dict[str, Any]) -> bool:
        """Validate camera data"""
        image = raw_data.get('image')

        if image is None:
            return False

        if not isinstance(image, np.ndarray):
            return False

        if len(image.shape) != 3 or image.shape[2] != 3:
            return False

        # Check reasonable image size
        height, width = image.shape[:2]
        if width < 100 or height < 100 or width > 2000 or height > 2000:
            return False

        return True

    def detect_pillars(self, image: np.ndarray) -> List[Pillar]:
        """Detect colored pillars with optimized computer vision"""
        pillars = []

        if image is None or image.size == 0:
            return pillars

        height, width = image.shape[:2]

        # Convert to HSV once
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Apply Gaussian blur for noise reduction
        hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        for color_name, color_ranges in self.color_ranges.items():
            # Skip red for pillar detection (used for wall classification)
            if color_name == 'red':
                continue

            # Create combined mask for all color ranges
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for lower, upper in color_ranges:
                color_mask = cv2.inRange(hsv_blurred, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, color_mask)

            # Morphological operations for noise reduction
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                pillar = self._analyze_contour(contour, color_name, width, height)
                if pillar:
                    pillars.append(pillar)

        return pillars

    def classify_walls(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify walls by color (red indicates dead end)"""
        if image is None or image.size == 0:
            return {'red_walls': [], 'normal_walls_ratio': 1.0}

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_walls = []

        # Detect red walls
        for lower, upper in self.color_ranges['red']:
            red_mask = cv2.inRange(hsv, lower, upper)

            # Find large red areas
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)

                if area > 1000:  # Large red area
                    rect = cv2.boundingRect(contour)
                    aspect_ratio = rect[2] / rect[3] if rect[3] > 0 else 0

                    # Check if it looks like a wall
                    if aspect_ratio > 1.5 or aspect_ratio < 0.67:
                        red_walls.append({
                            'area': float(area),
                            'bounding_rect': rect,
                            'is_deadend': True,
                            'aspect_ratio': float(aspect_ratio)
                        })

        # Calculate wall area ratio
        total_image_area = image.shape[0] * image.shape[1]
        red_wall_area = sum(wall['area'] for wall in red_walls)
        normal_walls_ratio = 1.0 - (red_wall_area / total_image_area)

        return {
            'red_walls': red_walls,
            'normal_walls_ratio': float(normal_walls_ratio),
            'red_wall_count': len(red_walls)
        }

    def estimate_pillar_distance(self, pillar_area: float, pillar_color: str) -> float:
        """Estimate distance using known pillar size"""
        if pillar_area <= 0:
            return 5.0

        # Calculate focal length in pixels (approximate)
        image_width_pixels = 640  # Typical camera width
        focal_length_pixels = image_width_pixels / (2 * math.tan(math.radians(self.camera_fov_degrees / 2)))

        # Estimate pillar diameter in pixels
        pillar_diameter_pixels = math.sqrt(pillar_area * 4 / math.pi)

        if pillar_diameter_pixels > 0:
            distance = (self.known_pillar_diameter * focal_length_pixels) / pillar_diameter_pixels
            return np.clip(distance, 0.3, 8.0)

        return 5.0

    def get_processing_stats(self) -> Dict[str, float]:
        """Get camera processing statistics"""
        if not self.processing_times:
            return {}

        return {
            'avg_processing_time_ms': np.mean(self.processing_times) * 1000,
            'max_processing_time_ms': np.max(self.processing_times) * 1000,
            'avg_detections_per_frame': np.mean(self.detection_counts),
            'total_frames_processed': len(self.processing_times)
        }

    def _analyze_contour(self, contour: np.ndarray, color: str, 
                        image_width: int, image_height: int) -> Optional[Pillar]:
        """Analyze contour to determine if it's a valid pillar"""
        area = cv2.contourArea(contour)

        # Area filter
        if not (self.min_pillar_area < area < self.max_pillar_area):
            return None

        # Shape analysis
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None

        circularity = 4 * math.pi * area / (perimeter ** 2)

        if circularity < self.circularity_threshold:
            return None

        # Aspect ratio check
        rect = cv2.boundingRect(contour)
        aspect_ratio = rect[2] / rect[3] if rect[3] > 0 else 0

        if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
            return None

        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Calculate bearing and distance
        bearing = (cx - image_width/2) * math.radians(self.camera_fov_degrees) / image_width
        distance = self.estimate_pillar_distance(area, color)

        # Create pillar object
        return Pillar(
            position=Point2D(0, 0),  # Will be set by controller with world coordinates
            color=color,
            confidence=min(1.0, circularity * area / 1000.0),
            detection_count=1,
            last_seen=time.time()
        )

if __name__ == "__main__":
    print("🔍 Optimized Sensor Processors Initialized")

    # Performance testing
    profiler = PerformanceProfiler()

    # Test LIDAR processor
    lidar = OptimizedLidarProcessor(profiler)
    print(f"✅ LIDAR Processor: {lidar.max_range}m range, {lidar.voxel_size*1000:.0f}mm voxels")

    # Test camera processor
    camera = OptimizedCameraProcessor(profiler)
    print(f"✅ Camera Processor: {len(camera.color_ranges)} colors, {camera.camera_fov_degrees}° FOV")

    # Memory optimization test
    test_points = [Point3D(i*0.1, i*0.1, 0) for i in range(1000)]
    optimized = MemoryOptimizer.optimize_point_cloud(test_points, voxel_size=0.1)
    print(f"📊 Memory optimization: {len(test_points)} → {len(optimized)} points")

    print("🚀 Ready for high-performance sensor processing")
    print("⚡ C++ optimization targets identified")
