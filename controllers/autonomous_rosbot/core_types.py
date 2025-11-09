#!/usr/bin/env python3
"""
Core Data Structures and Types
Professional autonomous exploration system
Author: Robotics Systems Engineer - October 2025
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from enum import Enum

class ExplorationState(Enum):
    """Robot exploration states for finite state machine"""
    SCANNING = "scanning"
    ALIGNING = "aligning"  
    MOVING_STRAIGHT = "moving_straight"
    DETECTING_TURN = "detecting_turn"
    TURNING = "turning"
    MAPPING_COMPLETE = "mapping_complete"
    PATH_PLANNING = "path_planning"
    COMPLETED = "completed"

class WallType(Enum):
    """Wall classification types for maze navigation"""
    NORMAL_WALL = "normal"
    RED_WALL_DEADEND = "red_deadend"
    PASSAGE_OPENING = "opening"
    UNKNOWN = "unknown"

class SensorType(Enum):
    """Available sensor types"""
    LIDAR = "lidar"
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    ENCODERS = "encoders"

@dataclass(frozen=True)
class Point3D:
    """3D point representation with memory optimization"""
    x: float
    y: float
    z: float

    def distance_to(self, other: 'Point3D') -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def to_2d(self) -> 'Point2D':
        """Project to 2D by dropping Z coordinate"""
        return Point2D(self.x, self.y)

@dataclass(frozen=True)
class Point2D:
    """2D point representation for top view mapping"""
    x: float
    y: float

    def distance_to(self, other: 'Point2D') -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def angle_to(self, other: 'Point2D') -> float:
        """Calculate angle to another point"""
        return math.atan2(other.y - self.y, other.x - self.x)

    def translate(self, dx: float, dy: float) -> 'Point2D':
        """Create translated point"""
        return Point2D(self.x + dx, self.y + dy)

    def rotate(self, angle: float, center: Optional['Point2D'] = None) -> 'Point2D':
        """Rotate point around center (origin if None)"""
        if center is None:
            center = Point2D(0, 0)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Translate to origin
        x = self.x - center.x
        y = self.y - center.y

        # Rotate
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a

        # Translate back
        return Point2D(new_x + center.x, new_y + center.y)

@dataclass
class RobotPose:
    """Robot pose with uncertainty representation"""
    x: float
    y: float
    theta: float  # Orientation in radians
    uncertainty: float = 0.0  # Pose uncertainty for future EKF implementation
    timestamp: float = 0.0

    def to_point2d(self) -> Point2D:
        """Convert to Point2D"""
        return Point2D(self.x, self.y)

    def distance_to(self, point: Point2D) -> float:
        """Calculate distance to a point"""
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)

@dataclass
class PassageSegment:
    """Passage segment information for navigation"""
    start_point: Point2D
    end_point: Point2D
    width: float
    length: float
    direction: float  # Angle in radians
    wall_type: WallType = WallType.NORMAL_WALL
    confidence: float = 0.0
    timestamp: float = 0.0

    def get_center_point(self) -> Point2D:
        """Get center point of passage"""
        return Point2D(
            (self.start_point.x + self.end_point.x) / 2,
            (self.start_point.y + self.end_point.y) / 2
        )

    def is_accessible(self, robot_width: float = 0.3) -> bool:
        """Check if passage is accessible for robot"""
        return self.width > robot_width * 1.2  # 20% safety margin

@dataclass
class Pillar:
    """Pillar/landmark information with tracking data"""
    position: Point2D
    color: str
    confidence: float
    detection_count: int = 0
    last_seen: float = 0.0
    world_position_history: List[Point2D] = field(default_factory=list)

    def update_position(self, new_position: Point2D, new_confidence: float, timestamp: float):
        """Update pillar position with weighted fusion"""
        self.world_position_history.append(new_position)

        # Keep only recent history
        if len(self.world_position_history) > 10:
            self.world_position_history = self.world_position_history[-10:]

        # Weighted average update
        total_confidence = self.confidence + new_confidence
        if total_confidence > 0:
            weight = new_confidence / total_confidence
            self.position = Point2D(
                self.position.x * (1 - weight) + new_position.x * weight,
                self.position.y * (1 - weight) + new_position.y * weight
            )
            self.confidence = min(1.0, total_confidence)

        self.detection_count += 1
        self.last_seen = timestamp

@dataclass
class SensorData:
    """Unified sensor data container"""
    timestamp: float
    sensor_type: SensorType
    data: Dict[str, Any]
    processing_time: float = 0.0

class PerformanceProfiler:
    """Performance profiling for optimization identification"""

    def __init__(self):
        self.timing_data: Dict[str, List[float]] = {}
        self.memory_data: Dict[str, List[float]] = {}

    def record_timing(self, operation: str, duration: float):
        """Record operation timing"""
        if operation not in self.timing_data:
            self.timing_data[operation] = []
        self.timing_data[operation].append(duration)

    def record_memory(self, operation: str, memory_mb: float):
        """Record memory usage"""
        if operation not in self.memory_data:
            self.memory_data[operation] = []
        self.memory_data[operation].append(memory_mb)

    def get_performance_report(self) -> Dict[str, Dict[str, float]]:
        """Generate performance analysis report"""
        report = {}

        for operation, timings in self.timing_data.items():
            if timings:
                report[operation] = {
                    'avg_time_ms': np.mean(timings) * 1000,
                    'max_time_ms': np.max(timings) * 1000,
                    'min_time_ms': np.min(timings) * 1000,
                    'std_time_ms': np.std(timings) * 1000,
                    'call_count': len(timings)
                }

        return report

    def identify_bottlenecks(self, threshold_ms: float = 10.0) -> List[str]:
        """Identify performance bottlenecks for C++ optimization"""
        bottlenecks = []
        report = self.get_performance_report()

        for operation, stats in report.items():
            if stats['avg_time_ms'] > threshold_ms:
                bottlenecks.append(operation)

        return sorted(bottlenecks, key=lambda op: report[op]['avg_time_ms'], reverse=True)

# Memory optimization utilities
class MemoryOptimizer:
    """Memory optimization utilities for large-scale mapping"""

    @staticmethod
    def compress_occupancy_grid(grid: np.ndarray) -> bytes:
        """Compress occupancy grid for memory efficiency"""
        # Simple run-length encoding for sparse occupancy grids
        compressed_data = []
        current_value = grid.flat[0]
        count = 1

        for value in grid.flat[1:]:
            if value == current_value and count < 255:
                count += 1
            else:
                compressed_data.extend([current_value, count])
                current_value = value
                count = 1

        compressed_data.extend([current_value, count])
        return bytes(compressed_data)

    @staticmethod
    def decompress_occupancy_grid(compressed_data: bytes, shape: Tuple[int, int]) -> np.ndarray:
        """Decompress occupancy grid"""
        data = list(compressed_data)
        decompressed = []

        for i in range(0, len(data), 2):
            value = data[i]
            count = data[i + 1]
            decompressed.extend([value] * count)

        return np.array(decompressed, dtype=np.int8).reshape(shape)

    @staticmethod
    def optimize_point_cloud(points: List[Point3D], voxel_size: float = 0.05) -> List[Point3D]:
        """Voxel grid filtering for point cloud optimization"""
        if not points:
            return []

        # Create voxel grid
        voxel_dict = {}

        for point in points:
            voxel_key = (
                int(point.x / voxel_size),
                int(point.y / voxel_size),
                int(point.z / voxel_size)
            )

            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = []
            voxel_dict[voxel_key].append(point)

        # Average points in each voxel
        optimized_points = []
        for voxel_points in voxel_dict.values():
            if voxel_points:
                avg_x = sum(p.x for p in voxel_points) / len(voxel_points)
                avg_y = sum(p.y for p in voxel_points) / len(voxel_points)
                avg_z = sum(p.z for p in voxel_points) / len(voxel_points)
                optimized_points.append(Point3D(avg_x, avg_y, avg_z))

        return optimized_points

# C++ integration utilities
class CppIntegrationConfig:
    """Configuration for C++ integration via pybind11"""

    CRITICAL_COMPONENTS = [
        "occupancy_mapping",
        "astar_pathfinding", 
        "lidar_processing",
        "point_cloud_processing"
    ]

    MEMORY_THRESHOLDS = {
        "occupancy_grid_mb": 1.0,
        "point_cloud_mb": 0.5,
        "sensor_buffer_mb": 0.2
    }

    PERFORMANCE_THRESHOLDS = {
        "mapping_update_ms": 5.0,
        "pathfinding_ms": 20.0,
        "sensor_processing_ms": 10.0
    }

# Global constants for system configuration
class SystemConfig:
    """System-wide configuration constants"""

    # Mapping parameters
    DEFAULT_MAP_WIDTH = 500
    DEFAULT_MAP_HEIGHT = 500
    DEFAULT_MAP_RESOLUTION = 0.04  # 4cm per cell
    DEFAULT_MAP_ORIGIN = (-10.0, -10.0)

    # Robot parameters
    ROBOT_WHEEL_RADIUS = 0.043
    ROBOT_WHEEL_BASE = 0.22
    ROBOT_MAX_VELOCITY = 26.0
    ROBOT_WIDTH = 0.3  # For passage accessibility checks

    # Control parameters
    MAX_LINEAR_SPEED = 0.5
    MAX_ANGULAR_SPEED = 1.2
    ALIGNMENT_TOLERANCE = 0.1  # radians
    TURN_TOLERANCE = 0.15      # radians

    # Sensor parameters
    LIDAR_MAX_RANGE = 15.0
    LIDAR_MIN_RANGE = 0.05
    CAMERA_FOV_DEGREES = 60.0

    # Performance parameters
    VISUALIZATION_FPS = 10
    SENSOR_PROCESSING_HZ = 20
    MAPPING_UPDATE_HZ = 10

    # Memory limits (MB)
    MAX_OCCUPANCY_GRID_MB = 2.0
    MAX_POINT_CLOUD_MB = 1.0
    MAX_SENSOR_BUFFER_MB = 0.5

if __name__ == "__main__":
    # Example usage and testing
    print("🏗️ Core Data Structures Initialized")

    # Test point operations
    p1 = Point3D(1.0, 2.0, 3.0)
    p2 = Point3D(4.0, 6.0, 8.0)
    print(f"3D Distance: {p1.distance_to(p2):.2f}m")

    p2d = p1.to_2d()
    print(f"2D Point: ({p2d.x:.2f}, {p2d.y:.2f})")

    # Test robot pose
    pose = RobotPose(x=0.0, y=0.0, theta=math.pi/4)
    print(f"Robot pose: ({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.theta):.1f}°)")

    # Test performance profiler
    profiler = PerformanceProfiler()
    profiler.record_timing("test_operation", 0.015)  # 15ms
    profiler.record_timing("test_operation", 0.012)  # 12ms

    bottlenecks = profiler.identify_bottlenecks(threshold_ms=10.0)
    print(f"Performance bottlenecks: {bottlenecks}")

    print("✅ Core types module ready for distributed system")
