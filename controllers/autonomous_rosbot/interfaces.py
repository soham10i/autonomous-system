#!/usr/bin/env python3
"""
Abstract Interfaces for Autonomous Exploration System
Clean separation of concerns for C++ integration
Author: Software Architecture Engineer - October 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from core_types import (
    Point2D, Point3D, RobotPose, SensorData, PassageSegment, 
    Pillar, PerformanceProfiler, SensorType
)

class ISensorProcessor(ABC):
    """Abstract interface for sensor data processing

    This interface defines the contract for all sensor processors,
    enabling easy swapping between Python and C++ implementations.
    """

    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        self.profiler = profiler
        self.sensor_type = SensorType.LIDAR  # Override in subclasses

    @abstractmethod
    def process_data(self, raw_data: Dict[str, Any]) -> SensorData:
        """Process raw sensor data and return structured sensor data

        Args:
            raw_data: Raw sensor data from hardware

        Returns:
            SensorData: Processed sensor data with extracted features
        """
        pass

    @abstractmethod
    def is_data_valid(self, raw_data: Dict[str, Any]) -> bool:
        """Validate raw sensor data before processing

        Args:
            raw_data: Raw sensor data to validate

        Returns:
            bool: True if data is valid for processing
        """
        pass

    @abstractmethod
    def get_processing_stats(self) -> Dict[str, float]:
        """Get processing performance statistics

        Returns:
            Dict containing timing and memory statistics
        """
        pass

class ILidarProcessor(ISensorProcessor):
    """Specialized interface for LIDAR processing"""

    @abstractmethod
    def detect_passages(self, ranges: np.ndarray, angles: np.ndarray) -> List[Dict[str, Any]]:
        """Detect passages from LIDAR data

        Args:
            ranges: Distance measurements
            angles: Corresponding angles

        Returns:
            List of detected passages with properties
        """
        pass

    @abstractmethod
    def detect_walls(self, ranges: np.ndarray, angles: np.ndarray) -> List[Dict[str, Any]]:
        """Detect wall segments from LIDAR data

        Args:
            ranges: Distance measurements  
            angles: Corresponding angles

        Returns:
            List of detected walls with properties
        """
        pass

    @abstractmethod
    def get_front_clearance(self, ranges: np.ndarray, angles: np.ndarray) -> float:
        """Get clear distance in front of robot

        Args:
            ranges: Distance measurements
            angles: Corresponding angles

        Returns:
            Clear distance in meters
        """
        pass

class ICameraProcessor(ISensorProcessor):
    """Specialized interface for camera processing"""

    @abstractmethod
    def detect_pillars(self, image: np.ndarray) -> List[Pillar]:
        """Detect colored pillars in camera image

        Args:
            image: RGB camera image

        Returns:
            List of detected pillars
        """
        pass

    @abstractmethod
    def classify_walls(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify walls by color (red = dead end)

        Args:
            image: RGB camera image

        Returns:
            Wall classification results
        """
        pass

    @abstractmethod
    def estimate_pillar_distance(self, pillar_area: float, pillar_color: str) -> float:
        """Estimate distance to pillar based on apparent size

        Args:
            pillar_area: Pixel area of detected pillar
            pillar_color: Color of the pillar

        Returns:
            Estimated distance in meters
        """
        pass

class IMapper(ABC):
    """Abstract interface for mapping functionality

    Defines the contract for occupancy grid mapping with support
    for both Python and C++ implementations.
    """

    def __init__(self, width: int, height: int, resolution: float, 
                 origin: Tuple[float, float], profiler: Optional[PerformanceProfiler] = None):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = origin
        self.profiler = profiler

    @abstractmethod
    def update_map(self, sensor_data: SensorData, robot_pose: RobotPose) -> None:
        """Update map with new sensor data

        Args:
            sensor_data: Processed sensor information
            robot_pose: Current robot pose
        """
        pass

    @abstractmethod
    def get_occupancy_grid(self) -> np.ndarray:
        """Get current occupancy grid

        Returns:
            Occupancy grid array (-1: unknown, 0: free, 100: occupied)
        """
        pass

    @abstractmethod
    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates

        Args:
            wx, wy: World coordinates

        Returns:
            Grid coordinates (gx, gy)
        """
        pass

    @abstractmethod
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates

        Args:
            gx, gy: Grid coordinates

        Returns:
            World coordinates (wx, wy)
        """
        pass

    @abstractmethod
    def is_cell_free(self, gx: int, gy: int) -> bool:
        """Check if grid cell is free space

        Args:
            gx, gy: Grid coordinates

        Returns:
            True if cell is free space
        """
        pass

    @abstractmethod
    def is_cell_occupied(self, gx: int, gy: int) -> bool:
        """Check if grid cell is occupied

        Args:
            gx, gy: Grid coordinates

        Returns:
            True if cell is occupied
        """
        pass

    @abstractmethod
    def get_2d_visualization(self) -> np.ndarray:
        """Generate 2D top-view visualization

        Returns:
            RGB image array for visualization
        """
        pass

class IPathPlanner(ABC):
    """Abstract interface for path planning

    Supports multiple path planning algorithms with unified interface.
    """

    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        self.profiler = profiler

    @abstractmethod
    def plan_path(self, start: Point2D, goal: Point2D, 
                  occupancy_grid: np.ndarray) -> List[Point2D]:
        """Plan path from start to goal

        Args:
            start: Starting position
            goal: Goal position  
            occupancy_grid: Current occupancy grid

        Returns:
            List of waypoints from start to goal
        """
        pass

    @abstractmethod
    def is_path_clear(self, start: Point2D, goal: Point2D, 
                      occupancy_grid: np.ndarray) -> bool:
        """Check if direct path is clear

        Args:
            start: Starting position
            goal: Goal position
            occupancy_grid: Current occupancy grid

        Returns:
            True if direct path is clear
        """
        pass

    @abstractmethod
    def calculate_path_cost(self, path: List[Point2D]) -> float:
        """Calculate total path cost/distance

        Args:
            path: List of waypoints

        Returns:
            Total path cost in meters
        """
        pass

    @abstractmethod
    def smooth_path(self, path: List[Point2D]) -> List[Point2D]:
        """Apply path smoothing algorithm

        Args:
            path: Original path waypoints

        Returns:
            Smoothed path waypoints
        """
        pass

class IExplorationPlanner(ABC):
    """Abstract interface for exploration planning"""

    @abstractmethod
    def find_exploration_targets(self, occupancy_grid: np.ndarray, 
                                robot_pose: RobotPose) -> List[Point2D]:
        """Find next exploration targets (frontiers)

        Args:
            occupancy_grid: Current map
            robot_pose: Current robot position

        Returns:
            List of exploration target positions
        """
        pass

    @abstractmethod
    def select_best_target(self, targets: List[Point2D], 
                          robot_pose: RobotPose) -> Optional[Point2D]:
        """Select best exploration target from candidates

        Args:
            targets: Candidate exploration targets
            robot_pose: Current robot position

        Returns:
            Best target position or None
        """
        pass

class IRobotController(ABC):
    """Abstract interface for robot control"""

    @abstractmethod
    def set_velocities(self, linear: float, angular: float) -> None:
        """Set robot velocities

        Args:
            linear: Linear velocity (m/s)
            angular: Angular velocity (rad/s)
        """
        pass

    @abstractmethod
    def stop_robot(self) -> None:
        """Emergency stop robot"""
        pass

    @abstractmethod
    def get_odometry(self) -> RobotPose:
        """Get current robot pose from odometry

        Returns:
            Current robot pose
        """
        pass

    @abstractmethod
    def is_moving(self) -> bool:
        """Check if robot is currently moving

        Returns:
            True if robot is moving
        """
        pass

class IVisualization(ABC):
    """Abstract interface for visualization"""

    @abstractmethod
    def update_display(self, occupancy_grid: np.ndarray, robot_pose: RobotPose,
                      pillars: List[Pillar], path: Optional[List[Point2D]] = None) -> None:
        """Update visualization display

        Args:
            occupancy_grid: Current map
            robot_pose: Current robot position  
            pillars: Detected pillars
            path: Current path (optional)
        """
        pass

    @abstractmethod
    def save_visualization(self, filename: str) -> bool:
        """Save current visualization to file

        Args:
            filename: Output filename

        Returns:
            True if save successful
        """
        pass

class IDataLogger(ABC):
    """Abstract interface for data logging and persistence"""

    @abstractmethod
    def log_sensor_data(self, sensor_data: SensorData) -> None:
        """Log sensor data for analysis

        Args:
            sensor_data: Processed sensor data
        """
        pass

    @abstractmethod
    def log_robot_pose(self, pose: RobotPose) -> None:
        """Log robot pose data

        Args:
            pose: Robot pose to log
        """
        pass

    @abstractmethod
    def save_exploration_results(self, results: Dict[str, Any]) -> str:
        """Save complete exploration results

        Args:
            results: Exploration results dictionary

        Returns:
            Saved filename
        """
        pass

    @abstractmethod
    def load_previous_session(self) -> Optional[Dict[str, Any]]:
        """Load previous exploration session

        Returns:
            Previous session data or None
        """
        pass

# Factory interfaces for dependency injection
class ISensorFactory(ABC):
    """Factory interface for creating sensor processors"""

    @abstractmethod
    def create_lidar_processor(self) -> ILidarProcessor:
        """Create LIDAR processor instance"""
        pass

    @abstractmethod
    def create_camera_processor(self) -> ICameraProcessor:
        """Create camera processor instance"""
        pass

class ISystemFactory(ABC):
    """Factory interface for creating system components"""

    @abstractmethod
    def create_mapper(self) -> IMapper:
        """Create mapper instance"""
        pass

    @abstractmethod
    def create_path_planner(self) -> IPathPlanner:
        """Create path planner instance"""
        pass

    @abstractmethod
    def create_exploration_planner(self) -> IExplorationPlanner:
        """Create exploration planner instance"""
        pass

if __name__ == "__main__":
    print("🔌 Abstract Interfaces Initialized")
    print("✅ Clean separation of concerns for C++ integration")
    print("✅ Performance profiling integration ready")
    print("✅ Factory pattern support for dependency injection")
    print("🏗️ Ready for distributed implementation")
