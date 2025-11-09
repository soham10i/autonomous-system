#!/usr/bin/env python3

"""
Complete Advanced ROSBot Controller with Full SLAM Integration
Integrates all advanced SLAM components based on detailed module analysis
"""

import sys
import os
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import time

# Webots imports
from controller import Robot, Motor, DistanceSensor, Camera, Lidar, Keyboard

# Import all SLAM modules with error handling
try:
    from robust_slam_manager import RobustSLAMManager
    from enhanced_occupancy_grid import EnhancedOccupancyGrid
    from frontier_explorer import FrontierExplorer, AutonomousExplorer
    from pillar_detector import PillarDetector, PillarMapper
    from a_star import AStar
    from sensor_fusion import SensorFusion, KalmanFilter, PoseEstimate
    from visual_odometry import VisualOdometry
    from loop_closure_detector import LoopClosureDetector
    from webots_rosbot_constants import *
    SLAM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SLAM modules import error: {e}")
    SLAM_AVAILABLE = False

# Import SLAMLogger separately to avoid circular dependencies if SLAM_AVAILABLE is False
from slam_logging_config import get_slam_logger, performance_monitor

# Initialize SLAM logger
slam_logger = get_slam_logger(module_name="rosbot_controller")

# Constants and default values
MAX_SPEED = 6.28
SAFE_DISTANCE = 1.0
EMERGENCY_DISTANCE = 0.3
ROBOT_WIDTH = 0.200

# Grid parameters (from analysis)
GRID_RESOLUTION = 0.05  # 5cm resolution
MAZE_WIDTH = 200        # 200 cells = 10m at 5cm resolution
MAZE_HEIGHT = 200       # 200 cells = 10m at 5cm resolution
MAZE_ORIGIN_X = -5.0    # Center the grid
MAZE_ORIGIN_Y = -5.0


class RobotState(Enum):
    """Advanced robot operational states"""
    INITIALIZING = "initializing"
    EXPLORING = "exploring"
    PILLAR_DETECTED = "pillar_detected"
    NAVIGATING_TO_PILLAR = "navigating_to_pillar"
    PLANNING_BETWEEN_PILLARS = "planning_between_pillars"
    LOOP_CLOSURE_ACTIVE = "loop_closure_active"
    MISSION_COMPLETE = "mission_complete"
    MANUAL_CONTROL = "manual_control"
    ERROR_RECOVERY = "error_recovery"


class AdvancedROSBotController:
    """Complete advanced ROSBot controller with full SLAM integration"""

    def __init__(self):
        # Initialize Webots robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Initialize hardware
        self._init_motors()
        self._init_sensors()

        # Initialize advanced SLAM system
        self._init_advanced_slam()

        # State management
        self.current_state = RobotState.INITIALIZING
        self.detected_pillars = []
        self.current_target = None
        self.exploration_complete = False
        self.mission_start_time = time.time()

        # Advanced navigation parameters
        self.goal_tolerance = 0.15
        self.max_speed = 2.0
        self.exploration_timeout = 600  # 10 minutes
        self.pillar_scan_frequency = 3  # Check every 3 steps
        self.step_counter = 0

        # Robot state
        self.current_pose = [0.0, 0.0, 0.0]  # x, y, theta
        self.pose_confidence = 1.0
        self.current_goal = None
        self.path_to_goal = []
        self.path_index = 0

        # Performance monitoring
        self.last_update_time = time.time()
        self.update_frequency = 0.0

        print("🤖 Advanced ROSBot Controller with Full SLAM initialized")
        print("🔬 All advanced SLAM components integrated")

    def _init_motors(self):
        """Initialize ROSBot 4-wheel motors"""
        try:
            self.front_left_motor = self.robot.getDevice(
                "front left wheel motor")
            self.front_right_motor = self.robot.getDevice(
                "front right wheel motor")
            self.rear_left_motor = self.robot.getDevice(
                "rear left wheel motor")
            self.rear_right_motor = self.robot.getDevice(
                "rear right wheel motor")

            self.motors = [self.front_left_motor, self.front_right_motor,
                           self.rear_left_motor, self.rear_right_motor]

            for motor in self.motors:
                motor.setPosition(float("inf"))
                motor.setVelocity(0.0)

            slam_logger.info("4-wheel motor system initialized")
        except Exception as e:
            slam_logger.error(f"FATAL: Motor initialization failed: {e}")
            sys.exit()

    def _init_sensors(self):
        """Initialize all robot sensors"""
        # RGB Camera
        self.camera = None
        self.camera_width, self.camera_height = 640, 480
        try:
            self.camera = self.robot.getDevice("camera rgb")
            if self.camera:
                self.camera.enable(self.timestep)
                self.camera_width = self.camera.getWidth()
                self.camera_height = self.camera.getHeight()
                slam_logger.info("RGB Camera enabled")
        except Exception as e:
            slam_logger.warning(f"RGB camera error: {e}")

        # Depth Camera
        self.depth_camera = None
        try:
            self.depth_camera = self.robot.getDevice("camera depth")
            if self.depth_camera:
                self.depth_camera.enable(self.timestep)
                slam_logger.info("Depth Camera enabled")
        except Exception as e:
            slam_logger.warning(f"Depth camera error: {e}")

        # LiDAR
        self.lidar = None
        try:
            self.lidar = self.robot.getDevice("lidar")
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
            slam_logger.info("LiDAR enabled")
        except Exception as e:
            slam_logger.error(f"FATAL: LiDAR error: {e}")
            sys.exit()

        # Position sensors for wheel odometry
        self.position_sensors = []
        try:
            sensor_names = ["front left wheel motor sensor", "front right wheel motor sensor",
                            "rear left wheel motor sensor", "rear right wheel motor sensor"]

            for name in sensor_names:
                sensor = self.robot.getDevice(name)
                sensor.enable(self.timestep)
                self.position_sensors.append(sensor)

            self.last_positions = [0.0, 0.0, 0.0, 0.0]
            slam_logger.info("Wheel encoders initialized")
        except Exception as e:
            slam_logger.warning(f"Encoder error: {e}")
            self.position_sensors = []

        # Distance sensors with LiDAR fallback
        self.distance_sensors = []
        sensor_names = [
            "front left distance sensor", "front right distance sensor",
            "rear left distance sensor", "rear right distance sensor"
        ]
        for name in sensor_names:
            try:
                sensor = self.robot.getDevice(name)
                sensor.enable(self.timestep)
                self.distance_sensors.append(sensor)
            except Exception as e:
                slam_logger.warning(f"Distance sensor {name} error: {e}")

        self.distance_readings = [SAFE_DISTANCE] * len(self.distance_sensors)

        # Keyboard for manual control
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.timestep)

        slam_logger.info("Complete sensor suite initialized")

    def _init_advanced_slam(self):
        """Initialize complete advanced SLAM system"""
        self.slam_components = {}

        try:
            # 1. Enhanced Occupancy Grid (foundation)
            self.occupancy_grid = EnhancedOccupancyGrid(
                resolution=GRID_RESOLUTION,
                width=MAZE_WIDTH,
                height=MAZE_HEIGHT,
                origin_x=MAZE_ORIGIN_X,
                origin_y=MAZE_ORIGIN_Y
            )
            self.slam_components["occupancy_grid"] = True
            slam_logger.info("Enhanced Occupancy Grid initialized")

            # 2. A* Path Planner
            self.path_planner = AStar(occupancy_grid=self.occupancy_grid)
            self.slam_components["path_planner"] = True
            slam_logger.info("A* Path Planner initialized")

            # 3. Frontier Explorer with Autonomous Explorer
            self.frontier_explorer = FrontierExplorer(
                occupancy_grid=self.occupancy_grid,
                min_frontier_size=5,
                exploration_radius=1.0
            )
            self.autonomous_explorer = AutonomousExplorer(
                occupancy_grid=self.occupancy_grid,
                a_star_planner=self.path_planner,
                robot_radius=0.15
            )
            self.slam_components["exploration"] = True
            slam_logger.info("Frontier & Autonomous Explorer initialized")

            # 4. Pillar Detection System
            self.pillar_detector = PillarDetector(
                camera_width=self.camera_width,
                camera_height=self.camera_height,
                camera_fov=1.047  # 60 degrees
            )
            self.pillar_mapper = PillarMapper(
                occupancy_grid=self.occupancy_grid)
            self.slam_components["pillar_detection"] = True
            slam_logger.info("Pillar Detection & Mapping initialized")

            # 5. Advanced Sensor Fusion
            self.kalman_filter = KalmanFilter(initial_pose=(0.0, 0.0, 0.0))
            self.sensor_fusion = SensorFusion(initial_pose=(0.0, 0.0, 0.0))
            self.slam_components["sensor_fusion"] = True
            slam_logger.info("Advanced Sensor Fusion initialized")

            # 6. Visual Odometry
            self.visual_odometry = VisualOdometry()
            self.slam_components["visual_odometry"] = True
            slam_logger.info("Visual Odometry initialized")

            # 7. Loop Closure Detection
            self.loop_closure_detector = LoopClosureDetector(
                keyframe_distance_threshold=1.0,
                keyframe_angle_threshold=0.5
            )
            self.slam_components["loop_closure"] = True
            slam_logger.info("Loop Closure Detection initialized")

            # 8. Robust SLAM Manager (orchestrates everything)
            self.slam_manager = RobustSLAMManager(
                initial_grid_size=(MAZE_WIDTH, MAZE_HEIGHT),
                resolution=GRID_RESOLUTION,
                loop_closure_detector=self.loop_closure_detector
            )
            self.slam_components["slam_manager"] = True
            slam_logger.info("Robust SLAM Manager initialized")

        except Exception as e:
            slam_logger.error(f"Advanced SLAM initialization error: {e}")
            slam_logger.warning("Running with basic navigation fallback")
            self.slam_components = {}

    @performance_monitor
    def update_sensors(self):
        """Update all sensor readings with performance monitoring"""
        start_time = time.time()

        # Distance sensors
        if self.distance_sensors:
            self.distance_readings = [sensor.getValue()
                                      for sensor in self.distance_sensors]
        else:
            self.distance_readings = self._lidar_to_distance_readings()

        # Camera image
        self.camera_image = None
        if self.camera:
            try:
                raw_image = self.camera.getImage()
                self.camera_image = np.frombuffer(raw_image, dtype=np.uint8)
                self.camera_image = self.camera_image.reshape(
                    (self.camera_height, self.camera_width, 4))
                self.camera_image = cv2.cvtColor(
                    self.camera_image, cv2.COLOR_BGRA2RGB)
            except Exception as e:
                slam_logger.warning(f"Camera read error: {e}")

        # LiDAR data
        self.lidar_data = []  # Initialize lidar_data to an empty list
        if self.lidar:
            try:
                # Get point cloud data (x, y, z coordinates relative to the sensor)
                self.lidar_data = self.lidar.getPointCloud()
            except Exception as e:
                slam_logger.warning(f"LiDAR read error: {e}")

        # Wheel odometry
        self._update_wheel_odometry()

        # Performance monitoring
        update_time = time.time() - start_time
        if update_time > 0:
            self.update_frequency = 1.0 / update_time

    def _update_wheel_odometry(self):
        """Update wheel-based odometry"""
        if not self.position_sensors:
            return

        try:
            current_positions = [sensor.getValue()
                                 for sensor in self.position_sensors]

            # Calculate wheel movements
            deltas = [current - last for current,
                      last in zip(current_positions, self.last_positions)]
            self.last_positions = current_positions

            # Convert to robot motion (simplified differential drive)
            wheel_radius = 0.0325
            left_delta = (deltas[0] + deltas[2]) / 2.0  # Average left wheels
            right_delta = (deltas[1] + deltas[3]) / 2.0  # Average right wheels

            distance_left = left_delta * wheel_radius
            distance_right = right_delta * wheel_radius

            distance = (distance_left + distance_right) / 2.0
            angle_change = (distance_right - distance_left) / ROBOT_WIDTH

            # Update pose
            self.current_pose[0] += distance * np.cos(self.current_pose[2])
            self.current_pose[1] += distance * np.sin(self.current_pose[2])
            self.current_pose[2] += angle_change

            # Normalize angle
            while self.current_pose[2] > np.pi:
                self.current_pose[2] -= 2 * np.pi
            while self.current_pose[2] < -np.pi:
                self.current_pose[2] += 2 * np.pi

        except Exception as e:
            slam_logger.warning(f"Odometry error: {e}")

    def _lidar_to_distance_readings(self) -> List[float]:
        """Convert LiDAR point cloud to distance sensor readings"""
        if not self.lidar_data:
            return [SAFE_DISTANCE] * len(self.distance_sensors)

        # Project 3D points to 2D distances for obstacle avoidance
        # This is a simplified approach, assuming the robot is on a flat plane
        distances = [SAFE_DISTANCE] * 8  # 8 sectors for distance sensors

        # Assuming lidar_data is a list of Webots PointCloudPoint objects or similar
        for point in self.lidar_data:
            # Calculate 2D distance from robot origin (0,0) to the point (x,y)
            # We ignore the z-coordinate for 2D distance readings
            dist = np.sqrt(point.x**2 + point.y**2)

            # Calculate angle of the point relative to robot's forward direction
            angle = np.arctan2(point.y, point.x)  # Angle in robot's frame

            # Normalize angle to [0, 2*pi) and map to 8 sectors
            normalized_angle = (angle + np.pi) % (2 * np.pi)
            sector_idx = int(normalized_angle / (2 * np.pi / 8))

            if dist < distances[sector_idx]:
                distances[sector_idx] = dist

        return distances

    def run(self):
        """Advanced main control loop"""
        print("🚀 Starting Advanced Autonomous Navigation...")

        while self.robot.step(self.timestep) != -1:
            self.step_counter += 1

            # Update all sensors
            self.update_sensors()

            # Advanced SLAM updates
            self._update_advanced_slam()

            # State machine execution
            try:
                if self.current_state == RobotState.INITIALIZING:
                    self.handle_initialization()
                elif self.current_state == RobotState.EXPLORING:
                    self.handle_advanced_exploration()
                elif self.current_state == RobotState.PILLAR_DETECTED:
                    self.handle_pillar_detection()
                elif self.current_state == RobotState.NAVIGATING_TO_PILLAR:
                    self.handle_advanced_navigation()
                elif self.current_state == RobotState.PLANNING_BETWEEN_PILLARS:
                    self.handle_pillar_planning()
                elif self.current_state == RobotState.LOOP_CLOSURE_ACTIVE:
                    self.handle_loop_closure()
                elif self.current_state == RobotState.MANUAL_CONTROL:
                    self.handle_manual_control()
                elif self.current_state == RobotState.ERROR_RECOVERY:
                    self.handle_error_recovery()

            except Exception as e:
                slam_logger.error(f"State machine error: {e}")
                self.current_state = RobotState.ERROR_RECOVERY

            # Always handle keyboard and safety
            self.handle_keyboard_input()
            self.check_advanced_safety()

            # Status reporting
            if self.step_counter % 100 == 0:
                self._print_status()

    @performance_monitor
    def _update_advanced_slam(self):
        """Update all SLAM components"""
        if not self.slam_components:
            return

        try:
            # Update sensor fusion
            if 'sensor_fusion' in self.slam_components:
                current_time = time.time()
                # Create PoseEstimate for visual odometry
                if self.camera_image is not None and 'visual_odometry' in self.slam_components:
                    visual_pose_estimate = self.visual_odometry.get_pose_estimate(self.camera_image, current_time)
                    if visual_pose_estimate:
                        self.sensor_fusion.update_visual_odometry(visual_pose_estimate)

                # Create PoseEstimate for wheel odometry
                # The current_pose is already updated by _update_wheel_odometry
                wheel_pose_estimate = PoseEstimate(
                    x=self.current_pose[0],
                    y=self.current_pose[1],
                    theta=self.current_pose[2],
                    timestamp=current_time,
                    source='wheel',
                    confidence=1.0  # Assuming high confidence for direct wheel odometry
                )
                self.sensor_fusion.update_wheel_odometry(wheel_pose_estimate)
                # For LiDAR, we pass raw data and current pose for scan matching
                self.sensor_fusion.update_lidar_scan(
                    self.lidar_data, self.current_pose)

                # Get fused pose estimate
                fused_pose = self.sensor_fusion.get_current_pose()
                if fused_pose:
                    self.current_pose = [fused_pose.x,
                                         fused_pose.y, fused_pose.theta]
                    self.pose_confidence = fused_pose.confidence

            # Update occupancy grid
            if 'occupancy_grid' in self.slam_components and self.lidar_data is not None:
                # Update grid with LiDAR data
                self.occupancy_grid.update_grid(
                    self.current_pose, self.lidar_data, self.camera_image)

                # Save map periodically for visualization
                if self.step_counter % 100 == 0:  # Save every 100 steps
                    map_filename = f"occupancy_grid_step_{self.step_counter}.png"
                    self.occupancy_grid.save_map_as_image(
                        map_filename, self.lidar_data)
                    slam_logger.info(
                        f"Saved occupancy grid map to {map_filename}")
            # Update loop closure detection
            if 'loop_closure' in self.slam_components and self.camera_image is not None:
                loop_closure = self.loop_closure_detector.detect_loop_closure(
                    self.current_pose, self.camera_image, self.lidar_data
                )
                if loop_closure:
                    slam_logger.info("Loop closure detected!")
                    self.current_state = RobotState.LOOP_CLOSURE_ACTIVE
            # Update SLAM manager
            if 'slam_manager' in self.slam_components:
                self.slam_manager.update_system(
                    sensor_data={
                        'lidar': self.lidar_data,
                        'camera': self.camera_image
                    },
                    current_pose=self.current_pose
                )
        except Exception as e:
            slam_logger.warning(f"SLAM update error: {e}")

    def handle_initialization(self):
        """Advanced initialization"""
        print("🔧 Advanced SLAM initialization...")

        if self.step_counter > 20:  # More time for advanced systems
            # Initialize SLAM manager
            if 'slam_manager' in self.slam_components:
                self.slam_manager.initialize()

            self.current_state = RobotState.EXPLORING
            slam_logger.info("Advanced systems ready - starting exploration")

    def handle_advanced_exploration(self):
        """Advanced exploration with frontier detection"""
        slam_logger.info(
            f"Exploring... Step: {self.step_counter}, Pillars found: {len(self.detected_pillars)}")

        # Less frequent pillar detection to avoid getting stuck
        if self.step_counter % (self.pillar_scan_frequency * 3) == 0:  # Reduced frequency
            pillars = self._detect_pillars()
            if pillars:
                # Log but don't immediately switch state
                for pillar in pillars:
                    if pillar not in self.detected_pillars:
                        self.detected_pillars.append(pillar)
                        slam_logger.info("Pillar detected and logged")

                # Only switch to pillar state if we have time to process
                if len(self.detected_pillars) >= 2 and self.step_counter > 100:
                    self.current_state = RobotState.PLANNING_BETWEEN_PILLARS
                    return

        # Use autonomous explorer if available
        if 'exploration' in self.slam_components:
            try:
                self.autonomous_explorer.update(self.current_pose)

                if self.autonomous_explorer.exploration_complete():
                    if len(self.detected_pillars) >= 2:
                        self.current_state = RobotState.PLANNING_BETWEEN_PILLARS
                    else:
                        slam_logger.warning(
                            "Exploration complete but insufficient pillars")
                else:
                    # Follow explorer's path
                    target = self.autonomous_explorer.get_current_target()
                    if target:
                        self._navigate_to_point(target)
                    else:
                        self.basic_obstacle_avoidance()
            except Exception as e:
                slam_logger.warning(f"Explorer error: {e}")
                self.basic_obstacle_avoidance()
        else:
            self.basic_obstacle_avoidance()

    def handle_pillar_detection(self):
        """Advanced pillar detection with avoidance behavior"""
        slam_logger.info("Pillar detected - analyzing...")

        # Quick pillar verification (reduced frames to prevent getting stuck)
        confirmed_pillars = []
        for _ in range(2):  # Reduced from 5 to 2 frames
            self.robot.step(self.timestep)
            self.update_sensors()
            pillars = self._detect_pillars()
            if pillars:
                confirmed_pillars.extend(pillars)

        if confirmed_pillars:
            # Log pillar detection
            for pillar in confirmed_pillars:
                # Check if a pillar with similar world_position already exists in detected_pillars
                # Using a small tolerance for comparison
                already_detected = False
                for existing_pillar in self.detected_pillars:
                    # 10cm tolerance
                    if np.linalg.norm(np.array(pillar.world_position) - np.array(existing_pillar.world_position)) < 0.1:
                        already_detected = True
                        break

                if not already_detected:
                    self.detected_pillars.append(pillar)
                    slam_logger.info(
                        f"New pillar logged at ({pillar.world_position[0]:.2f}, {pillar.world_position[1]:.2f})")

            # Update pillar mapper if available
            if "pillar_detection" in self.slam_components:
                try:
                    for pillar in confirmed_pillars:
                        self.pillar_mapper.update_pillar_map(pillar)
                except Exception as e:
                    slam_logger.warning(f"Pillar mapper error: {e}")

        # IMPORTANT: Don't get stuck at pillars - continue exploring
        slam_logger.info(
            "Pillar logged - continuing exploration to avoid getting stuck")

        # Add pillar avoidance behavior - move away from detected pillar
        if confirmed_pillars:
            # Turn away from the pillar and continue
            self.set_motor_speeds(-1.5, 1.5)  # Turn left to avoid
            slam_logger.info(
                "Turning to avoid pillar and continue exploration")

        # Continue exploration regardless
        self.current_state = RobotState.EXPLORING

    def handle_advanced_navigation(self):
        """Advanced A* navigation with dynamic replanning"""
        if not self.current_target:
            self.current_state = RobotState.EXPLORING
            return

        # Check if reached target
        distance = np.sqrt((self.current_pose[0] - self.current_target[0])**2 +
                           (self.current_pose[1] - self.current_target[1])**2)

        if distance < self.goal_tolerance:
            slam_logger.info(f"Reached target!")
            self.current_target = None

            # Check for more pillars
            remaining = [
                p for p in self.detected_pillars if not p.get("visited", False)]
            if remaining:
                self.current_state = RobotState.PLANNING_BETWEEN_PILLARS
            else:
                self.current_state = RobotState.MISSION_COMPLETE
        else:
            # Advanced A* navigation
            if "path_planner" in self.slam_components and "occupancy_grid" in self.slam_components:
                try:
                    start = self.occupancy_grid.world_to_grid(
                        self.current_pose[0], self.current_pose[1])
                    goal = self.occupancy_grid.world_to_grid(
                        self.current_target[0], self.current_target[1])

                    path = self.path_planner.plan_path(
                        start, goal, robot_radius_grid=int(ROBOT_WIDTH / (2 * GRID_RESOLUTION)))
                    if path and len(path) > 1:
                        next_point = self.occupancy_grid.grid_to_world(
                            path[1][0], path[1][1])
                        self._navigate_to_point(next_point)
                    else:
                        # Fallback navigation
                        self._navigate_to_point(self.current_target)
                except Exception as e:
                    slam_logger.warning(f"A* navigation error: {e}")
                    self._navigate_to_point(self.current_target)
            else:
                self._navigate_to_point(self.current_target)

    def handle_pillar_planning(self):
        """Plan optimal route between pillars"""
        if len(self.detected_pillars) < 2:
            self.current_state = RobotState.EXPLORING
            return

        # Use pillar mapper for optimal planning
        if 'pillar_detection' in self.slam_components:
            nearest_pillars = self.pillar_mapper.find_nearest_pillars(
                self.current_pose, max_count=5)
            if nearest_pillars:
                self.current_target = (
                    nearest_pillars[0]['x'], nearest_pillars[0]['y'])
                self.current_state = RobotState.NAVIGATING_TO_PILLAR
        else:
            # Simple nearest pillar selection
            distances = []
            for pillar in self.detected_pillars:
                if not pillar.get('visited', False):
                    dist = np.sqrt((pillar['x'] - self.current_pose[0])**2 +
                                   (pillar['y'] - self.current_pose[1])**2)
                    distances.append((dist, pillar))

            if distances:
                nearest = min(distances, key=lambda x: x[0])[1]
                self.current_target = (nearest['x'], nearest['y'])
                self.current_state = RobotState.NAVIGATING_TO_PILLAR

    def handle_loop_closure(self):
        """Handle loop closure correction"""
        slam_logger.info("Processing loop closure...")

        # Let loop closure detector process
        if 'loop_closure' in self.slam_components:
            # This would update pose estimates and map
            pass

        # Return to exploration after loop closure
        self.current_state = RobotState.EXPLORING

    def handle_manual_control(self):
        """Manual control mode"""
        pass  # Handled in keyboard input

    def handle_error_recovery(self):
        """Advanced error recovery"""
        slam_logger.error("Error recovery mode")
        self.stop_robot()

        # Try to recover after some steps
        if self.step_counter % 50 == 0:
            self.current_state = RobotState.INITIALIZING

    def _detect_pillars(self) -> List[Dict[str, Any]]:
        """Detect pillars using advanced detection"""
        if not self.camera_image or 'pillar_detection' not in self.slam_components:
            return []

        try:
            pillars = self.pillar_detector.detect_pillars(
                self.camera_image, self.current_pose)
            return pillars if pillars else []
        except Exception as e:
            slam_logger.warning(f"Pillar detection error: {e}")
            return []

    def _navigate_to_point(self, target: Tuple[float, float]):
        """Navigate to a specific point"""
        angle_to_target = np.arctan2(target[1] - self.current_pose[1],
                                     target[0] - self.current_pose[0])
        angle_diff = angle_to_target - self.current_pose[2]

        # Normalize angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Check for obstacles
        min_distance = min(
            self.distance_readings[:3] + self.distance_readings[-2:])

        if min_distance < SAFE_DISTANCE:
            self.basic_obstacle_avoidance()
        else:
            if abs(angle_diff) > 0.1:
                # Turn towards target
                turn_speed = max(-2.0, min(2.0, angle_diff * 3.0))
                self.set_motor_speeds(turn_speed, -turn_speed)
            else:
                # Move forward
                speed = min(self.max_speed, 1.5)
                self.set_motor_speeds(speed, speed)

    def basic_obstacle_avoidance(self):
        """Enhanced obstacle avoidance with stuck prevention"""
        # Get sensor readings
        front_distances = [self.distance_readings[i]
                           for i in [0, 1, 7] if i < len(self.distance_readings)]
        left_distances = [self.distance_readings[i]
                          for i in [5, 6] if i < len(self.distance_readings)]
        right_distances = [self.distance_readings[i]
                           for i in [2, 3] if i < len(self.distance_readings)]

        min_front = min(front_distances) if front_distances else SAFE_DISTANCE
        avg_left = sum(left_distances) / \
            len(left_distances) if left_distances else SAFE_DISTANCE
        avg_right = sum(right_distances) / \
            len(right_distances) if right_distances else SAFE_DISTANCE

        # Stuck detection
        if not hasattr(self, 'last_pose'):
            self.last_pose = self.current_pose.copy()
            self.stuck_counter = 0
            self.stuck_recovery_direction = 1  # 1 for right, -1 for left

        # Check if robot is stuck (hasn't moved much)
        pose_diff = abs(self.current_pose[0] - self.last_pose[0]) + \
            abs(self.current_pose[1] - self.last_pose[1])
        if pose_diff < 0.01:  # Very little movement
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_pose = self.current_pose.copy()

        # Recovery behavior when stuck
        if self.stuck_counter > 20:  # Stuck for 20 steps
            slam_logger.info("Robot stuck - executing recovery maneuver")
            # Aggressive recovery: back up and turn
            if self.stuck_counter < 40:
                self.set_motor_speeds(-1.5, -1.5)  # Back up
            else:
                # Turn in recovery direction
                turn_speed = 2.0 * self.stuck_recovery_direction
                self.set_motor_speeds(turn_speed, -turn_speed)

                if self.stuck_counter > 60:
                    # Switch recovery direction and reset
                    self.stuck_recovery_direction *= -1
                    self.stuck_counter = 0

        # Normal obstacle avoidance
        elif min_front < SAFE_DISTANCE:
            # Choose direction with more space, but add randomness to avoid cycles
            space_difference = avg_left - avg_right

            # Add small random component to break symmetry
            import random
            randomness = (random.random() - 0.5) * 0.2

            if space_difference + randomness > 0.1:
                self.set_motor_speeds(-1.5, 1.5)  # Turn left
                slam_logger.info("Turning left to avoid obstacle")
            else:
                self.set_motor_speeds(1.5, -1.5)  # Turn right
                slam_logger.info("Turning right to avoid obstacle")

        # Clear path - move forward
        else:
            speed = min(self.max_speed, 2.0)
            self.set_motor_speeds(speed, speed)
            slam_logger.info("Moving forward")

    def handle_keyboard_input(self):
        """Enhanced keyboard control"""
        key = self.keyboard.getKey()

        if key == ord("M"):
            self.current_state = (RobotState.MANUAL_CONTROL if self.current_state != RobotState.MANUAL_CONTROL
                                  else RobotState.EXPLORING)
            slam_logger.info(
                f"Switched to {"Manual" if self.current_state == RobotState.MANUAL_CONTROL else "Autonomous"}")

        elif key == ord("R"):  # Reset
            self.current_state = RobotState.INITIALIZING
            slam_logger.info("System reset")

        elif key == ord("S"):  # Status
            self._print_detailed_status()

        if self.current_state == RobotState.MANUAL_CONTROL:
            if key == self.keyboard.UP:
                self.set_motor_speeds(2.0, 2.0)
            elif key == self.keyboard.DOWN:
                self.set_motor_speeds(-2.0, -2.0)
            elif key == self.keyboard.LEFT:
                self.set_motor_speeds(-1.0, 1.0)
            elif key == self.keyboard.RIGHT:
                self.set_motor_speeds(1.0, -1.0)
            else:
                self.set_motor_speeds(0.0, 0.0)

    def check_advanced_safety(self):
        """Advanced safety monitoring"""
        # Emergency obstacle detection
        min_distance = min(
            self.distance_readings) if self.distance_readings else SAFE_DISTANCE

        if min_distance < EMERGENCY_DISTANCE:
            self.stop_robot()
            slam_logger.critical("Emergency stop - obstacle detected!")

        # Mission timeout
        if time.time() - self.mission_start_time > self.exploration_timeout:
            slam_logger.info("Mission timeout - completing...")
            self.current_state = RobotState.MISSION_COMPLETE

        # Pose confidence monitoring
        if hasattr(self, "pose_confidence") and self.pose_confidence < 0.3:
            slam_logger.warning("Low pose confidence - may need recalibration")

    def _print_status(self):
        """Print current system status"""
        slam_logger.info(
            f"Status: {self.current_state.value} | Pose: ({self.current_pose[0]:.2f}, {self.current_pose[1]:.2f}, {np.degrees(self.current_pose[2]):.1f}°)")
        slam_logger.info(
            f"Pillars: {len(self.detected_pillars)} | Components: {len(self.slam_components)}")

    def _print_detailed_status(self):
        """Print detailed system status"""
        slam_logger.info("\n" + "="*50)
        slam_logger.info("🔍 DETAILED SYSTEM STATUS")
        slam_logger.info("="*50)
        slam_logger.info(f"State: {self.current_state.value}")
        slam_logger.info(
            f"Pose: ({self.current_pose[0]:.3f}, {self.current_pose[1]:.3f}, {np.degrees(self.current_pose[2]):.1f}°)")
        slam_logger.info(
            f"Confidence: {getattr(self, "pose_confidence", 1.0):.2f}")
        slam_logger.info(f"Pillars detected: {len(self.detected_pillars)}")
        slam_logger.info(
            f"SLAM components active: {list(self.slam_components.keys())}")
        slam_logger.info(f"Update frequency: {self.update_frequency:.1f} Hz")
        slam_logger.info("="*50 + "\n")

    def set_motor_speeds(self, left_speed: float, right_speed: float):
        """Set motor speeds with safety limits"""
        left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

        self.front_left_motor.setVelocity(left_speed)
        self.rear_left_motor.setVelocity(left_speed)
        self.front_right_motor.setVelocity(right_speed)
        self.rear_right_motor.setVelocity(right_speed)

    def stop_robot(self):
        """Emergency stop all motors"""
        for motor in self.motors:
            motor.setVelocity(0.0)


# Main execution
if __name__ == "__main__":
    try:
        controller = AdvancedROSBotController()
        controller.run()
    except KeyboardInterrupt:
        slam_logger.info("Program interrupted by user")
    except Exception as e:
        slam_logger.critical(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
