#!/usr/bin/env python3
"""
Professional Robot Control Components
High-performance motor control and odometry
Author: Control Systems Engineer - October 2025
"""

import time
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from collections import deque

from core_types import Point2D, RobotPose, PerformanceProfiler, SystemConfig
from interfaces import IRobotController

class PrecisionRobotController(IRobotController):
    """High-precision robot controller with advanced kinematics

    Features:
    - Differential drive kinematics
    - Wheel odometry integration
    - Velocity smoothing and limiting
    - Performance monitoring
    """

    def __init__(self, robot, profiler: Optional[PerformanceProfiler] = None):
        self.robot = robot
        self.profiler = profiler
        self.timestep = int(robot.getBasicTimeStep())

        # Robot physical parameters
        self.wheel_radius = SystemConfig.ROBOT_WHEEL_RADIUS
        self.wheel_base = SystemConfig.ROBOT_WHEEL_BASE
        self.max_wheel_velocity = SystemConfig.ROBOT_MAX_VELOCITY

        # Control parameters
        self.max_linear_speed = SystemConfig.MAX_LINEAR_SPEED
        self.max_angular_speed = SystemConfig.MAX_ANGULAR_SPEED

        # Velocity smoothing
        self.enable_velocity_smoothing = True
        self.velocity_smoothing_factor = 0.8
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

        # Safety parameters
        self.emergency_stop_active = False
        self.velocity_limit_factor = 0.9  # 10% safety margin

        # Odometry
        self.robot_pose = RobotPose(x=0.0, y=0.0, theta=0.0)
        self.odometry_initialized = False

        # Performance tracking
        self.control_times = []
        self.velocity_commands = []

        # Initialize hardware
        self._init_motors()
        self._init_encoders()

        print("🚗 Precision Robot Controller initialized")
        print(f"   Wheel base: {self.wheel_base*1000:.0f}mm")
        print(f"   Wheel radius: {self.wheel_radius*1000:.0f}mm")
        print(f"   Max speeds: {self.max_linear_speed:.1f}m/s, {math.degrees(self.max_angular_speed):.0f}°/s")

    def _init_motors(self) -> None:
        """Initialize robot motors"""
        motor_names = [
            'front left wheel motor', 'front right wheel motor',
            'rear left wheel motor', 'rear right wheel motor'
        ]

        self.motors = {}
        motor_keys = ['fl', 'fr', 'rl', 'rr']

        for i, key in enumerate(motor_keys):
            try:
                motor = self.robot.getDevice(motor_names[i])
                if motor:
                    motor.setPosition(float('inf'))  # Velocity control mode
                    motor.setVelocity(0.0)
                    self.motors[key] = motor
            except Exception as e:
                print(f"Warning: Motor {key} initialization failed: {e}")

        print(f"✅ Motors initialized: {len(self.motors)}/4")

    def _init_encoders(self) -> None:
        """Initialize wheel encoders for odometry"""
        encoder_names = [
            'front left wheel motor sensor', 'front right wheel motor sensor',
            'rear left wheel motor sensor', 'rear right wheel motor sensor'
        ]

        self.encoders = {}
        encoder_keys = ['fl', 'fr', 'rl', 'rr']

        for i, key in enumerate(encoder_keys):
            try:
                encoder = self.robot.getDevice(encoder_names[i])
                if encoder:
                    encoder.enable(self.timestep)
                    self.encoders[key] = encoder
            except Exception as e:
                print(f"Warning: Encoder {key} initialization failed: {e}")

        self.prev_encoder_values = {key: 0.0 for key in self.encoders}

        print(f"✅ Encoders initialized: {len(self.encoders)}/4")

    def set_velocities(self, linear: float, angular: float) -> None:
        """Set robot velocities with safety checks and smoothing"""
        start_time = time.time()

        try:
            # Emergency stop check
            if self.emergency_stop_active:
                linear, angular = 0.0, 0.0

            # Apply velocity limits with safety margin
            max_linear = self.max_linear_speed * self.velocity_limit_factor
            max_angular = self.max_angular_speed * self.velocity_limit_factor

            linear = np.clip(linear, -max_linear, max_linear)
            angular = np.clip(angular, -max_angular, max_angular)

            # Velocity smoothing
            if self.enable_velocity_smoothing:
                alpha = self.velocity_smoothing_factor
                linear = alpha * self.current_linear_vel + (1 - alpha) * linear
                angular = alpha * self.current_angular_vel + (1 - alpha) * angular

            # Store current velocities
            self.current_linear_vel = linear
            self.current_angular_vel = angular

            # Convert to wheel velocities using differential drive kinematics
            left_wheel_vel, right_wheel_vel = self._differential_drive_kinematics(linear, angular)

            # Apply wheel velocity limits
            left_wheel_vel = np.clip(left_wheel_vel, -self.max_wheel_velocity, self.max_wheel_velocity)
            right_wheel_vel = np.clip(right_wheel_vel, -self.max_wheel_velocity, self.max_wheel_velocity)

            # Set motor velocities
            self._set_wheel_velocities(left_wheel_vel, right_wheel_vel)

            # Performance tracking
            control_time = time.time() - start_time
            self.control_times.append(control_time)
            self.velocity_commands.append((linear, angular))

            if self.profiler:
                self.profiler.record_timing("robot_control", control_time)

        except Exception as e:
            print(f"Robot control error: {e}")
            self.emergency_stop()

    def _differential_drive_kinematics(self, linear: float, angular: float) -> Tuple[float, float]:
        """Convert linear and angular velocities to wheel velocities"""
        # Differential drive kinematics:
        # v_left = (linear - angular * wheelbase/2) / wheel_radius
        # v_right = (linear + angular * wheelbase/2) / wheel_radius

        left_wheel_vel = (linear - angular * self.wheel_base / 2.0) / self.wheel_radius
        right_wheel_vel = (linear + angular * self.wheel_base / 2.0) / self.wheel_radius

        return left_wheel_vel, right_wheel_vel

    def _set_wheel_velocities(self, left_vel: float, right_vel: float) -> None:
        """Set individual wheel velocities"""
        # Left wheels
        for key in ['fl', 'rl']:
            if key in self.motors:
                self.motors[key].setVelocity(left_vel)

        # Right wheels
        for key in ['fr', 'rr']:
            if key in self.motors:
                self.motors[key].setVelocity(right_vel)

    def stop_robot(self) -> None:
        """Emergency stop robot"""
        self.emergency_stop_active = True
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

        # Set all motors to zero velocity
        for motor in self.motors.values():
            if motor:
                motor.setVelocity(0.0)

        print("🛑 Emergency stop activated")

    def resume_operation(self) -> None:
        """Resume robot operation after emergency stop"""
        self.emergency_stop_active = False
        print("✅ Robot operation resumed")

    def get_odometry(self) -> RobotPose:
        """Get current robot pose from wheel odometry"""
        try:
            # Get current encoder readings
            current_encoders = {}
            for key, encoder in self.encoders.items():
                if encoder:
                    current_encoders[key] = encoder.getValue()

            if len(current_encoders) < 2:
                return self.robot_pose

            # Initialize odometry system
            if not self.odometry_initialized:
                self.prev_encoder_values.update(current_encoders)
                self.odometry_initialized = True
                return self.robot_pose

            # Calculate encoder changes
            encoder_deltas = {}
            for key in current_encoders:
                if key in self.prev_encoder_values:
                    encoder_deltas[key] = current_encoders[key] - self.prev_encoder_values[key]
                    self.prev_encoder_values[key] = current_encoders[key]

            if len(encoder_deltas) < 2:
                return self.robot_pose

            # Calculate wheel displacements
            left_keys = [k for k in encoder_deltas.keys() if 'l' in k]
            right_keys = [k for k in encoder_deltas.keys() if 'r' in k]

            if not left_keys or not right_keys:
                return self.robot_pose

            left_displacement = np.mean([encoder_deltas[k] for k in left_keys]) * self.wheel_radius
            right_displacement = np.mean([encoder_deltas[k] for k in right_keys]) * self.wheel_radius

            # Differential drive odometry
            linear_displacement = (left_displacement + right_displacement) / 2.0
            angular_displacement = (right_displacement - left_displacement) / self.wheel_base

            # Update robot pose using exact kinematics
            if abs(angular_displacement) < 1e-6:
                # Pure translation
                self.robot_pose.x += linear_displacement * math.cos(self.robot_pose.theta)
                self.robot_pose.y += linear_displacement * math.sin(self.robot_pose.theta)
            else:
                # Curved motion with instantaneous center of rotation (ICR)
                radius = linear_displacement / angular_displacement

                # ICR coordinates
                icr_x = self.robot_pose.x - radius * math.sin(self.robot_pose.theta)
                icr_y = self.robot_pose.y + radius * math.cos(self.robot_pose.theta)

                # Apply rotation transformation
                cos_theta = math.cos(angular_displacement)
                sin_theta = math.sin(angular_displacement)

                # Current position relative to ICR
                rel_x = self.robot_pose.x - icr_x
                rel_y = self.robot_pose.y - icr_y

                # New position after rotation
                self.robot_pose.x = icr_x + rel_x * cos_theta - rel_y * sin_theta
                self.robot_pose.y = icr_y + rel_x * sin_theta + rel_y * cos_theta

            # Update orientation
            self.robot_pose.theta = self._normalize_angle(self.robot_pose.theta + angular_displacement)
            self.robot_pose.timestamp = time.time()

            return self.robot_pose

        except Exception as e:
            print(f"Odometry error: {e}")
            return self.robot_pose

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]"""
        return math.atan2(math.sin(angle), math.cos(angle))

    def is_moving(self) -> bool:
        """Check if robot is currently moving"""
        return (abs(self.current_linear_vel) > 0.01 or 
                abs(self.current_angular_vel) > 0.01)

    def set_pose(self, pose: RobotPose) -> None:
        """Manually set robot pose (for initialization)"""
        self.robot_pose = pose
        print(f"📍 Robot pose set: ({pose.x:.3f}, {pose.y:.3f}, {math.degrees(pose.theta):.1f}°)")

    def get_control_stats(self) -> Dict[str, float]:
        """Get robot control performance statistics"""
        if not self.control_times:
            return {}

        # Calculate velocity statistics
        linear_vels = [cmd[0] for cmd in self.velocity_commands]
        angular_vels = [cmd[1] for cmd in self.velocity_commands]

        return {
            'avg_control_time_ms': np.mean(self.control_times) * 1000,
            'max_control_time_ms': np.max(self.control_times) * 1000,
            'control_calls': len(self.control_times),
            'avg_linear_velocity': np.mean(np.abs(linear_vels)) if linear_vels else 0,
            'avg_angular_velocity': np.mean(np.abs(angular_vels)) if angular_vels else 0,
            'max_linear_velocity': np.max(np.abs(linear_vels)) if linear_vels else 0,
            'max_angular_velocity': np.max(np.abs(angular_vels)) if angular_vels else 0,
            'emergency_stops': 1 if self.emergency_stop_active else 0
        }

class AdvancedNavigationController:
    """Advanced navigation controller with path following"""

    def __init__(self, robot_controller: IRobotController, 
                 profiler: Optional[PerformanceProfiler] = None):
        self.robot_controller = robot_controller
        self.profiler = profiler

        # Path following parameters
        self.waypoint_tolerance = 0.15  # 15cm tolerance
        self.lookahead_distance = 0.5   # 50cm lookahead
        self.max_lateral_error = 0.3    # Maximum lateral deviation

        # Control gains
        self.kp_linear = 1.0
        self.kp_angular = 2.0
        self.ki_angular = 0.1

        # Navigation state
        self.current_path: List[Point2D] = []
        self.current_waypoint_index = 0
        self.angular_error_integral = 0.0

        # Performance tracking
        self.navigation_times = []
        self.path_errors = []

        print("🧭 Advanced Navigation Controller initialized")
        print(f"   Waypoint tolerance: {self.waypoint_tolerance*100:.0f}cm")
        print(f"   Lookahead distance: {self.lookahead_distance*100:.0f}cm")

    def follow_path(self, path: List[Point2D], robot_pose: RobotPose) -> bool:
        """Follow a planned path using advanced control algorithms"""
        start_time = time.time()

        try:
            if not path:
                self.robot_controller.set_velocities(0.0, 0.0)
                return True

            self.current_path = path

            # Find current target waypoint
            target_waypoint = self._get_target_waypoint(robot_pose)

            if target_waypoint is None:
                # Path completed
                self.robot_controller.set_velocities(0.0, 0.0)
                return True

            # Calculate control commands
            linear_vel, angular_vel = self._calculate_control_commands(
                robot_pose, target_waypoint
            )

            # Apply commands
            self.robot_controller.set_velocities(linear_vel, angular_vel)

            # Performance tracking
            navigation_time = time.time() - start_time
            self.navigation_times.append(navigation_time)

            path_error = robot_pose.distance_to(target_waypoint)
            self.path_errors.append(path_error)

            if self.profiler:
                self.profiler.record_timing("navigation_control", navigation_time)

            return False  # Still following path

        except Exception as e:
            print(f"Navigation error: {e}")
            self.robot_controller.set_velocities(0.0, 0.0)
            return True

    def _get_target_waypoint(self, robot_pose: RobotPose) -> Optional[Point2D]:
        """Get current target waypoint using lookahead algorithm"""
        if not self.current_path:
            return None

        robot_point = robot_pose.to_point2d()

        # Check if current waypoint is reached
        if self.current_waypoint_index < len(self.current_path):
            current_target = self.current_path[self.current_waypoint_index]

            if robot_point.distance_to(current_target) < self.waypoint_tolerance:
                self.current_waypoint_index += 1
                print(f"✅ Waypoint {self.current_waypoint_index} reached")

        # Check if path completed
        if self.current_waypoint_index >= len(self.current_path):
            return None

        # Lookahead algorithm
        lookahead_target = None

        # Start from current waypoint
        for i in range(self.current_waypoint_index, len(self.current_path)):
            waypoint = self.current_path[i]
            distance = robot_point.distance_to(waypoint)

            if distance >= self.lookahead_distance:
                lookahead_target = waypoint
                break

        # If no lookahead target found, use the last waypoint
        if lookahead_target is None:
            lookahead_target = self.current_path[-1]

        return lookahead_target

    def _calculate_control_commands(self, robot_pose: RobotPose, 
                                  target: Point2D) -> Tuple[float, float]:
        """Calculate control commands using advanced control theory"""
        # Calculate desired heading
        dx = target.x - robot_pose.x
        dy = target.y - robot_pose.y
        desired_angle = math.atan2(dy, dx)

        # Calculate angular error
        angular_error = self._normalize_angle(desired_angle - robot_pose.theta)

        # Integrate angular error for I-term
        self.angular_error_integral += angular_error

        # Limit integral windup
        max_integral = 1.0
        self.angular_error_integral = np.clip(
            self.angular_error_integral, -max_integral, max_integral
        )

        # Calculate distance to target
        distance_to_target = robot_pose.distance_to(target)

        # Linear velocity control (proportional to distance, with limits)
        linear_velocity = self.kp_linear * distance_to_target
        linear_velocity = np.clip(linear_velocity, 0.0, 0.6)

        # Reduce linear velocity for sharp turns
        if abs(angular_error) > math.radians(30):
            linear_velocity *= 0.5

        # Angular velocity control (PI controller)
        angular_velocity = (self.kp_angular * angular_error + 
                           self.ki_angular * self.angular_error_integral)
        angular_velocity = np.clip(angular_velocity, -1.5, 1.5)

        return linear_velocity, angular_velocity

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]"""
        return math.atan2(math.sin(angle), math.cos(angle))

    def reset_path(self) -> None:
        """Reset navigation state for new path"""
        self.current_path = []
        self.current_waypoint_index = 0
        self.angular_error_integral = 0.0

    def get_navigation_stats(self) -> Dict[str, float]:
        """Get navigation performance statistics"""
        if not self.navigation_times:
            return {}

        return {
            'avg_navigation_time_ms': np.mean(self.navigation_times) * 1000,
            'avg_path_error': np.mean(self.path_errors),
            'max_path_error': np.max(self.path_errors),
            'navigation_calls': len(self.navigation_times),
            'current_waypoint_index': self.current_waypoint_index,
            'path_length': len(self.current_path)
        }

if __name__ == "__main__":
    print("🚗 Professional Robot Control Components Initialized")
    print("✅ Precision kinematics and odometry")
    print("✅ Velocity smoothing and safety systems")
    print("✅ Advanced navigation with lookahead control")
    print("✅ Performance monitoring and statistics")
    print("🚀 Ready for high-precision robot control")
    print("⚡ C++ optimization targets: Kinematics and control loops")
