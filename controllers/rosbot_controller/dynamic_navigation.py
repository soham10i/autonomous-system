
import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time

@dataclass
class SafetyStatus:
    """Current safety status of the robot"""
    is_safe: bool
    emergency_stop: bool
    collision_risk: float  # [0.0, 1.0]
    min_obstacle_distance: float
    safety_message: str
    timestamp: float

@dataclass
class NavigationCommand:
    """Navigation command with safety constraints"""
    linear_velocity: float
    angular_velocity: float
    confidence: float
    safety_override: bool
    command_source: str

class ObstacleDetector:
    """
    Real-time obstacle detection using LiDAR data with safety zones.
    """
    
    def __init__(self, robot_radius=0.15, safety_margin=0.3):
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        
        # Safety zones (distances in meters)
        self.critical_zone = robot_radius + 0.1    # Emergency stop zone
        self.warning_zone = robot_radius + 0.2     # Slow down zone
        self.caution_zone = robot_radius + safety_margin  # Path replan zone
        
        # Detection parameters
        self.max_detection_range = 3.0
        self.angular_resolution = math.pi / 180.0  # 1 degree
        self.obstacle_persistence_time = 2.0  # seconds
        
        # Obstacle tracking
        self.detected_obstacles = {}
        self.last_detection_time = {}
        
    def detect_obstacles(self, lidar_ranges, robot_position, robot_orientation):
        """
        Detect obstacles from LiDAR data and classify by safety zones.
        Returns SafetyStatus object.
        """
        current_time = time.time()
        
        if not lidar_ranges or len(lidar_ranges) == 0:
            return SafetyStatus(
                is_safe=False,
                emergency_stop=True,
                collision_risk=1.0,
                min_obstacle_distance=0.0,
                safety_message="No LiDAR data available",
                timestamp=current_time
            )
        
        # Analyze LiDAR data
        min_distance = float('inf')
        critical_obstacles = []
        warning_obstacles = []
        caution_obstacles = []
        
        num_rays = len(lidar_ranges)
        angle_per_ray = 2 * math.pi / num_rays
        
        for i, distance in enumerate(lidar_ranges):
            if distance <= 0 or distance > self.max_detection_range:
                continue
            
            # Calculate obstacle angle relative to robot
            ray_angle = i * angle_per_ray
            absolute_angle = robot_orientation + ray_angle - math.pi  # Adjust for forward direction
            
            # Calculate obstacle position
            obs_x = robot_position[0] + distance * math.cos(absolute_angle)
            obs_y = robot_position[1] + distance * math.sin(absolute_angle)
            
            min_distance = min(min_distance, distance)
            
            # Classify by safety zone
            if distance <= self.critical_zone:
                critical_obstacles.append((obs_x, obs_y, distance, ray_angle))
            elif distance <= self.warning_zone:
                warning_obstacles.append((obs_x, obs_y, distance, ray_angle))
            elif distance <= self.caution_zone:
                caution_obstacles.append((obs_x, obs_y, distance, ray_angle))
        
        # Determine safety status
        emergency_stop = len(critical_obstacles) > 0
        collision_risk = self._calculate_collision_risk(critical_obstacles, warning_obstacles, caution_obstacles)
        
        # Generate safety message
        if emergency_stop:
            safety_message = f"EMERGENCY: {len(critical_obstacles)} critical obstacles detected"
        elif len(warning_obstacles) > 0:
            safety_message = f"WARNING: {len(warning_obstacles)} close obstacles detected"
        elif len(caution_obstacles) > 0:
            safety_message = f"CAUTION: {len(caution_obstacles)} obstacles in path"
        else:
            safety_message = "All clear"
        
        return SafetyStatus(
            is_safe=not emergency_stop and len(warning_obstacles) == 0,
            emergency_stop=emergency_stop,
            collision_risk=collision_risk,
            min_obstacle_distance=min_distance if min_distance != float('inf') else self.max_detection_range,
            safety_message=safety_message,
            timestamp=current_time
        )
    
    def _calculate_collision_risk(self, critical, warning, caution):
        """Calculate overall collision risk based on detected obstacles."""
        if len(critical) > 0:
            return 1.0  # Maximum risk
        elif len(warning) > 0:
            return 0.7 + 0.3 * (len(warning) / 10.0)  # High risk
        elif len(caution) > 0:
            return 0.3 + 0.4 * (len(caution) / 20.0)  # Moderate risk
        else:
            return 0.0  # No risk
    
    def get_obstacle_free_directions(self, lidar_ranges, robot_orientation, num_sectors=8):
        """
        Find obstacle-free directions for emergency navigation.
        Returns list of (angle, clearance) tuples.
        """
        if not lidar_ranges:
            return []
        
        sector_size = 2 * math.pi / num_sectors
        sector_clearances = []
        
        for sector in range(num_sectors):
            sector_angle = sector * sector_size
            min_clearance = float('inf')
            
            # Check rays in this sector
            num_rays = len(lidar_ranges)
            rays_per_sector = max(1, num_rays // num_sectors)
            start_ray = sector * rays_per_sector
            end_ray = min(start_ray + rays_per_sector, num_rays)
            
            for ray_idx in range(start_ray, end_ray):
                distance = lidar_ranges[ray_idx]
                if distance > 0:
                    min_clearance = min(min_clearance, distance)
            
            if min_clearance > self.caution_zone:
                absolute_angle = robot_orientation + sector_angle - math.pi
                sector_clearances.append((absolute_angle, min_clearance))
        
        # Sort by clearance (best first)
        sector_clearances.sort(key=lambda x: x[1], reverse=True)
        return sector_clearances

class DynamicPathPlanner:
    """
    Dynamic path planning that replans when obstacles block the current path.
    """
    
    def __init__(self, occupancy_grid, a_star_planner, replan_threshold=0.5):
        self.occupancy_grid = occupancy_grid
        self.a_star = a_star_planner
        self.replan_threshold = replan_threshold  # Distance ahead to check for obstacles
        
        # Current navigation state
        self.current_path = None
        self.current_target = None
        self.path_index = 0
        self.last_replan_time = 0
        self.replan_cooldown = 2.0  # Minimum time between replans
        
        # Path validation
        self.path_check_distance = 1.0  # Check path this far ahead
        self.path_validity_threshold = 0.7  # Occupancy probability threshold
        
    def set_target(self, target_position, robot_position):
        """Set a new navigation target and plan initial path."""
        self.current_target = target_position
        self.current_path = self.a_star.find_path(robot_position, target_position)
        self.path_index = 0
        
        if self.current_path:
            print(f"Path planned to target {target_position}: {len(self.current_path)} waypoints")
            return True
        else:
            print(f"Failed to plan path to target {target_position}")
            return False
    
    def update_navigation(self, robot_position, robot_orientation, safety_status):
        """
        Update navigation with dynamic replanning based on safety status.
        Returns NavigationCommand.
        """
        current_time = time.time()
        
        # Emergency stop override
        if safety_status.emergency_stop:
            return NavigationCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
                confidence=1.0,
                safety_override=True,
                command_source="emergency_stop"
            )
        
        # Check if we have a valid path
        if not self.current_path or not self.current_target:
            return NavigationCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
                confidence=0.0,
                safety_override=False,
                command_source="no_path"
            )
        
        # Check if we've reached the target
        target_distance = math.sqrt(
            (robot_position[0] - self.current_target[0])**2 + 
            (robot_position[1] - self.current_target[1])**2
        )
        
        if target_distance < 0.2:  # 20cm tolerance
            print("Target reached!")
            self.current_path = None
            self.current_target = None
            return NavigationCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
                confidence=1.0,
                safety_override=False,
                command_source="target_reached"
            )
        
        # Check if path needs replanning
        needs_replan = False
        
        # Check for obstacles in current path
        if (current_time - self.last_replan_time > self.replan_cooldown and
            safety_status.collision_risk > 0.5):
            path_blocked = self._is_path_blocked(robot_position)
            if path_blocked:
                needs_replan = True
                print("Path blocked by obstacles, replanning...")
        
        # Check if robot has deviated too far from path
        if self.path_index < len(self.current_path):
            next_waypoint = self.current_path[self.path_index]
            deviation = math.sqrt(
                (robot_position[0] - next_waypoint[0])**2 + 
                (robot_position[1] - next_waypoint[1])**2
            )
            if deviation > 0.5:  # 50cm deviation threshold
                needs_replan = True
                print("Robot deviated from path, replanning...")
        
        # Perform replanning if needed
        if needs_replan:
            self._replan_path(robot_position)
            self.last_replan_time = current_time
        
        # Generate navigation command
        return self._follow_path(robot_position, robot_orientation, safety_status)
    
    def _is_path_blocked(self, robot_position):
        """Check if the current path is blocked by obstacles."""
        if not self.current_path:
            return False
        
        # Check waypoints ahead of current position
        check_distance = 0.0
        for i in range(self.path_index, len(self.current_path)):
            waypoint = self.current_path[i]
            
            # Check if waypoint is in occupied space
            grid_x, grid_y = self.occupancy_grid.world_to_grid(waypoint[0], waypoint[1])
            if self.occupancy_grid.is_valid_grid_coord(grid_x, grid_y):
                prob_grid = self.occupancy_grid.get_probability_grid()
                if prob_grid[grid_x, grid_y] > self.path_validity_threshold:
                    return True
            
            # Only check a certain distance ahead
            if i > self.path_index:
                prev_waypoint = self.current_path[i-1]
                segment_length = math.sqrt(
                    (waypoint[0] - prev_waypoint[0])**2 + 
                    (waypoint[1] - prev_waypoint[1])**2
                )
                check_distance += segment_length
                
                if check_distance > self.path_check_distance:
                    break
        
        return False
    
    def _replan_path(self, robot_position):
        """Replan path from current position to target."""
        if not self.current_target:
            return
        
        print(f"Replanning path from {robot_position} to {self.current_target}")
        new_path = self.a_star.find_path(robot_position, self.current_target)
        
        if new_path:
            self.current_path = new_path
            self.path_index = 0
            print(f"Replanned path with {len(new_path)} waypoints")
        else:
            print("Failed to replan path - keeping original path")
    
    def _follow_path(self, robot_position, robot_orientation, safety_status):
        """Generate navigation commands to follow the current path."""
        if not self.current_path or self.path_index >= len(self.current_path):
            return NavigationCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
                confidence=0.0,
                safety_override=False,
                command_source="path_complete"
            )
        
        # Get next waypoint
        target_waypoint = self.current_path[self.path_index]
        
        # Check if we've reached current waypoint
        waypoint_distance = math.sqrt(
            (robot_position[0] - target_waypoint[0])**2 + 
            (robot_position[1] - target_waypoint[1])**2
        )
        
        if waypoint_distance < 0.15:  # 15cm waypoint tolerance
            self.path_index += 1
            if self.path_index < len(self.current_path):
                target_waypoint = self.current_path[self.path_index]
            else:
                return NavigationCommand(
                    linear_velocity=0.0,
                    angular_velocity=0.0,
                    confidence=1.0,
                    safety_override=False,
                    command_source="path_complete"
                )
        
        # Calculate desired heading
        dx = target_waypoint[0] - robot_position[0]
        dy = target_waypoint[1] - robot_position[1]
        desired_heading = math.atan2(dy, dx)
        
        # Calculate heading error
        heading_error = desired_heading - robot_orientation
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Calculate velocities with safety adjustments
        max_linear_speed = 0.5
        max_angular_speed = 1.0
        
        # Reduce speed based on safety status
        speed_factor = 1.0
        if safety_status.collision_risk > 0.3:
            speed_factor = max(0.1, 1.0 - safety_status.collision_risk)
        
        # Proportional control
        angular_velocity = max_angular_speed * np.clip(heading_error / (math.pi/4), -1, 1)
        linear_velocity = max_linear_speed * speed_factor * (1.0 - abs(heading_error) / math.pi)
        
        # Minimum forward motion (unless unsafe)
        if not safety_status.emergency_stop and safety_status.collision_risk < 0.7:
            linear_velocity = max(linear_velocity, 0.05)
        
        # Calculate confidence based on path validity and safety
        confidence = min(1.0, (1.0 - safety_status.collision_risk) * 0.8 + 0.2)
        
        return NavigationCommand(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            confidence=confidence,
            safety_override=safety_status.collision_risk > 0.5,
            command_source="path_following"
        )
    
    def get_navigation_status(self):
        """Get current navigation status for monitoring."""
        return {
            'has_path': self.current_path is not None,
            'has_target': self.current_target is not None,
            'path_length': len(self.current_path) if self.current_path else 0,
            'path_progress': self.path_index,
            'target_position': self.current_target,
            'time_since_last_replan': time.time() - self.last_replan_time
        }

class EmergencyNavigator:
    """
    Emergency navigation for when normal path planning fails.
    Uses reactive behaviors to avoid obstacles and find safe directions.
    """
    
    def __init__(self, obstacle_detector):
        self.obstacle_detector = obstacle_detector
        
        # Emergency behavior parameters
        self.escape_speed = 0.2  # Slow speed for emergency maneuvers
        self.escape_turn_rate = 0.8  # Turn rate for escape maneuvers
        self.stuck_threshold = 10  # Timesteps before declaring stuck
        self.stuck_counter = 0
        
    def emergency_navigate(self, robot_position, robot_orientation, lidar_ranges, safety_status):
        """
        Emergency navigation when normal planning fails.
        Uses reactive obstacle avoidance.
        """
        if safety_status.emergency_stop:
            return NavigationCommand(
                linear_velocity=0.0,
                angular_velocity=0.0,
                confidence=1.0,
                safety_override=True,
                command_source="emergency_stop"
            )
        
        # Get obstacle-free directions
        free_directions = self.obstacle_detector.get_obstacle_free_directions(
            lidar_ranges, robot_orientation
        )
        
        if not free_directions:
            # No free directions - rotate in place to find escape route
            return NavigationCommand(
                linear_velocity=0.0,
                angular_velocity=self.escape_turn_rate,
                confidence=0.5,
                safety_override=True,
                command_source="emergency_rotation"
            )
        
        # Choose best escape direction (highest clearance)
        best_direction, clearance = free_directions[0]
        
        # Calculate turn to escape direction
        heading_error = best_direction - robot_orientation
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Emergency escape command
        angular_velocity = self.escape_turn_rate * np.clip(heading_error / (math.pi/4), -1, 1)
        
        # Move forward if heading roughly correct and path is clear
        linear_velocity = 0.0
        if abs(heading_error) < math.pi/4 and clearance > 0.5:
            linear_velocity = self.escape_speed
        
        return NavigationCommand(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            confidence=0.3,  # Low confidence for emergency maneuvers
            safety_override=True,
            command_source="emergency_escape"
        )

class SafetyController:
    """
    Main safety controller that coordinates obstacle detection, path planning,
    and emergency behaviors.
    """
    
    def __init__(self, occupancy_grid, a_star_planner, robot_radius=0.15):
        self.obstacle_detector = ObstacleDetector(robot_radius)
        self.path_planner = DynamicPathPlanner(occupancy_grid, a_star_planner)
        self.emergency_navigator = EmergencyNavigator(self.obstacle_detector)
        
        # Control mode
        self.control_mode = "NORMAL"  # NORMAL, EMERGENCY, STUCK
        self.mode_transition_time = 0
        
        # Safety monitoring
        self.safety_history = []
        self.max_safety_history = 20
        
    def update(self, robot_position, robot_orientation, lidar_ranges, target_position=None):
        """
        Main safety controller update loop.
        Returns NavigationCommand with safety guarantees.
        """
        current_time = time.time()
        
        # Detect obstacles and assess safety
        safety_status = self.obstacle_detector.detect_obstacles(
            lidar_ranges, robot_position, robot_orientation
        )
        
        # Update safety history
        self.safety_history.append(safety_status)
        if len(self.safety_history) > self.max_safety_history:
            self.safety_history.pop(0)
        
        # Set target if provided
        if target_position and target_position != self.path_planner.current_target:
            self.path_planner.set_target(target_position, robot_position)
        
        # Determine control mode
        self._update_control_mode(safety_status, current_time)
        
        # Generate navigation command based on mode
        if self.control_mode == "EMERGENCY":
            command = self.emergency_navigator.emergency_navigate(
                robot_position, robot_orientation, lidar_ranges, safety_status
            )
        else:
            command = self.path_planner.update_navigation(
                robot_position, robot_orientation, safety_status
            )
            
            # Override with emergency behavior if needed
            if safety_status.emergency_stop:
                command.linear_velocity = 0.0
                command.angular_velocity = 0.0
                command.safety_override = True
                command.command_source = "safety_override"
        
        return command, safety_status
    
    def _update_control_mode(self, safety_status, current_time):
        """Update control mode based on safety status."""
        previous_mode = self.control_mode
        
        # Emergency mode conditions
        if safety_status.emergency_stop or safety_status.collision_risk > 0.8:
            self.control_mode = "EMERGENCY"
        
        # Return to normal mode conditions
        elif (self.control_mode == "EMERGENCY" and 
              safety_status.collision_risk < 0.3 and
              current_time - self.mode_transition_time > 3.0):  # 3 second cooldown
            self.control_mode = "NORMAL"
        
        # Track mode transitions
        if previous_mode != self.control_mode:
            self.mode_transition_time = current_time
            print(f"Control mode changed: {previous_mode} -> {self.control_mode}")
    
    def get_safety_status(self):
        """Get comprehensive safety and navigation status."""
        nav_status = self.path_planner.get_navigation_status()
        
        # Calculate recent safety metrics
        recent_emergencies = sum(1 for s in self.safety_history[-10:] if s.emergency_stop)
        avg_collision_risk = np.mean([s.collision_risk for s in self.safety_history[-5:]] if self.safety_history else [0.0])
        
        return {
            'control_mode': self.control_mode,
            'navigation_status': nav_status,
            'recent_emergencies': recent_emergencies,
            'average_collision_risk': avg_collision_risk,
            'safety_history_length': len(self.safety_history)
        }