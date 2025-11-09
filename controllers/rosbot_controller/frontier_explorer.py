
import numpy as np
import cv2
from typing import List, Tuple, Optional
import math
import logging
import time
from slam_logging_config import get_slam_logger, performance_monitor, SLAMLogger

class FrontierExplorer:
    """
    Frontier-based exploration for autonomous SLAM mapping.
    Detects boundaries between known free space and unknown areas.
    """
    
    def __init__(self, occupancy_grid, min_frontier_size=5, exploration_radius=1.0):
        self.occupancy_grid = occupancy_grid
        self.min_frontier_size = min_frontier_size  # Minimum cells to form a frontier
        self.exploration_radius = exploration_radius  # meters from frontier to target
        self.frontiers = []
        self.current_target = None
        self.exploration_complete = False
        
    def detect_frontiers(self, robot_position_world):
        """
        Detect frontier cells - boundaries between free and unknown space.
        Returns list of frontier regions as (centroid_world, size, cells).
        """
        prob_grid = self.occupancy_grid.get_probability_grid()
        
        # Define cell states
        FREE_THRESHOLD = 0.3    # Below this = free space
        OCCUPIED_THRESHOLD = 0.7  # Above this = occupied
        # Between thresholds = unknown
        
        frontier_cells = []
        rows, cols = prob_grid.shape
        
        # Find cells that are unknown and adjacent to free space
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # Current cell must be unknown
                if FREE_THRESHOLD <= prob_grid[i, j] <= OCCUPIED_THRESHOLD:
                    
                    # Check if adjacent to free space (8-connectivity)
                    has_free_neighbor = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                if prob_grid[ni, nj] < FREE_THRESHOLD:
                                    has_free_neighbor = True
                                    break
                        if has_free_neighbor:
                            break
                    
                    if has_free_neighbor:
                        frontier_cells.append((i, j))
        
        # Group nearby frontier cells into regions
        frontier_regions = self._group_frontier_cells(frontier_cells)
        
        # Convert to world coordinates and filter by size
        valid_frontiers = []
        for region_cells in frontier_regions:
            if len(region_cells) >= self.min_frontier_size:
                # Calculate centroid in grid coordinates
                centroid_grid = np.mean(region_cells, axis=0)
                
                # Convert to world coordinates
                centroid_world = self.occupancy_grid.grid_to_world(
                    int(centroid_grid[0]), int(centroid_grid[1])
                )
                
                # Calculate exploration target (slightly back from frontier)
                target_world = self._calculate_exploration_target(
                    centroid_world, robot_position_world
                )
                
                valid_frontiers.append({
                    'centroid_world': centroid_world,
                    'target_world': target_world,
                    'size': len(region_cells),
                    'cells': region_cells
                })
        
        self.frontiers = valid_frontiers
        return valid_frontiers
    
    def _group_frontier_cells(self, frontier_cells):
        """Group nearby frontier cells using connected components."""
        if not frontier_cells:
            return []
        
        # Create binary image of frontier cells
        max_row = max(cell[0] for cell in frontier_cells) + 1
        max_col = max(cell[1] for cell in frontier_cells) + 1
        frontier_image = np.zeros((max_row, max_col), dtype=np.uint8)
        
        for row, col in frontier_cells:
            frontier_image[row, col] = 255
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(frontier_image)
        
        # Group cells by component
        regions = []
        for label in range(1, num_labels):  # Skip background (label 0)
            component_cells = []
            for row, col in frontier_cells:
                if labels[row, col] == label:
                    component_cells.append((row, col))
            if component_cells:
                regions.append(component_cells)
        
        return regions
    
    def _calculate_exploration_target(self, frontier_centroid, robot_position):
        """Calculate a safe exploration target near the frontier."""
        # Vector from robot to frontier
        dx = frontier_centroid[0] - robot_position[0]
        dy = frontier_centroid[1] - robot_position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 1e-6:  # Avoid division by zero
            return frontier_centroid
        
        # Move target slightly back from frontier toward robot
        unit_dx = dx / distance
        unit_dy = dy / distance
        
        target_x = frontier_centroid[0] - unit_dx * self.exploration_radius
        target_y = frontier_centroid[1] - unit_dy * self.exploration_radius
        
        return (target_x, target_y)
    
    def select_best_frontier(self, robot_position_world, previous_target=None):
        """
        Select the best frontier to explore next.
        Considers distance, size, and avoids recent targets.
        """
        if not self.frontiers:
            self.exploration_complete = True
            return None
        
        best_frontier = None
        best_score = float('-inf')
        
        for frontier in self.frontiers:
            score = self._evaluate_frontier(frontier, robot_position_world, previous_target)
            if score > best_score:
                best_score = score
                best_frontier = frontier
        
        if best_frontier:
            self.current_target = best_frontier['target_world']
            return best_frontier
        
        return None
    
    def _evaluate_frontier(self, frontier, robot_position, previous_target):
        """
        Evaluate frontier quality based on multiple criteria.
        Higher score = better frontier.
        """
        target_pos = frontier['target_world']
        
        # Distance factor (closer is better, but not too close)
        distance = math.sqrt(
            (target_pos[0] - robot_position[0])**2 + 
            (target_pos[1] - robot_position[1])**2
        )
        
        # Optimal distance range: 0.5 to 2.0 meters
        if distance < 0.5:
            distance_score = distance / 0.5  # Penalize very close targets
        elif distance <= 2.0:
            distance_score = 1.0  # Optimal range
        else:
            distance_score = 2.0 / distance  # Penalize distant targets
        
        # Size factor (larger frontiers are more promising)
        size_score = min(frontier['size'] / 20.0, 1.0)  # Normalize, cap at 1.0
        
        # Avoid recently visited areas
        previous_penalty = 0.0
        if previous_target:
            prev_distance = math.sqrt(
                (target_pos[0] - previous_target[0])**2 + 
                (target_pos[1] - previous_target[1])**2
            )
            if prev_distance < 1.0:  # Within 1 meter of previous target
                previous_penalty = 0.5
        
        # Combined score (weights can be tuned)
        score = (0.4 * distance_score + 
                0.4 * size_score - 
                0.2 * previous_penalty)
        
        return score
    
    def is_exploration_complete(self):
        """Check if exploration is complete (no more frontiers)."""
        return self.exploration_complete or len(self.frontiers) == 0
    
    def get_exploration_status(self):
        """Get current exploration status for debugging."""
        return {
            'num_frontiers': len(self.frontiers),
            'current_target': self.current_target,
            'exploration_complete': self.exploration_complete,
            'frontiers': self.frontiers
        }
    
    def visualize_frontiers(self, robot_position_world):
        """
        Create a visualization of detected frontiers.
        Returns RGB image with frontiers marked.
        """
        # Get occupancy grid as image
        prob_grid = self.occupancy_grid.get_probability_grid()
        
        # Convert to RGB (0=black/unknown, 0.5=gray/uncertain, 1=white/free)
        vis_image = np.zeros((prob_grid.shape[0], prob_grid.shape[1], 3), dtype=np.uint8)
        vis_image[:, :, 0] = (prob_grid * 255).astype(np.uint8)  # Red channel
        vis_image[:, :, 1] = (prob_grid * 255).astype(np.uint8)  # Green channel  
        vis_image[:, :, 2] = (prob_grid * 255).astype(np.uint8)  # Blue channel
        
        # Mark frontier cells in blue
        for frontier in self.frontiers:
            for cell in frontier['cells']:
                row, col = cell
                if 0 <= row < vis_image.shape[0] and 0 <= col < vis_image.shape[1]:
                    vis_image[row, col] = [0, 0, 255]  # Blue for frontiers
        
        # Mark robot position in red
        robot_grid = self.occupancy_grid.world_to_grid(
            robot_position_world[0], robot_position_world[1]
        )
        if 0 <= robot_grid[0] < vis_image.shape[0] and \
            0 <= robot_grid[1] < vis_image.shape[1]:
            vis_image[robot_grid[0], robot_grid[1]] = [255, 0, 0]  # Red for robot
        
        # Mark current target in green
        if self.current_target:
            target_grid = self.occupancy_grid.world_to_grid(
                self.current_target[0], self.current_target[1]
            )
            if 0 <= target_grid[0] < vis_image.shape[0] and \
                0 <= target_grid[1] < vis_image.shape[1]:
                vis_image[target_grid[0], target_grid[1]] = [0, 255, 0]  # Green for target
        
        return vis_image


class AutonomousExplorer:
    """
    High-level autonomous exploration controller.
    Integrates frontier detection with path planning and movement.
    """
    
    def __init__(self, occupancy_grid, a_star_planner, robot_radius=0.15):
        self.frontier_explorer = FrontierExplorer(occupancy_grid)
        self.occupancy_grid = occupancy_grid
        self.a_star = a_star_planner
        self.robot_radius = robot_radius
        
        self.current_path = None
        self.current_target = None
        self.previous_target = None
        self.state = "EXPLORING"  # EXPLORING, MOVING_TO_TARGET, COMPLETED
        self.path_index = 0
        
        # Navigation parameters
        self.target_tolerance = 0.3  # meters
        self.stuck_threshold = 10    # timesteps before replanning
        self.stuck_counter = 0
        
    @performance_monitor
    def update(self, robot_position_world, robot_orientation=0.0):
        """
        Main update loop for autonomous exploration.
        Returns: (linear_velocity, angular_velocity) commands.
        """
        
        if self.state == "COMPLETED":
            return 0.0, 0.0
            
        # Update frontier detection
        frontiers = self.frontier_explorer.detect_frontiers(robot_position_world)
        
        # Check if exploration is complete
        if self.frontier_explorer.is_exploration_complete():
            self.state = "COMPLETED"
            print("Exploration completed! No more frontiers detected.")
            return 0.0, 0.0
        
        # Select new target if needed
        if self.current_target is None or \
            self._reached_target(robot_position_world) or \
            self.stuck_counter > self.stuck_threshold:
            
            if self.stuck_counter > self.stuck_threshold:
                print("Robot appears stuck, selecting new target...")
                self.previous_target = self.current_target
                
            frontier = self.frontier_explorer.select_best_frontier(
                robot_position_world, self.previous_target
            )
            
            if frontier:
                self.current_target = frontier['target_world']
                print(f"New exploration target: {self.current_target}")
                
                # Plan path to target
                self.current_path = self.a_star.find_path(
                    robot_position_world, 
                    self.current_target,
                    robot_radius_grid=int(self.robot_radius / self.occupancy_grid.resolution)
                )
                
                if self.current_path:
                    self.path_index = 0
                    self.state = "MOVING_TO_TARGET"
                    self.stuck_counter = 0
                    print(f"Path planned with {len(self.current_path)} waypoints")
                else:
                    print("Failed to plan path to target")
                    self.current_target = None
                    return 0.0, 0.0
            else:
                print("No suitable frontier found")
                return 0.0, 0.0
        
        # Execute movement toward target
        if self.state == "MOVING_TO_TARGET" and self.current_path:
            return self._follow_path(robot_position_world, robot_orientation)
        
        return 0.0, 0.0
    
    def _reached_target(self, robot_position):
        """Check if robot has reached the current target."""
        if not self.current_target:
            return False
            
        distance = math.sqrt(
            (robot_position[0] - self.current_target[0])**2 + 
            (robot_position[1] - self.current_target[1])**2
        )
        
        return distance < self.target_tolerance
    
    def _follow_path(self, robot_position, robot_orientation):
        """
        Follow the planned path using simple proportional control.
        Returns: (linear_velocity, angular_velocity)
        """
        if not self.current_path or self.path_index >= len(self.current_path):
            return 0.0, 0.0
        
        # Get current waypoint
        waypoint = self.current_path[self.path_index]
        
        # Calculate distance to waypoint
        dx = waypoint[0] - robot_position[0]
        dy = waypoint[1] - robot_position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if reached current waypoint
        if distance < 0.2:  # 20cm tolerance for waypoints
            self.path_index += 1
            if self.path_index >= len(self.current_path):
                print("Reached target!")
                self.previous_target = self.current_target
                self.current_target = None
                self.state = "EXPLORING"
                return 0.0, 0.0
            else:
                waypoint = self.current_path[self.path_index]
                dx = waypoint[0] - robot_position[0]
                dy = waypoint[1] - robot_position[1]
                distance = math.sqrt(dx*dx + dy*dy)
        
        # Calculate desired heading
        desired_heading = math.atan2(dy, dx)
        
        # Calculate heading error
        heading_error = desired_heading - robot_orientation
        
        # Normalize heading error to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Proportional control
        max_linear_speed = 0.5   # m/s
        max_angular_speed = 1.0  # rad/s
        
        # Reduce linear speed when turning
        angular_velocity = max_angular_speed * np.clip(heading_error / (math.pi/4), -1, 1)
        linear_velocity = max_linear_speed * (1.0 - abs(heading_error) / math.pi)
        
        # Ensure minimum forward motion
        linear_velocity = max(linear_velocity, 0.1)
        
        # Check for potential stuck condition
        if linear_velocity < 0.15 and abs(angular_velocity) < 0.1:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        return linear_velocity, angular_velocity
    
    def get_status(self):
        """Get current exploration status for monitoring."""
        status = self.frontier_explorer.get_exploration_status()
        status.update({
            'state': self.state,
            'current_target': self.current_target,
            'path_length': len(self.current_path) if self.current_path else 0,
            'path_progress': self.path_index,
            'stuck_counter': self.stuck_counter
        })
        return status
    
    def visualize_exploration(self, robot_position_world):
        """Create visualization of current exploration state."""
        return self.frontier_explorer.visualize_frontiers(robot_position_world)


