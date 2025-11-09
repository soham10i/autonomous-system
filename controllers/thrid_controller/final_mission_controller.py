#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
final_controller_v2.py (Complete Mission with Advanced Goal Generation)

This is the final, fully integrated controller. It uses the ADAS-style
architecture and incorporates an advanced goal generation method for the
FOLLOW_WALL behavior, inspired by principles from robotics research papers.
This prevents the robot from getting stuck at intersections and enables robust
exploration and mission completion.
"""
# -----------------------------------------------------------------------------
# --- IMPORTS -----------------------------------------------------------------
# -----------------------------------------------------------------------------
import math
import numpy as np
from controller import Supervisor
import cv2
from enum import Enum
import heapq
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

# -----------------------------------------------------------------------------
# --- CONSTANTS ---------------------------------------------------------------
# -----------------------------------------------------------------------------
TIME_STEP = 32
# Robot Constraints
MAX_LINEAR_VELOCITY = 1.5
MAX_ANGULAR_VELOCITY = 15.0
MAX_LINEAR_ACCELERATION = 0.5
MAX_ANGULAR_ACCELERATION = 2.0

# DWA Planner Config
DWA_TIME_HORIZON = 2.0
DWA_V_SAMPLES = 10
DWA_W_SAMPLES = 30
DWA_COST_GOAL = 2.0
DWA_COST_VELOCITY = 0.5
DWA_COST_OBSTACLE = 1.0
DWA_ROBOT_RADIUS = 0.3
BEHAVIOR_COMMIT_TIME = 3.0
WALL_FOLLOW_DISTANCE = 0.6

# Map and A* Config
MAP_SIZE_PIXELS = 600
GRID_BOUNDS_M = 10.0
GRID_RESOLUTION = GRID_BOUNDS_M / MAP_SIZE_PIXELS
LIDAR_MAPPING_RANGE = 5.0
ASTAR_WAYPOINT_THRESHOLD = 0.2
PILLAR_DISTANCE_THRESHOLD = 0.3

# Color Detection
HSV_RANGES = {
    'red': [(0, 150, 100), (10, 255, 255)],
    'blue': [(100, 150, 50), (140, 255, 255)],
    'yellow': [(20, 100, 100), (30, 255, 255)]
}
MIN_CONTOUR_AREA = 100

# -----------------------------------------------------------------------------
# --- STATE MACHINES & DATA STRUCTURES ----------------------------------------
# -----------------------------------------------------------------------------
class MissionState(Enum):
    EXPLORING = "Exploring with DWA"
    PLANNING_PATH = "Planning A* Path"
    NAVIGATING_PATH = "Navigating A* Path"
    FINISHED = "Mission Finished"

class Behavior(Enum):
    FOLLOW_WALL = "Following Wall"
    ENTER_PASSAGE = "Entering Passage"
    CROSS_OPEN_SPACE = "Crossing Open Space"

class WorldModel:
    def __init__(self):
        self.wall_segments = []
        self.lidar_points = np.array([])

# -----------------------------------------------------------------------------
# --- CORE MODULES (UNCHANGED) ------------------------------------------------
# -----------------------------------------------------------------------------
class OccupancyGrid:
    def __init__(self, map_size_pixels, resolution):
        self.map_size = map_size_pixels
        self.resolution = resolution
        self.log_odds_grid = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.L_OCC, self.L_FREE = 1.0, -0.5
        self.semantic_objects = {}

    def world_to_map(self, world_x, world_y):
        map_x = int(world_x / self.resolution + self.map_size / 2)
        map_y = int(-world_y / self.resolution + self.map_size / 2)
        return map_x, map_y

    def update_cell(self, map_x, map_y, is_occupied):
        if 0 <= map_x < self.map_size and 0 <= map_y < self.map_size:
            update = self.L_OCC if is_occupied else self.L_FREE
            self.log_odds_grid[map_y, map_x] = np.clip(self.log_odds_grid[map_y, map_x] + update, -10, 10)

    def add_semantic_object(self, world_x, world_y, color_name):
        if color_name in self.semantic_objects:
            ex, ey = self.semantic_objects[color_name]
            self.semantic_objects[color_name] = ((ex * 9 + world_x) / 10.0, (ey * 9 + world_y) / 10.0)
        else:
            self.semantic_objects[color_name] = (world_x, world_y)

    def raytrace(self, x0, y0, x1, y1):
        dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
        dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            if x0 == x1 and y0 == y1: break
            self.update_cell(x0, y0, is_occupied=False)
            e2 = 2 * err
            if e2 >= dy: err += dy; x0 += sx
            if e2 <= dx: err += dx; y0 += sy

class AStarPlanner:
    def __init__(self, grid):
        self.grid = grid
    def heuristic(self, a, b): return math.hypot(b[0] - a[0], b[1] - a[1])
    def plan_path(self, start_world, goal_world):
        start_grid, goal_grid = self.grid.world_to_map(*start_world), self.grid.world_to_map(*goal_world)
        open_set, came_from, g_score = [], {}, {start_grid: 0}
        heapq.heappush(open_set, (self.heuristic(start_grid, goal_grid), start_grid))
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_grid:
                return self._reconstruct_path(came_from, current)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy)
                    if not (0 <= neighbor[0] < self.grid.map_size and 0 <= neighbor[1] < self.grid.map_size):
                        continue
                    if self.grid.log_odds_grid[neighbor[1], neighbor[0]] > 0.1:
                        continue
                    tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None
    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from: path.append(came_from[current])
        path.reverse()
        return [(self.grid.resolution * (pt[0] - self.grid.map_size / 2), self.grid.resolution * -(pt[1] - self.grid.map_size / 2)) for pt in path]

class PerceptionSystem:
    def __init__(self, sensor_spec):
        self.lidar_res = sensor_spec['lidar_res']
        self.lidar_fov = 2 * math.pi
        self.lidar_angles = np.array([(i / self.lidar_res - 0.5) * self.lidar_fov for i in range(self.lidar_res)])
    def process_lidar(self, ranges):
        ranges = np.array(ranges)
        valid_indices = np.isfinite(ranges) & (ranges > 0.01)
        angles, valid_ranges = self.lidar_angles[valid_indices], ranges[valid_indices]
        px, py = valid_ranges * np.cos(angles), valid_ranges * np.sin(angles)
        points = np.vstack((px, py)).T
        if len(points) < 10:
            return [], points
        db = DBSCAN(eps=0.15, min_samples=5).fit(points)
        labels = db.labels_
        wall_segments = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = points[labels == label]
            if len(cluster_points) < 5:
                continue
            if np.var(cluster_points[:, 0]) > np.var(cluster_points[:, 1]):
                X, y = cluster_points[:, 0].reshape(-1, 1), cluster_points[:, 1]
                ransac = RANSACRegressor(min_samples=2).fit(X, y)
                x_min, x_max = X.min(), X.max()
                y_min_pred, y_max_pred = ransac.predict([[x_min]]), ransac.predict([[x_max]])
                wall_segments.append(((x_min, y_min_pred[0]), (x_max, y_max_pred[0])))
            else:
                X, y = cluster_points[:, 1].reshape(-1, 1), cluster_points[:, 0]
                ransac = RANSACRegressor(min_samples=2).fit(X, y)
                x_min, x_max = X.min(), X.max()
                y_min_pred, y_max_pred = ransac.predict([[x_min]]), ransac.predict([[x_max]])
                wall_segments.append(((y_min_pred[0], x_min), (y_max_pred[0], x_max)))
        return wall_segments, points
    def update(self, lidar_ranges):
        world = WorldModel()
        world.wall_segments, world.lidar_points = self.process_lidar(lidar_ranges)
        return world

class LocalPlannerDWA:
    def __init__(self, constraints, config):
        self.constraints = constraints
        self.config = config
    def _point_to_line_segment_dist(self, px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0: return math.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
    def _calculate_dynamic_window(self, current_v, current_w, dt):
        v_min = max(0.0, current_v - self.constraints['max_linear_accel'] * dt)
        v_max = min(self.constraints['max_linear_vel'], current_v + self.constraints['max_linear_accel'] * dt)
        w_min = max(-self.constraints['max_angular_vel'], current_w - self.constraints['max_angular_accel'] * dt)
        w_max = min(self.constraints['max_angular_vel'], current_w + self.constraints['max_angular_accel'] * dt)
        return [v_min, v_max, w_min, w_max]
    def _simulate_trajectory(self, v, w, dt):
        x, y, theta = 0.0, 0.0, 0.0; path = []
        time = 0.0
        while time <= self.config['time_horizon']:
            time += dt; x += v * math.cos(theta) * dt; y += v * math.sin(theta) * dt
            theta += w * dt; path.append((x, y))
        return path
    def _score_trajectory(self, path, goal, v, wall_segments):
        cost_goal = self.config['cost_goal'] * math.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1])
        cost_velocity = self.config['cost_velocity'] * (self.constraints['max_linear_vel'] - v)
        min_obs_dist = float('inf')
        if not wall_segments:
            cost_obstacle = 0.0
        else:
            for pos in path:
                for p1, p2 in wall_segments:
                    dist = self._point_to_line_segment_dist(pos[0], pos[1], p1[0], p1[1], p2[0], p2[1])
                    if dist < min_obs_dist:
                        min_obs_dist = dist
            if min_obs_dist <= self.config['robot_radius']:
                return float('inf')
            cost_obstacle = self.config['cost_obstacle'] * (1.0 / min_obs_dist)
        return cost_goal + cost_velocity + cost_obstacle
    def find_best_trajectory(self, current_velocity, goal, world_model, dt):
        current_v, current_w = current_velocity
        v_min, v_max, w_min, w_max = self._calculate_dynamic_window(current_v, current_w, dt)
        best_cost, best_v_w, all_trajectories = float('inf'), (0.0, 0.0), []
        for v in np.linspace(v_min, v_max, DWA_V_SAMPLES):
            for w in np.linspace(w_min, w_max, DWA_W_SAMPLES):
                path = self._simulate_trajectory(v, w, dt)
                all_trajectories.append(path)
                cost = self._score_trajectory(path, goal, v, world_model.wall_segments)
                if cost < best_cost:
                    best_cost, best_v_w = cost, (v, w)
        return best_v_w, all_trajectories

# -----------------------------------------------------------------------------
# --- MAIN CONTROLLER CLASS ---------------------------------------------------
# -----------------------------------------------------------------------------
class FinalMissionController(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        
        # --- High-Level State ---
        self.mission_state = MissionState.EXPLORING
        self.behavior = Behavior.FOLLOW_WALL
        
        # --- Modules ---
        self.motors = [self.getDevice(name) for name in ["front left wheel motor", "front right wheel motor", "rear left wheel motor", "rear right wheel motor"]]
        for m in self.motors: m.setPosition(float('inf')); m.setVelocity(0.0)
        self.lidar = self.getDevice("lidar"); self.lidar.enable(self.timestep)
        
        # Check if camera devices exist before enabling
        self.camera = self.getDevice("camera rgb")
        if self.camera: self.camera.enable(self.timestep)
        
        self.depth_camera = self.getDevice("camera depth")
        if self.depth_camera: self.depth_camera.enable(self.timestep)
        
        actual_lidar_res = self.lidar.getHorizontalResolution()
        sensor_spec = {'lidar_res': actual_lidar_res}
        constraints = {'max_linear_vel': MAX_LINEAR_VELOCITY, 'max_angular_vel': MAX_ANGULAR_VELOCITY, 'max_linear_accel': MAX_LINEAR_ACCELERATION, 'max_angular_accel': MAX_ANGULAR_ACCELERATION}
        dwa_config = {'time_horizon': DWA_TIME_HORIZON, 'cost_goal': DWA_COST_GOAL, 'cost_velocity': DWA_COST_VELOCITY, 'cost_obstacle': DWA_COST_OBSTACLE, 'robot_radius': DWA_ROBOT_RADIUS}

        # --- Main System Components ---
        self.grid = OccupancyGrid(MAP_SIZE_PIXELS, GRID_RESOLUTION)
        self.astar_planner = AStarPlanner(self.grid)
        self.perception = PerceptionSystem(sensor_spec)
        self.local_planner = LocalPlannerDWA(constraints, dwa_config)
        
        # --- State Variables ---
        self.robot_node = self.getSelf()
        self.pose = np.array([0.0, 0.0, 0.0]) # x, y, theta
        self.world_model = WorldModel()
        self.current_velocity = (0.0, 0.0)
        self.astar_path = []
        self.astar_waypoint_idx = 0
        self.behavior_timer = 0.0
        self.locked_goal_angle = 0.0

    # --- Perception Methods ---
    def update_global_map(self, lidar_ranges):
        robot_map_x, robot_map_y = self.grid.world_to_map(*self.pose[:2])
        fov, num_rays = self.lidar.getFov(), self.lidar.getHorizontalResolution()
        for i, dist in enumerate(lidar_ranges):
            dist = dist if math.isfinite(dist) and dist < LIDAR_MAPPING_RANGE else LIDAR_MAPPING_RANGE
            angle = self.pose[2] + (i / num_rays - 0.5) * fov
            end_x, end_y = self.pose[0] + dist * math.cos(angle), self.pose[1] + dist * math.sin(angle)
            end_map_x, end_map_y = self.grid.world_to_map(end_x, end_y)
            self.grid.raytrace(robot_map_x, robot_map_y, end_map_x, end_map_y)
            if dist < LIDAR_MAPPING_RANGE: self.grid.update_cell(end_map_x, end_map_y, is_occupied=True)

    def detect_pillars(self):
        if not self.camera or not self.depth_camera: return # Check if cameras are initialized
        bgr_image = self.camera.getImage()
        if not bgr_image: return
        cam_width, cam_height, cam_fov = self.camera.getWidth(), self.camera.getHeight(), self.camera.getFov()
        frame = np.frombuffer(bgr_image, np.uint8).reshape((cam_height, cam_width, 4))[:,:,:3]
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for color_name, (lower, upper) in HSV_RANGES.items():
            mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < MIN_CONTOUR_AREA: continue
            M = cv2.moments(max(contours, key=cv2.contourArea)); cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            distance = self.depth_camera.getRangeImage()[cy * cam_width + cx] # Access depth image as 1D array
            if math.isinf(distance) or math.isnan(distance) or distance > 5.0: continue
            x_coord = (cx - cam_width/2)/(cam_width/2)
            x_cam, z_cam = -x_coord * distance * math.tan(cam_fov/2)*(cam_width/cam_height), distance
            x_robot, y_robot = z_cam, -x_cam
            rx, ry, r_theta = self.pose
            world_x = rx + x_robot * math.cos(r_theta) - y_robot * math.sin(r_theta)
            world_y = ry + x_robot * math.sin(r_theta) + y_robot * math.cos(r_theta)
            self.grid.add_semantic_object(world_x, world_y, color_name)
    
    # --- Planning & Behavior Methods ---
    def decide_exploration_behavior(self, current_time):
        # If a behavior is committed, stick to it until the timer runs out
        if current_time < self.behavior_timer: 
            return

        # Parameters for passage detection
        min_passage_width_m = 0.6 # Minimum width of a passage in meters
        max_passage_depth_m = 5.0 # Maximum depth to consider for a passage
        
        lidar_ranges = np.array(self.lidar.getRangeImage())
        if not lidar_ranges.size: # If no lidar data, assume open space
            self.behavior = Behavior.CROSS_OPEN_SPACE
            self.behavior_timer = current_time + BEHAVIOR_COMMIT_TIME * 4
            return

        # Filter out invalid readings and points beyond max_passage_depth
        valid_indices = np.isfinite(lidar_ranges) & (lidar_ranges > 0.01) & (lidar_ranges < max_passage_depth_m)
        angles = self.perception.lidar_angles[valid_indices]
        ranges = lidar_ranges[valid_indices]

        if not valid_indices.any(): # If no valid points, assume open space
            self.behavior = Behavior.CROSS_OPEN_SPACE
            self.behavior_timer = current_time + BEHAVIOR_COMMIT_TIME * 4
            return

        # Convert to Cartesian coordinates for easier width calculation
        points_x = ranges * np.cos(angles)
        points_y = ranges * np.sin(angles)
        
        # Find the largest continuous free space (gap)
        max_gap_angle = 0.0
        best_gap_mid_angle = 0.0
        
        # Iterate through sorted angles to find gaps
        # Sort points by angle for easier gap detection
        sorted_indices = np.argsort(angles)
        angles_sorted = angles[sorted_indices]
        points_x_sorted = points_x[sorted_indices]
        points_y_sorted = points_y[sorted_indices]

        # Consider wrapping around the LiDAR scan (from last point to first point)
        # Append the first point to the end with angle + 2*pi to check the wrap-around gap
        angles_extended = np.append(angles_sorted, angles_sorted[0] + 2 * math.pi)
        points_x_extended = np.append(points_x_sorted, points_x_sorted[0])
        points_y_extended = np.append(points_y_sorted, points_y_sorted[0])

        for i in range(len(angles_sorted)):
            p1_angle, p1_x, p1_y = angles_extended[i], points_x_extended[i], points_y_extended[i]
            p2_angle, p2_x, p2_y = angles_extended[i+1], points_x_extended[i+1], points_y_extended[i+1]

            # Calculate the Euclidean distance between consecutive points
            gap_distance = math.hypot(p2_x - p1_x, p2_y - p1_y)
            
            # Calculate the angular width of the gap
            gap_angular_width = p2_angle - p1_angle

            # If the gap is wider than min_passage_width_m and within the forward cone
            # and the angular width is significant
            if gap_distance > min_passage_width_m and abs(p1_angle + gap_angular_width / 2) < math.radians(150) / 2: # Check within a wide forward cone
                if gap_angular_width > max_gap_angle:
                    max_gap_angle = gap_angular_width
                    best_gap_mid_angle = p1_angle + gap_angular_width / 2

        # Normalize best_gap_mid_angle to be within -pi to pi
        best_gap_mid_angle = math.atan2(math.sin(best_gap_mid_angle), math.cos(best_gap_mid_angle))

        if max_gap_angle > math.radians(20): # A significant angular gap indicates a passage
            self.behavior = Behavior.ENTER_PASSAGE
            self.locked_goal_angle = best_gap_mid_angle
            self.behavior_timer = current_time + BEHAVIOR_COMMIT_TIME * 3
        elif self.world_model.wall_segments: # If walls are detected but no clear passage, follow wall
            self.behavior = Behavior.FOLLOW_WALL
            self.behavior_timer = current_time + BEHAVIOR_COMMIT_TIME / 3
        else: # No walls, no clear passage, cross open space
            self.behavior = Behavior.CROSS_OPEN_SPACE
            self.behavior_timer = current_time + BEHAVIOR_COMMIT_TIME * 4

    def get_dwa_goal(self, current_time):
        if self.mission_state == MissionState.NAVIGATING_PATH:
            if not self.astar_path or self.astar_waypoint_idx >= len(self.astar_path): return (0.0, 0.0)
            waypoint = self.astar_path[self.astar_waypoint_idx]
            dx, dy = waypoint[0] - self.pose[0], waypoint[1] - self.pose[1]
            theta = -self.pose[2]
            return (dx * math.cos(theta) - dy * math.sin(theta), dx * math.sin(theta) + dy * math.cos(theta))

        elif self.mission_state == MissionState.EXPLORING:
            if self.behavior == Behavior.ENTER_PASSAGE:
                # Goal is a point in the middle of the detected passage, dynamically adjusted by depth
                # Use the range of the LiDAR point at the best_gap_mid_angle to determine depth
                # Find the LiDAR range closest to the best_gap_mid_angle
                lidar_ranges = np.array(self.lidar.getRangeImage())
                angles = self.perception.lidar_angles
                
                # Find the index of the angle closest to locked_goal_angle
                angle_diffs = np.abs(angles - self.locked_goal_angle)
                closest_angle_idx = np.argmin(angle_diffs)
                
                # Get the range at that angle, clamp it to a reasonable value
                passage_depth = lidar_ranges[closest_angle_idx]
                if not math.isfinite(passage_depth) or passage_depth < 0.5: # If invalid or too close, use a default
                    passage_depth = 2.0
                passage_depth = min(passage_depth, 4.0) # Cap the depth to avoid overly ambitious goals

                goal_dist = passage_depth * 0.8 # Set goal slightly before the end of the passage
                return (goal_dist * math.cos(self.locked_goal_angle), goal_dist * math.sin(self.locked_goal_angle))
            
            elif self.behavior == Behavior.FOLLOW_WALL:
                # --- ADVANCED WALL-FOLLOWING GOAL LOGIC ---
                # Find the closest wall segment on the right side
                right_walls = [w for w in self.world_model.wall_segments if w[0][1] < 0 or w[1][1] < 0]
                
                if not right_walls: 
                    # If no right walls, try to move forward or cross open space
                    self.behavior = Behavior.CROSS_OPEN_SPACE # Switch to cross open space if no wall to follow
                    self.behavior_timer = current_time + BEHAVIOR_COMMIT_TIME * 2 # Commit longer to cross open space
                    return (2.0, 0.0) # Simple forward goal if no wall to follow

                # Find the wall segment that is closest to the robot and is on its right side
                closest_wall = None
                min_dist = float('inf')
                for wall in right_walls:
                    p1, p2 = wall
                    # Calculate distance from robot origin (0,0) to the wall segment
                    dist = self.local_planner._point_to_line_segment_dist(0, 0, p1[0], p1[1], p2[0], p2[1])
                    if dist < min_dist:
                        min_dist = dist
                        closest_wall = wall
                
                if closest_wall is None: # Should not happen if right_walls is not empty
                    self.behavior = Behavior.CROSS_OPEN_SPACE
                    self.behavior_timer = current_time + BEHAVIOR_COMMIT_TIME * 2
                    return (2.0, 0.0)

                p1, p2 = closest_wall
                
                # Calculate the vector of the wall segment
                wall_vec_x, wall_vec_y = p2[0] - p1[0], p2[1] - p1[1]
                wall_len = math.hypot(wall_vec_x, wall_vec_y)
                if wall_len == 0: return (1.0, 0.0) # Avoid division by zero
                
                # Normalize wall vector
                wall_vec_x /= wall_len
                wall_vec_y /= wall_len

                # Calculate the normal vector pointing away from the wall (assuming right-hand wall following)
                # The robot is on the right of the wall. If wall_vec is (dx, dy), then normal is (dy, -dx)
                normal_x = wall_vec_y
                normal_y = -wall_vec_x

                # Calculate a point on the wall that is closest to the robot's current position (0,0)
                # This is a projection of (0,0) onto the line segment (p1, p2)
                dot_product = (0 - p1[0]) * wall_vec_x + (0 - p1[1]) * wall_vec_y
                t = max(0, min(1, dot_product / wall_len**2)) if wall_len > 0 else 0
                closest_point_on_wall_x = p1[0] + t * wall_vec_x
                closest_point_on_wall_y = p1[1] + t * wall_vec_y

                # Goal point is offset from the closest point on the wall by WALL_FOLLOW_DISTANCE
                goal_x = closest_point_on_wall_x + normal_x * WALL_FOLLOW_DISTANCE
                goal_y = closest_point_on_wall_y + normal_y * WALL_FOLLOW_DISTANCE
                
                # Add a component that encourages forward movement along the wall
                # This helps prevent the robot from getting stuck or oscillating
                forward_component_x = wall_vec_x * 1.5 # Increased forward component
                forward_component_y = wall_vec_y * 1.5
                
                goal_x += forward_component_x
                goal_y += forward_component_y
                
                return (goal_x, goal_y)
            
            else: # CROSS_OPEN_SPACE
                self.behavior = Behavior.CROSS_OPEN_SPACE # Explicitly set behavior
                self.behavior_timer = current_time + BEHAVIOR_COMMIT_TIME * 4 # Even longer commitment to cross open space
                return (4.0, 0.0) # Drive even further forward in open space
        return (0.0, 0.0)

    # --- Main Loop ---
    def run(self):
        print("Final Mission Controller Initializing...")
        while self.step(self.timestep) != -1:
            current_time = self.getTime()
            pos, orientation = self.robot_node.getPosition(), self.robot_node.getOrientation()
            # Corrected yaw calculation for XZ ground plane (Y is up)
            # Webots orientation matrix is column-major. Yaw is rotation around Y.
            # yaw = atan2(-R31, R11) = atan2(-orientation[2], orientation[0])
            self.pose = np.array([pos[0], pos[2], math.atan2(-orientation[2], orientation[0])])
            lidar_ranges = self.lidar.getRangeImage()
            if not lidar_ranges:
                print("DEBUG: Lidar ranges are empty. Skipping current step.")
                continue
            
            self.update_global_map(lidar_ranges)
            self.detect_pillars()
            self.world_model = self.perception.update(lidar_ranges)

            goal_vector, v, w = (0,0), 0, 0
            
            if self.mission_state == MissionState.EXPLORING:
                self.decide_exploration_behavior(current_time)
                goal_vector = self.get_dwa_goal(current_time)
                (v, w), _ = self.local_planner.find_best_trajectory(self.current_velocity, goal_vector, self.world_model, self.timestep / 1000.0)
                if 'blue' in self.grid.semantic_objects and 'yellow' in self.grid.semantic_objects:
                    print("Found both pillars! Switching to A* planning.")
                    self.mission_state, (v, w) = MissionState.PLANNING_PATH, (0, 0)
            
            elif self.mission_state == MissionState.PLANNING_PATH:
                start_pos, end_pos = self.grid.semantic_objects['blue'], self.grid.semantic_objects['yellow']
                print(f"Planning path from BLUE {start_pos} to YELLOW {end_pos}...")
                self.astar_path = self.astar_planner.plan_path(start_pos, end_pos)
                if self.astar_path:
                    self.astar_waypoint_idx = 0
                    self.mission_state = MissionState.NAVIGATING_PATH
                    print("Path found! Navigating...")
                else:
                    print("A* path planning failed! Mission stuck."); self.mission_state = MissionState.FINISHED
                v, w = 0, 0

            elif self.mission_state == MissionState.NAVIGATING_PATH:
                waypoint = self.astar_path[self.astar_waypoint_idx]
                if math.hypot(self.pose[0] - waypoint[0], self.pose[1] - waypoint[1]) < ASTAR_WAYPOINT_THRESHOLD:
                    self.astar_waypoint_idx += 1
                    if self.astar_waypoint_idx >= len(self.astar_path):
                        print("Final destination reached! Mission complete."); self.mission_state = MissionState.FINISHED; v, w = 0, 0
                if self.mission_state == MissionState.NAVIGATING_PATH:
                    goal_vector = self.get_dwa_goal(current_time)
                    (v, w), _ = self.local_planner.find_best_trajectory(self.current_velocity, goal_vector, self.world_model, self.timestep / 1000.0)

            self.current_velocity = (v, w)
            wheel_radius, track_width = 0.043, 0.22
            left_speed = (v - w * track_width / 2) / wheel_radius
            right_speed = (v + w * track_width / 2) / wheel_radius
            print(f"DEBUG: Calculated v={v:.2f}, w={w:.2f}, left_speed={left_speed:.2f}, right_speed={right_speed:.2f}")
            for i in [0, 2]: self.motors[i].setVelocity(left_speed)
            for i in [1, 3]: self.motors[i].setVelocity(right_speed)

if __name__ == "__main__":
    controller = FinalMissionController()
    controller.run()