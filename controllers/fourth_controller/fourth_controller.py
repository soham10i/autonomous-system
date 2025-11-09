#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
full_slam_controller_v2.py (Complete Mission with EKF SLAM - Corrected)

This controller implements the full architecture specified in the engineering
prompt plan. It performs autonomous exploration, mapping, and navigation
without relying on the Supervisor for localization. This version includes
corrections for Webots device names.

Architecture:
1.  EKF_SLAM: Estimates robot pose and landmark positions.
2.  LandmarkExtractor: Identifies corners from Lidar scans for the SLAM update.
3.  GlobalPlanner: Finds exploration frontiers aand plans A* paths on the map.
4.  LocalPlannerDWA: Generates smooth, collision-free local trajectories.
5.  FinalMissionController: Integrates all modules and manages the mission state.
"""
# -----------------------------------------------------------------------------
# --- IMPORTS -----------------------------------------------------------------
# -----------------------------------------------------------------------------
import math
import numpy as np
from controller import Supervisor, Robot
import cv2
from enum import Enum
import heapq
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

# -----------------------------------------------------------------------------
# --- CONSTANTS ---------------------------------------------------------------
# -----------------------------------------------------------------------------
TIME_STEP = 32
WHEEL_RADIUS = 0.043
TRACK_WIDTH = 0.22

# EKF Noise Parameters (Crucial for tuning)
MOTION_NOISE = np.diag([0.1, 0.1, np.deg2rad(1.0)])**2 # Odometry noise
MEASUREMENT_NOISE = np.diag([0.1, np.deg2rad(5.0)])**2 # Landmark measurement noise

# Robot velocity limits (tune as needed)
MAX_LINEAR_VELOCITY = 0.6  # meters per second
MAX_ANGULAR_VELOCITY = 2.0 # radians per second

# DWA Planner Config
DWA_TIME_HORIZON = 1.5
DWA_V_SAMPLES = 5
DWA_W_SAMPLES = 11
DWA_COST_GOAL = 1.0
DWA_COST_VELOCITY = 0.2
DWA_COST_OBSTACLE = 2.0
DWA_ROBOT_RADIUS = 0.25

# Map and A* Config
MAP_SIZE_PIXELS = 600
GRID_BOUNDS_M = 10.0
GRID_RESOLUTION = GRID_BOUNDS_M / MAP_SIZE_PIXELS
ASTAR_WAYPOINT_THRESHOLD = 0.3
PILLAR_DISTANCE_THRESHOLD = 0.4

# Color Detection
HSV_RANGES = {
    'blue': [(100, 150, 50), (140, 255, 255)],
    'yellow': [(20, 100, 100), (30, 255, 255)]
}
MIN_CONTOUR_AREA = 100

# -----------------------------------------------------------------------------
# --- STATE MACHINES & DATA STRUCTURES ----------------------------------------
# -----------------------------------------------------------------------------
class MissionState(Enum):
    EXPLORING = "Exploring (Frontier)"
    PLANNING_PATH = "Planning A* Path"
    NAVIGATING_PATH = "Navigating A* Path"
    FINISHED = "Mission Finished"

class WorldModel:
    def __init__(self):
        self.landmarks = [] # List of (range, bearing, signature)
        self.lidar_points = np.array([])
        self.wall_segments = []

# -----------------------------------------------------------------------------
# --- HELPER FUNCTIONS --------------------------------------------------------
# -----------------------------------------------------------------------------
def normalize_angle(angle):
    while angle > math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

# -----------------------------------------------------------------------------
# --- CORE MODULES ------------------------------------------------------------
# -----------------------------------------------------------------------------
class EKF_SLAM:
    """Implements the Extended Kalman Filter for SLAM."""
    def __init__(self, initial_pose):
        self.state = np.array(initial_pose).reshape(-1, 1)
        self.covariance = np.diag([0.01, 0.01, 0.01])
        self.landmark_map = {}
        self.next_landmark_id = 0

    def predict(self, odometry, dt):
        v, w = odometry
        theta = self.state[2, 0]
        G = np.eye(len(self.state))
        G[0, 2] = -v * dt * math.sin(theta)
        G[1, 2] = v * dt * math.cos(theta)
        self.state[0, 0] += v * dt * math.cos(theta)
        self.state[1, 0] += v * dt * math.sin(theta)
        self.state[2, 0] = normalize_angle(self.state[2, 0] + w * dt)
        F = np.zeros((3, len(self.state)))
        F[:3, :3] = np.eye(3)
        self.covariance = G @ self.covariance @ G.T + F.T @ MOTION_NOISE @ F

    def update(self, landmark_observations):
        for r, b, signature in landmark_observations:
            if signature not in self.landmark_map:
                self._add_new_landmark(r, b, signature)
            else:
                self._update_existing_landmark(r, b, signature)

    def _add_new_landmark(self, r, b, signature):
        idx = len(self.state)
        self.landmark_map[signature] = idx
        rx, ry, r_theta = self.state[:3, 0]
        lx = rx + r * math.cos(b + r_theta)
        ly = ry + r * math.sin(b + r_theta)
        new_state = np.vstack([self.state, [[lx], [ly]]])
        old_size = len(self.state)
        new_cov = np.eye(old_size + 2) * 0.01
        new_cov[:old_size, :old_size] = self.covariance
        self.covariance = new_cov
        self.state = new_state

    def _update_existing_landmark(self, r, b, signature):
        idx = self.landmark_map[signature]
        rx, ry, r_theta = self.state[:3, 0]
        lx, ly = self.state[idx:idx+2, 0]
        dx, dy = lx - rx, ly - ry
        q = dx**2 + dy**2
        if q < 0.0001: return
        expected_r = math.sqrt(q)
        expected_b = normalize_angle(math.atan2(dy, dx) - r_theta)
        innovation = np.array([r - expected_r, normalize_angle(b - expected_b)]).reshape(2, 1)
        H_small = np.array([[-dx/expected_r, -dy/expected_r, 0, dx/expected_r, dy/expected_r],
                            [dy/q, -dx/q, -1, -dy/q, dx/q]])
        F = np.zeros((5, len(self.state)))
        F[:3, :3] = np.eye(3)
        F[3:, idx:idx+2] = np.eye(2)
        H = H_small @ F
        S = H @ self.covariance @ H.T + MEASUREMENT_NOISE
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state += K @ innovation
        self.state[2, 0] = normalize_angle(self.state[2, 0])
        self.covariance = (np.eye(len(self.state)) - K @ H) @ self.covariance

class LandmarkExtractor:
    def __init__(self):
        self.lidar_angles = None
        self.next_signature = 0
        self.known_landmarks = {}

    def _get_line_intersection(self, line1, line2):
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6: return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        if 0 < t < 1 and 0 < u < 1:
            return x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        return None

    def update(self, lidar_ranges, robot_pose):
        if self.lidar_angles is None:
            self.lidar_angles = np.linspace(-math.pi, math.pi, len(lidar_ranges), endpoint=False)
        
        ranges = np.array(lidar_ranges)
        valid_indices = np.isfinite(ranges) & (ranges > 0.01)
        angles, valid_ranges = self.lidar_angles[valid_indices], ranges[valid_indices]
        px, py = valid_ranges * np.cos(angles), valid_ranges * np.sin(angles)
        points = np.vstack((px, py)).T
        if len(points) < 10: return [], points, []
        
        db = DBSCAN(eps=0.2, min_samples=5).fit(points)
        labels, wall_segments = db.labels_, []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1: continue
            cluster_points = points[labels == label]
            if len(cluster_points) < 5: continue
            if np.var(cluster_points[:, 0]) > np.var(cluster_points[:, 1]):
                X, y = cluster_points[:, 0].reshape(-1, 1), cluster_points[:, 1]
                ransac = RANSACRegressor(min_samples=5).fit(X, y)
                x_min, x_max = X.min(), X.max()
                y_min_pred, y_max_pred = ransac.predict([[x_min]]), ransac.predict([[x_max]])
                wall_segments.append(((x_min, y_min_pred[0]), (x_max, y_max_pred[0])))
            else:
                X, y = cluster_points[:, 1].reshape(-1, 1), cluster_points[:, 0]
                ransac = RANSACRegressor(min_samples=5).fit(X, y)
                x_min, x_max = X.min(), X.max()
                y_min_pred, y_max_pred = ransac.predict([[x_min]]), ransac.predict([[x_max]])
                wall_segments.append(((y_min_pred[0], x_min), (y_max_pred[0], x_max)))

        landmarks_in_robot_frame = []
        for i in range(len(wall_segments)):
            for j in range(i + 1, len(wall_segments)):
                intersection = self._get_line_intersection(wall_segments[i], wall_segments[j])
                if intersection:
                    landmarks_in_robot_frame.append(intersection)

        landmark_observations = []
        rx, ry, r_theta = robot_pose
        for lx_robot, ly_robot in landmarks_in_robot_frame:
            lx_world = rx + lx_robot * math.cos(r_theta) - ly_robot * math.sin(r_theta)
            ly_world = ry + lx_robot * math.sin(r_theta) + ly_robot * math.cos(r_theta)
            
            min_dist, assoc_sig = 0.5, -1
            for sig, pos in self.known_landmarks.items():
                dist = math.hypot(lx_world - pos[0], ly_world - pos[1])
                if dist < min_dist:
                    min_dist, assoc_sig = dist, sig
            
            if assoc_sig == -1:
                assoc_sig = self.next_signature
                self.next_signature += 1
            
            self.known_landmarks[assoc_sig] = (lx_world, ly_world)
            
            r, b = math.hypot(lx_robot, ly_robot), math.atan2(ly_robot, lx_robot)
            landmark_observations.append((r, b, assoc_sig))

        return landmark_observations, points, wall_segments

# --- FIX: Add the missing OccupancyGrid and AStarPlanner classes ---
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
            
    def update_global_map(self, lidar_ranges, robot_pose):
        robot_map_x, robot_map_y = self.world_to_map(*robot_pose[:2])
        # This is a simplified map update. A proper implementation would get FOV from the lidar.
        fov = 2 * math.pi 
        num_rays = len(lidar_ranges)
        for i, dist in enumerate(lidar_ranges):
            dist = dist if math.isfinite(dist) and dist < 5.0 else 5.0
            angle = robot_pose[2] + (i / num_rays - 0.5) * fov
            end_x, end_y = robot_pose[0] + dist * math.cos(angle), robot_pose[1] + dist * math.sin(angle)
            end_map_x, end_map_y = self.world_to_map(end_x, end_y)
            self.raytrace(robot_map_x, robot_map_y, end_map_x, end_map_y)
            if dist < 5.0: self.update_cell(end_map_x, end_map_y, is_occupied=True)

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
            if current == goal_grid: return self._reconstruct_path(came_from, current)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0: continue
                    neighbor = (current[0] + dx, current[1] + dy)
                    if not (0 <= neighbor[0] < self.grid.map_size and 0 <= neighbor[1] < self.grid.map_size) or \
                       self.grid.log_odds_grid[neighbor[1], neighbor[0]] > 0.1: continue
                    tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor], g_score[neighbor] = current, tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None
    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from: path.append(came_from[current])
        path.reverse()
        return [(self.grid.resolution * (pt[0] - self.grid.map_size / 2), self.grid.resolution * -(pt[1] - self.grid.map_size / 2)) for pt in path]

class GlobalPlanner:
    def __init__(self):
        self.grid = OccupancyGrid(MAP_SIZE_PIXELS, GRID_RESOLUTION)
        self.astar = AStarPlanner(self.grid)

    def find_exploration_goal(self, robot_pose):
        robot_map_x, robot_map_y = self.grid.world_to_map(*robot_pose[:2])
        for r in range(1, int(MAP_SIZE_PIXELS / 4)):
            for i in range(-r, r + 1):
                for j in range(-r, r + 1):
                    if abs(i) != r and abs(j) != r: continue
                    px, py = robot_map_x + i, robot_map_y + j
                    if 0 <= px < MAP_SIZE_PIXELS and 0 <= py < MAP_SIZE_PIXELS and self.grid.log_odds_grid[py, px] == 0:
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                if 0 <= px+dx < MAP_SIZE_PIXELS and 0 <= py+dy < MAP_SIZE_PIXELS and self.grid.log_odds_grid[py+dy, px+dx] < -0.1:
                                    return (GRID_RESOLUTION * (px - MAP_SIZE_PIXELS / 2),
                                            GRID_RESOLUTION * -(py - MAP_SIZE_PIXELS / 2))
        return None

class PillarDetector:
    def __init__(self, camera, depth_camera):
        self.camera = camera
        self.depth_camera = depth_camera
        
    def detect(self, robot_pose):
        bgr_image = self.camera.getImage()
        if not bgr_image: return {}
        
        detected_pillars = {}
        cam_width, cam_height, cam_fov = self.camera.getWidth(), self.camera.getHeight(), self.camera.getFov()
        frame = np.frombuffer(bgr_image, np.uint8).reshape((cam_height, cam_width, 4))[:,:,:3]
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for color_name, (lower, upper) in HSV_RANGES.items():
            mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < MIN_CONTOUR_AREA: continue
            
            M = cv2.moments(max(contours, key=cv2.contourArea))
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            distance = self.depth_camera.getRange(cx, cy)
            if math.isinf(distance) or math.isnan(distance) or distance > 5.0: continue
            
            x_coord = (cx - cam_width/2)/(cam_width/2)
            x_cam = -x_coord * distance * math.tan(cam_fov/2)*(cam_width/cam_height)
            z_cam = distance
            x_robot, y_robot = z_cam, -x_cam
            
            rx, ry, r_theta = robot_pose
            world_x = rx + x_robot * math.cos(r_theta) - y_robot * math.sin(r_theta)
            world_y = ry + x_robot * math.sin(r_theta) + y_robot * math.cos(r_theta)
            detected_pillars[color_name] = (world_x, world_y)
            
        return detected_pillars

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
        if not wall_segments: cost_obstacle = 0.0
        else:
            for pos in path:
                for p1, p2 in wall_segments:
                    dist = self._point_to_line_segment_dist(pos[0], pos[1], p1[0], p1[1], p2[0], p2[1])
                    if dist < min_obs_dist: min_obs_dist = dist
            if min_obs_dist <= self.config['robot_radius']: return float('inf')
            cost_obstacle = self.config['cost_obstacle'] * (1.0 / min_obs_dist)
        return cost_goal + cost_velocity + cost_obstacle
    def find_best_trajectory(self, current_velocity, goal, world_model, dt):
        current_v, current_w = current_velocity
        v_min, v_max, w_min, w_max = self._calculate_dynamic_window(current_v, current_w, dt)
        best_cost, best_v_w = float('inf'), (0.0, 0.0)
        for v in np.linspace(v_min, v_max, DWA_V_SAMPLES):
            for w in np.linspace(w_min, w_max, DWA_W_SAMPLES):
                path = self._simulate_trajectory(v, w, dt)
                cost = self._score_trajectory(path, goal, v, world_model.wall_segments)
                if cost < best_cost: best_cost, best_v_w = cost, (v, w)
        return best_v_w

# -----------------------------------------------------------------------------
# --- MAIN CONTROLLER CLASS ---------------------------------------------------
# -----------------------------------------------------------------------------
class FinalMissionController(Robot):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        
        self.mission_state = MissionState.EXPLORING
        
        motor_names = ["front left wheel motor", "front right wheel motor", "rear left wheel motor", "rear right wheel motor"]
        self.motors = [self.getDevice(name) for name in motor_names]
        for m in self.motors:
            if m is None: print(f"Error: Motor device not found."); return
            m.setPosition(float('inf')); m.setVelocity(0.0)
            
        self.lidar = self.getDevice("lidar"); self.lidar.enable(self.timestep)
        self.camera = self.getDevice("camera rgb"); self.camera.enable(self.timestep)
        self.depth_camera = self.getDevice("camera depth"); self.depth_camera.enable(self.timestep)

        ps_names = ["front left wheel motor sensor", "front right wheel motor sensor", 
                    "rear left wheel motor sensor", "rear right wheel motor sensor"]
        self.position_sensors = [self.getDevice(name) for name in ps_names]
        for ps in self.position_sensors:
            if ps is None: print(f"Error: Position sensor not found."); return
            ps.enable(self.timestep)

        self.ekf = EKF_SLAM([0.0, 0.0, 0.0])
        self.landmark_extractor = LandmarkExtractor()
        self.global_planner = GlobalPlanner()
        self.pillar_detector = PillarDetector(self.camera, self.depth_camera)
        self.local_planner = LocalPlannerDWA(
            {'max_linear_vel': MAX_LINEAR_VELOCITY, 'max_angular_vel': MAX_ANGULAR_VELOCITY, 'max_linear_accel': 0.5, 'max_angular_accel': 2.0},
            {'time_horizon': DWA_TIME_HORIZON, 'cost_goal': DWA_COST_GOAL, 'cost_velocity': DWA_COST_VELOCITY, 'cost_obstacle': DWA_COST_OBSTACLE, 'robot_radius': DWA_ROBOT_RADIUS}
        )
        
        self.pose = np.array([0.0, 0.0, 0.0])
        self.world_model = WorldModel()
        self.current_velocity = (0.0, 0.0)
        self.astar_path = []
        self.astar_waypoint_idx = 0
        self.last_positions = [0.0, 0.0, 0.0, 0.0]

    def get_odometry(self, dt):
        current_positions = [ps.getValue() for ps in self.position_sensors]
        
        if not all(math.isfinite(p) for p in self.last_positions):
             self.last_positions = current_positions
             return 0.0, 0.0

        deltas = [(current - last) for current, last in zip(current_positions, self.last_positions)]
        self.last_positions = current_positions
        
        left_delta = (deltas[0] + deltas[2]) / 2.0
        right_delta = (deltas[1] + deltas[3]) / 2.0
        
        dl = left_delta * WHEEL_RADIUS
        dr = right_delta * WHEEL_RADIUS
        
        v = (dl + dr) / (2 * dt)
        w = (dr - dl) / (TRACK_WIDTH * dt)
        return v, w

    def get_dwa_goal(self):
        if self.mission_state == MissionState.EXPLORING:
            goal_world = self.global_planner.find_exploration_goal(self.pose)
            if goal_world is None: return (0,0)
        elif self.mission_state == MissionState.NAVIGATING_PATH:
            if not self.astar_path or self.astar_waypoint_idx >= len(self.astar_path): return (0,0)
            goal_world = self.astar_path[self.astar_waypoint_idx]
        else:
            return (0,0)

        dx, dy = goal_world[0] - self.pose[0], goal_world[1] - self.pose[1]
        theta = -self.pose[2]
        return (dx * math.cos(theta) - dy * math.sin(theta), dx * math.sin(theta) + dy * math.cos(theta))

    def run(self):
        print("Final SLAM Mission Controller Initializing...")
        dt = self.timestep / 1000.0
        
        self.step(self.timestep)
        self.last_positions = [ps.getValue() for ps in self.position_sensors]
        
        while self.step(self.timestep) != -1:
            odometry = self.get_odometry(dt)
            self.ekf.predict(odometry, dt)
            self.pose = self.ekf.state[:3, 0]
            
            lidar_ranges = self.lidar.getRangeImage()
            if not lidar_ranges: continue
            
            landmarks, points, walls = self.landmark_extractor.update(lidar_ranges, self.pose)
            self.ekf.update(landmarks)
            self.pose = self.ekf.state[:3, 0]
            
            self.world_model.lidar_points = points
            self.world_model.wall_segments = walls

            self.global_planner.grid.update_global_map(lidar_ranges, self.pose)
            detected_pillars = self.pillar_detector.detect(self.pose)
            for name, pos in detected_pillars.items():
                self.global_planner.grid.add_semantic_object(pos[0], pos[1], name)

            v, w = 0, 0
            
            if self.mission_state == MissionState.EXPLORING:
                goal_vector = self.get_dwa_goal()
                v, w = self.local_planner.find_best_trajectory(self.current_velocity, goal_vector, self.world_model, dt)
                if 'blue' in self.global_planner.grid.semantic_objects and 'yellow' in self.global_planner.grid.semantic_objects:
                    self.mission_state, (v, w) = MissionState.PLANNING_PATH, (0, 0)
            
            elif self.mission_state == MissionState.PLANNING_PATH:
                start_pos = self.global_planner.grid.semantic_objects['blue']
                end_pos = self.global_planner.grid.semantic_objects['yellow']
                self.astar_path = self.global_planner.astar.plan_path(start_pos, end_pos)
                if self.astar_path: self.mission_state, self.astar_waypoint_idx = MissionState.NAVIGATING_PATH, 0
                else: self.mission_state = MissionState.FINISHED
                v, w = 0, 0

            elif self.mission_state == MissionState.NAVIGATING_PATH:
                waypoint = self.astar_path[self.astar_waypoint_idx]
                if math.hypot(self.pose[0] - waypoint[0], self.pose[1] - waypoint[1]) < ASTAR_WAYPOINT_THRESHOLD:
                    self.astar_waypoint_idx += 1
                    if self.astar_waypoint_idx >= len(self.astar_path): self.mission_state, (v, w) = MissionState.FINISHED, (0,0)
                if self.mission_state == MissionState.NAVIGATING_PATH:
                    goal_vector = self.get_dwa_goal()
                    v, w = self.local_planner.find_best_trajectory(self.current_velocity, goal_vector, self.world_model, dt)

            self.current_velocity = (v, w)
            left_speed = (v - w * TRACK_WIDTH / 2) / WHEEL_RADIUS
            right_speed = (v + w * TRACK_WIDTH / 2) / WHEEL_RADIUS
            for i in [0, 2]: self.motors[i].setVelocity(left_speed)
            for i in [1, 3]: self.motors[i].setVelocity(right_speed)

if __name__ == "__main__":
    controller = FinalMissionController()
    controller.run()
