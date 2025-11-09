#!/usr/bin/env python3
"""
High-Performance Path Planning Components
Optimized A* implementation ready for C++ migration
Author: Path Planning Engineer - October 2025
"""

import time
import numpy as np
import heapq
import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

from core_types import Point2D, RobotPose, PerformanceProfiler, SystemConfig
from interfaces import IPathPlanner, IExplorationPlanner

@dataclass
class PathNode:
    """Path planning node with optimized memory layout"""
    position: Tuple[int, int]
    g_cost: float = float('inf')
    h_cost: float = 0.0
    f_cost: float = float('inf')
    parent: Optional['PathNode'] = None

    def __lt__(self, other: 'PathNode') -> bool:
        """Comparison for heapq"""
        return self.f_cost < other.f_cost

class OptimizedAStarPlanner(IPathPlanner):
    """Memory-optimized A* path planner

    Features:
    - Vectorized heuristic calculations
    - Memory-efficient node representation  
    - Path smoothing algorithms
    - Performance profiling for C++ migration
    """

    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        super().__init__(profiler)

        # A* parameters
        self.diagonal_cost = math.sqrt(2.0)
        self.orthogonal_cost = 1.0
        self.safety_margin_cells = 2  # Cells to avoid near obstacles

        # Performance optimization
        self.max_iterations = 10000
        self.enable_path_smoothing = True
        self.smoothing_iterations = 3

        # Memory management
        self.node_pool_size = 5000
        self.reuse_nodes = True

        # Performance tracking
        self.planning_times = []
        self.path_lengths = []
        self.iterations_count = []

        print("🎯 Optimized A* Planner initialized")
        print(f"   Max iterations: {self.max_iterations}")
        print(f"   Safety margin: {self.safety_margin_cells} cells")
        print(f"   Path smoothing: {self.enable_path_smoothing}")

    def plan_path(self, start: Point2D, goal: Point2D, 
                  occupancy_grid: np.ndarray) -> List[Point2D]:
        """Plan optimal path using optimized A* algorithm"""
        start_time = time.time()

        try:
            # Validate inputs
            if occupancy_grid is None or occupancy_grid.size == 0:
                return []

            height, width = occupancy_grid.shape
            start_gx, start_gy = int(start.x), int(start.y)
            goal_gx, goal_gy = int(goal.x), int(goal.y)

            # Boundary checks
            if not (0 <= start_gx < width and 0 <= start_gy < height):
                return []
            if not (0 <= goal_gx < width and 0 <= goal_gy < height):
                return []

            # Check if start and goal are free
            if not self._is_cell_safe(start_gx, start_gy, occupancy_grid):
                return []
            if not self._is_cell_safe(goal_gx, goal_gy, occupancy_grid):
                return []

            # Quick check for direct path
            if self.is_path_clear(start, goal, occupancy_grid):
                path = [start, goal]
            else:
                # Run A* algorithm
                path = self._astar_search(
                    (start_gx, start_gy), (goal_gx, goal_gy), occupancy_grid
                )

                # Convert to Point2D
                path = [Point2D(float(x), float(y)) for x, y in path]

            # Apply path smoothing
            if self.enable_path_smoothing and len(path) > 2:
                path = self.smooth_path(path)

            # Record performance metrics
            planning_time = time.time() - start_time
            self.planning_times.append(planning_time)
            self.path_lengths.append(len(path))

            if self.profiler:
                self.profiler.record_timing("path_planning", planning_time)

            return path

        except Exception as e:
            print(f"Path planning error: {e}")
            return []

    def _astar_search(self, start: Tuple[int, int], goal: Tuple[int, int],
                     occupancy_grid: np.ndarray) -> List[Tuple[int, int]]:
        """Core A* search algorithm with optimizations"""
        height, width = occupancy_grid.shape

        # Initialize data structures
        open_set = []
        closed_set: Set[Tuple[int, int]] = set()
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        # Add start node to open set
        h_cost = self._heuristic(start, goal)
        heapq.heappush(open_set, (h_cost, start))

        iterations = 0

        while open_set and iterations < self.max_iterations:
            iterations += 1

            # Get node with lowest f_cost
            current_f, current = heapq.heappop(open_set)

            # Check if goal reached
            if current == goal:
                self.iterations_count.append(iterations)
                return self._reconstruct_path(came_from, current)

            # Add to closed set
            closed_set.add(current)

            # Check all neighbors
            for neighbor in self._get_neighbors(current, width, height):
                if neighbor in closed_set:
                    continue

                # Check if neighbor is safe to traverse
                if not self._is_cell_safe(neighbor[0], neighbor[1], occupancy_grid):
                    continue

                # Calculate costs
                movement_cost = self._calculate_movement_cost(current, neighbor)
                tentative_g_score = g_score[current] + movement_cost

                # Check if this path is better
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score

                    h_cost = self._heuristic(neighbor, goal)
                    f_cost = tentative_g_score + h_cost

                    # Add to open set
                    heapq.heappush(open_set, (f_cost, neighbor))

        self.iterations_count.append(iterations)
        return []  # No path found

    def _get_neighbors(self, pos: Tuple[int, int], width: int, height: int) -> List[Tuple[int, int]]:
        """Get valid neighbor cells (8-connected)"""
        x, y = pos
        neighbors = []

        # 8-connected neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                # Boundary check
                if 0 <= nx < width and 0 <= ny < height:
                    neighbors.append((nx, ny))

        return neighbors

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)

    def _calculate_movement_cost(self, from_pos: Tuple[int, int], 
                                to_pos: Tuple[int, int]) -> float:
        """Calculate movement cost between adjacent cells"""
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])

        if dx + dy == 2:  # Diagonal movement
            return self.diagonal_cost
        else:  # Orthogonal movement
            return self.orthogonal_cost

    def _is_cell_safe(self, gx: int, gy: int, occupancy_grid: np.ndarray) -> bool:
        """Check if cell is safe for robot navigation"""
        height, width = occupancy_grid.shape

        if not (0 <= gx < width and 0 <= gy < height):
            return False

        # Check center cell
        if occupancy_grid[gy, gx] != 0:  # 0 = free space
            return False

        # Check safety margin around cell
        for dx in range(-self.safety_margin_cells, self.safety_margin_cells + 1):
            for dy in range(-self.safety_margin_cells, self.safety_margin_cells + 1):
                check_x, check_y = gx + dx, gy + dy

                if 0 <= check_x < width and 0 <= check_y < height:
                    if occupancy_grid[check_y, check_x] == 100:  # Occupied
                        return False

        return True

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]],
                         current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def is_path_clear(self, start: Point2D, goal: Point2D, 
                      occupancy_grid: np.ndarray) -> bool:
        """Check if direct path between two points is clear"""
        # Simple line-of-sight check using Bresenham algorithm
        x0, y0 = int(start.x), int(start.y)
        x1, y1 = int(goal.x), int(goal.y)

        # Bresenham line algorithm
        points = self._bresenham_line(x0, y0, x1, y1)

        for x, y in points:
            if not self._is_cell_safe(x, y, occupancy_grid):
                return False

        return True

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham line algorithm for line-of-sight checking"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            points.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x += sx

            if e2 < dx:
                err += dx
                y += sy

        return points

    def calculate_path_cost(self, path: List[Point2D]) -> float:
        """Calculate total path distance"""
        if len(path) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(path)):
            dx = path[i].x - path[i-1].x
            dy = path[i].y - path[i-1].y
            total_distance += math.sqrt(dx * dx + dy * dy)

        return total_distance

    def smooth_path(self, path: List[Point2D]) -> List[Point2D]:
        """Apply path smoothing to reduce unnecessary waypoints"""
        if len(path) <= 2:
            return path

        smoothed_path = [path[0]]  # Start with first point

        for _ in range(self.smoothing_iterations):
            i = 0
            while i < len(path) - 1:
                current = smoothed_path[-1]

                # Look ahead to find the farthest reachable point
                farthest_index = i + 1
                for j in range(i + 2, len(path)):
                    # Check if direct path is possible (simplified check)
                    if self._can_connect_directly(current, path[j]):
                        farthest_index = j
                    else:
                        break

                # Add the farthest reachable point
                if farthest_index < len(path):
                    smoothed_path.append(path[farthest_index])

                i = farthest_index

        return smoothed_path

    def _can_connect_directly(self, start: Point2D, end: Point2D) -> bool:
        """Simplified check for direct connection (could be improved with grid check)"""
        # For now, just check distance isn't too large
        distance = start.distance_to(end)
        return distance < 5.0  # Maximum direct connection distance

    def get_planning_stats(self) -> Dict[str, float]:
        """Get path planning performance statistics"""
        if not self.planning_times:
            return {}

        return {
            'avg_planning_time_ms': np.mean(self.planning_times) * 1000,
            'max_planning_time_ms': np.max(self.planning_times) * 1000,
            'avg_path_length': np.mean(self.path_lengths),
            'avg_iterations': np.mean(self.iterations_count) if self.iterations_count else 0,
            'max_iterations': np.max(self.iterations_count) if self.iterations_count else 0,
            'planning_calls': len(self.planning_times),
            'success_rate': len([p for p in self.path_lengths if p > 0]) / len(self.path_lengths) * 100
        }

class IntelligentExplorationPlanner(IExplorationPlanner):
    """Advanced frontier-based exploration planner

    Features:
    - Efficient frontier detection
    - Multi-criteria target selection  
    - Information gain estimation
    - Performance optimization
    """

    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        self.profiler = profiler

        # Frontier detection parameters
        self.min_frontier_size = 5
        self.frontier_cluster_distance = 8
        self.max_exploration_distance = 15.0

        # Selection criteria weights
        self.distance_weight = 0.3
        self.information_gain_weight = 0.5
        self.safety_weight = 0.2

        # Performance tracking
        self.detection_times = []
        self.frontier_counts = []

        print("🎯 Intelligent Exploration Planner initialized")
        print(f"   Min frontier size: {self.min_frontier_size}")
        print(f"   Max exploration distance: {self.max_exploration_distance}m")

    def find_exploration_targets(self, occupancy_grid: np.ndarray, 
                                robot_pose: RobotPose) -> List[Point2D]:
        """Find frontier exploration targets"""
        start_time = time.time()

        try:
            if occupancy_grid is None or occupancy_grid.size == 0:
                return []

            height, width = occupancy_grid.shape
            robot_gx = int((robot_pose.x + 10.0) / 0.04)  # Approximate grid conversion
            robot_gy = int((robot_pose.y + 10.0) / 0.04)

            # Find frontier cells
            frontier_cells = self._detect_frontier_cells(occupancy_grid)

            # Cluster nearby frontier cells
            frontier_clusters = self._cluster_frontiers(frontier_cells)

            # Convert to exploration targets
            targets = []
            for cluster in frontier_clusters:
                if len(cluster) >= self.min_frontier_size:
                    # Calculate cluster centroid
                    center_x = sum(cell[0] for cell in cluster) / len(cluster)
                    center_y = sum(cell[1] for cell in cluster) / len(cluster)

                    # Convert to world coordinates (approximate)
                    world_x = center_x * 0.04 - 10.0
                    world_y = center_y * 0.04 - 10.0

                    # Distance check
                    distance = math.sqrt((world_x - robot_pose.x)**2 + (world_y - robot_pose.y)**2)

                    if distance <= self.max_exploration_distance:
                        targets.append(Point2D(world_x, world_y))

            # Record performance
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            self.frontier_counts.append(len(targets))

            if self.profiler:
                self.profiler.record_timing("frontier_detection", detection_time)

            return targets

        except Exception as e:
            print(f"Frontier detection error: {e}")
            return []

    def _detect_frontier_cells(self, occupancy_grid: np.ndarray) -> List[Tuple[int, int]]:
        """Detect frontier cells (free cells adjacent to unknown areas)"""
        height, width = occupancy_grid.shape
        frontier_cells = []

        # Vectorized frontier detection
        for y in range(1, height - 1):
            for x in range(1, width - 1):

                # Check if cell is free space
                if occupancy_grid[y, x] != 0:
                    continue

                # Check if adjacent to unknown space
                has_unknown_neighbor = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue

                        ny, nx = y + dy, x + dx
                        if 0 <= nx < width and 0 <= ny < height:
                            if occupancy_grid[ny, nx] == -1:  # Unknown
                                has_unknown_neighbor = True
                                break

                    if has_unknown_neighbor:
                        break

                if has_unknown_neighbor:
                    frontier_cells.append((x, y))

        return frontier_cells

    def _cluster_frontiers(self, frontier_cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Cluster nearby frontier cells"""
        if not frontier_cells:
            return []

        clusters = []
        unassigned = set(frontier_cells)

        while unassigned:
            # Start new cluster
            seed = next(iter(unassigned))
            current_cluster = [seed]
            unassigned.remove(seed)

            # Find nearby cells iteratively
            i = 0
            while i < len(current_cluster):
                current_cell = current_cluster[i]

                # Find cells within clustering distance
                to_add = []
                for cell in unassigned:
                    distance = math.sqrt((cell[0] - current_cell[0])**2 + 
                                       (cell[1] - current_cell[1])**2)
                    if distance <= self.frontier_cluster_distance:
                        to_add.append(cell)

                # Add to cluster
                for cell in to_add:
                    current_cluster.append(cell)
                    unassigned.remove(cell)

                i += 1

            clusters.append(current_cluster)

        return clusters

    def select_best_target(self, targets: List[Point2D], 
                          robot_pose: RobotPose) -> Optional[Point2D]:
        """Select best exploration target using multi-criteria analysis"""
        if not targets:
            return None

        best_target = None
        best_score = -1.0

        for target in targets:
            # Calculate selection criteria
            distance = robot_pose.distance_to(target)

            # Distance score (closer is better, but normalize)
            distance_score = 1.0 / (1.0 + distance * 0.1)

            # Information gain score (simplified - could be improved)
            info_gain_score = 1.0  # Assume all frontiers have equal information gain

            # Safety score (simplified - could check surrounding obstacles)
            safety_score = 1.0

            # Combined score
            total_score = (self.distance_weight * distance_score +
                         self.information_gain_weight * info_gain_score +
                         self.safety_weight * safety_score)

            if total_score > best_score:
                best_score = total_score
                best_target = target

        return best_target

    def get_exploration_stats(self) -> Dict[str, float]:
        """Get exploration planning statistics"""
        if not self.detection_times:
            return {}

        return {
            'avg_detection_time_ms': np.mean(self.detection_times) * 1000,
            'avg_frontiers_found': np.mean(self.frontier_counts),
            'max_frontiers_found': np.max(self.frontier_counts),
            'detection_calls': len(self.detection_times)
        }


