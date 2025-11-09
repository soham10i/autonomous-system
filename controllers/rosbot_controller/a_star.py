import heapq
import numpy as np
import logging
import time
from slam_logging_config import get_slam_logger, performance_monitor, SLAMLogger


class AStar:
    def __init__(self, occupancy_grid):
        self.grid = occupancy_grid

    def heuristic(self, a, b):
        # Euclidean distance as heuristic
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    @performance_monitor
    def plan_path(self, start_grid, goal_grid, robot_radius_grid=0):
        if not self.grid.is_valid_grid_coord(start_grid[0], start_grid[1]):
            print(f"Start point {start_grid} is outside the grid.")
            return None
        if not self.grid.is_valid_grid_coord(goal_grid[0], goal_grid[1]):
            print(f"Goal point {goal_grid} is outside the grid.")
            return None

        # Check if start or goal is occupied, using grid coordinates directly
        if self.grid.is_occupied_grid(start_grid[0], start_grid[1]):
            print(f"Start point {start_grid} is occupied.")
            return None
        if self.grid.is_occupied_grid(goal_grid[0], goal_grid[1]):
            print(f"Goal point {goal_grid} is occupied.")
            return None

        open_set = []
        heapq.heappush(open_set, (0, start_grid))  # (f_score, node)

        came_from = {}

        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current_f_score, current_node = heapq.heappop(open_set)

            if current_node == goal_grid:
                return self._reconstruct_path(came_from, current_node)

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    neighbor = (current_node[0] + dx, current_node[1] + dy)

                    if not self.grid.is_valid_grid_coord(neighbor[0], neighbor[1]):
                        continue

                    # Check for obstacles, considering robot radius
                    is_obstacle = False
                    for r_dx in range(-robot_radius_grid, robot_radius_grid + 1):
                        for r_dy in range(-robot_radius_grid, robot_radius_grid + 1):
                            check_x = neighbor[0] + r_dx
                            check_y = neighbor[1] + r_dy
                            if not self.grid.is_valid_grid_coord(check_x, check_y) or \
                               self.grid.is_occupied_grid(check_x, check_y):
                                is_obstacle = True
                                break
                        if is_obstacle:
                            break
                    if is_obstacle:
                        continue

                    tentative_g_score = g_score[current_node] + self.heuristic(current_node, neighbor)

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current_node
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def _reconstruct_path(self, came_from, current_node):
        path = []
        while current_node in came_from:
            path.append(current_node)
            current_node = came_from[current_node]
        path.append(current_node) # Add the start node
        path.reverse()
        
        # Convert grid path to world coordinates
        world_path = []
        for grid_x, grid_y in path:
            world_path.append(self.grid.grid_to_world(grid_x, grid_y))
        return world_path
