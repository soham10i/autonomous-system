
import numpy as np
import heapq

class GlobalPlanner:
    def __init__(self):
        """
        Initializes the GlobalPlanner class.
        """
        pass

    def _world_to_grid(self, x_world, y_world, occupancy_grid):
        """
        Converts world coordinates to grid coordinates using the occupancy grid's properties.
        """
        grid_x = int((x_world - occupancy_grid.origin_x) / occupancy_grid.resolution)
        grid_y = int((y_world - occupancy_grid.origin_y) / occupancy_grid.resolution)
        return grid_x, grid_y

    def _grid_to_world(self, grid_x, grid_y, occupancy_grid):
        """
        Converts grid coordinates to world coordinates (center of cell) using the occupancy grid's properties.
        """
        x_world = grid_x * occupancy_grid.resolution + occupancy_grid.origin_x + occupancy_grid.resolution / 2
        y_world = grid_y * occupancy_grid.resolution + occupancy_grid.origin_y + occupancy_grid.resolution / 2
        return x_world, y_world

    def plan_path(self, start_world, end_world, occupancy_grid):
        """
        Finds a path from start to end using the A* algorithm on the occupancy grid.

        :param start_world: Start point in world coordinates [x, y].
        :param end_world: End point in world coordinates [x, y].
        :param occupancy_grid: The OccupancyGrid object.
        :return: A list of world coordinates representing the path, or None if no path is found.
        """
        start_grid = self._world_to_grid(start_world[0], start_world[1], occupancy_grid)
        end_grid = self._world_to_grid(end_world[0], end_world[1], occupancy_grid)

        cells_x, cells_y = occupancy_grid.get_grid_dimensions()
        grid_data = occupancy_grid.get_occupancy_grid()

        # Check if start or end are occupied
        if occupancy_grid.is_occupied(start_world[0], start_world[1]) or \
           occupancy_grid.is_occupied(end_world[0], end_world[1]):
            return None

        # A* algorithm implementation
        open_set = []
        heapq.heappush(open_set, (0, start_grid))  # (f_score, (x, y))

        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, end_grid)}

        while open_set:
            current_f_score, current_grid = heapq.heappop(open_set)

            if current_grid == end_grid:
                return self._reconstruct_path(came_from, current_grid, occupancy_grid)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor_grid = (current_grid[0] + dx, current_grid[1] + dy)

                if not (0 <= neighbor_grid[0] < cells_x and 0 <= neighbor_grid[1] < cells_y):
                    continue

                # Check if neighbor is occupied
                neighbor_world_x, neighbor_world_y = self._grid_to_world(neighbor_grid[0], neighbor_grid[1], occupancy_grid)
                if occupancy_grid.is_occupied(neighbor_world_x, neighbor_world_y):
                    continue

                tentative_g_score = g_score[current_grid] + self._distance(current_grid, neighbor_grid)

                if neighbor_grid not in g_score or tentative_g_score < g_score[neighbor_grid]:
                    came_from[neighbor_grid] = current_grid
                    g_score[neighbor_grid] = tentative_g_score
                    f_score[neighbor_grid] = tentative_g_score + self._heuristic(neighbor_grid, end_grid)
                    heapq.heappush(open_set, (f_score[neighbor_grid], neighbor_grid))

        return None  # No path found

    def _heuristic(self, a, b):
        """
        Manhattan distance heuristic.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _distance(self, a, b):
        """
        Euclidean distance between two grid points.
        """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _reconstruct_path(self, came_from, current_grid, occupancy_grid):
        """
        Reconstructs the path from the came_from dictionary.
        """
        path = []
        while current_grid in came_from:
            path.append(self._grid_to_world(current_grid[0], current_grid[1], occupancy_grid))
            current_grid = came_from[current_grid]
        path.append(self._grid_to_world(current_grid[0], current_grid[1], occupancy_grid)) # Add start node
        return path[::-1]  # Reverse to get path from start to end

    def find_exploration_goal(self, occupancy_grid, robot_pose):
        """
        Identifies the closest frontier (boundary between known and unknown space) and returns it as a goal point.

        :param occupancy_grid: The OccupancyGrid object.
        :param robot_pose: Current robot pose [x, y, theta] in world coordinates.
        :return: A goal point [x, y] in world coordinates, or None if no frontier is found.
        """
        cells_x, cells_y = occupancy_grid.get_grid_dimensions()
        grid_data = occupancy_grid.get_occupancy_grid()

        robot_grid_x, robot_grid_y = self._world_to_grid(robot_pose[0], robot_pose[1], occupancy_grid)

        frontiers = []
        # Iterate through the grid to find frontier cells
        for x in range(cells_x):
            for y in range(cells_y):
                # A cell is a frontier if it is unknown (around 0.5 probability) and has a known (occupied or free) neighbor
                prob = grid_data[x, y]
                if 0.4 < prob < 0.6: # Unknown cell
                    # Check neighbors
                    is_frontier = False
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < cells_x and 0 <= ny < cells_y:
                            n_prob = grid_data[nx, ny]
                            if n_prob >= 0.6 or n_prob <= 0.4: # Known neighbor
                                is_frontier = True
                                break
                    if is_frontier:
                        frontiers.append((x, y))
        
        if not frontiers:
            return None

        # Find the closest frontier to the robot
        closest_frontier = None
        min_distance = float("inf")

        for fx, fy in frontiers:
            dist = self._distance((robot_grid_x, robot_grid_y), (fx, fy))
            if dist < min_distance:
                min_distance = dist
                closest_frontier = (fx, fy)

        if closest_frontier:
            return self._grid_to_world(closest_frontier[0], closest_frontier[1], occupancy_grid)
        return None


