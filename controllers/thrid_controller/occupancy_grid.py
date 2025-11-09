
import numpy as np

class OccupancyGrid:
    def __init__(self, resolution=0.05, width=20.0, height=20.0, origin_x=0.0, origin_y=0.0, prob_occ=0.9, prob_free=0.1, prob_prior=0.5):
        """
        Initializes the OccupancyGrid.

        :param resolution: Resolution of the grid (meters/cell).
        :param width: Width of the grid in meters.
        :param height: Height of the grid in meters.
        :param origin_x: X-coordinate of the grid origin in world frame.
        :param origin_y: Y-coordinate of the grid origin in world frame.
        :param prob_occ: Probability of occupancy for an occupied cell.
        :param prob_free: Probability of occupancy for a free cell.
        :param prob_prior: Prior probability of occupancy.
        """
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.cells_x = int(self.width / self.resolution)
        self.cells_y = int(self.height / self.resolution)

        # Log-odds representation
        self.log_odds_grid = np.full((self.cells_x, self.cells_y), self._prob_to_log_odds(prob_prior))

        self.prob_occ_log_odds = self._prob_to_log_odds(prob_occ)
        self.prob_free_log_odds = self._prob_to_log_odds(prob_free)

    def _prob_to_log_odds(self, p):
        """
        Converts a probability to log-odds.
        """
        return np.log(p / (1 - p))

    def _log_odds_to_prob(self, l):
        """
        Converts log-odds to a probability.
        """
        return 1 - (1 / (1 + np.exp(l)))

    def _world_to_grid(self, x_world, y_world):
        """
        Converts world coordinates to grid coordinates.
        """
        grid_x = int((x_world - self.origin_x) / self.resolution)
        grid_y = int((y_world - self.origin_y) / self.resolution)
        return grid_x, grid_y

    def _grid_to_world(self, grid_x, grid_y):
        """
        Converts grid coordinates to world coordinates (center of cell).
        """
        x_world = grid_x * self.resolution + self.origin_x + self.resolution / 2
        y_world = grid_y * self.resolution + self.origin_y + self.resolution / 2
        return x_world, y_world

    def update_grid(self, robot_pose, lidar_scan):
        """
        Updates the occupancy grid based on a new LiDAR scan and robot pose.

        :param robot_pose: Current robot pose [x, y, theta] in world coordinates.
        :param lidar_scan: List of (distance, angle) tuples from the LiDAR.
        """
        robot_x, robot_y, robot_theta = robot_pose
        robot_grid_x, robot_grid_y = self._world_to_grid(robot_x, robot_y)

        # Mark robot's current cell as free
        if 0 <= robot_grid_x < self.cells_x and 0 <= robot_grid_y < self.cells_y:
            self.log_odds_grid[robot_grid_x, robot_grid_y] += self.prob_free_log_odds - self._prob_to_log_odds(0.5)

        for distance, angle in lidar_scan:
            # Calculate the world coordinates of the hit point
            hit_x_world = robot_x + distance * np.cos(robot_theta + angle)
            hit_y_world = robot_y + distance * np.sin(robot_theta + angle)

            hit_grid_x, hit_grid_y = self._world_to_grid(hit_x_world, hit_y_world)

            # Update occupied cell
            if 0 <= hit_grid_x < self.cells_x and 0 <= hit_grid_y < self.cells_y:
                self.log_odds_grid[hit_grid_x, hit_grid_y] += self.prob_occ_log_odds - self._prob_to_log_odds(0.5)

            # Update free cells along the ray (Bresenham's line algorithm)
            # This is a simplified version; a more robust implementation would use a proper ray-tracing algorithm
            # to mark all cells along the ray as free.
            # For now, we'll just mark cells between robot and hit point as free.
            
            # Convert robot and hit points to grid coordinates
            x0, y0 = robot_grid_x, robot_grid_y
            x1, y1 = hit_grid_x, hit_grid_y

            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            x, y = x0, y0
            while True:
                if 0 <= x < self.cells_x and 0 <= y < self.cells_y:
                    # Only mark as free if not the hit cell itself
                    if not (x == hit_grid_x and y == hit_grid_y):
                        self.log_odds_grid[x, y] += self.prob_free_log_odds - self._prob_to_log_odds(0.5)
                
                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

    def get_occupancy_grid(self):
        """
        Returns the occupancy grid as probabilities.
        """
        return self._log_odds_to_prob(self.log_odds_grid)

    def is_occupied(self, x_world, y_world, threshold=0.7):
        """
        Checks if a world coordinate is occupied.
        """
        grid_x, grid_y = self._world_to_grid(x_world, y_world)
        if 0 <= grid_x < self.cells_x and 0 <= grid_y < self.cells_y:
            return self._log_odds_to_prob(self.log_odds_grid[grid_x, grid_y]) > threshold
        return False # Out of bounds is considered not occupied for simplicity

    def get_grid_dimensions(self):
        """
        Returns the dimensions of the grid in cells.
        """
        return self.cells_x, self.cells_y

    def get_resolution(self):
        """
        Returns the resolution of the grid.
        """
        return self.resolution

    def get_origin(self):
        """
        Returns the origin of the grid in world coordinates.
        """
        return self.origin_x, self.origin_y

    def get_grid_value(self, x_world, y_world):
        """
        Returns the probability value of a specific cell in world coordinates.
        """
        grid_x, grid_y = self._world_to_grid(x_world, y_world)
        if 0 <= grid_x < self.cells_x and 0 <= grid_y < self.cells_y:
            return self._log_odds_to_prob(self.log_odds_grid[grid_x, grid_y])
        return 0.5 # Unknown if out of bounds



