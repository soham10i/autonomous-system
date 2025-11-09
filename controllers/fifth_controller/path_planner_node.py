#!/usr/bin/env python3
"""
Path Planner Node - A* Algorithm for Maze Navigation
Plans optimal paths from Start->Blue->Yellow pillars
Author: RosBot Navigation Team
"""

import rospy
import numpy as np
import heapq
import math
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String
import yaml

class AStarPathPlanner:
    """A* path planning for maze navigation"""

    def __init__(self):
        rospy.init_node('path_planner_node', anonymous=True)

        # Publishers
        self.path_pub = rospy.Publisher('/path', Path, queue_size=1, latch=True)

        # Subscribers  
        rospy.Subscriber('/mission_target', String, self.target_callback)

        # Load maze map and pillar locations
        self.load_maze_configuration()

        # Current target
        self.current_target = "blue"  # Start with blue pillar

        rospy.loginfo("Path Planner initialized - Ready for A* planning")

    def load_maze_configuration(self):
        """Load maze map and pillar positions"""
        # Static maze configuration (can be loaded from file)
        # This is a simplified example - replace with actual maze dimensions

        # Maze parameters (adjust based on actual maze)
        self.map_width = 100  # cells
        self.map_height = 100  # cells
        self.map_resolution = 0.05  # 5cm per cell

        # Create simple occupancy grid (0=free, 100=occupied, -1=unknown)
        self.occupancy_grid = np.zeros((self.map_height, self.map_width), dtype=np.int8)

        # Pillar positions in world coordinates (meters)
        # These should be measured/estimated from the actual maze
        self.pillar_positions = {
            "blue": {"x": 2.0, "y": 1.5},    # Adjust coordinates
            "yellow": {"x": -1.5, "y": 3.0}  # Adjust coordinates
        }

        # Robot starting position
        self.start_position = {"x": 0.0, "y": 0.0}

        rospy.loginfo("Maze configuration loaded")

    def target_callback(self, msg):
        """Handle target change requests"""
        new_target = msg.data.lower()
        if new_target in ["blue", "yellow"]:
            self.current_target = new_target
            rospy.loginfo(f"Target changed to: {self.current_target}")
            self.plan_path()

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x + self.map_width * self.map_resolution / 2) / self.map_resolution)
        grid_y = int((y + self.map_height * self.map_resolution / 2) / self.map_resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x = (grid_x * self.map_resolution) - (self.map_width * self.map_resolution / 2)
        y = (grid_y * self.map_resolution) - (self.map_height * self.map_resolution / 2)
        return x, y

    def heuristic(self, a, b):
        """Euclidean distance heuristic for A*"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, pos):
        """Get valid neighboring cells"""
        neighbors = []
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

        for dx, dy in directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy

            # Check bounds
            if (0 <= new_x < self.map_width and 
                0 <= new_y < self.map_height and
                self.occupancy_grid[new_y, new_x] != 100):  # Not occupied
                neighbors.append((new_x, new_y))

        return neighbors

    def astar_search(self, start, goal):
        """A* pathfinding algorithm"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)

                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def plan_path(self):
        """Plan path to current target"""
        # Get current robot position (assume starting position for now)
        start_world = (self.start_position["x"], self.start_position["y"])
        start_grid = self.world_to_grid(*start_world)

        # Get target position
        if self.current_target in self.pillar_positions:
            target_pos = self.pillar_positions[self.current_target]
            target_world = (target_pos["x"], target_pos["y"])
            target_grid = self.world_to_grid(*target_world)

            rospy.loginfo(f"Planning path to {self.current_target} pillar")

            # Run A* algorithm
            grid_path = self.astar_search(start_grid, target_grid)

            if grid_path:
                # Convert to ROS Path message
                path_msg = Path()
                path_msg.header.stamp = rospy.Time.now()
                path_msg.header.frame_id = "world"

                for grid_x, grid_y in grid_path:
                    world_x, world_y = self.grid_to_world(grid_x, grid_y)

                    pose = PoseStamped()
                    pose.header.stamp = rospy.Time.now()
                    pose.header.frame_id = "world"
                    pose.pose.position.x = world_x
                    pose.pose.position.y = world_y
                    pose.pose.position.z = 0.0
                    pose.pose.orientation.w = 1.0

                    path_msg.poses.append(pose)

                self.path_pub.publish(path_msg)
                rospy.loginfo(f"Path published: {len(grid_path)} waypoints")
            else:
                rospy.logwarn(f"No path found to {self.current_target} pillar")

    def run(self):
        """Main node loop"""
        # Initial path planning
        self.plan_path()
        rospy.spin()

if __name__ == '__main__':
    try:
        planner = AStarPathPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
