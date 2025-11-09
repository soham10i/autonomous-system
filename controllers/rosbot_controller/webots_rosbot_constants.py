import numpy as np

WHEEL_RADIUS = 0.043  # meters (from Rosbot model in Webots, WHEEL_CYLINDER radius)
TRACK_WIDTH = 0.22   # meters (distance between wheel centers, approx 2 * 0.11 from HingeJoint anchors)
# Lidar properties (from Rosbot documentation/model)
LIDAR_MAX_RANGE = 3.5 # meters
LIDAR_MIN_RANGE = 0.2 # meters (minimum valid range)
LIDAR_NUMBER_OF_RAYS = 256 # number of rays
LIDAR_HORIZONTAL_FOV = 6.28 # radians (360 degrees)

# Robot physical dimensions
ROBOT_RADIUS = 0.15 # meters (approximate, for obstacle inflation)
ROBOT_WIDTH = 0.22 # meters (approximate, for obstacle inflation)

# Maze dimensions (approximate, based on Maze1.wbt)
# These values define the area the occupancy grid will cover.
# They should be large enough to encompass the entire maze.
MAZE_WIDTH = 5.0  # meters
MAZE_HEIGHT = 5.0 # meters

# Origin of the occupancy grid in world coordinates.
# This should be the bottom-left corner of the area covered by the grid.
# If the maze is centered at (0,0) and is 5x5, then origin would be (-2.5, -2.5)
MAZE_ORIGIN_X = -2.5 # meters
MAZE_ORIGIN_Y = -2.5 # meters

# Occupancy Grid parameters
GRID_RESOLUTION = 0.05 # meters per cell

# Pillar locations (from Maze1.wbt)
BLUE_PILLAR_POS = (1.28, 0.83) # (x, y) world coordinates
YELLOW_PILLAR_POS = (-0.03, 0.31) # (x, y) world coordinates

# Controller parameters
TIME_STEP = 64  # milliseconds
MAX_VELOCITY = 3.28  # rad/s (for wheels)
MAX_SPEED = 4.0  # m/s (linear speed for the robot)


# PID controller gains for path following (tuned values)
KP_ANGULAR = 1.5
KP_LINEAR = 0.5

# Thresholds
DISTANCE_THRESHOLD_WAYPOINT = 0.1 # meters
DISTANCE_THRESHOLD_PILLAR = 0.3 # meters
ANGLE_THRESHOLD_WAYPOINT = np.deg2rad(5) # radians

# Exploration parameters
EXPLORATION_LINEAR_VEL = 0.5
EXPLORATION_ANGULAR_VEL_TURN = 0.5
OBSTACLE_AVOIDANCE_DISTANCE = 0.7 # meters (Increased to avoid premature detection)
OBSTACLE_AVOIDANCE_ANGLE_FOV = np.deg2rad(30) # radians (Increased FOV to consider wider front area)
SAFE_DISTANCE = 0.2  # meters (minimum distance to obstacles for safe navigation)
EMERGENCY_DISTANCE = 0.15
