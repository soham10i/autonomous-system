from controller import Robot, Lidar, DistanceSensor, Motor, Supervisor, Keyboard
import numpy as np
import math
import cv2
import sys

from enhanced_occupancy_grid import EnhancedOccupancyGrid
from a_star import AStar
from webots_rosbot_constants import *
from visual_odometry import VisualOdometry

# --- Global Variables and Constants ---
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Keyboard
keyboard = None
try:
    keyboard = Keyboard()
    keyboard.enable(timestep)
    print("Keyboard enabled for manual control.")
except Exception:
    print("Warning: Could not enable keyboard.")

# Camera
camera = None
camera_width, camera_height = 0, 0
try:
    camera = robot.getDevice("camera rgb") # Name from Rosbot PROTO
    if camera:
        camera.enable(timestep)
        camera_width = camera.getWidth()
        camera_height = camera.getHeight()
        print("Camera \'camera\' enabled.")
    else:
        print("Warning: Camera device \'camera\' not found. Camera features will be disabled.")
except Exception as e:
    print(f"Warning: Error initializing camera: {e}")

# Depth Camera (RangeFinder)
depth_camera = None
depth_camera_width, depth_camera_height = 0, 0
try:
    depth_camera = robot.getDevice("camera depth") # Name from Astra PROTO
    if depth_camera:
        depth_camera.enable(timestep)
        depth_camera_width = depth_camera.getWidth()
        depth_camera_height = depth_camera.getHeight()
        print("Depth Camera \'camera depth\' enabled.")
    else:
        print("Warning: Depth Camera device \'camera depth\' not found. Depth features will be disabled.")
except Exception as e:
    print(f"Warning: Error initializing depth camera: {e}")

# LiDAR
lidar = None
try:
    lidar = robot.getDevice("lidar") # Corrected LiDAR name
    lidar.enable(timestep)
    lidar.enablePointCloud()
    print("LiDAR device \'lidar\' enabled.")
except Exception:
    print("FATAL: LiDAR device \'lidar\' not found."); sys.exit()

# Motors and Position Sensors
motor_names = ["front left wheel motor", "front right wheel motor", "rear left wheel motor", "rear right wheel motor"]
motors = []
position_sensors = []

front_left_motor = None
front_right_motor = None
rear_left_motor = None
rear_right_motor = None

front_left_position_sensor = None
front_right_position_sensor = None
rear_left_position_sensor = None
rear_right_position_sensor = None

last_front_left_position = 0.0
last_front_right_position = 0.0
last_rear_left_position = 0.0
last_rear_right_position = 0.0

try:
    front_left_motor = robot.getDevice("front left wheel motor")
    front_right_motor = robot.getDevice("front right wheel motor")
    rear_left_motor = robot.getDevice("rear left wheel motor")
    rear_right_motor = robot.getDevice("rear right wheel motor")

    motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]

    for motor in motors:
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)

    front_left_position_sensor = robot.getDevice("front left wheel motor sensor")
    front_right_position_sensor = robot.getDevice("front right wheel motor sensor")
    rear_left_position_sensor = robot.getDevice("rear left wheel motor sensor")
    rear_right_position_sensor = robot.getDevice("rear right wheel motor sensor")

    position_sensors = [front_left_position_sensor, front_right_position_sensor, rear_left_position_sensor, rear_right_position_sensor]

    for ps in position_sensors:
        ps.enable(timestep)
    print("All 4 motors and position sensors initialized.")
except Exception as e:
    print(f"FATAL: Could not initialize motors/sensors. Error: {e}"); sys.exit()

# Initial robot pose from supervisor
robot_node = robot.getSelf()
robot_pose_matrix = robot_node.getPose()

robot_x = robot_pose_matrix[3] # X translation
robot_y = robot_pose_matrix[7] # Y translation
robot_theta = math.atan2(robot_pose_matrix[4], robot_pose_matrix[0])

# Odometry and Path History
path_history = []

# Occupancy Grid and A* Pathfinding
occupancy_grid = EnhancedOccupancyGrid(GRID_RESOLUTION, MAZE_WIDTH, MAZE_HEIGHT, MAZE_ORIGIN_X, MAZE_ORIGIN_Y)
a_star = AStar(occupancy_grid)
visual_odometry = VisualOdometry()

# State machine variables
state = 'EXPLORING' # or 'NAVIGATING_TO_BLUE', 'NAVIGATING_TO_YELLOW', 'FINISHED'
path = []
path_index = 0

# Pillar locations (approximate, will need to be refined)
blue_pillar_pos = BLUE_PILLAR_POS
yellow_pillar_pos = YELLOW_PILLAR_POS

reached_blue_pillar = False

# --- Utility Functions ---
def update_odometry():
    global robot_x, robot_y, robot_theta, last_front_left_position, last_front_right_position, last_rear_left_position, last_rear_right_position, path_history

    current_front_left_position = front_left_position_sensor.getValue()
    current_front_right_position = front_right_position_sensor.getValue()
    current_rear_left_position = rear_left_position_sensor.getValue()
    current_rear_right_position = rear_right_position_sensor.getValue()

    delta_front_left = current_front_left_position - last_front_left_position
    delta_front_right = current_front_right_position - last_front_right_position
    delta_rear_left = current_rear_left_position - last_rear_left_position
    delta_rear_right = current_rear_right_position - last_rear_right_position

    last_front_left_position = current_front_left_position
    last_front_right_position = current_front_right_position
    last_rear_left_position = current_rear_left_position
    last_rear_right_position = current_rear_right_position

    # Average the deltas for left and right sides
    delta_left = (delta_front_left + delta_rear_left) / 2.0
    delta_right = (delta_front_right + delta_rear_right) / 2.0

    # Calculate linear and angular displacement
    delta_s_left = delta_left * WHEEL_RADIUS
    delta_s_right = delta_right * WHEEL_RADIUS

    delta_s = (delta_s_left + delta_s_right) / 2.0
    delta_theta = (delta_s_right - delta_s_left) / TRACK_WIDTH

    # Update robot pose
    robot_x += delta_s * math.cos(robot_theta + delta_theta / 2.0)
    robot_y += delta_s * math.sin(robot_theta + delta_theta / 2.0)
    robot_theta += delta_theta

    # Normalize theta to be within -pi to pi
    robot_theta = math.atan2(math.sin(robot_theta), math.cos(robot_theta))
    path_history.append((robot_x, robot_y))

def set_velocity(linear_velocity, angular_velocity):
    # Differential drive kinematics for 4 wheels
    left_wheel_speed = (linear_velocity - angular_velocity * TRACK_WIDTH / 2) / WHEEL_RADIUS
    right_wheel_speed = (linear_velocity + angular_velocity * TRACK_WIDTH / 2) / WHEEL_RADIUS

    # Cap velocities to MAX_VELOCITY
    left_wheel_speed = np.clip(left_wheel_speed, -MAX_VELOCITY, MAX_VELOCITY)
    right_wheel_speed = np.clip(right_wheel_speed, -MAX_VELOCITY, MAX_VELOCITY)

    front_left_motor.setVelocity(left_wheel_speed)
    rear_left_motor.setVelocity(left_wheel_speed)
    front_right_motor.setVelocity(right_wheel_speed)
    rear_right_motor.setVelocity(right_wheel_speed)

def stop_robot():
    for motor in motors:
        motor.setVelocity(0.0)

def get_lidar_data():
    range_image = lidar.getRangeImage()
    
    lidar_data = []
    if range_image:
        for i in range(LIDAR_NUMBER_OF_RAYS):
            distance = range_image[i]
            angle = -LIDAR_HORIZONTAL_FOV / 2 + i * (LIDAR_HORIZONTAL_FOV / (LIDAR_NUMBER_OF_RAYS - 1))
            
            if distance < LIDAR_MAX_RANGE and distance > 0.01:
                lidar_data.append((angle, distance))
    return lidar_data

def get_camera_image():
    global camera, camera_width, camera_height, depth_camera, depth_camera_width, depth_camera_height
    rgb_image = None
    depth_image = None

    if camera:
        image_data = camera.getImage()
        rgb_image = np.frombuffer(image_data, np.uint8).reshape((camera_height, camera_width, 4))

    if depth_camera:
        depth_data = depth_camera.getRangeImage()
        # RangeImage is a flat list of floats, reshape to 2D array
        depth_image = np.array(depth_data, dtype=np.float32).reshape((depth_camera_height, depth_camera_width))

    return rgb_image, depth_image

def detect_pillars(camera_image):
    if camera_image is None:
        return None, None

    # Convert BGR to HSV color space
    hsv_image = cv2.cvtColor(camera_image, cv2.COLOR_BGRA2BGR) # Convert BGRA to BGR first
    hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)

    # Define color ranges for blue and yellow pillars in HSV
    # These values might need tuning based on the actual pillar colors in Webots
    # Blue color range
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Yellow color range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create masks for blue and yellow colors
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Find contours in the masks
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_blue_pillar = None
    detected_yellow_pillar = None

    # Process blue contours
    if contours_blue:
        # Find the largest contour (assuming pillar is the largest blue object)
        largest_blue_contour = max(contours_blue, key=cv2.contourArea)
        if cv2.contourArea(largest_blue_contour) > 100: # Minimum contour area to avoid noise
            M = cv2.moments(largest_blue_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                detected_blue_pillar = (cX, cY) # Pixel coordinates of blue pillar center

    # Process yellow contours
    if contours_yellow:
        largest_yellow_contour = max(contours_yellow, key=cv2.contourArea)
        if cv2.contourArea(largest_yellow_contour) > 100: # Minimum contour area to avoid noise
            M = cv2.moments(largest_yellow_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                detected_yellow_pillar = (cX, cY) # Pixel coordinates of yellow pillar center

    return detected_blue_pillar, detected_yellow_pillar

def bresenham_line(x0, y0, x1, y1):
    """Yields integer coordinates on a line from (x0, y0) to (x1, y1)."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        yield (x0, y0)
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx: err += dx; y0 += sy

def update_map_and_get_points():
    global occupancy_grid, robot_x, robot_y, robot_theta, lidar
    """
    Projects LiDAR points onto the occupancy grid using ray-casting and
    returns the current scan\'s hit points for visualization.
    """
    ranges = lidar.getRangeImage()
    fov = LIDAR_HORIZONTAL_FOV
    res = LIDAR_NUMBER_OF_RAYS
    
    # Convert robot world coordinates to map pixel coordinates
    robot_map_x = int((robot_x - MAZE_ORIGIN_X) / GRID_RESOLUTION)
    robot_map_y = int((robot_y - MAZE_ORIGIN_Y) / GRID_RESOLUTION)
    
    current_scan_world_coords = []

    for i, r in enumerate(ranges):
        if not (LIDAR_MIN_RANGE < r < LIDAR_MAX_RANGE): continue
        angle = -fov / 2.0 + (i * fov / res)
        beam_angle = robot_theta + angle
        
        world_x = robot_x + r * np.cos(beam_angle)
        world_y = robot_y + r * np.sin(beam_angle)
        current_scan_world_coords.append((world_x, world_y))
        
        end_map_x = int((world_x - MAZE_ORIGIN_X) / GRID_RESOLUTION)
        end_map_y = int((world_y - MAZE_ORIGIN_Y) / GRID_RESOLUTION)
        
        # Ensure coordinates are within grid bounds
        if not (0 <= robot_map_x < occupancy_grid.shape[1] and \
                0 <= robot_map_y < occupancy_grid.shape[0] and \
                0 <= end_map_x < occupancy_grid.shape[1] and \
                0 <= end_map_y < occupancy_grid.shape[0]):
            continue

        for (px, py) in bresenham_line(robot_map_x, robot_map_y, end_map_x, end_map_y):
            if 0 <= px < occupancy_grid.shape[1] and 0 <= py < occupancy_grid.shape[0]:
                # Mark as free (lower probability of occupancy)
                occupancy_grid.grid[py, px] = max(0, occupancy_grid.grid[py, px] - 0.1) # Decrease occupancy
    
        if 0 <= end_map_x < occupancy_grid.shape[1] and 0 <= end_map_y < occupancy_grid.shape[0]:
            # Mark as occupied (higher probability of occupancy)
            occupancy_grid.grid[end_map_y, end_map_x] = min(1, occupancy_grid.grid[end_map_y, end_map_x] + 0.2) # Increase occupancy

    return current_scan_world_coords

def visualize_with_opencv(lidar_points, camera_image_data):
    global occupancy_grid, robot_x, robot_y, robot_theta, path_history, camera_width, camera_height
    """Visualizes the map, robot, LiDAR scan, path history, and camera feed."""
    # 1. Create the base map visualization from the occupancy grid
    # Convert probabilistic grid to grayscale image
    map_vis = (occupancy_grid.grid * 255).astype(np.uint8)
    map_vis = cv2.cvtColor(map_vis, cv2.COLOR_GRAY2BGR)
    
    # 2. Draw the robot\"s path history in red
    if len(path_history) > 1:
        path_pixels = []
        for wx, wy in path_history:
            px, py = occupancy_grid.world_to_grid(wx, wy)
            path_pixels.append((px, py))
        path_pixels = np.array(path_pixels).astype(np.int32)
        cv2.polylines(map_vis, [path_pixels], isClosed=False, color=(0, 0, 255), thickness=2)

    # 3. Draw the live LiDAR scan as blue rays
    robot_pixel_x, robot_pixel_y = occupancy_grid.world_to_grid(robot_x, robot_y)
    for point in lidar_points:
        point_pixel_x, point_pixel_y = occupancy_grid.world_to_grid(point[0], point[1])
        cv2.line(map_vis, (robot_pixel_x, robot_pixel_y), (point_pixel_x, point_pixel_y), (255, 0, 0), 1)

    # 4. Draw robot position and heading
    cv2.circle(map_vis, (robot_pixel_x, robot_pixel_y), 8, (0, 255, 0), -1)
    arrow_end_x = robot_pixel_x + int(30 * np.cos(robot_theta))
    arrow_end_y = robot_pixel_y + int(30 * np.sin(robot_theta))
    cv2.arrowedLine(map_vis, (robot_pixel_x, robot_pixel_y), (arrow_end_x, arrow_end_y), (0, 255, 0), 3)

    # Flip map vertically to match world orientation (Webots Y-up, OpenCV Y-down)
    map_vis = cv2.flip(map_vis, 0)
    
    # 5. Process and place camera image
    if camera_image_data is not None and camera_width > 0:
        cam_img_bgr = cv2.cvtColor(camera_image_data, cv2.COLOR_BGRA2BGR)
        vis_h, vis_w, _ = map_vis.shape
        new_cam_h = vis_h // 4
        new_cam_w = int(new_cam_h * (camera_width / camera_height))
        cam_img_resized = cv2.resize(cam_img_bgr, (new_cam_w, new_cam_h))
        map_vis[10:10+new_cam_h, vis_w-10-new_cam_w:vis_w-10] = cam_img_resized

    # 6. Show the final visualization
    cv2.imshow("Global Map View", map_vis)
    cv2.waitKey(1)
