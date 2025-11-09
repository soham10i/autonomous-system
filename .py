"""
main_controller controller.
This controller allows for manual mapping of an environment using keyboard controls.
It visualizes the map, robot pose, live LiDAR scan (as rays), the robot's
path history, and the live camera feed in a single, real-time dashboard window
using the OpenCV library.
- The robot is driven using the arrow keys.
- Odometry from wheel sensors is used to track the robot's position.
- LiDAR data is processed with a ray-casting algorithm to build an accurate 2D occupancy grid map.
- Obstacle points are "thickened" to create a dense point cloud map suitable for pathfinding.
"""

from controller import Supervisor, Keyboard, PositionSensor, Camera
import numpy as np
import math
import sys
import cv2

# --- Robot and Simulation Setup ---
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# --- Device Initialization ---
# Keyboard
try:
    keyboard = Keyboard()
    keyboard.enable(timestep)
    print("Keyboard enabled for manual control.")
except Exception:
    keyboard = None
    print("Warning: Could not enable keyboard.")

# Camera
camera = None
camera_width, camera_height = 0, 0
try:
    camera = robot.getDevice("camera rgb")
    camera.enable(timestep)
    camera_width = camera.getWidth()
    camera_height = camera.getHeight()
    print("Camera 'camera rgb' enabled.")
except Exception:
    print("Warning: Camera device 'camera rgb' not found. Camera feed will be disabled.")

# LiDAR
lidar = None
try:
    lidar = robot.getDevice('lidar')
    lidar.enable(timestep)
    print("LiDAR device 'lidar' enabled.")
except Exception:
    print("FATAL: LiDAR device 'lidar' not found."); sys.exit()

# Motors and Position Sensors
motor_names = ["front left wheel motor", "front right wheel motor", "rear left wheel motor", "rear right wheel motor"]
motors = []
position_sensors = []
try:
    for name in motor_names:
        motor = robot.getDevice(name)
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)
        motors.append(motor)
        ps_name = name + " sensor"
        ps = robot.getDevice(ps_name)
        ps.enable(timestep)
        position_sensors.append(ps)
    print("All 4 motors and position sensors initialized.")
except Exception as e:
    print(f"FATAL: Could not initialize motors/sensors. Error: {e}"); sys.exit()

# --- Parameters ---
max_speed = 0.5
max_omega = 0.8
wheelRadius = 0.04
axleLength = 0.18

# --- Mapping Parameters ---
MAP_SIZE_METERS = 40 # Increased map size to accommodate longer LiDAR range
MAP_SCALE = 0.05
MAP_SIZE_PIXELS = int(MAP_SIZE_METERS / MAP_SCALE)
UNKNOWN = 128
FREE = 255
OCCUPIED = 0
occupancy_grid = np.full((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS), UNKNOWN, dtype=np.uint8)

# --- Odometry and Path History ---
robot_pose = np.array([MAP_SIZE_METERS / 2.0, MAP_SIZE_METERS / 2.0, -np.pi / 2.0])
last_wheel_positions = np.full(4, np.nan)
path_history = []

# --- Utility Functions ---
def update_odometry():
    """Calculates robot's new pose and updates path history."""
    global robot_pose, last_wheel_positions, path_history
    current_wheel_positions = np.array([ps.getValue() for ps in position_sensors])
    if np.any(np.isnan(last_wheel_positions)):
        last_wheel_positions = current_wheel_positions
        return
    wheel_deltas = current_wheel_positions - last_wheel_positions
    delta_left = (wheel_deltas[0] + wheel_deltas[2]) / 2.0
    delta_right = (wheel_deltas[1] + wheel_deltas[3]) / 2.0
    last_wheel_positions = current_wheel_positions
    delta_dist = ((delta_left + delta_right) / 2.0) * wheelRadius
    delta_theta = ((delta_right - delta_left) * wheelRadius) / axleLength
    robot_pose[0] += delta_dist * np.cos(robot_pose[2] + delta_theta / 2.0)
    robot_pose[1] += delta_dist * np.sin(robot_pose[2] + delta_theta / 2.0)
    robot_pose[2] += delta_theta
    robot_pose[2] = (robot_pose[2] + np.pi) % (2 * np.pi) - np.pi
    path_history.append((robot_pose[0], robot_pose[1]))

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
    """
    Projects LiDAR points onto the occupancy grid using ray-casting and
    returns the current scan's hit points for visualization.
    """
    ranges = lidar.getRangeImage()
    fov = lidar.getFov()
    res = lidar.getHorizontalResolution()
    robot_map_x = int(robot_pose[0] / MAP_SCALE)
    robot_map_y = int(robot_pose[1] / MAP_SCALE)
    
    current_scan_world_coords = []

    for i, r in enumerate(ranges):
        if not (lidar.getMinRange() < r < lidar.getMaxRange()): continue
        angle = -fov / 2.0 + (i * fov / res)
        beam_angle = robot_pose[2] + angle
        
        world_x = robot_pose[0] + r * np.cos(beam_angle)
        world_y = robot_pose[1] + r * np.sin(beam_angle)
        current_scan_world_coords.append((world_x, world_y))
        
        end_map_x = int(world_x / MAP_SCALE)
        end_map_y = int(world_y / MAP_SCALE)
        
        for (px, py) in bresenham_line(robot_map_x, robot_map_y, end_map_x, end_map_y):
            if 0 <= px < MAP_SIZE_PIXELS and 0 <= py < MAP_SIZE_PIXELS:
                occupancy_grid[py, px] = FREE
    
        if 0 <= end_map_x < MAP_SIZE_PIXELS and 0 <= end_map_y < MAP_SIZE_PIXELS:
            cv2.rectangle(occupancy_grid, (end_map_x-1, end_map_y-1), (end_map_x+1, end_map_y+1), OCCUPIED, -1)

    return current_scan_world_coords

def visualize_with_opencv(lidar_points, camera_image_data):
    """Visualizes the map, robot, LiDAR scan, path history, and camera feed."""
    # 1. Create the base map visualization
    map_vis = np.full((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS, 3), UNKNOWN, dtype=np.uint8)
    map_vis[occupancy_grid == FREE] = [255, 255, 255]
    map_vis[occupancy_grid == OCCUPIED] = [0, 0, 0]
    
    # 2. Draw the robot's path history in red
    if len(path_history) > 1:
        path_pixels = np.array(path_history) / MAP_SCALE
        path_pixels = path_pixels.astype(np.int32)
        cv2.polylines(map_vis, [path_pixels], isClosed=False, color=(0, 0, 255), thickness=2)

    # 3. Draw the live LiDAR scan as blue rays
    robot_pixel_x = int(robot_pose[0] / MAP_SCALE)
    robot_pixel_y = int(robot_pose[1] / MAP_SCALE)
    for point in lidar_points:
        point_pixel_x = int(point[0] / MAP_SCALE)
        point_pixel_y = int(point[1] / MAP_SCALE)
        cv2.line(map_vis, (robot_pixel_x, robot_pixel_y), (point_pixel_x, point_pixel_y), (255, 0, 0), 1)

    # 4. Draw robot position and heading
    cv2.circle(map_vis, (robot_pixel_x, robot_pixel_y), 8, (0, 255, 0), -1)
    arrow_end_x = robot_pixel_x + int(30 * np.cos(robot_pose[2]))
    arrow_end_y = robot_pixel_y + int(30 * np.sin(robot_pose[2]))
    cv2.arrowedLine(map_vis, (robot_pixel_x, robot_pixel_y), (arrow_end_x, arrow_end_y), (0, 255, 0), 3)

    # Flip map vertically to match world orientation
    map_vis = cv2.flip(map_vis, 0)
    
    # 5. Process and place camera image
    if camera_image_data and camera_width > 0:
        cam_img = np.frombuffer(camera_image_data, np.uint8).reshape((camera_height, camera_width, 4))
        cam_img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_BGRA2BGR)
        vis_h, vis_w, _ = map_vis.shape
        new_cam_h = vis_h // 4
        new_cam_w = int(new_cam_h * (camera_width / camera_height))
        cam_img_resized = cv2.resize(cam_img_bgr, (new_cam_w, new_cam_h))
        map_vis[10:10+new_cam_h, vis_w-10-new_cam_w:vis_w-10] = cam_img_resized

    # 6. Show the final visualization
    cv2.imshow("Global Map View", map_vis)
    cv2.waitKey(1)

# --- Main Control Loop ---
while robot.step(timestep) != -1:
    v, omega = 0.0, 0.0
    if keyboard:
        key = keyboard.getKey()
        if key == Keyboard.UP: v = max_speed
        elif key == Keyboard.DOWN: v = -max_speed
        elif key == Keyboard.LEFT: omega = max_omega
        elif key == Keyboard.RIGHT: omega = -max_omega
    left_vel = (2*v - omega*axleLength) / (2*wheelRadius)
    right_vel = (2*v + omega*axleLength) / (2*wheelRadius)
    motors[0].setVelocity(left_vel); motors[2].setVelocity(left_vel)
    motors[1].setVelocity(right_vel); motors[3].setVelocity(right_vel)
    
    update_odometry()
    current_lidar_points = update_map_and_get_points()
    current_camera_image = camera.getImage() if camera else None
    
    visualize_with_opencv(current_lidar_points, current_camera_image)

# Cleanup OpenCV windows on exit
cv2.destroyAllWindows()
