from controller import Robot, Keyboard, Lidar, Camera, InertialUnit, PositionSensor
import numpy as np
import math
import struct

# Initialize the robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get robot motors and enable position sensors for odometry
left_motor = robot.getDevice("front left wheel motor")
right_motor = robot.getDevice("front right wheel motor")
rear_left_motor = robot.getDevice("rear left wheel motor")
rear_right_motor = robot.getDevice("rear right wheel motor")

motors = [left_motor, right_motor, rear_left_motor, rear_right_motor]
for motor in motors:
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

# Wheel sensors for odometry
left_position_sensor = robot.getDevice("front left wheel motor sensor")
right_position_sensor = robot.getDevice("front right wheel motor sensor")
left_position_sensor.enable(timestep)
camera = robot.getDevice("camera rgb")
camera.enable(timestep)

# Robust sensor initialization (as in rosbot_controller)
lidar = None
camera = None
depth_camera = None
imu = None
try:
    lidar = robot.getDevice("lidar")
    if lidar:
        lidar.enable(timestep)
        lidar.enablePointCloud()
        print("✅ LiDAR enabled")
except Exception as e:
    print(f"⚠️ LiDAR error: {e}")
    lidar = None

try:
    camera = robot.getDevice("camera rgb")
    if camera:
        camera.enable(timestep)
        print("✅ RGB camera enabled")
except Exception as e:
    print(f"⚠️ RGB camera error: {e}")
    camera = None

try:
    depth_camera = robot.getDevice("camera depth")
    if depth_camera:
        depth_camera.enable(timestep)
        print("✅ Depth camera enabled")
except Exception as e:
    print(f"⚠️ Depth camera error: {e}")
    depth_camera = None

try:
    imu = robot.getDevice("imu")
    if imu:
        imu.enable(timestep)
        print("✅ IMU enabled")
except Exception as e:
    print(f"⚠️ IMU error: {e}")
    imu = None

# Enable keyboard input
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Robot movement speeds
MAX_SPEED = 6.0

# Robot parameters for odometry
WHEEL_RADIUS = 0.043  # From RosBot.proto (radius of WHEEL_CYLINDER)
TRACK_WIDTH = 0.22    # Approximate distance between wheels (0.11 * 2)

# Robot pose (x, y, theta)
robot_x = 0.0
robot_y = 0.0
robot_theta = 0.0

# Previous wheel positions for odometry
prev_left_wheel_pos = 0.0
prev_right_wheel_pos = 0.0

# 3D Occupancy Grid Parameters
MAP_RESOLUTION = 0.1  # meters per cell
MAP_SIZE_X = 50       # cells
MAP_SIZE_Y = 50       # cells
MAP_SIZE_Z = 10       # cells (for height)
MAP_ORIGIN_X = -MAP_SIZE_X * MAP_RESOLUTION / 2
MAP_ORIGIN_Y = -MAP_SIZE_Y * MAP_RESOLUTION / 2
MAP_ORIGIN_Z = 0.0 # Assuming ground is at Z=0

# Initialize 3D occupancy grid (0: unknown, 1: free, 2: occupied)
occupancy_grid = np.zeros((MAP_SIZE_X, MAP_SIZE_Y, MAP_SIZE_Z), dtype=np.uint8)

def world_to_grid(x, y, z):
    grid_x = int((x - MAP_ORIGIN_X) / MAP_RESOLUTION)
    grid_y = int((y - MAP_ORIGIN_Y) / MAP_RESOLUTION)
    grid_z = int((z - MAP_ORIGIN_Z) / MAP_RESOLUTION)
    return grid_x, grid_y, grid_z

def is_valid_grid_coord(gx, gy, gz):
    return 0 <= gx < MAP_SIZE_X and 0 <= gy < MAP_SIZE_Y and 0 <= gz < MAP_SIZE_Z

def set_robot_velocity(left_speed, right_speed):
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
    rear_left_motor.setVelocity(left_speed)
    rear_right_motor.setVelocity(right_speed)

# Function to convert depth image to point cloud
def depth_image_to_point_cloud(depth_image, camera_width, camera_height, fov):
    points = []
    # Calculate focal length (assuming pinhole camera model)
    focal_length = (camera_width / 2) / np.tan(fov / 2)

    for y in range(camera_height):
        for x in range(camera_width):
            # Get depth value (assuming 16-bit depth image)
            depth = camera.imageGetDepth(depth_image, x, y)
            
            # Convert pixel coordinates to camera coordinates
            x_c = (x - camera_width / 2) * depth / focal_length
            y_c = (y - camera_height / 2) * depth / focal_length
            z_c = depth
            
            if depth > 0 and depth < camera.getMaxRange(): # Only consider valid depth values
                points.append([x_c, y_c, z_c])
    return np.array(points)

# Function to update occupancy grid with new points
def update_occupancy_grid(points, robot_pose):
    global occupancy_grid
    
    robot_x, robot_y, robot_theta = robot_pose

    for p in points:
        px, py, pz = p[0], p[1], p[2]
        rotated_x = px * math.cos(robot_theta) - py * math.sin(robot_theta)
        rotated_y = px * math.sin(robot_theta) + py * math.cos(robot_theta)
        global_x = robot_x + rotated_x
        global_y = robot_y + rotated_y
        global_z = pz # Assuming z is absolute or relative to robot base

        # Skip points with NaN or infinite values
        if (math.isnan(global_x) or math.isnan(global_y) or math.isnan(global_z) or
            math.isinf(global_x) or math.isinf(global_y) or math.isinf(global_z)):
            continue

        gx, gy, gz = world_to_grid(global_x, global_y, global_z)
        if is_valid_grid_coord(gx, gy, gz):
            occupancy_grid[gx, gy, gz] = 2 # Mark as occupied

# Function to save occupancy grid to a text file
def save_occupancy_grid(filename, grid):
    with open(filename, 'w') as f:
        for x in range(MAP_SIZE_X):
            for y in range(MAP_SIZE_Y):
                for z in range(MAP_SIZE_Z):
                    if grid[x, y, z] == 2: # Only save occupied cells
                        # Convert grid coordinates back to approximate world coordinates for easier visualization
                        world_x = MAP_ORIGIN_X + x * MAP_RESOLUTION
                        world_y = MAP_ORIGIN_Y + y * MAP_RESOLUTION
                        world_z = MAP_ORIGIN_Z + z * MAP_RESOLUTION
                        f.write(f"{world_x:.2f} {world_y:.2f} {world_z:.2f}\n")

# Main loop
while robot.step(timestep) != -1:
    # Odometry calculation
    current_left_wheel_pos = left_position_sensor.getValue()
    current_right_wheel_pos = right_position_sensor.getValue()

    delta_left = (current_left_wheel_pos - prev_left_wheel_pos) * WHEEL_RADIUS
    delta_right = (current_right_wheel_pos - prev_right_wheel_pos) * WHEEL_RADIUS

    prev_left_wheel_pos = current_left_wheel_pos
    prev_right_wheel_pos = current_right_wheel_pos

    delta_s = (delta_left + delta_right) / 2.0
    delta_theta = (delta_right - delta_left) / TRACK_WIDTH

    robot_x += delta_s * math.cos(robot_theta + delta_theta / 2.0)
    robot_y += delta_s * math.sin(robot_theta + delta_theta / 2.0)
    robot_theta += delta_theta

    # Normalize theta to be within -pi to pi
    robot_theta = math.atan2(math.sin(robot_theta), math.cos(robot_theta))

    key = keyboard.getKey()
    
    left_speed = 0.0
    right_speed = 0.0

    if key == Keyboard.UP:
        left_speed = MAX_SPEED
        right_speed = MAX_SPEED
    elif key == Keyboard.DOWN:
        left_speed = -MAX_SPEED
        right_speed = -MAX_SPEED
    elif key == Keyboard.LEFT:
        left_speed = -MAX_SPEED / 2
        right_speed = MAX_SPEED / 2
    elif key == Keyboard.RIGHT:
        left_speed = MAX_SPEED / 2
        right_speed = -MAX_SPEED / 2
    else:
        left_speed = 0.0
        right_speed = 0.0

    set_robot_velocity(left_speed, right_speed)

    current_robot_pose = (robot_x, robot_y, robot_theta)

    # Get LiDAR data and process
    lidar_data = lidar.getRangeImage()
    if lidar_data:
        lidar_points_local = []
        angle_increment = lidar.getFov() / lidar.getHorizontalResolution()
        
        # Lidar's position relative to robot's origin (from RosBot.proto: LIDAR_SLOT translation 0.02 0 0.1)
        lidar_offset_x = 0.02
        lidar_offset_y = 0.0
        lidar_offset_z = 0.1

        for i, distance in enumerate(lidar_data):
            if distance < lidar.getMaxRange(): # Valid measurement
                angle = -lidar.getFov() / 2 + i * angle_increment # Angle in lidar's frame
                
                # Convert to Cartesian in lidar's frame
                lx = distance * math.cos(angle)
                ly = distance * math.sin(angle)
                lz = 0.0 # Assuming 2D lidar on a flat plane
                
                # Transform to robot's frame (considering lidar's offset)
                rx = lx + lidar_offset_x
                ry = ly + lidar_offset_y
                rz = lz + lidar_offset_z
                lidar_points_local.append([rx, ry, rz])
        update_occupancy_grid(lidar_points_local, current_robot_pose)

    # Get camera data (RGB and Depth) and process
    camera_image = camera.getImage()
    camera_width = camera.getWidth()
    camera_height = camera.getHeight()
    camera_fov = camera.getFov()
    
    if camera_image:
        depth_point_cloud_local = depth_image_to_point_cloud(camera_image, camera_width, camera_height, camera_fov)
        
        # Camera's position relative to robot's origin (from RosBot.proto: CAMERA_SLOT translation -0.027 0 0.165)
        camera_offset_x = -0.027
        camera_offset_y = 0.0
        camera_offset_z = 0.165

        camera_points_robot_frame = []
        for p in depth_point_cloud_local:
            cx, cy, cz = p[0], p[1], p[2]
            
            # Transform from camera frame to robot frame (assuming camera's axes are aligned with robot's for simplicity)
            rx = cx + camera_offset_x
            ry = cy + camera_offset_y
            rz = cz + camera_offset_z
            camera_points_robot_frame.append([rx, ry, rz])
        update_occupancy_grid(camera_points_robot_frame, current_robot_pose)

    # Save occupancy grid periodically or on exit
    # For demonstration, let's save it every 1000 steps
    if robot.getTime() > 0 and int(robot.getTime() * 1000 / timestep) % 1000 == 0:
        save_occupancy_grid("occupancy_map.txt", occupancy_grid)
        # print(f"Saved occupancy grid.")

    # TODO: Further refinements:
    # - Implement a more robust localization (e.g., using IMU data, or a particle filter/Kalman filter)
    # - Implement ray tracing for free space marking in occupancy grid
    # - Implement loop closure for global consistency