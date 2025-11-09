from controller import Robot, Supervisor, Lidar, Camera, RangeFinder, Motor, Keyboard
import numpy as np
import cv2
import math
import time


# =============================================================================
# CONSTANTS
# =============================================================================

TIME_STEP = 64
MAX_SPEED = 3.5
SPEED_INCREMENT = 2.0

# Map parameters
MAP_SIZE_METERS = 10.0  # 10m x 10m map
MAP_RESOLUTION = 0.05   # 5cm per pixel (10m / 200 pixels = 0.05m/pixel)
MAP_SIZE_PIXELS = int(MAP_SIZE_METERS / MAP_RESOLUTION)  # 200x200 pixels

# Occupancy grid values
FREE = 255        # White - free space
UNKNOWN = 128     # Gray - unknown
OCCUPIED = 0      # Black - obstacle
ROBOT = (0, 255, 0)      # Green - robot position
LIDAR_POINT = (0, 0, 255) # Red - current lidar scan


# =============================================================================
# MAP MANAGER CLASS
# =============================================================================

class PersistentMapManager:
	"""Manages persistent 2D occupancy grid map"""

	def __init__(self, size_meters, resolution):
		self.size_meters = size_meters
		self.resolution = resolution
		self.size_pixels = int(size_meters / resolution)

		# Initialize map (grayscale: 0=occupied, 128=unknown, 255=free)
		self.occupancy_map = np.full((self.size_pixels, self.size_pixels), UNKNOWN, dtype=np.uint8)

		# Store all robot positions for trajectory
		self.robot_trajectory = []

		# Map origin in world coordinates (center of map)
		self.origin_x = 0.0
		self.origin_y = 0.0

		print(f"Map initialized: {self.size_pixels}x{self.size_pixels} pixels")
		print(f"Resolution: {resolution}m per pixel")

	def world_to_map(self, world_x, world_y):
		"""Convert world coordinates to map pixel coordinates"""
		# Map center is at (size_pixels/2, size_pixels/2)
		pixel_x = int((world_x - self.origin_x) / self.resolution + self.size_pixels / 2)
		pixel_y = int((world_y - self.origin_y) / self.resolution + self.size_pixels / 2)
		return (pixel_x, pixel_y)

	def is_valid_pixel(self, px, py):
		"""Check if pixel coordinates are within map bounds"""
		return 0 <= px < self.size_pixels and 0 <= py < self.size_pixels

	def update_map(self, robot_x, robot_y, robot_theta, lidar_data, max_range):
		"""
		Update occupancy map with lidar scan

		Args:
			robot_x, robot_y: Robot position in world coordinates
			robot_theta: Robot heading in radians
			lidar_data: Array of lidar distances
			max_range: Maximum lidar range
		"""
		# Convert robot position to map coordinates
		robot_px, robot_py = self.world_to_map(robot_x, robot_y)

		# Store robot position in trajectory
		if self.is_valid_pixel(robot_px, robot_py):
			self.robot_trajectory.append((robot_px, robot_py))

		if not lidar_data:
			return

		num_points = len(lidar_data)
		angle_step = 2 * math.pi / num_points

		for i, distance in enumerate(lidar_data):
			# Calculate point angle in world frame
			lidar_angle = i * angle_step
			world_angle = robot_theta + lidar_angle

			if distance < max_range * 0.95:  # Valid reading (not max range)
				# Calculate obstacle position in world coordinates
				obs_world_x = robot_x + distance * math.cos(world_angle)
				obs_world_y = robot_y + distance * math.sin(world_angle)

				# Convert to map coordinates
				obs_px, obs_py = self.world_to_map(obs_world_x, obs_world_y)

				# Mark obstacle as occupied
				if self.is_valid_pixel(obs_px, obs_py):
					self.occupancy_map[obs_py, obs_px] = OCCUPIED

				# Ray tracing: mark cells along the ray as free
				self._ray_trace(robot_px, robot_py, obs_px, obs_py)

	def _ray_trace(self, x0, y0, x1, y1):
		"""
		Bresenham's line algorithm to mark free cells along ray
		"""
		dx = abs(x1 - x0)
		dy = abs(y1 - y0)
		sx = 1 if x0 < x1 else -1
		sy = 1 if y0 < y1 else -1
		err = dx - dy

		x, y = x0, y0

		while True:
			# Mark as free (but don't overwrite occupied cells)
			if self.is_valid_pixel(x, y) and self.occupancy_map[y, x] == UNKNOWN:
				self.occupancy_map[y, x] = FREE

			# Stop before reaching obstacle
			if x == x1 and y == y1:
				break

			e2 = 2 * err
			if e2 > -dy:
				err -= dy
				x += sx
			if e2 < dx:
				err += dx
				y += sy

	def get_visualization(self, robot_x, robot_y, robot_theta, current_lidar_data=None):
		"""
		Create RGB visualization of the map

		Returns:
			RGB image showing map + robot + trajectory
		"""
		# Convert grayscale map to RGB
		vis_map = cv2.cvtColor(self.occupancy_map, cv2.COLOR_GRAY2BGR)

		# Draw robot trajectory (blue line)
		if len(self.robot_trajectory) > 1:
			points = np.array(self.robot_trajectory, dtype=np.int32)
			cv2.polylines(vis_map, [points], False, (255, 0, 0), 1)

		# Draw current lidar scan (red dots)
		if current_lidar_data:
			robot_px, robot_py = self.world_to_map(robot_x, robot_y)
			num_points = len(current_lidar_data)
			angle_step = 2 * math.pi / num_points

			for i, distance in enumerate(current_lidar_data):
				if distance < 7.9:  # Valid reading
					lidar_angle = i * angle_step
					world_angle = robot_theta + lidar_angle

					obs_x = robot_x + distance * math.cos(world_angle)
					obs_y = robot_y + distance * math.sin(world_angle)

					obs_px, obs_py = self.world_to_map(obs_x, obs_y)

					if self.is_valid_pixel(obs_px, obs_py):
						cv2.circle(vis_map, (obs_px, obs_py), 1, LIDAR_POINT, -1)

		# Draw robot position and heading
		robot_px, robot_py = self.world_to_map(robot_x, robot_y)
		if self.is_valid_pixel(robot_px, robot_py):
			# Robot body (green circle)
			cv2.circle(vis_map, (robot_px, robot_py), 5, ROBOT, -1)

			# Heading indicator (green arrow)
			arrow_length = 15
			end_x = int(robot_px + arrow_length * math.cos(robot_theta))
			end_y = int(robot_py + arrow_length * math.sin(robot_theta))
			cv2.arrowedLine(vis_map, (robot_px, robot_py), (end_x, end_y), 
						  ROBOT, 2, tipLength=0.3)

		# Flip vertically for correct orientation (OpenCV y-axis is inverted)
		vis_map = cv2.flip(vis_map, 0)

		return vis_map

	def save_map(self, filename="map_output.png"):
		"""Save the current map to file"""
		# Flip for correct orientation
		map_to_save = cv2.flip(self.occupancy_map, 0)
		cv2.imwrite(filename, map_to_save)
		print(f"Map saved to: {filename}")

	def save_map_with_trajectory(self, robot_x, robot_y, robot_theta, filename="map_with_trajectory.png"):
		"""Save map with robot trajectory"""
		vis = self.get_visualization(robot_x, robot_y, robot_theta)
		cv2.imwrite(filename, vis)
		print(f"Map with trajectory saved to: {filename}")


# =============================================================================
# MAIN CONTROLLER
# =============================================================================

class EnhancedRosBotController:
	"""Enhanced controller with persistent mapping"""

	def __init__(self):
		# Initialize Supervisor
		self.supervisor = Supervisor()

		# Enable keyboard
		self.keyboard = self.supervisor.getKeyboard()
		self.keyboard.enable(TIME_STEP)

		# Initialize sensors
		self._init_sensors()

		# Initialize motors
		self._init_motors()

		# Initialize map
		self.map_manager = PersistentMapManager(MAP_SIZE_METERS, MAP_RESOLUTION)

		# Motor speeds
		self.left_speed = 0
		self.right_speed = 0

		# Last save time
		self.last_save_time = time.time()

		print("\nEnhanced RosBot Controller Initialized")
		print("=" * 60)

	def _init_sensors(self):
		"""Initialize all sensors"""
		self.lidar = self.supervisor.getDevice('lidar')
		self.lidar.enable(TIME_STEP)
		self.lidar.enablePointCloud()
		print("✓ Lidar enabled")

		self.camera = self.supervisor.getDevice('camera rgb')
		self.camera.enable(TIME_STEP)
		print("✓ Camera enabled")

		self.depth = self.supervisor.getDevice('camera depth')
		self.depth.enable(TIME_STEP)
		print("✓ Depth sensor enabled")

	def _init_motors(self):
		"""Initialize motors"""
		self.front_left_motor = self.supervisor.getDevice('front left wheel motor')
		self.front_right_motor = self.supervisor.getDevice('front right wheel motor')
		self.rear_left_motor = self.supervisor.getDevice('rear left wheel motor')
		self.rear_right_motor = self.supervisor.getDevice('rear right wheel motor')

		# Set to velocity control
		self.front_left_motor.setPosition(float('inf'))
		self.front_right_motor.setPosition(float('inf'))
		self.rear_left_motor.setPosition(float('inf'))
		self.rear_right_motor.setPosition(float('inf'))

		# Initialize to zero
		self.set_motor_speeds(0, 0)
		print("✓ Motors initialized")

	def get_robot_pose(self):
		"""
		Get robot pose using Supervisor
		Returns: (x, y, theta) in world coordinates
		"""
		robot_node = self.supervisor.getSelf()
		if not robot_node:
			return (0, 0, 0)

		# Get position
		position = robot_node.getPosition()

		# Get orientation (rotation matrix)
		orientation = robot_node.getOrientation()

		# Extract yaw angle from rotation matrix
		# orientation[0], orientation[1], orientation[2] = first row
		# orientation[3], orientation[4], orientation[5] = second row
		theta = math.atan2(orientation[3], orientation[0])

		return (position[0], position[1], theta)

	def set_motor_speeds(self, left, right):
		"""Set speeds for all motors"""
		self.front_left_motor.setVelocity(left)
		self.rear_left_motor.setVelocity(left)
		self.front_right_motor.setVelocity(right)
		self.rear_right_motor.setVelocity(right)

	def process_keyboard_input(self):
		"""Handle keyboard input for robot control"""
		key = self.keyboard.getKey()

		# Reset speeds if no key is pressed
		if key == -1:
			self.left_speed = 0
			self.right_speed = 0
		else:
			# Forward
			if key == ord('W'):
				self.left_speed = min(self.left_speed + SPEED_INCREMENT, MAX_SPEED)
				self.right_speed = min(self.right_speed + SPEED_INCREMENT, MAX_SPEED)
			# Backward
			elif key == ord('S'):
				self.left_speed = max(self.left_speed - SPEED_INCREMENT, -MAX_SPEED)
				self.right_speed = max(self.right_speed - SPEED_INCREMENT, -MAX_SPEED)
			# Turn left
			elif key == ord('A'):
				self.left_speed = max(self.left_speed - SPEED_INCREMENT, -MAX_SPEED)
				self.right_speed = min(self.right_speed + SPEED_INCREMENT, MAX_SPEED)
			# Turn right
			elif key == ord('D'):
				self.left_speed = min(self.left_speed + SPEED_INCREMENT, MAX_SPEED)
				self.right_speed = max(self.right_speed - SPEED_INCREMENT, -MAX_SPEED)
			# Emergency stop
			elif key == ord(' '):
				self.left_speed = 0
				self.right_speed = 0
			# Save map
			elif key == ord('M'):
				robot_x, robot_y, robot_theta = self.get_robot_pose()
				self.map_manager.save_map_with_trajectory(robot_x, robot_y, robot_theta)

		self.set_motor_speeds(self.left_speed, self.right_speed)

	def run(self):
		"""Main control loop"""
		print("\n" + "=" * 60)
		print("CONTROLS:")
		print("  W - Move forward")
		print("  S - Move backward")
		print("  A - Turn left")
		print("  D - Turn right")
		print("  Space - Emergency stop")
		print("  M - Save map to file")
		print("  ESC - Exit and save final map")
		print("=" * 60 + "\n")

		# Create windows
		cv2.namedWindow('Camera View', cv2.WINDOW_NORMAL)
		cv2.namedWindow('Persistent 2D Map', cv2.WINDOW_NORMAL)

		frame_count = 0

		while self.supervisor.step(TIME_STEP) != -1:
			# Process keyboard input
			self.process_keyboard_input()

			# Get robot pose
			robot_x, robot_y, robot_theta = self.get_robot_pose()

			# Get lidar data
			lidar_data = self.lidar.getRangeImage()
			max_range = self.lidar.getMaxRange()

			# Update map every frame
			self.map_manager.update_map(robot_x, robot_y, robot_theta, lidar_data, max_range)

			# Get camera image
			camera_image = np.frombuffer(self.camera.getImage(), np.uint8).reshape(
				(self.camera.getHeight(), self.camera.getWidth(), 4))
			camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGRA2BGR)

			# Get map visualization
			map_view = self.map_manager.get_visualization(robot_x, robot_y, robot_theta, lidar_data)

			# Add info overlay on map
			info_text = [
				f"Position: ({robot_x:.2f}, {robot_y:.2f})",
				f"Heading: {math.degrees(robot_theta):.1f}°",
				f"Speed: L={self.left_speed:.1f} R={self.right_speed:.1f}",
				f"Frame: {frame_count}"
			]

			for i, text in enumerate(info_text):
				cv2.putText(map_view, text, (10, 20 + i*20), 
						   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

			# Display views
			cv2.imshow('Camera View', camera_image)
			cv2.imshow('Persistent 2D Map', map_view)

			# Auto-save every 30 seconds
			current_time = time.time()
			if current_time - self.last_save_time > 30:
				self.map_manager.save_map_with_trajectory(robot_x, robot_y, robot_theta, 
														 f"autosave_map_{int(current_time)}.png")
				self.last_save_time = current_time

			# Check for exit
			if cv2.waitKey(1) & 0xFF == 27:  # ESC key
				print("\nExiting and saving final map...")
				self.map_manager.save_map("final_map.png")
				self.map_manager.save_map_with_trajectory(robot_x, robot_y, robot_theta, 
														 "final_map_with_trajectory.png")
				break

			frame_count += 1

		# Cleanup
		cv2.destroyAllWindows()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
	controller = EnhancedRosBotController()
	controller.run()


if __name__ == "__main__":
	main()