from controller import Robot, Lidar, Camera
import numpy as np

# Constants
TIME_STEP = 64
MAX_SPEED = 6.28

# Color thresholds (these might need fine-tuning)
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([30, 255, 255])
BLUE_LOWER = np.array([100, 100, 100])
BLUE_UPPER = np.array([120, 255, 255])


class CarController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Get motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Get Lidar
        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.timestep)
        # Check if Lidar is a RangeFinder or Lidar type
        if self.lidar.getType() == Lidar.RANGE_FINDER:
            self.lidar_type = "range_finder"
            self.lidar.enable(self.timestep)
        elif self.lidar.getType() == Lidar.LIDAR:
            self.lidar_type = "lidar"
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
        else:
            print("Unsupported Lidar type.")
            self.lidar_type = "none"

        # Get Camera
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)

        self.state = 'SEARCHING_YELLOW'
        self.target_pillar_color = 'YELLOW'    
    
    def get_image_hsv(self):
        # Get camera image and convert to HSV
        img = self.camera.getImage()
        if img is None:
            return None
        
        # Webots camera image is RGB, convert to numpy array
        img_array = np.frombuffer(img, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
        rgb_image = img_array[:, :, :3] # Extract RGB channels

        # Manual RGB to HSV conversion
        hsv_image = np.zeros_like(rgb_image, dtype=np.float32)
        
        R, G, B = rgb_image[:,:,0] / 255.0, rgb_image[:,:,1] / 255.0, rgb_image[:,:,2] / 255.0

        Cmax = np.maximum(np.maximum(R, G), B)
        Cmin = np.minimum(np.minimum(R, G), B)
        delta = Cmax - Cmin

        # Hue calculation
        H = np.zeros_like(R)
        H[delta == 0] = 0
        H[(Cmax == R) & (delta != 0)] = (60 * (((G - B) / delta) % 6))[(Cmax == R) & (delta != 0)]
        H[(Cmax == G) & (delta != 0)] = (60 * (((B - R) / delta) + 2))[(Cmax == G) & (delta != 0)]
        H[(Cmax == B) & (delta != 0)] = (60 * (((R - G) / delta) + 4))[(Cmax == B) & (delta != 0)]
        H[H < 0] += 360 # Ensure hue is positive

        # Saturation calculation
        S = np.zeros_like(R)
        S[Cmax != 0] = (delta / Cmax)[Cmax != 0]

        # Value calculation
        V = Cmax

        hsv_image[:,:,0] = H / 2 # Hue in Webots is 0-180, so divide by 2
        hsv_image[:,:,1] = S * 255 # Saturation 0-255
        hsv_image[:,:,2] = V * 255 # Value 0-255

        return hsv_image.astype(np.uint8)

    def detect_pillar(self, image_hsv, lower_bound, upper_bound):
        if image_hsv is None:
            return False, 0

        # Apply HSV thresholding
        mask = (image_hsv[:, :, 0] >= lower_bound[0]) & (image_hsv[:, :, 0] <= upper_bound[0]) & \
               (image_hsv[:, :, 1] >= lower_bound[1]) & (image_hsv[:, :, 1] <= upper_bound[1]) & \
               (image_hsv[:, :, 2] >= lower_bound[2]) & (image_hsv[:, :, 2] <= upper_bound[2])

        if np.any(mask):
            # Calculate centroid of detected color
            y_coords, x_coords = np.where(mask)
            if len(x_coords) > 0:
                center_x = np.mean(x_coords)
                # Normalize center_x to be between -1 (left) and 1 (right)
                normalized_center_x = (center_x / self.camera.getWidth()) * 2 - 1
                return True, normalized_center_x
        return False, 0

    def obstacle_avoidance(self):
        if self.lidar_type == "lidar":
            lidar_data = self.lidar.getRangeImage()
        elif self.lidar_type == "range_finder":
            lidar_data = [self.lidar.getValue()]
        else:
            return 0, 0 # No lidar, no avoidance

        if lidar_data is None or len(lidar_data) == 0:
            return 0, 0 # No obstacle avoidance if no lidar data

        # Simple obstacle avoidance: check front and sides
        num_points = len(lidar_data)
        
        # Define sectors for obstacle detection
        front_sector_start = int(num_points * 0.4)
        front_sector_end = int(num_points * 0.6)
        left_sector_start = int(num_points * 0.1)
        left_sector_end = int(num_points * 0.3)
        right_sector_start = int(num_points * 0.7)
        right_sector_end = int(num_points * 0.9)

        front_distances = [d for d in lidar_data[front_sector_start:front_sector_end] if d > 0]
        left_distances = [d for d in lidar_data[left_sector_start:left_sector_end] if d > 0]
        right_distances = [d for d in lidar_data[right_sector_start:right_sector_end] if d > 0]

        front_distance = min(front_distances) if front_distances else float('inf')
        left_distance = min(left_distances) if left_distances else float('inf')
        right_distance = min(right_distances) if right_distances else float('inf')
        
        # Define a safe distance
        SAFE_DISTANCE = 0.5 # meters

        left_speed = MAX_SPEED
        right_speed = MAX_SPEED
        
        if front_distance < SAFE_DISTANCE:
            # Obstacle directly in front, turn
            if left_distance > right_distance: # More space on left
                left_speed = -MAX_SPEED / 2
                right_speed = MAX_SPEED / 2
            else: # More space on right
                left_speed = MAX_SPEED / 2
                right_speed = -MAX_SPEED / 2
        
        return left_speed, right_speed

    def run(self):
        while self.robot.step(self.timestep) != -1:
            left_speed = 0
            right_speed = 0

            image_hsv = self.get_image_hsv()

            if self.state == 'SEARCHING_YELLOW':
                found_yellow, yellow_x = self.detect_pillar(image_hsv, YELLOW_LOWER, YELLOW_UPPER)
                if found_yellow:
                    print("Found yellow pillar!")
                    self.state = 'NAVIGATING_TO_BLUE'
                    self.target_pillar_color = 'BLUE'
                else:
                    # Continue searching, use obstacle avoidance
                    left_speed, right_speed = self.obstacle_avoidance()
                    if left_speed == 0 and right_speed == 0: # If no immediate obstacle, just move forward
                        left_speed = MAX_SPEED
                        right_speed = MAX_SPEED
                    
            elif self.state == 'NAVIGATING_TO_BLUE':
                found_blue, blue_x = self.detect_pillar(image_hsv, BLUE_LOWER, BLUE_UPPER)
                if found_blue:
                    print("Found blue pillar!")
                    # Steer towards blue pillar
                    if blue_x < -0.1: # Blue pillar on left
                        left_speed = MAX_SPEED * 0.5
                        right_speed = MAX_SPEED
                    elif blue_x > 0.1: # Blue pillar on right
                        left_speed = MAX_SPEED
                        right_speed = MAX_SPEED * 0.5
                    else: # Blue pillar in center, move straight
                        left_speed = MAX_SPEED
                        right_speed = MAX_SPEED
                    
                    # Check if close enough to blue pillar (needs Lidar or distance sensor)
                    # For simplicity, let's assume if blue is detected and centered, we are close.
                    # A more robust solution would use Lidar to check distance to the blue pillar.
                    # For now, if blue is found and centered, we consider it reached.
                    
                    # Use Lidar to check distance to the blue pillar
                    lidar_data = None
                    if self.lidar_type == "lidar":
                        lidar_data = self.lidar.getRangeImage()
                    elif self.lidar_type == "range_finder":
                        lidar_data = [self.lidar.getValue()]

                    if lidar_data is not None and len(lidar_data) > 0:
                        # Assuming the blue pillar is directly in front when centered
                        # Find the minimum distance in the front sector
                        num_points = len(lidar_data)
                        front_sector_start = int(num_points * 0.4)
                        front_sector_end = int(num_points * 0.6)
                        front_distances = [d for d in lidar_data[front_sector_start:front_sector_end] if d > 0]
                        front_distance = min(front_distances) if front_distances else float('inf')

                        if abs(blue_x) < 0.1 and front_distance < 0.3: # If centered and very close
                            print("Reached blue pillar!")
                            self.state = 'REACHED_BLUE'
                            left_speed = 0
                            right_speed = 0
                else:
                    # Blue pillar not found, continue searching/avoiding obstacles
                    left_speed, right_speed = self.obstacle_avoidance()
                    if left_speed == 0 and right_speed == 0: # If no immediate obstacle, just move forward
                        left_speed = MAX_SPEED
                        right_speed = MAX_SPEED

            elif self.state == 'REACHED_BLUE':
                left_speed = 0
                right_speed = 0
                print("Task completed: Reached blue pillar.")
                break # Stop the simulation

            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)

if __name__ == '__main__':
    controller = CarController()
    controller.run()


