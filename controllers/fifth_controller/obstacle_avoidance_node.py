#!/usr/bin/env python3
"""
Obstacle Avoidance Node - LIDAR-based Safety Override
Provides emergency obstacle avoidance using Vector Field Histogram approach
Author: RosBot Navigation Team
"""

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

class ObstacleAvoidance:
    """Vector Field Histogram obstacle avoidance"""

    def __init__(self):
        rospy.init_node('obstacle_avoidance_node', anonymous=True)

        # Safety parameters
        self.safe_distance = rospy.get_param('~safe_distance', 0.4)     # 40cm safety distance
        self.critical_distance = rospy.get_param('~critical_distance', 0.2)  # 20cm critical distance
        self.max_speed = rospy.get_param('~max_speed', 0.3)             # Max allowed speed
        self.angular_resolution = rospy.get_param('~angular_resolution', 5)  # 5 degrees

        # State variables
        self.laser_data = None
        self.obstacle_detected = False
        self.emergency_stop = False

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_safe', Twist, queue_size=1)
        self.obstacle_pub = rospy.Publisher('/obstacle_detected', Bool, queue_size=1)

        # Subscribers
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)

        rospy.loginfo("Obstacle Avoidance initialized")

    def laser_callback(self, msg):
        """Process LIDAR data"""
        self.laser_data = msg
        self.detect_obstacles()

    def detect_obstacles(self):
        """Detect obstacles in robot path"""
        if not self.laser_data:
            return

        # Check for obstacles in front sector (±45 degrees)
        ranges = self.laser_data.ranges
        angle_min = self.laser_data.angle_min
        angle_increment = self.laser_data.angle_increment

        front_obstacles = []
        critical_obstacles = []

        for i, distance in enumerate(ranges):
            if math.isnan(distance) or math.isinf(distance):
                continue

            # Calculate angle for this reading
            angle = angle_min + i * angle_increment

            # Check if reading is in front sector
            if abs(angle) <= math.pi/4:  # ±45 degrees
                if distance < self.critical_distance:
                    critical_obstacles.append((angle, distance))
                elif distance < self.safe_distance:
                    front_obstacles.append((angle, distance))

        # Update obstacle detection status
        prev_obstacle = self.obstacle_detected
        prev_emergency = self.emergency_stop

        self.emergency_stop = len(critical_obstacles) > 0
        self.obstacle_detected = len(front_obstacles) > 0 or self.emergency_stop

        # Publish obstacle status
        if self.obstacle_detected != prev_obstacle or self.emergency_stop != prev_emergency:
            self.obstacle_pub.publish(Bool(data=self.obstacle_detected))

            if self.emergency_stop:
                rospy.logwarn("EMERGENCY STOP: Critical obstacle detected!")
            elif self.obstacle_detected:
                rospy.logwarn("Obstacle detected - reducing speed")

    def find_safe_direction(self):
        """Find safest direction using simplified VFH"""
        if not self.laser_data:
            return 0.0  # No turn

        # Create histogram of obstacle densities
        angle_sectors = int(360 / self.angular_resolution)
        histogram = np.zeros(angle_sectors)

        ranges = self.laser_data.ranges
        angle_min = self.laser_data.angle_min
        angle_increment = self.laser_data.angle_increment

        for i, distance in enumerate(ranges):
            if math.isnan(distance) or math.isinf(distance) or distance > self.safe_distance:
                continue

            # Calculate angle and sector
            angle = angle_min + i * angle_increment
            sector = int((math.degrees(angle) + 180) / self.angular_resolution) % angle_sectors

            # Add obstacle density (closer = higher density)
            if distance > 0:
                histogram[sector] += max(0, (self.safe_distance - distance) / distance)

        # Find the safest direction (minimum density)
        safest_sector = np.argmin(histogram)
        safest_angle = math.radians((safest_sector * self.angular_resolution) - 180)

        # Limit to reasonable turning angles (±90 degrees)
        safest_angle = max(-math.pi/2, min(math.pi/2, safest_angle))

        return safest_angle

    def cmd_vel_callback(self, msg):
        """Process velocity commands and apply safety override"""
        if not self.laser_data:
            # No LIDAR data, pass through command
            self.cmd_vel_pub.publish(msg)
            return

        # Create safe velocity command
        safe_cmd = Twist()

        if self.emergency_stop:
            # Emergency stop - all velocities to zero
            safe_cmd.linear.x = 0.0
            safe_cmd.angular.z = 0.0
        elif self.obstacle_detected:
            # Obstacle detected - modify command

            # Reduce linear speed based on obstacle proximity
            speed_factor = 0.3  # Reduce to 30% speed
            safe_cmd.linear.x = msg.linear.x * speed_factor

            # Find safe turning direction
            safe_turn = self.find_safe_direction()

            # Blend original angular command with safe direction
            if abs(safe_turn) > 0.1:  # Significant turning needed
                safe_cmd.angular.z = safe_turn * 0.5  # Moderate turn towards safe direction
            else:
                safe_cmd.angular.z = msg.angular.z * 0.8  # Slightly reduce original turn

            # Ensure velocities are within limits
            safe_cmd.linear.x = max(0.0, min(self.max_speed * 0.5, safe_cmd.linear.x))
            safe_cmd.angular.z = max(-1.0, min(1.0, safe_cmd.angular.z))

        else:
            # No obstacles - pass through with speed limit
            safe_cmd.linear.x = min(self.max_speed, msg.linear.x)
            safe_cmd.angular.z = msg.angular.z

        # Publish safe command
        self.cmd_vel_pub.publish(safe_cmd)

if __name__ == '__main__':
    try:
        obstacle_avoidance = ObstacleAvoidance()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
