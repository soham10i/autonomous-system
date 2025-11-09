#!/usr/bin/env python3
"""
Controller Node - Pure Pursuit Path Following
Controls robot to follow planned path using ground truth pose
Author: RosBot Navigation Team
"""

import rospy
import math
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool

class PurePursuitController:
    """Pure pursuit path following controller"""

    def __init__(self):
        rospy.init_node('controller_node', anonymous=True)

        # Control parameters
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 0.5)  # 50cm
        self.max_linear_speed = rospy.get_param('~max_linear_speed', 0.3)      # 30cm/s
        self.max_angular_speed = rospy.get_param('~max_angular_speed', 1.0)    # 1 rad/s
        self.path_tolerance = rospy.get_param('~path_tolerance', 0.2)          # 20cm

        # State variables
        self.current_pose = None
        self.current_path = None
        self.current_waypoint_index = 0
        self.path_complete = False
        self.enabled = True

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.path_complete_pub = rospy.Publisher('/path_complete', Bool, queue_size=1)

        # Subscribers
        rospy.Subscriber('/pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/path', Path, self.path_callback)
        rospy.Subscriber('/controller_enable', Bool, self.enable_callback)

        # Control loop timer
        self.control_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)  # 20Hz

        rospy.loginfo("Pure Pursuit Controller initialized")

    def pose_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = msg

    def path_callback(self, msg):
        """Update path to follow"""
        self.current_path = msg
        self.current_waypoint_index = 0
        self.path_complete = False
        rospy.loginfo(f"New path received: {len(msg.poses)} waypoints")

    def enable_callback(self, msg):
        """Enable/disable controller"""
        self.enabled = msg.data
        if not self.enabled:
            # Stop robot when disabled
            self.publish_zero_velocity()
        rospy.loginfo(f"Controller {'enabled' if self.enabled else 'disabled'}")

    def publish_zero_velocity(self):
        """Publish zero velocity command"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def get_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        dx = pose1.pose.position.x - pose2.pose.position.x
        dy = pose1.pose.position.y - pose2.pose.position.y
        return math.sqrt(dx*dx + dy*dy)

    def get_yaw_from_pose(self, pose):
        """Extract yaw angle from pose quaternion"""
        quat = pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def find_target_waypoint(self):
        """Find target waypoint using lookahead distance"""
        if not self.current_pose or not self.current_path:
            return None

        # Start from current waypoint index
        for i in range(self.current_waypoint_index, len(self.current_path.poses)):
            waypoint = self.current_path.poses[i]
            distance = self.get_distance(self.current_pose, waypoint)

            # Update current waypoint if we're close enough
            if i == self.current_waypoint_index and distance < self.path_tolerance:
                self.current_waypoint_index = min(i + 1, len(self.current_path.poses) - 1)

            # Return waypoint at lookahead distance
            if distance >= self.lookahead_distance:
                return waypoint

        # Return last waypoint if no waypoint at lookahead distance
        if self.current_path.poses:
            return self.current_path.poses[-1]

        return None

    def pure_pursuit_control(self, target_waypoint):
        """Calculate control commands using pure pursuit algorithm"""
        if not self.current_pose:
            return Twist()

        # Get current robot position and orientation
        robot_x = self.current_pose.pose.position.x
        robot_y = self.current_pose.pose.position.y
        robot_yaw = self.get_yaw_from_pose(self.current_pose)

        # Get target position
        target_x = target_waypoint.pose.position.x
        target_y = target_waypoint.pose.position.y

        # Transform target to robot frame
        dx = target_x - robot_x
        dy = target_y - robot_y

        # Rotate to robot frame
        target_x_robot = dx * math.cos(robot_yaw) + dy * math.sin(robot_yaw)
        target_y_robot = -dx * math.sin(robot_yaw) + dy * math.cos(robot_yaw)

        # Pure pursuit calculations
        L = math.sqrt(target_x_robot**2 + target_y_robot**2)

        if L < 0.01:  # Avoid division by zero
            cmd = Twist()
            return cmd

        # Calculate curvature
        curvature = 2 * target_y_robot / (L**2)

        # Calculate velocities
        cmd = Twist()

        # Linear velocity (reduce when turning sharply)
        cmd.linear.x = self.max_linear_speed * (1.0 - abs(curvature) * 0.5)
        cmd.linear.x = max(0.1, min(self.max_linear_speed, cmd.linear.x))

        # Angular velocity
        cmd.angular.z = curvature * cmd.linear.x
        cmd.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, cmd.angular.z))

        return cmd

    def check_path_completion(self):
        """Check if path is completed"""
        if (not self.current_path or 
            not self.current_pose or
            len(self.current_path.poses) == 0):
            return False

        # Check if we're close to the final waypoint
        final_waypoint = self.current_path.poses[-1]
        distance_to_goal = self.get_distance(self.current_pose, final_waypoint)

        if distance_to_goal < self.path_tolerance:
            return True

        return False

    def control_loop(self, event):
        """Main control loop"""
        if not self.enabled:
            return

        # Check path completion
        if self.check_path_completion():
            if not self.path_complete:
                self.path_complete = True
                self.publish_zero_velocity()
                self.path_complete_pub.publish(Bool(data=True))
                rospy.loginfo("Path completed!")
            return

        # Find target waypoint
        target_waypoint = self.find_target_waypoint()

        if target_waypoint:
            # Calculate and publish control commands
            cmd = self.pure_pursuit_control(target_waypoint)
            self.cmd_vel_pub.publish(cmd)
        else:
            # No valid target, stop robot
            self.publish_zero_velocity()

if __name__ == '__main__':
    try:
        controller = PurePursuitController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
