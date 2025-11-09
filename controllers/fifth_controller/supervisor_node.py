#!/usr/bin/env python3
"""
Supervisor Node - Webots Supervisor Interface
Publishes ground truth robot pose from Webots Supervisor
Author: RosBot Navigation Team
"""

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from controller import Robot, Supervisor
import tf.transformations
import time

class SupervisorNode:
    """Interface with Webots Supervisor to get robot ground truth pose"""

    def __init__(self):
        # Initialize ROS node
        rospy.init_node('supervisor_node', anonymous=True)

        # Initialize Webots Supervisor
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())

        # Get robot node from Webots
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        if self.robot_node is None:
            rospy.logerr("Robot node not found! Make sure robot is defined with DEF ROBOT")
            return

        # Publishers
        self.pose_pub = rospy.Publisher('/pose', PoseStamped, queue_size=10)

        # Node parameters
        self.publish_rate = rospy.get_param('~publish_rate', 20.0)  # 20 Hz

        rospy.loginfo("Supervisor Node initialized - Publishing robot pose")

    def run(self):
        """Main loop - publish robot pose"""
        rate = rospy.Rate(self.publish_rate)

        while not rospy.is_shutdown():
            # Step Webots simulation
            if self.supervisor.step(self.timestep) == -1:
                break

            # Get robot pose from Webots
            pose_msg = self.get_robot_pose()
            if pose_msg:
                self.pose_pub.publish(pose_msg)

            rate.sleep()

    def get_robot_pose(self):
        """Get robot pose from Webots Supervisor"""
        try:
            # Get position and orientation from Webots
            position = self.robot_node.getPosition()
            orientation = self.robot_node.getOrientation()

            # Convert to ROS PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "world"

            # Position (x, y, z)
            pose_msg.pose.position.x = position[0]
            pose_msg.pose.position.y = position[1]
            pose_msg.pose.position.z = position[2]

            # Convert rotation matrix to quaternion
            # Webots orientation is 3x3 rotation matrix (row-major)
            rotation_matrix = [
                [orientation[0], orientation[1], orientation[2]],
                [orientation[3], orientation[4], orientation[5]],
                [orientation[6], orientation[7], orientation[8]]
            ]

            # Extract yaw angle for 2D navigation
            yaw = tf.transformations.euler_from_matrix([
                [orientation[0], orientation[1], orientation[2], 0],
                [orientation[3], orientation[4], orientation[5], 0],
                [orientation[6], orientation[7], orientation[8], 0],
                [0, 0, 0, 1]
            ])[2]

            # Convert yaw to quaternion
            quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]

            return pose_msg

        except Exception as e:
            rospy.logerr(f"Error getting robot pose: {e}")
            return None

if __name__ == '__main__':
    try:
        supervisor = SupervisorNode()
        supervisor.run()
    except rospy.ROSInterruptException:
        pass
