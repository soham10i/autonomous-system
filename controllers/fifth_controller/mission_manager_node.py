#!/usr/bin/env python3
"""
Mission Manager Node - State Machine for Navigation Task
Orchestrates the complete mission: Start -> Blue Pillar -> Yellow Pillar
Author: RosBot Navigation Team
"""

import rospy
import time
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class MissionState:
    """Mission state enumeration"""
    INITIALIZING = "initializing"
    NAVIGATING_TO_BLUE = "navigating_to_blue"
    REACHED_BLUE = "reached_blue"
    NAVIGATING_TO_YELLOW = "navigating_to_yellow"
    REACHED_YELLOW = "reached_yellow"
    MISSION_COMPLETE = "mission_complete"
    ERROR = "error"

class MissionManager:
    """Mission state machine for pillar navigation task"""

    def __init__(self):
        rospy.init_node('mission_manager_node', anonymous=True)

        # Mission parameters
        self.pillar_detection_timeout = rospy.get_param('~pillar_detection_timeout', 30.0)  # 30 seconds
        self.pause_duration = rospy.get_param('~pause_duration', 2.0)  # 2 second pause at blue pillar

        # State variables
        self.current_state = MissionState.INITIALIZING
        self.mission_start_time = None
        self.blue_pillar_time = None
        self.yellow_pillar_time = None
        self.state_start_time = None

        # Detection tracking
        self.current_pillar_detected = None
        self.path_complete = False

        # Publishers
        self.mission_target_pub = rospy.Publisher('/mission_target', String, queue_size=1, latch=True)
        self.controller_enable_pub = rospy.Publisher('/controller_enable', Bool, queue_size=1)
        self.mission_status_pub = rospy.Publisher('/mission_status', String, queue_size=1)

        # Subscribers
        rospy.Subscriber('/pillar_detected', String, self.pillar_detected_callback)
        rospy.Subscriber('/path_complete', Bool, self.path_complete_callback)
        rospy.Subscriber('/pose', PoseStamped, self.pose_callback)

        # Mission timer
        self.state_timer = rospy.Timer(rospy.Duration(0.5), self.state_machine_update)  # 2Hz

        # Initialize mission
        self.initialize_mission()

        rospy.loginfo("Mission Manager initialized")

    def initialize_mission(self):
        """Initialize the mission"""
        self.mission_start_time = time.time()
        self.change_state(MissionState.NAVIGATING_TO_BLUE)

        # Start navigation to blue pillar
        self.mission_target_pub.publish(String(data="blue"))
        self.controller_enable_pub.publish(Bool(data=True))

        rospy.loginfo("🚀 MISSION STARTED - Navigating to BLUE pillar")

    def change_state(self, new_state):
        """Change mission state with logging"""
        if self.current_state != new_state:
            rospy.loginfo(f"State transition: {self.current_state} -> {new_state}")
            self.current_state = new_state
            self.state_start_time = time.time()
            self.mission_status_pub.publish(String(data=new_state))

    def pillar_detected_callback(self, msg):
        """Handle pillar detection"""
        detected_color = msg.data.lower()
        self.current_pillar_detected = detected_color
        rospy.loginfo(f"Pillar detected: {detected_color}")

    def path_complete_callback(self, msg):
        """Handle path completion"""
        self.path_complete = msg.data
        if self.path_complete:
            rospy.loginfo("Path completion received")

    def pose_callback(self, msg):
        """Handle robot pose updates"""
        # Can be used for additional logic if needed
        pass

    def get_elapsed_time(self):
        """Get total mission elapsed time"""
        if self.mission_start_time:
            return time.time() - self.mission_start_time
        return 0

    def get_state_elapsed_time(self):
        """Get time elapsed in current state"""
        if self.state_start_time:
            return time.time() - self.state_start_time
        return 0

    def log_timing_milestone(self, milestone):
        """Log timing milestones"""
        elapsed = self.get_elapsed_time()
        rospy.loginfo(f"🕐 TIMING: {milestone} at {elapsed:.2f} seconds")

        if milestone == "Blue Pillar Reached":
            self.blue_pillar_time = elapsed
        elif milestone == "Yellow Pillar Reached":
            self.yellow_pillar_time = elapsed

    def state_machine_update(self, event):
        """Main state machine update"""

        if self.current_state == MissionState.NAVIGATING_TO_BLUE:
            self.handle_navigating_to_blue()

        elif self.current_state == MissionState.REACHED_BLUE:
            self.handle_reached_blue()

        elif self.current_state == MissionState.NAVIGATING_TO_YELLOW:
            self.handle_navigating_to_yellow()

        elif self.current_state == MissionState.REACHED_YELLOW:
            self.handle_reached_yellow()

        elif self.current_state == MissionState.MISSION_COMPLETE:
            self.handle_mission_complete()

    def handle_navigating_to_blue(self):
        """Handle navigation to blue pillar"""
        # Check for blue pillar detection
        if self.current_pillar_detected == "blue":
            # Stop robot and log timing
            self.controller_enable_pub.publish(Bool(data=False))
            self.log_timing_milestone("Blue Pillar Reached")
            self.change_state(MissionState.REACHED_BLUE)
            return

        # Check for timeout
        if self.get_state_elapsed_time() > self.pillar_detection_timeout:
            rospy.logwarn("⚠️ Timeout waiting for blue pillar detection")
            # Continue anyway - maybe we missed the detection
            self.change_state(MissionState.NAVIGATING_TO_YELLOW)
            self.mission_target_pub.publish(String(data="yellow"))

    def handle_reached_blue(self):
        """Handle reaching blue pillar"""
        # Pause at blue pillar
        if self.get_state_elapsed_time() >= self.pause_duration:
            rospy.loginfo("🔄 Continuing to YELLOW pillar")

            # Start navigation to yellow pillar
            self.mission_target_pub.publish(String(data="yellow"))
            self.controller_enable_pub.publish(Bool(data=True))
            self.change_state(MissionState.NAVIGATING_TO_YELLOW)

    def handle_navigating_to_yellow(self):
        """Handle navigation to yellow pillar"""
        # Check for yellow pillar detection
        if self.current_pillar_detected == "yellow":
            # Stop robot and log timing
            self.controller_enable_pub.publish(Bool(data=False))
            self.log_timing_milestone("Yellow Pillar Reached")
            self.change_state(MissionState.REACHED_YELLOW)
            return

        # Check for timeout
        if self.get_state_elapsed_time() > self.pillar_detection_timeout:
            rospy.logwarn("⚠️ Timeout waiting for yellow pillar detection")
            self.change_state(MissionState.ERROR)

    def handle_reached_yellow(self):
        """Handle reaching yellow pillar"""
        # Mission complete
        self.change_state(MissionState.MISSION_COMPLETE)

    def handle_mission_complete(self):
        """Handle mission completion"""
        # Log final results
        total_time = self.get_elapsed_time()

        rospy.loginfo("🎉 MISSION COMPLETED SUCCESSFULLY!")
        rospy.loginfo("=" * 50)
        rospy.loginfo("📊 MISSION TIMING RESULTS:")

        if self.blue_pillar_time:
            rospy.loginfo(f"   Start → Blue Pillar: {self.blue_pillar_time:.2f} seconds")

        if self.yellow_pillar_time and self.blue_pillar_time:
            blue_to_yellow = self.yellow_pillar_time - self.blue_pillar_time
            rospy.loginfo(f"   Blue → Yellow Pillar: {blue_to_yellow:.2f} seconds")

        rospy.loginfo(f"   Total Mission Time: {total_time:.2f} seconds")
        rospy.loginfo("=" * 50)

        # Save results to file
        self.save_mission_results()

    def save_mission_results(self):
        """Save mission results to file"""
        try:
            results = {
                "mission_completed": True,
                "total_time": self.get_elapsed_time(),
                "blue_pillar_time": self.blue_pillar_time,
                "yellow_pillar_time": self.yellow_pillar_time
            }

            # Write results to file
            import json
            import os

            results_dir = os.path.expanduser("~/mission_results")
            os.makedirs(results_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{results_dir}/mission_results_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)

            rospy.loginfo(f"📁 Results saved: {filename}")

        except Exception as e:
            rospy.logerr(f"Error saving results: {e}")

if __name__ == '__main__':
    try:
        mission_manager = MissionManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
