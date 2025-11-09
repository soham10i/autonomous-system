#!/usr/bin/env python3
"""
Pillar Detection Node - HSV Color-based Detection
Detects blue and yellow pillars using camera and color segmentation
Author: RosBot Navigation Team
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import math

class PillarDetector:
    """Color-based pillar detection using HSV segmentation"""

    def __init__(self):
        rospy.init_node('pillar_detector_node', anonymous=True)

        # OpenCV bridge for ROS-OpenCV conversion
        self.bridge = CvBridge()

        # Detection parameters
        self.min_contour_area = rospy.get_param('~min_contour_area', 500)
        self.detection_confidence = rospy.get_param('~detection_confidence', 0.7)
        self.detection_distance_threshold = rospy.get_param('~detection_distance', 1.0)  # 1 meter

        # HSV color ranges for pillar detection
        # Blue pillar HSV range
        self.blue_lower = np.array([100, 50, 50])   # Lower blue threshold
        self.blue_upper = np.array([130, 255, 255]) # Upper blue threshold

        # Yellow pillar HSV range  
        self.yellow_lower = np.array([20, 100, 100])  # Lower yellow threshold
        self.yellow_upper = np.array([30, 255, 255])  # Upper yellow threshold

        # State variables
        self.camera_info = None
        self.last_detection = None
        self.detection_count = {"blue": 0, "yellow": 0}
        self.min_detections = 5  # Require multiple detections for confidence

        # Publishers
        self.pillar_detected_pub = rospy.Publisher('/pillar_detected', String, queue_size=1)
        self.pillar_position_pub = rospy.Publisher('/pillar_position', Point, queue_size=1)
        self.debug_image_pub = rospy.Publisher('/pillar_debug_image', Image, queue_size=1)

        # Subscribers
        rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        rospy.Subscriber('/camera/camera_info', CameraInfo, self.camera_info_callback)

        rospy.loginfo("Pillar Detector initialized")

    def camera_info_callback(self, msg):
        """Store camera calibration information"""
        self.camera_info = msg

    def image_callback(self, msg):
        """Process camera images for pillar detection"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Detect pillars
            detected_pillar, pillar_center, debug_image = self.detect_pillars(cv_image)

            # Publish debug image
            if debug_image is not None:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                self.debug_image_pub.publish(debug_msg)

            # Process detection results
            if detected_pillar:
                self.process_detection(detected_pillar, pillar_center)
            else:
                # Decay detection counts
                for color in self.detection_count:
                    self.detection_count[color] = max(0, self.detection_count[color] - 1)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def detect_pillars(self, image):
        """Detect blue and yellow pillars in image"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create debug image
        debug_image = image.copy()

        detected_pillar = None
        pillar_center = None
        max_area = 0

        # Detect blue pillar
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        blue_result = self.process_color_mask(blue_mask, "blue", debug_image)

        if blue_result and blue_result[1] > max_area:
            detected_pillar = "blue"
            pillar_center = blue_result[0]
            max_area = blue_result[1]

        # Detect yellow pillar
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        yellow_result = self.process_color_mask(yellow_mask, "yellow", debug_image)

        if yellow_result and yellow_result[1] > max_area:
            detected_pillar = "yellow"
            pillar_center = yellow_result[0]
            max_area = yellow_result[1]

        return detected_pillar, pillar_center, debug_image

    def process_color_mask(self, mask, color_name, debug_image):
        """Process color mask to find pillar contours"""
        # Apply morphological operations to clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Check minimum area threshold
        if area < self.min_contour_area:
            return None

        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Draw detection on debug image
        cv2.drawContours(debug_image, [largest_contour], -1, (0, 255, 0), 2)
        cv2.circle(debug_image, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(debug_image, f"{color_name}: {area:.0f}", (cx-50, cy-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return ((cx, cy), area)

    def process_detection(self, pillar_color, pillar_center):
        """Process pillar detection and publish results"""
        # Increment detection count
        self.detection_count[pillar_color] += 1

        # Check if we have enough confident detections
        if self.detection_count[pillar_color] >= self.min_detections:
            # Reset other color counts
            for color in self.detection_count:
                if color != pillar_color:
                    self.detection_count[color] = 0

            # Publish detection
            if self.last_detection != pillar_color:
                self.last_detection = pillar_color
                rospy.loginfo(f"✅ {pillar_color.upper()} PILLAR DETECTED!")

                # Publish pillar detection message
                self.pillar_detected_pub.publish(String(data=pillar_color))

                # Estimate 3D position (simplified)
                if self.camera_info and pillar_center:
                    pillar_3d_pos = self.estimate_pillar_position(pillar_center)
                    if pillar_3d_pos:
                        self.pillar_position_pub.publish(pillar_3d_pos)

    def estimate_pillar_position(self, image_center):
        """Estimate 3D position of pillar (simplified approach)"""
        if not self.camera_info:
            return None

        # Simplified position estimation
        # In a real system, this would use stereo vision or known pillar size

        # Get image center and pillar center
        img_center_x = self.camera_info.width / 2
        img_center_y = self.camera_info.height / 2

        pillar_x, pillar_y = image_center

        # Calculate angular offset
        fx = self.camera_info.K[0]  # Focal length x
        fy = self.camera_info.K[4]  # Focal length y

        # Horizontal and vertical angles
        angle_x = math.atan2(pillar_x - img_center_x, fx)
        angle_y = math.atan2(pillar_y - img_center_y, fy)

        # Estimate distance (rough approximation)
        # This assumes pillars have a known size and uses image size
        estimated_distance = 1.5  # meters (rough estimate)

        # Convert to 3D position in camera frame
        pos = Point()
        pos.x = estimated_distance * math.sin(angle_x)
        pos.y = -estimated_distance * math.sin(angle_y)  # Negative for camera frame
        pos.z = estimated_distance * math.cos(angle_x)

        return pos

if __name__ == '__main__':
    try:
        detector = PillarDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
