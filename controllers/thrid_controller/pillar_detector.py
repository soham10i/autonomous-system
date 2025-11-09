
import cv2
import numpy as np

class PillarDetector:
    def __init__(self):
        """
        Initializes the PillarDetector class.
        Defines color ranges for blue and yellow pillars in HSV.
        These values might need to be tuned based on the Webots environment lighting.
        """
        # HSV color ranges for blue
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])

        # HSV color ranges for yellow
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])

        self.detected_pillars = {}

    def detect_pillars(self, image):
        """
        Detects colored pillars (blue and yellow) in the input image.

        :param image: A NumPy array representing the camera image (RGB or BGR).
        :return: A dictionary of detected pillars: {color: (x_center, y_center, radius)}
        """
        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only blue colors
        mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # Threshold the HSV image to get only yellow colors
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        # Find contours in the masks
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_detections = {}

        # Process blue contours
        for contour in contours_blue:
            # Approximate contour to a circle to find its center and radius
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Filter by size (e.g., minimum radius to avoid noise)
            if radius > 10:  # Adjust this threshold as needed
                current_detections["blue"] = (center[0], center[1], radius)
                break # Assume only one blue pillar for now

        # Process yellow contours
        for contour in contours_yellow:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            if radius > 10:  # Adjust this threshold as needed
                current_detections["yellow"] = (center[0], center[1], radius)
                break # Assume only one yellow pillar for now

        self.detected_pillars = current_detections
        return self.detected_pillars

    def get_pillar_positions(self):
        """
        Returns the last detected pillar positions.
        """
        return self.detected_pillars


