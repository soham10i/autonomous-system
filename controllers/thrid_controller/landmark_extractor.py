
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

class LandmarkExtractor:
    def __init__(self, min_points_for_cluster=2, min_points_for_line=2):
        """
        Initializes the LandmarkExtractor class.

        :param min_points_for_cluster: Minimum number of points to form a cluster (for DBSCAN).
        :param min_points_for_line: Minimum number of points to fit a line (for RANSAC).
        """
        self.min_points_for_cluster = min_points_for_cluster
        self.min_points_for_line = min_points_for_line

    def extract_landmarks(self, lidar_points):
        """
        Processes raw Lidar scans to identify distinct, reliable landmarks.

        :param lidar_points: A list of Lidar points in Cartesian coordinates [(x1, y1), (x2, y2), ...].
        :return: A list of landmark observations in a (range, bearing, signature) format.
        """
        if not lidar_points:
            return []

        # 1. Cluster the points using DBSCAN
        clustering = DBSCAN(eps=0.2, min_samples=self.min_points_for_cluster).fit(lidar_points)
        labels = clustering.labels_

        # 2. For each cluster, fit a line using RANSAC
        lines = []
        for label in set(labels):
            if label == -1:  # Ignore noise points
                continue

            cluster_points = np.array(lidar_points)[labels == label]
            if len(cluster_points) < self.min_points_for_line:
                continue

            try:
                # RANSAC needs X and y, so we'll fit x -> y and y -> x and choose the better fit
                # This handles both horizontal and vertical lines better
                ransac_xy = RANSACRegressor(min_samples=self.min_points_for_line).fit(cluster_points[:, 0].reshape(-1, 1), cluster_points[:, 1])
                score_xy = ransac_xy.score(cluster_points[:, 0].reshape(-1, 1), cluster_points[:, 1])

                ransac_yx = RANSACRegressor(min_samples=self.min_points_for_line).fit(cluster_points[:, 1].reshape(-1, 1), cluster_points[:, 0])
                score_yx = ransac_yx.score(cluster_points[:, 1].reshape(-1, 1), cluster_points[:, 0])

                if score_xy >= score_yx:
                    # Line is more horizontal
                    line_points = cluster_points[ransac_xy.inlier_mask_]
                    lines.append(line_points)
                else:
                    # Line is more vertical
                    line_points = cluster_points[ransac_yx.inlier_mask_]
                    lines.append(line_points)

            except ValueError:
                # RANSAC can fail if all points are on a vertical line
                # In this case, we can treat it as a vertical line
                lines.append(cluster_points)

        # 3. Feature Extraction: Extract endpoints and corners as landmarks
        landmarks = []
        endpoints = []
        for line in lines:
            # Endpoints of the line segment
            p1 = line[0]
            p2 = line[-1]
            endpoints.append(p1)
            endpoints.append(p2)

        # Corners where two line segments meet at a steep angle
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i]
                line2 = lines[j]

                # Check for proximity and angle between lines
                # For simplicity, we'll check if the endpoints are close
                for p1 in [line1[0], line1[-1]]:
                    for p2 in [line2[0], line2[-1]]:
                        if np.linalg.norm(p1 - p2) < 0.3: # Proximity threshold
                            # Calculate angle between lines
                            v1 = line1[-1] - line1[0]
                            v2 = line2[-1] - line2[0]
                            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                            if np.abs(angle) > np.pi / 4 and np.abs(angle) < 3 * np.pi / 4: # Steep angle
                                corner = (p1 + p2) / 2
                                landmarks.append(corner)

        # Combine endpoints and corners, remove duplicates
        all_potential_landmarks = endpoints + landmarks
        if not all_potential_landmarks:
            return []

        # Use DBSCAN to cluster potential landmarks and find the center of each cluster
        landmark_clustering = DBSCAN(eps=0.3, min_samples=1).fit(all_potential_landmarks)
        final_landmarks = []
        for label in set(landmark_clustering.labels_):
            landmark_cluster = np.array(all_potential_landmarks)[landmark_clustering.labels_ == label]
            final_landmarks.append(np.mean(landmark_cluster, axis=0))

        # 4. Output: Convert to (range, bearing, signature) format
        landmark_observations = []
        for lm in final_landmarks:
            r = np.linalg.norm(lm)
            b = np.arctan2(lm[1], lm[0])
            signature = f"{lm[0]:.2f}_{lm[1]:.2f}" # Simple signature based on position
            landmark_observations.append((r, b, signature))

        return landmark_observations


