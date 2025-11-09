
import numpy as np

class LocalPlannerDWA:
    def __init__(self, max_linear_velocity=0.5, max_angular_velocity=1.0, 
                 linear_acceleration=0.2, angular_acceleration=0.5,
                 dt=0.1, predict_time=3.0, num_samples=10,
                 heading_cost_gain=1.0, distance_cost_gain=1.0, obstacle_cost_gain=1.0):
        """
        Initializes the LocalPlannerDWA class.

        :param max_linear_velocity: Maximum linear velocity of the robot.
        :param max_angular_velocity: Maximum angular velocity of the robot.
        :param linear_acceleration: Maximum linear acceleration of the robot.
        :param angular_acceleration: Maximum angular acceleration of the robot.
        :param dt: Time step for trajectory prediction.
        :param predict_time: Time horizon for trajectory prediction.
        :param num_samples: Number of samples for velocity space.
        :param heading_cost_gain: Gain for heading cost.
        :param distance_cost_gain: Gain for distance to goal cost.
        :param obstacle_cost_gain: Gain for obstacle avoidance cost.
        """
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.linear_acceleration = linear_acceleration
        self.angular_acceleration = angular_acceleration
        self.dt = dt
        self.predict_time = predict_time
        self.num_samples = num_samples

        self.heading_cost_gain = heading_cost_gain
        self.distance_cost_gain = distance_cost_gain
        self.obstacle_cost_gain = obstacle_cost_gain

    def _generate_trajectories(self, current_pose, current_linear_velocity, current_angular_velocity):
        """
        Generates a set of possible trajectories based on dynamic window.

        :param current_pose: Current robot pose [x, y, theta].
        :param current_linear_velocity: Current linear velocity.
        :param current_angular_velocity: Current angular velocity.
        :return: List of trajectories, where each trajectory is a list of poses.
        """
        trajectories = []

        # Calculate dynamic window
        min_v = max(0, current_linear_velocity - self.linear_acceleration * self.dt)
        max_v = min(self.max_linear_velocity, current_linear_velocity + self.linear_acceleration * self.dt)
        min_omega = max(-self.max_angular_velocity, current_angular_velocity - self.angular_acceleration * self.dt)
        max_omega = min(self.max_angular_velocity, current_angular_velocity + self.angular_acceleration * self.dt)

        # Sample velocities
        linear_velocities = np.linspace(min_v, max_v, self.num_samples)
        angular_velocities = np.linspace(min_omega, max_omega, self.num_samples)

        for v in linear_velocities:
            for omega in angular_velocities:
                trajectory = []
                x, y, theta = current_pose
                for t in np.arange(0, self.predict_time, self.dt):
                    if omega == 0:
                        x += v * self.dt * np.cos(theta)
                        y += v * self.dt * np.sin(theta)
                    else:
                        x += -v/omega * np.sin(theta) + v/omega * np.sin(theta + omega * self.dt)
                        y += v/omega * np.cos(theta) - v/omega * np.cos(theta + omega * self.dt)
                    theta += omega * self.dt
                    trajectory.append([x, y, theta])
                trajectories.append((trajectory, v, omega))
        return trajectories

    def _score_trajectory(self, trajectory, goal_pose, ekf_slam_object):
        """
        Scores a trajectory based on heading, distance to goal, and obstacle avoidance.

        :param trajectory: A list of poses in the trajectory.
        :param goal_pose: The target goal pose [x, y].
        :param ekf_slam_object: The EKF_SLAM object to get current state and covariance.
        :return: A cost value for the trajectory.
        """
        # Heading cost: penalize trajectories that don't point towards the goal
        last_pose = trajectory[-1]
        angle_to_goal = np.arctan2(goal_pose[1] - last_pose[1], goal_pose[0] - last_pose[0])
        heading_error = abs(angle_to_goal - last_pose[2])
        heading_cost = heading_error

        # Distance cost: penalize trajectories that end far from the goal
        distance_to_goal = np.linalg.norm(np.array(last_pose[:2]) - np.array(goal_pose[:2]))
        distance_cost = distance_to_goal

        # Obstacle cost: calculate probability of collision using EKF_SLAM covariance
        obstacle_cost = 0.0
        robot_covariance = ekf_slam_object.get_covariance()[:3, :3] # Robot's pose covariance
        landmarks = ekf_slam_object.get_landmarks()

        for pose in trajectory:
            robot_x, robot_y, _ = pose
            
            # Consider uncertainty in robot's position
            # For simplicity, we'll use a fixed radius around the robot for collision checking
            # A more advanced approach would sample from the robot's pose distribution
            
            # Check collision with known landmarks, considering their uncertainty
            for signature, lm_pos in landmarks.items():
                lm_x, lm_y = lm_pos
                lm_idx = ekf_slam_object.landmarks[signature]
                lm_covariance = ekf_slam_object.get_covariance()[lm_idx:lm_idx+2, lm_idx:lm_idx+2]

                # Calculate Mahalanobis distance between robot and landmark
                # This is a simplified approach; a full collision probability would integrate over the uncertainties
                combined_covariance = np.zeros((4,4))
                combined_covariance[:2,:2] = robot_covariance[:2,:2] # Only x,y covariance for robot
                combined_covariance[2:,2:] = lm_covariance

                mean_diff = np.array([robot_x - lm_x, robot_y - lm_y, 0, 0]) # Placeholder for full state diff
                
                # For now, a simpler distance-based collision check with a buffer for uncertainty
                distance = np.linalg.norm(np.array([robot_x, robot_y]) - np.array([lm_x, lm_y]))
                
                # A simple heuristic: if distance is below a threshold, increase cost based on uncertainty
                # The threshold should account for robot size and some buffer
                collision_threshold = 0.5 # Example threshold
                if distance < collision_threshold:
                    # Increase cost based on the inverse of the determinant of the combined covariance
                    # A smaller determinant means less uncertainty, so higher confidence in collision
                    # This is a very rough heuristic for 'probability of collision'
                    try:
                        uncertainty_factor = 1.0 / np.sqrt(np.linalg.det(combined_covariance + np.eye(4)*1e-6)) # Add small value for stability
                        obstacle_cost += uncertainty_factor * (collision_threshold - distance)
                    except np.linalg.LinAlgError:
                        # Handle singular matrix case, e.g., by assigning a high cost
                        obstacle_cost += 100.0 # High penalty

        total_cost = (self.heading_cost_gain * heading_cost +
                      self.distance_cost_gain * distance_cost +
                      self.obstacle_cost_gain * obstacle_cost)
        return total_cost

    def find_best_trajectory(self, current_pose, current_linear_velocity, current_angular_velocity, goal_pose, ekf_slam_object):
        """
        Finds the best trajectory and corresponding velocities using DWA.

        :param current_pose: Current robot pose [x, y, theta].
        :param current_linear_velocity: Current linear velocity.
        :param current_angular_velocity: Current angular velocity.
        :param goal_pose: The target goal pose [x, y].
        :param ekf_slam_object: The EKF_SLAM object to get current state and covariance.
        :return: Tuple of (best_linear_velocity, best_angular_velocity) or (0, 0) if no valid trajectory.
        """
        trajectories = self._generate_trajectories(current_pose, current_linear_velocity, current_angular_velocity)

        best_cost = float("inf")
        best_velocities = (0.0, 0.0)

        for trajectory, v, omega in trajectories:
            cost = self._score_trajectory(trajectory, goal_pose, ekf_slam_object)
            if cost < best_cost:
                best_cost = cost
                best_velocities = (v, omega)

        return best_velocities


