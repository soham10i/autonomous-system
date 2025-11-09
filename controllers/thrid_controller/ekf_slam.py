
import numpy as np

class EKF_SLAM:
    def __init__(self, initial_pose):
        """
        Initializes the EKF_SLAM class.

        State Vector: [x, y, theta, L1_x, L1_y, L2_x, L2_y, ...]
        - (x, y, theta): Robot's pose
        - (Li_x, Li_y): Coordinates of observed landmarks

        :param initial_pose: Initial pose of the robot [x, y, theta]
        """
        self.state = np.array([initial_pose[0], initial_pose[1], initial_pose[2]])  # Initial robot pose
        self.covariance = np.diag([0.1, 0.1, np.deg2rad(5.0)])  # Initial covariance matrix for robot pose
        self.landmarks = {}  # Dictionary to store landmarks: {signature: index_in_state_vector}
        self.next_landmark_idx = 3  # Starting index for landmarks in the state vector

    def predict(self, odometry, dt):
        """
        Implements the prediction step of the EKF-SLAM.

        :param odometry: Odometry data [linear_velocity, angular_velocity]
        :param dt: Time difference
        """
        v, omega = odometry
        theta = self.state[2]

        # Predict robot's pose
        if omega == 0:
            self.state[0] += v * dt * np.cos(theta)
            self.state[1] += v * dt * np.sin(theta)
        else:
            self.state[0] += -v/omega * np.sin(theta) + v/omega * np.sin(theta + omega * dt)
            self.state[1] += v/omega * np.cos(theta) - v/omega * np.cos(theta + omega * dt)
        self.state[2] += omega * dt

        # Normalize theta
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

        # Jacobian of the state transition function (F_x)
        # F_x = [[1, 0, -v*dt*sin(theta)],
        #        [0, 1,  v*dt*cos(theta)],
        #        [0, 0,  1]]
        # For non-zero omega:
        # F_x = [[1, 0, (-v/omega)*cos(theta) + (v/omega)*cos(theta + omega*dt)],
        #        [0, 1, (-v/omega)*sin(theta) + (v/omega)*sin(theta + omega*dt)],
        #        [0, 0, 1]]
        
        # Jacobian of the control input (F_u)
        # F_u = [[dt*cos(theta), 0],
        #        [dt*sin(theta), 0],
        #        [0, dt]]
        # For non-zero omega:
        # F_u = [[(sin(theta + omega*dt) - sin(theta))/omega, v*(cos(theta + omega*dt)*dt*omega - sin(theta + omega*dt) + sin(theta))/(omega**2)],
        #        [(-cos(theta + omega*dt) + cos(theta))/omega, v*(sin(theta + omega*dt)*dt*omega + cos(theta + omega*dt) - cos(theta))/(omega**2)],
        #        [0, dt]]

        # Simplified F_x for robot pose only (assuming landmarks are constant during prediction)
        F_x_robot = np.array([
            [1, 0, -v * dt * np.sin(theta)] if omega == 0 else [1, 0, (-v/omega)*np.cos(theta) + (v/omega)*np.cos(theta + omega*dt)],
            [0, 1, v * dt * np.cos(theta)] if omega == 0 else [0, 1, (-v/omega)*np.sin(theta) + (v/omega)*np.sin(theta + omega*dt)],
            [0, 0, 1]
        ])

        # Construct the full Jacobian F
        F = np.eye(len(self.state))
        F[:3, :3] = F_x_robot

        # Covariance prediction: P = F * P * F^T + Q
        # Q is the process noise covariance matrix
        # For simplicity, let's assume a diagonal Q for now
        Q = np.diag([0.01, 0.01, np.deg2rad(1.0)])  # Process noise for robot pose
        full_Q = np.zeros((len(self.state), len(self.state)))
        full_Q[:3, :3] = Q

        self.covariance = F @ self.covariance @ F.T + full_Q

    def update(self, landmark_observations):
        """
        Implements the update step of the EKF-SLAM.

        :param landmark_observations: List of (range, bearing, signature) tuples
        """
        for r, b, signature in landmark_observations:
            if signature not in self.landmarks:
                # New landmark: Initialize and add to state vector and covariance
                lm_x = self.state[0] + r * np.cos(self.state[2] + b)
                lm_y = self.state[1] + r * np.sin(self.state[2] + b)
                
                self.state = np.append(self.state, [lm_x, lm_y])
                lm_idx = self.next_landmark_idx
                self.landmarks[signature] = lm_idx
                self.next_landmark_idx += 2

                # Expand covariance matrix for new landmark
                # Jacobian of observation model for new landmark initialization (G_z)
                # G_z = [[cos(theta+b), -r*sin(theta+b)],
                #        [sin(theta+b), r*cos(theta+b)]]
                # For simplicity, using a more direct approach for covariance expansion
                
                # J_lm = Jacobian of landmark position w.r.t. robot pose and observation
                # J_lm = [[1, 0, -r*sin(theta+b)],
                #         [0, 1,  r*cos(theta+b)]]
                # This is for the case where landmark is observed in robot frame and converted to world frame

                # Simplified approach for adding landmark covariance
                # R_obs = observation noise covariance
                R_obs = np.diag([0.1, np.deg2rad(1.0)]) # Noise for range and bearing
                
                # G_z_inv = inverse of Jacobian of observation model (from landmark to observation)
                # G_z_inv = [[cos(theta+b), sin(theta+b)],
                #            [-sin(theta+b)/r, cos(theta+b)/r]]
                # This is not directly used here, but for understanding

                # The covariance of the new landmark is initialized based on the robot's current covariance
                # and the observation noise. A common way is to use the Jacobian of the transformation
                # from robot pose + observation to landmark position.
                
                # Let h(x_robot, z) be the function that transforms robot pose and observation to landmark position
                # x_lm = x_robot + r * cos(theta_robot + b)
                # y_lm = y_robot + r * sin(theta_robot + b)
                
                # H_x = Jacobian of h w.r.t. x_robot
                # H_x = [[1, 0, -r*sin(theta_robot + b)],
                #        [0, 1,  r*cos(theta_robot + b)]]
                
                # H_z = Jacobian of h w.r.t. z (observation)
                # H_z = [[cos(theta_robot + b), -r*sin(theta_robot + b)],
                #        [sin(theta_robot + b),  r*cos(theta_robot + b)]]

                # For simplicity, we'll use a common approximation for initializing landmark covariance
                # This is often done by taking a block from the robot's covariance and adding observation noise
                # A more rigorous approach involves the Jacobian of the inverse observation model

                new_lm_cov = np.zeros((2, 2))
                new_lm_cov[0,0] = r*r*np.sin(b)*np.sin(b)*self.covariance[2,2] + np.cos(b)*np.cos(b)*R_obs[0,0] + r*r*np.cos(b)*np.cos(b)*R_obs[1,1]
                new_lm_cov[0,1] = -r*r*np.sin(b)*np.cos(b)*self.covariance[2,2] + np.cos(b)*np.sin(b)*R_obs[0,0] - r*r*np.cos(b)*np.sin(b)*R_obs[1,1]
                new_lm_cov[1,0] = new_lm_cov[0,1]
                new_lm_cov[1,1] = r*r*np.cos(b)*np.cos(b)*self.covariance[2,2] + np.sin(b)*np.sin(b)*R_obs[0,0] + r*r*np.sin(b)*np.sin(b)*R_obs[1,1]

                # Expand the main covariance matrix
                old_covariance_size = self.covariance.shape[0]
                new_covariance = np.zeros((old_covariance_size + 2, old_covariance_size + 2))
                new_covariance[:old_covariance_size, :old_covariance_size] = self.covariance
                new_covariance[lm_idx:lm_idx+2, lm_idx:lm_idx+2] = new_lm_cov
                self.covariance = new_covariance

            else:
                # Re-observed landmark: Update robot's pose and landmark's position
                lm_idx = self.landmarks[signature]
                lm_x = self.state[lm_idx]
                lm_y = self.state[lm_idx + 1]

                # Predicted observation (h)
                delta_x = lm_x - self.state[0]
                delta_y = lm_y - self.state[1]
                q = delta_x**2 + delta_y**2
                predicted_r = np.sqrt(q)
                predicted_b = np.arctan2(delta_y, delta_x) - self.state[2]
                predicted_b = np.arctan2(np.sin(predicted_b), np.cos(predicted_b)) # Normalize angle

                predicted_observation = np.array([predicted_r, predicted_b])
                actual_observation = np.array([r, b])

                # Innovation (y)
                innovation = actual_observation - predicted_observation
                innovation[1] = np.arctan2(np.sin(innovation[1]), np.cos(innovation[1])) # Normalize angle

                # Jacobian of observation model (H)
                # H = [[-delta_x/sqrt(q), -delta_y/sqrt(q), 0, delta_x/sqrt(q), delta_y/sqrt(q)],
                #      [delta_y/q, -delta_x/q, -1, -delta_y/q, delta_x/q]]
                
                H_robot = np.array([
                    [-delta_x / predicted_r, -delta_y / predicted_r, 0],
                    [delta_y / q, -delta_x / q, -1]
                ])
                H_landmark = np.array([
                    [delta_x / predicted_r, delta_y / predicted_r],
                    [-delta_y / q, delta_x / q]
                ])

                H = np.zeros((2, len(self.state)))
                H[:, :3] = H_robot
                H[:, lm_idx:lm_idx+2] = H_landmark

                # Observation noise covariance (R)
                R = np.diag([0.1, np.deg2rad(1.0)])  # Noise for range and bearing

                # Kalman Gain (K)
                S = H @ self.covariance @ H.T + R
                K = self.covariance @ H.T @ np.linalg.inv(S)

                # Update state and covariance
                self.state = self.state + (K @ innovation)
                self.covariance = (np.eye(len(self.state)) - K @ H) @ self.covariance

                # Normalize robot's theta after update
                self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

    def get_robot_pose(self):
        """
        Returns the current estimated robot pose.
        """
        return self.state[:3]

    def get_landmarks(self):
        """
        Returns the current estimated landmark positions.
        """
        landmarks_data = {}
        for signature, idx in self.landmarks.items():
            landmarks_data[signature] = self.state[idx:idx+2]
        return landmarks_data

    def get_covariance(self):
        """
        Returns the current covariance matrix.
        """
        return self.covariance



