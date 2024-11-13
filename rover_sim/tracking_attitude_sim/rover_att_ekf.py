import numpy as np
from scipy.spatial.transform import Rotation 

process_noise_var = 0.01
measurement_noise_var = 0.01

Q = np.eye(7) * process_noise_var  # Process noise covariance matrix
R = np.eye(4) * measurement_noise_var  # Measurement noise covariance matrix (quaternion for TRIAD)


def initialize_kalman_filter():
    """
    Initializes the Kalman filter parameters.
    
    Returns:
    - state (np.ndarray): Initial state vector (quaternion and angular velocity).
    - P (np.ndarray): Initial covariance matrix.
    - Q (np.ndarray): Process noise covariance matrix.
    - R (np.ndarray): Measurement noise covariance matrix.
    """
    # Initial state vector: quaternion [w, x, y, z] and angular velocity [wx, wy, wz]
    initial_quat = np.array([1, 0, 0, 0])  # Starting with no rotation
    initial_angular_velocity = np.array([0, 0, 0])  # Assume no initial angular velocity
    state = np.hstack((initial_quat, initial_angular_velocity))
    
    # Covariance matrices
    P = np.eye(7) * 0.1  # Initial state covariance matrix
    
    return state, P

def quaternion_multiply(q, r):
    """ Multiplies two quaternions q and r. """
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
    ])

def predict(state, P, dt, angular_velocity):
    """
    Prediction step of the Kalman filter with quaternion dynamics discretized using angular velocity.
    
    Parameters:
    - state (np.ndarray): Current state vector (quaternion and angular velocity).
    - P (np.ndarray): Current covariance matrix.
    - dt (float): Time step.
    - angular_velocity (np.ndarray): Angular velocity measurement.
    
    Returns:
    - state (np.ndarray): Predicted state vector.
    - P (np.ndarray): Predicted covariance matrix.
    """
    # Extract quaternion and angular velocity
    current_quat = state[:4]
    current_omega = angular_velocity

    # Construct the continuous-time quaternion dynamics matrix
    omega_x, omega_y, omega_z = angular_velocity
    Omega = np.array([
        [0, -omega_x, -omega_y, -omega_z],
        [omega_x, 0, omega_z, -omega_y],
        [omega_y, -omega_z, 0, omega_x],
        [omega_z, omega_y, -omega_x, 0]
    ])

    # Discretize the quaternion dynamics
    F_quat = np.eye(4) + 0.5 * Omega * dt  # State transition matrix for the quaternion part

    # Apply the quaternion update
    predicted_quat = F_quat @ current_quat
    predicted_quat /= np.linalg.norm(predicted_quat)  # Normalize to avoid drift

    # State transition model for angular velocity is identity
    predicted_state = np.hstack((predicted_quat, angular_velocity))

    # Construct the full 7x7 state transition matrix F for both quaternion and angular velocity
    F = np.eye(7)
    F[:4, :4] = F_quat  # Quaternion transition part
    F[4:, 4:] = np.eye(3)  # Angular velocity part remains identity

    # Update the state covariance P
    P = F @ P @ F.T + Q  # Add process noise
    
    return predicted_state, P

def measure_with_triad(acceleration, magnetic_field):
    """
    Uses TRIAD method to estimate orientation quaternion from accelerometer and magnetometer data.
    
    Parameters:
    - acceleration (np.ndarray): Accelerometer measurement.
    - magnetic_field (np.ndarray): Magnetometer measurement.
    
    Returns:
    - measured_quat (np.ndarray): Quaternion representing the TRIAD-based orientation estimate.
    """
    # Fixed vectors in the inertial frame
    gravity_vector = np.array([0, 0, -1])  # Down direction
    magnetic_vector = np.array([1, 0, 0])  # Magnetic north

    # Normalize the vectors
    accel_body = acceleration / np.linalg.norm(acceleration)
    mag_body = magnetic_field / np.linalg.norm(magnetic_field)
    
    # Ensure that align_vectors receives lists of two 3D vectors each
    triad_rotation, _ = Rotation.align_vectors([gravity_vector, magnetic_vector], [accel_body, mag_body])
    
    # Return quaternion
    measured_quat = triad_rotation.as_quat()
    return measured_quat

def update(state, P, measurement, R):
    """
    Update step of the Kalman filter.
    
    Parameters:
    - state (np.ndarray): Predicted state vector.
    - P (np.ndarray): Predicted covariance matrix.
    - measurement (np.ndarray): Quaternion measurement from TRIAD.
    - R (np.ndarray): Measurement noise covariance matrix.
    
    Returns:
    - state (np.ndarray): Updated state vector.
    - P (np.ndarray): Updated covariance matrix.
    """
    # Measurement model (only quaternion part of the state)
    H = np.hstack((np.eye(4), np.zeros((4, 3))))  # Measurement matrix
    
    # Convert rotation difference to quaternion residual
    y = measurement - state[:4]
    y /= np.linalg.norm(y)  # Normalize residual

    # Kalman gain calculation and update
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)  # K is now a 7x4 matrix
    
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    
    state = state + K @ y
    state[:4] /= np.linalg.norm(state[:4])

    # Update covariance estimate
    P = P - K @ H @ P  # Expanded form of the covariance update
    
    return state, P
