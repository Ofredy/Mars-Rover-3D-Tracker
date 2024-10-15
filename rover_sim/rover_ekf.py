# library imports
import numpy as np


######### ekf constants #########
imu_hz = 25
ranging_hz = 50

NUM_STATES = 9
process_noise_variance = 0.0125
measurement_noise_variance = 0.01
x_0_guess_variance = 2

# Time step T is now 1/imu_hz
T = 1 / imu_hz
tau_a = 0.0001    # Replace with actual tau_a
tau_b = 0.0001    # Replace with actual tau_b

# Process noise submatrix Q0 for each axis, replacing T with 1/imu_hz
Q0 = np.array([
    [T**3 / 3 * tau_a + T**5 / 20 * tau_b, T**2 / 2 * tau_a + T**4 / 8 * tau_b, -T**3 / 6 * tau_b],
    [T**2 / 2 * tau_a + T**4 / 8 * tau_b, T * tau_a + T**3 / 3 * tau_b, -T**2 / 2 * tau_b],
    [-T**3 / 6 * tau_b, -T**2 / 2 * tau_b, T * tau_b]
])

# Full process noise matrix Qk-1 for the state vector [x, y, z, vx, vy, vz, ax, ay, az]
Q = np.block([
    [Q0, np.zeros_like(Q0), np.zeros_like(Q0)],  # x-axis
    [np.zeros_like(Q0), Q0, np.zeros_like(Q0)],  # y-axis
    [np.zeros_like(Q0), np.zeros_like(Q0), Q0]   # z-axis
])

R = measurement_noise_variance * np.sqrt(measurement_noise_variance)

# A0 matrix
A0_k = np.array([
    [1, T, -(T**2) / 2],
    [0, 1, -T],
    [0, 0, 1]
])

# Ak matrix (block diagonal form)
A = np.block([
    [A0_k, np.zeros((3, 3)), np.zeros((3, 3))],
    [np.zeros((3, 3)), A0_k, np.zeros((3, 3))],
    [np.zeros((3, 3)), np.zeros((3, 3)), A0_k]
])

# B0 matrix
B0_k = np.array([
    [(T**2) / 2, 0, 0],
    [T, 0, 0],
    [0, 0, 0]
])

# Bk matrix (block diagonal form)
B = np.block([
    [B0_k, np.zeros((3, 3)), np.zeros((3, 3))],
    [np.zeros((3, 3)), B0_k, np.zeros((3, 3))],
    [np.zeros((3, 3)), np.zeros((3, 3)), B0_k]
])

NUM_BEACONS = 3
beacons = np.array([[ -10.0,  10.0,  0.0 ],   # Beacon 1
                    [  10.0,  10.0,  0.0 ],   # Beacon 2
                    [  0.0,  -10.0,  0.0 ]])  # Beacon 3


def rover_state_update(x_n, acc_measurement):

    u_n = np.array([acc_measurement[0], 0, 0, acc_measurement[1], 0, 0, acc_measurement[2], 0, 0])

    return A @ x_n + B @ u_n

def rover_jacobian():

    # since state model is linear just return A
    return A

def ekf_predict_t(x_n, P_n, acc_measurement):

    # prediction
    x_pred_n = rover_state_update(x_n, acc_measurement)

    # uncertainty propagation
    rover_jacobian_n = rover_jacobian()

    P_n = rover_jacobian_n @ P_n @ np.transpose(rover_jacobian_n) + Q

    return x_pred_n, P_n

def get_estimate_range(x_n, beacon_idx):

    estimate_position = np.array([x_n[0], x_n[3], x_n[6]])

    return np.linalg.norm( estimate_position - beacons[beacon_idx] )

def observation_jacobian(x_n, estimate_range, beacon_idx):

    return np.array([ ( x_n[0] - beacons[beacon_idx][0] ) / estimate_range, 0, 0, ( x_n[3] - beacons[beacon_idx][1] ) / estimate_range, 0.0, 0.0, ( x_n[6] - beacons[beacon_idx][2] ) / estimate_range, 0.0, 0.0 ] ).reshape(1, -1)

def ekf_update_t(x_n, P_n, ranging_buffer, beacon_idx):

    # calculate jacobian
    estimate_range = get_estimate_range(x_n, beacon_idx)
    observation_jacobian_n = observation_jacobian(x_n, estimate_range, beacon_idx)

    # calculate kalman gain
    k_n = P_n @ np.transpose(observation_jacobian_n) @ np.linalg.inv( observation_jacobian_n @ P_n @ np.transpose(observation_jacobian_n) + R )

    # estimate state
    x_n = x_n + k_n.reshape(-1) * ( ranging_buffer - estimate_range )

    # update estimate covariance
    P_n = ( np.eye(NUM_STATES) - k_n @ observation_jacobian_n ) @ P_n @ np.transpose( ( np.eye(NUM_STATES) - k_n @ observation_jacobian_n ) ) + k_n @ np.transpose(k_n) * R

    return x_n, P_n
