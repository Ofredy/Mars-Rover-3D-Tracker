# library imports
import numpy as np


######### ekf constants #########
imu_hz = 10
ranging_hz = 0

NUM_STATES = 6
NUM_RANGING_MEASUREMENTS = 3
process_noise_variance = 0.01
measurement_noise_variance = 0.01
x_0_guess_variance = 0.01
Q = np.eye(NUM_STATES) * process_noise_variance * (1/imu_hz)
R = np.eye(NUM_RANGING_MEASUREMENTS) * measurement_noise_variance

A = np.array([[ 1, 0, 0, (1/imu_hz), 0, 0 ],
              [ 0, 1, 0, 0, (1/imu_hz), 0 ],
              [ 0, 0, 1, 0, 0, (1/imu_hz) ],
              [ 0, 0, 0, 1, 0, 0 ],
              [ 0, 0, 0, 0, 1, 0 ],
              [ 0, 0, 0, 0, 0, 1 ],])

B = np.array([[ (1/imu_hz)**2, 0, 0 ],
              [ 0, (1/imu_hz)**2, 0 ],
              [ 0, 0, (1/imu_hz)**2 ],
              [ (1/imu_hz), 0, 0 ],
              [ 0, (1/imu_hz), 0 ],
              [ 0, 0, (1/imu_hz) ]])

beacons = np.array([[ -5, 5, 0.1 ],
                    [ -5, -5, 3 ],
                    [ 5, -5, 0.15 ]])


def rover_state_update(x_n, acc_measurement):

    return A @ x_n + B @ acc_measurement

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

def get_estimate_ranges(x_n):

    return np.linalg.norm( x_n[0:3] - beacons, axis=1)

def observation_jacobian(x_pred_n):

    estimate_ranges = get_estimate_ranges(x_pred_n)

    return np.array([[ ( x_pred_n[0] ) / estimate_ranges[0], ( x_pred_n[1] ) / estimate_ranges[0], ( x_pred_n[2] ) / estimate_ranges[0], 0, 0, 0, 0, 0, 0 ],
                     [ ( x_pred_n[0] ) / estimate_ranges[1], ( x_pred_n[1] ) / estimate_ranges[1], ( x_pred_n[2] ) / estimate_ranges[1], 0, 0, 0, 0, 0, 0 ],
                     [ ( x_pred_n[0] ) / estimate_ranges[2], ( x_pred_n[1] ) / estimate_ranges[2], ( x_pred_n[2] ) / estimate_ranges[2], 0, 0, 0, 0, 0, 0 ]])

def ekf_update_t(x_pred_n, P_pred_n, z):

    observation_jacobian_n = observation_jacobian(x_pred_n)

    # calculate kalman gain
    k_n = P_pred_n @ np.transpose(observation_jacobian_n) @ np.linalg.inv( observation_jacobian_n @ P_pred_n @ np.transpose(observation_jacobian_n) + R )

    # estimate state
    x_n = x_pred_n + k_n @ ( z - observation_jacobian_n @ x_pred_n )

    # update estimate covariance
    P_pred_n = ( np.eye(NUM_STATES) - k_n @ observation_jacobian_n ) @ P_pred_n @ np.transpose( ( np.eye(NUM_STATES) - k_n @ observation_jacobian_n ) ) + k_n @ R @ np.transpose(k_n)

    return x_n, P_pred_n, k_n
