# library imports
import numpy as np


######### ekf constants #########
imu_hz = 20
ranging_hz = 20

NUM_STATES = 6
NUM_RANGING_MEASUREMENTS = 3
process_noise_variance = 0.1
measurement_noise_variance = 0.01
x_0_guess_variance = 5
Q = np.eye(NUM_STATES) * process_noise_variance * (1/imu_hz)
R = np.eye(NUM_RANGING_MEASUREMENTS) * measurement_noise_variance

beacons = np.array([[ -10.0, 0.2, 0.1 ],
                    [ 0, 0.25, 0.3 ],
                    [ 10, 0.3, 0.15 ]])


def rover_state_update(x_n, imu_measurements_n):

    x_n[0] += imu_measurements_n[0] * (1/imu_hz) + imu_measurements_n[3] * (1/imu_hz)**2 
    x_n[1] += imu_measurements_n[1] * (1/imu_hz) + imu_measurements_n[4] * (1/imu_hz)**2  
    x_n[2] += imu_measurements_n[2] * (1/imu_hz) + imu_measurements_n[5] * (1/imu_hz)**2  

    x_n[3] += imu_measurements_n[3] * (1/imu_hz) 
    x_n[4] += imu_measurements_n[4] * (1/imu_hz) 
    x_n[5] += imu_measurements_n[5] * (1/imu_hz) 

    return x_n

def rover_jacobian():

    return np.array([
                        [1, 0, 0, (1/imu_hz), 0, 0],
                        [0, 1, 0, 0, (1/imu_hz), 0],
                        [0, 0, 1, 0, 0, (1/imu_hz)],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1]
                    ])

def ekf_predict_t(x_n, P_n, imu_measurements_n):

    # prediction
    x_pred_n = rover_state_update(x_n, imu_measurements_n)

    # uncertainty propagation
    rover_jacobian_n = rover_jacobian()

    P_n = rover_jacobian_n @ P_n @ np.transpose(rover_jacobian_n) + Q

    return x_pred_n, P_n

def get_estimate_ranges(x_n):

    return np.linalg.norm( x_n[0:3] - beacons, axis=1)

def observation_jacobian(x_pred_n):

    estimate_ranges = get_estimate_ranges(x_pred_n, )

    return np.array([[ ( x_pred_n[0] ) / estimate_ranges[0], ( x_pred_n[1] ) / estimate_ranges[0], ( x_pred_n[2] ) / estimate_ranges[0], 0, 0, 0 ],
                     [ ( x_pred_n[0] ) / estimate_ranges[1], ( x_pred_n[1] ) / estimate_ranges[1], ( x_pred_n[2] ) / estimate_ranges[1], 0, 0, 0 ],
                     [ ( x_pred_n[0] ) / estimate_ranges[2], ( x_pred_n[1] ) / estimate_ranges[2], ( x_pred_n[2] ) / estimate_ranges[2], 0, 0, 0 ]])

def ekf_update_t(x_pred_n, P_pred_n, z):

    observation_jacobian_n = observation_jacobian(x_pred_n)

    # calculate kalman gain
    k_n = P_pred_n @ np.transpose(observation_jacobian_n) @ np.linalg.inv( observation_jacobian_n @ P_pred_n @ np.transpose(observation_jacobian_n) + R )

    # estimate state
    x_n = x_pred_n + k_n @ ( z - observation_jacobian_n @ x_pred_n )

    # update estimate covariance
    P_pred_n = ( np.eye(NUM_STATES) - k_n @ observation_jacobian_n ) @ P_pred_n @ np.transpose( ( np.eye(NUM_STATES) - k_n @ observation_jacobian_n ) ) + k_n @ R @ np.transpose(k_n)

    return x_n, P_pred_n, k_n
