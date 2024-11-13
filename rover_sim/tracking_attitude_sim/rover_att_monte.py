import math

import numpy as np
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt


simulation_hz = 100  # 10 Hz simulation frequency
simulation_time = 5  # Run for 5 seconds
NUM_MONTE_RUNS = 10  # Number of Monte Carlo runs

process_noise_var = 0.00000001


def get_beta_p_t(beta_t, t, freq_modifier):
    b0, b1, b2, b3 = beta_t[0], beta_t[1], beta_t[2], beta_t[3]
    beta_matrix = np.array([[-b1, -b2, -b3],
                            [b0, -b3, b2],
                            [b3, b0, -b1],
                            [-b2, b1, b0]])
    # Angular velocity with varying frequency
    w_t = np.array([np.sin(0.1 * t), np.cos(freq_modifier * t), np.cos(0.1 * t)]) * 20 * np.pi / 180
    return 0.5 * np.dot(beta_matrix, w_t), w_t  # Returns quaternion derivative and angular velocity vector

def integrator(func_x, x_0, t_0, t_f, dt=(1/simulation_hz), norm_value=False):
    state_summary = []
    x_n = x_0
    state_summary.append(x_n)
    t = t_0
    while t <= t_f:
        f = func_x(x_n, t)
        x_n = x_n + f * dt
        if norm_value:
            x_n = x_n / np.linalg.norm(x_n)
        state_summary.append(x_n)
        t += dt
    return state_summary

def generate_attitude_changes(initial_quat=np.array([1.0, 0.0, 0.0, 0.0]), time_step=(1 / simulation_hz), num_steps=int(simulation_time / (1 / simulation_hz)), monte_runs=NUM_MONTE_RUNS):
    attitudes = np.zeros((monte_runs, num_steps, 4))
    accelerations = np.zeros((monte_runs, num_steps, 3))
    magnetic_fields = np.zeros((monte_runs, num_steps, 3))
    gyros = np.zeros((monte_runs, num_steps, 3))
    
    gravity_vector = np.array([0.0, 0.0, -1])
    magnetic_field_vector = np.array([1.0, 0.0, 0.0])

    for monte_idx in range(monte_runs):
        # Generate a unique frequency modifier for each Monte Carlo run
        freq_modifier = 0.1 + 0.05 * monte_idx  # Example variation in frequency
        
        # Integrate with the varying frequency using lambda to pass freq_modifier to get_beta_p_t
        t_f = (num_steps - 1) * time_step
        attitude_sequence = integrator(lambda beta, t: get_beta_p_t(beta, t, freq_modifier)[0], initial_quat, 0, t_f, time_step, norm_value=True)
        attitudes[monte_idx, :, :] = np.array(attitude_sequence[:num_steps])

        for step in range(num_steps):
            current_quat = attitudes[monte_idx, step]
            
            # Get quaternion derivative and angular velocity with frequency modifier
            _, angular_velocity = get_beta_p_t(current_quat, step * time_step, freq_modifier)
            gyros[monte_idx, step] = angular_velocity  # Store angular velocity vector
            
            # Simulate accelerometer and magnetometer readings
            rotation = Rotation.from_quat(current_quat)
            accel_body_frame = rotation.apply(gravity_vector)
            noise = np.random.normal(0, np.sqrt(process_noise_var))
            accelerations[monte_idx, step] = accel_body_frame + noise

            mag_body_frame = rotation.apply(magnetic_field_vector)
            noise = np.random.normal(0, np.sqrt(process_noise_var))
            magnetic_fields[monte_idx, step] = mag_body_frame + noise

    return {'att': attitudes, 'acc': accelerations, 'mag': magnetic_fields, 'gyro': gyros}

def visualize_rotation(attitudes, time_step=(1/simulation_hz)):
    """
    Visualizes the rotation of all Monte Carlo runs by plotting roll, pitch, and yaw over time.

    Parameters:
    - attitudes (np.ndarray): 3D NumPy array with shape (monte_runs, num_steps, 4) storing quaternions.
    - time_step (float): Time interval between each step, in seconds.
    """
    monte_runs, num_steps, _ = attitudes.shape
    time_array = np.linspace(0, num_steps * time_step, num_steps)

    # Initialize figure and subplots for Roll, Pitch, and Yaw
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle("Roll, Pitch, and Yaw over Time for All Monte Carlo Runs")

    for monte_idx in range(monte_runs):
        # Extract the specific Monte Carlo run quaternions
        quaternions = attitudes[monte_idx]

        # Convert quaternions to Euler angles (roll, pitch, yaw)
        euler_angles = Rotation.from_quat(quaternions).as_euler('xyz', degrees=True)  # 'xyz' for roll, pitch, yaw
        
        # Plot roll, pitch, and yaw over time for this Monte Carlo run
        axs[0].plot(time_array, euler_angles[:, 0])
        axs[1].plot(time_array, euler_angles[:, 1])
        axs[2].plot(time_array, euler_angles[:, 2])

    # Set labels and titles for each subplot
    axs[0].set_title("Roll over Time")
    axs[0].set_ylabel("Roll (degrees)")
    axs[0].set_xlabel("Time (s)")

    axs[1].set_title("Pitch over Time")
    axs[1].set_ylabel("Pitch (degrees)")
    axs[1].set_xlabel("Time (s)")

    axs[2].set_title("Yaw over Time")
    axs[2].set_ylabel("Yaw (degrees)")
    axs[2].set_xlabel("Time (s)")

    # Display legends and adjust layout
    for ax in axs:
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the main title

def monte_att_estimate(monte_sum, dt=(1/simulation_hz), accel_noise_std=0.000000000005, mag_noise_std=0.0000000001):
    """
    Runs the TRIAD method for attitude estimation over multiple Monte Carlo runs 
    using accelerometer and magnetometer data with added noise.
    
    Parameters:
    - monte_sum (dict): Contains arrays for true attitudes, accelerometer, magnetometer, and gyroscope data.
    - dt (float): Time step.
    - accel_noise_std (float): Standard deviation of noise to add to accelerometer data.
    - mag_noise_std (float): Standard deviation of noise to add to magnetometer data.
    
    Returns:
    - estimated_states_all (np.ndarray): 3D array with shape (monte_runs, num_steps, 4) storing estimated quaternions.
    """
    monte_runs, num_steps = monte_sum['gyro'].shape[0], monte_sum['gyro'].shape[1]
    estimated_states_all = np.zeros((monte_runs, num_steps, 4))  # To store estimated quaternions for all runs

    for monte_idx in range(monte_runs):
        accelerations = monte_sum['acc'][monte_idx]
        magnetic_fields = monte_sum['mag'][monte_idx]

        estimated_states = np.zeros((num_steps, 4))  # To store quaternion estimates for this run

        for step in range(num_steps):
            # Measurement step using TRIAD with accelerometer and magnetometer data
            acceleration = accelerations[step]
            magnetic_field = magnetic_fields[step]
            
            # Add Gaussian noise to accelerometer and magnetometer measurements
            noisy_accel = acceleration + np.random.normal(0, accel_noise_std)
            noisy_mag = magnetic_field + np.random.normal(0, mag_noise_std)
            
            # Estimate the quaternion using TRIAD with noisy measurements
            estimated_quat = triad_method(noisy_accel, noisy_mag)
            
            # Store quaternion estimate
            estimated_states[step] = estimated_quat

        # Store the estimated states for this Monte Carlo run
        estimated_states_all[monte_idx] = estimated_states

    return estimated_states_all

def triad_method(sensor_1, sensor_2, inertial_1=np.array([0, 0, -1]), inertial_2=np.array([1, 0, 0])):

    # norming inputs
    sensor_1 = sensor_1 / np.linalg.norm(sensor_1)
    sensor_2 = sensor_2 / np.linalg.norm(sensor_2)
    inertial_1 = inertial_1 / np.linalg.norm(inertial_1)
    inertial_2 = inertial_2 / np.linalg.norm(inertial_2)

    # t frame for b & n
    B_t1 = sensor_1
    B_t2 = np.cross(sensor_1, sensor_2) / np.linalg.norm(np.cross(sensor_1, sensor_2))
    B_t3 = np.cross(B_t1, B_t2)

    N_t1 = inertial_1 / np.linalg.norm(inertial_1)
    N_t2 = np.cross(inertial_1, inertial_2) / np.linalg.norm(np.cross(inertial_1, inertial_2))
    N_t3 = np.cross(N_t1, N_t2)

    Bbar_T_dcm = np.stack((B_t1, B_t2, B_t3), axis=1)
    NT_dcm = np.stack((N_t1, N_t2, N_t3), axis=1) 

    return dcm_to_quaternions(np.dot(Bbar_T_dcm, np.transpose(NT_dcm)))

def dcm_to_quaternions(dcm):

    beta_vec_squared = [ (1/4) * ( 1 + np.trace(dcm) ),
                         (1/4) * ( 1 + 2*dcm[0][0] - np.trace(dcm) ),
                         (1/4) * ( 1 + 2*dcm[1][1] - np.trace(dcm) ),
                         (1/4) * ( 1 + 2*dcm[2][2] - np.trace(dcm) ) ]

    beta_idx = beta_vec_squared.index(max(beta_vec_squared))

    beta_vec = shepperd_hash[beta_idx](dcm, math.sqrt(beta_vec_squared[beta_idx]))

    if is_short_way_quaternion(beta_vec[0]):
        return beta_vec

    else:
        return shepperd_hash[beta_idx](dcm, -math.sqrt(beta_vec_squared[beta_idx]))

def shepperd_method_0(dcm, b0):

    b1 = ( dcm[1][2] - dcm[2][1] ) / ( 4*b0 )
    b2 = ( dcm[2][0] - dcm[0][2] ) / ( 4*b0 )
    b3 = ( dcm[0][1] - dcm[1][0] ) / ( 4*b0 )

    return np.array([b0, b1, b2, b3])

def shepperd_method_1(dcm, b1):

    b0 = ( dcm[1][2] - dcm[2][1] ) / ( 4*b1 )
    b2 = ( dcm[0][1] + dcm[1][0] ) / ( 4*b1 )
    b3 = ( dcm[2][0] + dcm[0][2] ) / ( 4*b1 )

    return np.array([b0, b1, b2, b3])

def shepperd_method_2(dcm, b2):

    b0 = ( dcm[2][0] - dcm[0][2] ) / ( 4*b2 )
    b1 = ( dcm[0][1] + dcm[1][0] ) / ( 4*b2 )
    b3 = ( dcm[1][2] + dcm[2][1] ) / ( 4*b2 )

    return np.array([b0, b1, b2, b3])

def shepperd_method_3(dcm, b3):

    b0 = ( dcm[0][1] - dcm[1][0] ) / ( 4*b3 )
    b1 = ( dcm[2][0] + dcm[0][2] ) / ( 4*b3 )
    b2 = ( dcm[1][2] + dcm[2][1] ) / ( 4*b3 )

    return np.array([b0, b1, b2, b3])

def is_short_way_quaternion(b0):

    if b0 > 0:
        return True

    else:
        return False

shepperd_hash = { 0: shepperd_method_0,
                  1: shepperd_method_1,
                  2: shepperd_method_2,
                  3: shepperd_method_3 }

def plot_attitude_error(true_attitudes_all, estimated_attitudes_all, dt=(1/simulation_hz)):
    monte_runs, num_steps = true_attitudes_all.shape[0], true_attitudes_all.shape[1]
    time_array = np.linspace(0, num_steps * dt, num_steps)

    plt.figure(figsize=(15, 10))
    
    # Set up subplots for Roll, Pitch, and Yaw errors
    for angle_idx, angle_name in enumerate(['Roll', 'Pitch', 'Yaw']):
        plt.subplot(3, 1, angle_idx + 1)
        
        for monte_idx in range(monte_runs):
            true_attitudes = true_attitudes_all[monte_idx]
            estimated_attitudes = estimated_attitudes_all[monte_idx]
            angle_errors = np.zeros(num_steps)
            
            for i in range(num_steps):
                # Convert both true and estimated quaternions to Euler angles
                true_euler = Rotation.from_quat(true_attitudes[i]).as_euler('xyz', degrees=True)
                est_euler = Rotation.from_quat(estimated_attitudes[i]).as_euler('xyz', degrees=True)
                
                # Calculate the error for the specific angle (roll, pitch, or yaw)
                angle_errors[i] = true_euler[angle_idx] - est_euler[angle_idx]
            
            # Plot the error for the current Monte Carlo run
            plt.plot(time_array, angle_errors)

        plt.xlabel('Time (s)')
        plt.ylabel(f'{angle_name} Error (degrees)')
        plt.title(f'{angle_name} Error Over Time')
        plt.grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Euler Angle Errors (Roll, Pitch, Yaw) for All Monte Carlo Runs')

if __name__ == "__main__":

    monte_sum = generate_attitude_changes()

    ekf_sum = monte_att_estimate(monte_sum)

    visualize_rotation(monte_sum['att'])
    plot_attitude_error(monte_sum['att'], ekf_sum)

    plt.show()
