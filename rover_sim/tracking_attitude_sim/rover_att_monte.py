import math

import numpy as np
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt


simulation_hz = 100  # 10 Hz simulation frequency
simulation_time = 5  # Run for 5 seconds
NUM_MONTE_RUNS = 10  # Number of Monte Carlo runs

process_noise_var = 0.01

estimation_hz = 10   # Estimation frequency (Hz)


def get_beta_p_t(beta_t, t, freq_modifier=1.0, angular_velocity_profile=None):
    """
    Computes the quaternion derivative and angular velocity vector.

    Parameters:
    - beta_t: Current quaternion state.
    - t: Current time.
    - freq_modifier: Frequency modifier for angular velocity.
    - angular_velocity_profile: Callable for angular velocity as a function of time.

    Returns:
    - Quaternion derivative and angular velocity vector.
    """
    b0, b1, b2, b3 = beta_t[0], beta_t[1], beta_t[2], beta_t[3]
    beta_matrix = np.array([[-b1, -b2, -b3],
                            [b0, -b3, b2],
                            [b3, b0, -b1],
                            [-b2, b1, b0]])

    # Angular velocity for yaw-only motion
    if angular_velocity_profile:
        w_t = angular_velocity_profile(t)
    else:
        w_t = np.array([0.0, 0.0, 35.0]) * np.pi / 180  # Default yaw angular velocity

    w_t *= freq_modifier  # Scale by frequency modifier
    return 0.5 * np.dot(beta_matrix, w_t), w_t

def integrator(func_x, x_0, t_0, t_f, dt=(1 / simulation_hz), norm_value=False, **kwargs):
    """
    Numerical integrator to compute state evolution over time.
    
    Parameters:
    - func_x: Function to compute the derivative.
    - x_0: Initial state.
    - t_0: Start time.
    - t_f: End time.
    - dt: Time step.
    - norm_value: Whether to normalize the state after each step.
    - kwargs: Additional arguments to pass to the derivative function.
    
    Returns:
    - state_summary: List of states over time.
    """
    state_summary = []
    x_n = x_0
    state_summary.append(x_n)
    t = t_0
    while t <= t_f:
        f = func_x(x_n, t, **kwargs)  # Pass additional parameters to func_x
        x_n = x_n + f * dt
        if norm_value:
            x_n = x_n / np.linalg.norm(x_n)
        state_summary.append(x_n)
        t += dt
    return state_summary
    
# Define yaw-only angular velocity profiles
def angular_velocity_profile_1(t):
    return np.array([0.0, 0.0, 35.0 * np.sin(0.1 * t)]) * np.pi / 180

def angular_velocity_profile_2(t):
    return np.array([0.0, 0.0, 30.0 * np.cos(0.05 * t)]) * np.pi / 180

def angular_velocity_profile_3(t):
    return np.array([0.0, 0.0, 40.0]) * np.pi / 180  # Constant yaw velocity

def generate_attitude_changes(
        initial_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        time_step=(1 / simulation_hz),
        num_steps=int(simulation_time / (1 / simulation_hz)),
        monte_runs=NUM_MONTE_RUNS,
        acceleration_std=0.1 ):

    attitudes = np.zeros((monte_runs, num_steps, 4))
    accelerations = np.zeros((monte_runs, num_steps, 3))
    magnetic_fields = np.zeros((monte_runs, num_steps, 3))
    gyros = np.zeros((monte_runs, num_steps, 3))

    gravity_vector = np.array([0.0, 0.0, -1])
    magnetic_field_vector = np.array([1.0, 0.0, 0.0])

    profiles = [angular_velocity_profile_1, angular_velocity_profile_2, angular_velocity_profile_3]

    for monte_idx in range(monte_runs):
        # Assign yaw-only profile for each run
        angular_velocity_profile = profiles[monte_idx % len(profiles)]

        freq_modifier = 0.1 + 0.05 * monte_idx
        t_f = (num_steps - 1) * time_step
        attitude_sequence = integrator(
            lambda beta, t: get_beta_p_t(beta, t, freq_modifier, angular_velocity_profile=angular_velocity_profile)[0],
            initial_quat,
            0,
            t_f,
            time_step,
            norm_value=True
        )
        attitudes[monte_idx, :, :] = np.array(attitude_sequence[:num_steps])

        for step in range(num_steps):
            current_quat = attitudes[monte_idx, step]
            _, angular_velocity = get_beta_p_t(current_quat, step * time_step, freq_modifier,
                                               angular_velocity_profile=angular_velocity_profile)
            gyros[monte_idx, step] = angular_velocity

            # Simulate accelerometer readings with gravity and noise in x, y directions
            rotation = Rotation.from_quat(current_quat[[1, 2, 3, 0]])  # Convert to [qx, qy, qz, qw] order for SciPy
            accel_body_frame = rotation.apply(gravity_vector)

            # Add random noise to x, y directions
            random_acceleration = np.array([
                np.random.normal(0, acceleration_std),  # Random acceleration in x
                np.random.normal(0, acceleration_std),  # Random acceleration in y
                0  # No noise in z
            ])
            accelerations[monte_idx, step] = accel_body_frame + random_acceleration

            # Simulate magnetometer readings
            mag_body_frame = rotation.apply(magnetic_field_vector)
            magnetic_fields[monte_idx, step] = mag_body_frame + np.random.normal(0, np.sqrt(process_noise_var))

    return {'att': attitudes, 'acc': accelerations, 'mag': magnetic_fields, 'gyro': gyros}

def visualize_quaternion_trajectories(attitudes, dt=(1 / simulation_hz)):
    """
    Visualizes the quaternion trajectories for all Monte Carlo runs.

    Parameters:
    - attitudes (np.ndarray): 3D NumPy array with shape (monte_runs, num_steps, 4) storing quaternions.
    - dt (float): Time step between each step, in seconds.
    """
    monte_runs, num_steps, _ = attitudes.shape
    time_array = np.linspace(0, num_steps * dt, num_steps)

    fig, axs = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle("Quaternion Components Over Time for All Monte Carlo Runs")

    labels = ['q0 (scalar)', 'q1', 'q2', 'q3']
    for i in range(4):
        for monte_idx in range(monte_runs):
            axs[i].plot(time_array, attitudes[monte_idx, :, i], label=f'Run {monte_idx + 1}')
        axs[i].set_title(labels[i])
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def monte_att_estimate(monte_sum):
    """
    Runs the TRIAD method for attitude estimation at a lower frequency than the simulation.
    
    Parameters:
    - monte_sum: Contains arrays for true attitudes, accelerometer, magnetometer, and gyroscope data.
    
    Returns:
    - estimated_states_all: 3D array with shape (monte_runs, num_estimation_steps, 4).
    """
    estimation_step_interval = simulation_hz // estimation_hz
    monte_runs, num_steps = monte_sum['gyro'].shape[0], monte_sum['gyro'].shape[1]
    num_estimation_steps = num_steps // estimation_step_interval

    estimated_states_all = np.zeros((monte_runs, num_estimation_steps, 4))

    for monte_idx in range(monte_runs):
        accelerations = monte_sum['acc'][monte_idx]
        magnetic_fields = monte_sum['mag'][monte_idx]
        estimated_states = np.zeros((num_estimation_steps, 4))

        for step in range(num_estimation_steps):
            simulation_step = step * estimation_step_interval
            acceleration = accelerations[simulation_step]
            magnetic_field = magnetic_fields[simulation_step]

            # Use TRIAD method for estimation
            estimated_states[step] = triad_method(acceleration, magnetic_field)

        estimated_states_all[monte_idx] = estimated_states

    return estimated_states_all

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def triad_method(sensor_1, sensor_2, inertial_1=np.array([0, 0, -1]), inertial_2=np.array([1, 0, 0])):
    # Normalize inputs
    sensor_1 = normalize_quaternion(sensor_1)
    sensor_2 = normalize_quaternion(sensor_2)
    inertial_1 = normalize_quaternion(inertial_1)
    inertial_2 = normalize_quaternion(inertial_2)

    # TRIAD reference frames
    B_t1 = sensor_1
    B_t2 = np.cross(sensor_1, sensor_2) / np.linalg.norm(np.cross(sensor_1, sensor_2))
    B_t3 = np.cross(B_t1, B_t2)

    N_t1 = inertial_1
    N_t2 = np.cross(inertial_1, inertial_2) / np.linalg.norm(np.cross(inertial_1, inertial_2))
    N_t3 = np.cross(N_t1, N_t2)

    Bbar_T_dcm = np.stack((B_t1, B_t2, B_t3), axis=1)
    NT_dcm = np.stack((N_t1, N_t2, N_t3), axis=1)

    return dcm_to_quaternions(np.dot(Bbar_T_dcm, NT_dcm.T))

def dcm_to_quaternions(dcm):
    T = np.trace(dcm)
    M = max(dcm[0, 0], dcm[1, 1], dcm[2, 2], T)

    if M == dcm[0, 0]:
        qx = 0.5 * math.sqrt(1 + 2 * dcm[0, 0] - T)
        qy = (dcm[0, 1] + dcm[1, 0]) / (4 * qx)
        qz = (dcm[0, 2] + dcm[2, 0]) / (4 * qx)
        qw = (dcm[2, 1] - dcm[1, 2]) / (4 * qx)
    elif M == dcm[1, 1]:
        qy = 0.5 * math.sqrt(1 + 2 * dcm[1, 1] - T)
        qx = (dcm[0, 1] + dcm[1, 0]) / (4 * qy)
        qz = (dcm[1, 2] + dcm[2, 1]) / (4 * qy)
        qw = (dcm[0, 2] - dcm[2, 0]) / (4 * qy)
    elif M == dcm[2, 2]:
        qz = 0.5 * math.sqrt(1 + 2 * dcm[2, 2] - T)
        qx = (dcm[0, 2] + dcm[2, 0]) / (4 * qz)
        qy = (dcm[1, 2] + dcm[2, 1]) / (4 * qz)
        qw = (dcm[1, 0] - dcm[0, 1]) / (4 * qz)
    else:  # T is the largest value
        qw = 0.5 * math.sqrt(1 + T)
        qx = (dcm[2, 1] - dcm[1, 2]) / (4 * qw)
        qy = (dcm[0, 2] - dcm[2, 0]) / (4 * qw)
        qz = (dcm[1, 0] - dcm[0, 1]) / (4 * qw)

    return normalize_quaternion(np.array([qw, qx, qy, qz]))

def quaternion_error(true_quat, estimated_quat):
    true_rot = Rotation.from_quat(true_quat[[1, 2, 3, 0]])
    estimated_rot = Rotation.from_quat(estimated_quat[[1, 2, 3, 0]])
    relative_rot = true_rot.inv() * estimated_rot
    angle_error = relative_rot.magnitude()
    return np.degrees(angle_error)

def plot_quaternion_error(true_attitudes_all, estimated_attitudes_all, estimation_step_interval=simulation_hz // estimation_hz, dt=(1 / simulation_hz)):
    """
    Plots the quaternion error magnitude over time for each Monte Carlo run.

    Parameters:
    - true_attitudes_all (np.ndarray): True quaternions (shape: monte_runs x num_steps x 4).
    - estimated_attitudes_all (np.ndarray): Estimated quaternions (shape: monte_runs x num_estimation_steps x 4).
    - estimation_step_interval (int): Interval between simulation and estimation steps.
    - dt (float): Time step between each sample in seconds.
    """
    monte_runs, num_estimation_steps = estimated_attitudes_all.shape[0], estimated_attitudes_all.shape[1]
    time_array = np.linspace(0, num_estimation_steps * dt * estimation_step_interval, num_estimation_steps)

    plt.figure(figsize=(12, 6))
    plt.title("Quaternion Error Magnitude Over Time")

    for monte_idx in range(monte_runs):
        quaternion_errors = np.zeros(num_estimation_steps)
        for step in range(num_estimation_steps):
            true_quat = true_attitudes_all[monte_idx, step * estimation_step_interval]
            estimated_quat = estimated_attitudes_all[monte_idx, step]
            quaternion_errors[step] = quaternion_error(true_quat, estimated_quat)

        plt.plot(time_array, quaternion_errors)

    plt.xlabel('Time (s)')
    plt.ylabel('Quaternion Error (degrees)')
    plt.grid(True)
    plt.tight_layout()

def quaternion_to_euler(quat):
    """
    Converts a quaternion to Euler angles (roll, pitch, yaw).

    Parameters:
    - quat: Quaternion in the format [w, x, y, z].

    Returns:
    - euler_angles: Euler angles [roll, pitch, yaw] in degrees.
    """
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]])  # Convert to [x, y, z, w] for SciPy
    euler_angles = rotation.as_euler('xyz', degrees=True)
    return euler_angles

def plot_euler_angle_error(true_attitudes_all, estimated_attitudes_all, estimation_step_interval=simulation_hz // estimation_hz, dt=(1 / simulation_hz)):
    """
    Plots the error in Euler angles (roll, pitch, yaw) over time for each Monte Carlo run.

    Parameters:
    - true_attitudes_all (np.ndarray): True quaternions (shape: monte_runs x num_steps x 4).
    - estimated_attitudes_all (np.ndarray): Estimated quaternions (shape: monte_runs x num_estimation_steps x 4).
    - estimation_step_interval (int): Interval between simulation and estimation steps.
    - dt (float): Time step between each sample in seconds.
    """
    monte_runs, num_estimation_steps = estimated_attitudes_all.shape[0], estimated_attitudes_all.shape[1]
    time_array = np.linspace(0, num_estimation_steps * dt * estimation_step_interval, num_estimation_steps)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Euler Angle Error Over Time for All Monte Carlo Runs")

    euler_labels = ['Roll (degrees)', 'Pitch (degrees)', 'Yaw (degrees)']
    for monte_idx in range(monte_runs):
        euler_errors = np.zeros((num_estimation_steps, 3))  # Store errors for roll, pitch, yaw
        for step in range(num_estimation_steps):
            true_quat = true_attitudes_all[monte_idx, step * estimation_step_interval]
            estimated_quat = estimated_attitudes_all[monte_idx, step]
            
            # Convert quaternions to Euler angles
            true_euler = quaternion_to_euler(true_quat)
            estimated_euler = quaternion_to_euler(estimated_quat)
            
            # Compute error as difference in Euler angles
            euler_errors[step] = true_euler - estimated_euler

        for i in range(3):  # Roll, Pitch, Yaw
            axs[i].plot(time_array, euler_errors[:, i])

    for i in range(3):
        axs[i].set_title(euler_labels[i])
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel(euler_labels[i])
        axs[i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


if __name__ == "__main__":

    monte_sum = generate_attitude_changes()

    ekf_sum = monte_att_estimate(monte_sum)

    visualize_quaternion_trajectories(monte_sum['att'])
    plot_quaternion_error(monte_sum['att'], ekf_sum)
    plot_euler_angle_error(monte_sum['att'], ekf_sum)

    plt.show()
