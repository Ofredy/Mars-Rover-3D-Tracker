import numpy as np
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt

from rover_att_ekf import *

simulation_hz = 100  # 10 Hz simulation frequency
simulation_time = 5  # Run for 5 seconds
NUM_MONTE_RUNS = 10  # Number of Monte Carlo runs


def generate_attitude_changes(initial_quat=np.array([1, 0, 0, 0]), time_step=(1/simulation_hz), num_steps=int(simulation_time/(1/simulation_hz)), monte_runs=NUM_MONTE_RUNS):
    """
    Generates a Monte Carlo array of attitude changes by integrating random angular velocities using RK4.
    
    Parameters:
    - initial_quat (np.ndarray): Initial quaternion as a 4-element array [w, x, y, z].
    - time_step (float): Time step for integration, in seconds.
    - num_steps (int): Number of attitude changes (time steps) to generate per Monte Carlo run.
    - monte_runs (int): Number of Monte Carlo runs (independent attitude sequences).
    
    Returns:
    - A dictionary with:
      - attitudes (np.ndarray): 3D array with shape (monte_runs, num_steps, 4) storing quaternions.
      - accelerations (np.ndarray): 3D array with shape (monte_runs, num_steps, 3) storing accelerometer data.
      - magnetic_fields (np.ndarray): 3D array with shape (monte_runs, num_steps, 3) storing magnetometer data.
      - gyros (np.ndarray): 3D array with shape (monte_runs, num_steps, 3) storing gyroscope (angular velocity) data.
    """
    # Initialize arrays to store quaternions, accelerometer, magnetometer, and gyroscope data
    attitudes = np.zeros((monte_runs, num_steps, 4))
    accelerations = np.zeros((monte_runs, num_steps, 3))
    magnetic_fields = np.zeros((monte_runs, num_steps, 3))
    gyros = np.zeros((monte_runs, num_steps, 3))  # To store gyroscope data

    # Fixed gravity and magnetic field vectors in the inertial frame
    gravity_vector = np.array([0, 0, -9.81])  # Assume gravity along -Z in m/s^2
    magnetic_field_vector = np.array([0.2, 0, 0.5])  # Example magnetic field in the X-Z plane

    for monte_idx in range(monte_runs):
        # Set initial attitude for each Monte Carlo run
        current_quat = initial_quat

        for step in range(num_steps):
            # Store the current quaternion
            attitudes[monte_idx, step] = current_quat

            # Generate a random angular velocity vector (radians per second)
            angular_velocity = np.random.uniform(-0.1, 0.1, 3)
            gyros[monte_idx, step] = angular_velocity  # Store the gyroscope measurement

            # Perform RK4 integration
            current_quat = rk4_quaternion_integration(current_quat, angular_velocity, time_step)

            # Simulate accelerometer measurement in the body frame
            rotation = Rotation.from_quat(current_quat)
            accel_body_frame = rotation.apply(gravity_vector)
            noise = np.random.normal(0, np.sqrt(process_noise_var), 3)  # Add some noise
            accelerations[monte_idx, step] = accel_body_frame + noise

            # Simulate magnetometer measurement in the body frame
            mag_body_frame = rotation.apply(magnetic_field_vector)
            noise = np.random.normal(0, np.sqrt(process_noise_var), 3)  # Add some smaller noise
            magnetic_fields[monte_idx, step] = mag_body_frame + noise

    # Create a dictionary to return the generated data
    monte_sum = {
        'att': attitudes,
        'acc': accelerations,
        'mag': magnetic_fields,
        'gyro': gyros
    }
    return monte_sum

def rk4_quaternion_integration(quat, angular_velocity, dt):
    """
    Performs Runge-Kutta 4th order integration to update quaternion based on angular velocity.
    
    Parameters:
    - quat (np.ndarray): Current quaternion [w, x, y, z].
    - angular_velocity (np.ndarray): Angular velocity vector [wx, wy, wz].
    - dt (float): Time step.
    
    Returns:
    - np.ndarray: Updated quaternion [w, x, y, z].
    """
    def quaternion_derivative(q, omega):
        omega_quat = np.hstack(([0], omega))  # Quaternion for angular velocity
        return 0.5 * quaternion_multiply(q, omega_quat)

    k1 = quaternion_derivative(quat, angular_velocity) * dt
    k2 = quaternion_derivative(quat + 0.5 * k1, angular_velocity) * dt
    k3 = quaternion_derivative(quat + 0.5 * k2, angular_velocity) * dt
    k4 = quaternion_derivative(quat + k3, angular_velocity) * dt

    # Update quaternion
    new_quat = quat + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return new_quat / np.linalg.norm(new_quat)  # Normalize to prevent drift

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
    plt.show()

def monte_att_ekf(monte_sum, dt=(1/simulation_hz)):
    """
    Runs the Kalman filter over multiple Monte Carlo runs of attitude, accelerometer, magnetometer, and gyroscope data.
    
    Parameters:
    - monte_sum (dict): Contains arrays for true attitudes, accelerometer, magnetometer, and gyroscope data.
    - dt (float): Time step.
    
    Returns:
    - estimated_states_all (np.ndarray): 3D array with shape (monte_runs, num_steps, 4) storing estimated quaternions.
    """
    monte_runs, num_steps = monte_sum['gyro'].shape[0], monte_sum['gyro'].shape[1]
    estimated_states_all = np.zeros((monte_runs, num_steps, 4))  # To store estimated quaternions for all runs

    for monte_idx in range(monte_runs):
        accelerations = monte_sum['acc'][monte_idx]
        magnetic_fields = monte_sum['mag'][monte_idx]
        gyros = monte_sum['gyro'][monte_idx]

        # Initialize the Kalman filter for each run
        state, P = initialize_kalman_filter()
        estimated_states = np.zeros((num_steps, 4))  # To store quaternion estimates for this run

        for step in range(num_steps):
            # Prediction step using the actual gyro measurement from the data
            angular_velocity = gyros[step]
            state, P = predict(state, P, dt, angular_velocity)
            
            # Measurement step using TRIAD with accelerometer and magnetometer data
            acceleration = accelerations[step]
            magnetic_field = magnetic_fields[step]
            measured_quat = measure_with_triad(acceleration, magnetic_field)
            
            # Update step
            state, P = update(state, P, measured_quat, R)
            
            # Store quaternion estimate
            estimated_states[step] = state[:4]

        # Store the estimated states for this Monte Carlo run
        estimated_states_all[monte_idx] = estimated_states

    return estimated_states_all

def plot_attitude_error(true_attitudes_all, estimated_attitudes_all, dt=(1/simulation_hz)):
    """
    Plots the attitude error (in degrees) between the true and estimated quaternions over time for all Monte Carlo runs.
    
    Parameters:
    - true_attitudes_all (np.ndarray): 3D array with shape (monte_runs, num_steps, 4) storing true quaternions.
    - estimated_attitudes_all (np.ndarray): 3D array with shape (monte_runs, num_steps, 4) storing estimated quaternions.
    - dt (float): Time step in seconds.
    """
    monte_runs, num_steps = true_attitudes_all.shape[0], true_attitudes_all.shape[1]
    time_array = np.linspace(0, num_steps * dt, num_steps)

    plt.figure(figsize=(12, 6))
    
    for monte_idx in range(monte_runs):
        true_attitudes = true_attitudes_all[monte_idx]
        estimated_attitudes = estimated_attitudes_all[monte_idx]
        
        # Calculate attitude error for this run
        attitude_errors = np.zeros(num_steps)
        for i in range(num_steps):
            true_rot = Rotation.from_quat(true_attitudes[i])
            est_rot = Rotation.from_quat(estimated_attitudes[i])
            
            # Compute relative rotation between true and estimated orientation
            relative_rotation = true_rot.inv() * est_rot
            attitude_errors[i] = relative_rotation.magnitude() * (180 / np.pi)  # Convert radians to degrees

        # Plot the error for this run
        plt.plot(time_array, attitude_errors)

    plt.xlabel('Time (s)')
    plt.ylabel('Attitude Error (degrees)')
    plt.title('Kalman Filter Attitude Error Over Time')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    monte_sum = generate_attitude_changes()

    ekf_sum = monte_att_ekf(monte_sum)

    plot_attitude_error(monte_sum['att'], ekf_sum)
