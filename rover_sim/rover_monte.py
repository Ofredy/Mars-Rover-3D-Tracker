# library imports
import numpy as np
import matplotlib.pyplot as plt

# our imports
from rover_ekf import *


######### monte constants #########
NUM_MONTE_RUNS = 50

acceleration_std = 0.3

initial_state_std = 0.01

simulation_time = 50 # s
simulation_hz = 100

np.random.seed(69)


def rover_x_dot(x_n, acceleration_n):

    # EXTRA -> Add attitude

    return np.array([
                     x_n[1],
                     acceleration_n[0],
                     0,
                     x_n[4],
                     acceleration_n[1],
                     0,
                     0,
                     0,
                     0,
                    ])

def runge_kutta(get_x_dot, x_0, t_0, t_f, dt=0.1):

    # EXTRA -> Add attitude

    state_summary = np.zeros(shape=((int(simulation_time/(1/simulation_hz)), NUM_STATES))) 
    acceleration_summary =  np.zeros(shape=((int(simulation_time/(1/simulation_hz)), 3))) 

    t = t_0
    time_step_idx = 0
    x_n = x_0

    while t < t_f:

        # Generate random acceleration (constant during the whole time step)
        acceleration = np.array([np.random.normal(0, acceleration_std),  # Random acceleration in x,
                                 np.random.normal(0, acceleration_std),  # Random acceleration in y,
                                 0]) # Random acceleration in z direction

        state_summary[time_step_idx] = x_n
        acceleration_summary[time_step_idx] = acceleration

        time_step_idx += 1

        # Runge-Kutta stages using the same acceleration
        k1 = dt * get_x_dot(x_n, acceleration)
        k2 = dt * get_x_dot(x_n + 0.5 * k1, acceleration)
        k3 = dt * get_x_dot(x_n + 0.5 * k2, acceleration)
        k4 = dt * get_x_dot(x_n + k3, acceleration)

        # Update the state using the weighted average of the stages
        x_n = x_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        t = dt * time_step_idx

    return { 'state_sum': state_summary,
             'acc_sum': acceleration_summary }

def generate_rover_trajectories():

    rover_trajectories = []

    for _ in range(NUM_MONTE_RUNS):

        initial_state = np.random.normal(0, initial_state_std, size=NUM_STATES)

        temp_monte_sum = runge_kutta(rover_x_dot, initial_state, 0, simulation_time, (1/simulation_hz))

        rover_trajectories.append(temp_monte_sum)

    return rover_trajectories

def add_process_noise_to_rover_trajectories(rover_trajectories):

    for monte_idx, run_hash in enumerate(rover_trajectories):

        run_hash['state_sum'] += np.random.normal(0, np.sqrt(process_noise_variance))
        run_hash['acc_sum'] += np.random.normal(0, np.sqrt(process_noise_variance))
        
        rover_trajectories[monte_idx]['state_sum'] = run_hash['state_sum']
        rover_trajectories[monte_idx]['acc_sum'] = run_hash['acc_sum']

    return rover_trajectories

def plot_rover_trajectory(rover_trajectories, fig_num=1, save_as_png=False, dpi=300):

    plt.figure(fig_num)

    ax = plt.axes(projection='3d')  # Create a 3D axis

    for run_hash in rover_trajectories:
        ax.plot(run_hash['state_sum'][:, 0], run_hash['state_sum'][:, 3], run_hash['state_sum'][:, 6])
    
    ax.set_title('Positions of Mars Rover Trajectories')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position') 

    # Save as PNG with higher DPI if requested
    if save_as_png:
        plt.savefig(f'rover_trajectories.png', format='png', dpi=dpi)

def state_init():

    tau_a = np.sqrt(x_0_guess_variance)    # Replace with actual tau_a
    tau_b = np.sqrt(x_0_guess_variance)    # Replace with actual tau_b

    # Process noise submatrix Q0 for each axis, replacing T with 1/imu_hz
    Q0 = np.array([
        [T**3 / 3 * tau_a + T**5 / 20 * tau_b, T**2 / 2 * tau_a + T**4 / 8 * tau_b, -T**3 / 6 * tau_b],
        [T**2 / 2 * tau_a + T**4 / 8 * tau_b, T * tau_a + T**3 / 3 * tau_b, -T**2 / 2 * tau_b],
        [-T**3 / 6 * tau_b, -T**2 / 2 * tau_b, T * tau_b]
    ])

    # Full process noise matrix Qk-1 for the state vector [x, y, z, vx, vy, vz, ax, ay, az]
    P_n = np.block([
        [Q0, np.zeros_like(Q0), np.zeros_like(Q0)],  # x-axis
        [np.zeros_like(Q0), Q0, np.zeros_like(Q0)],  # y-axis
        [np.zeros_like(Q0), np.zeros_like(Q0), Q0]   # z-axis
    ])

    return np.array([np.random.normal(0, np.sqrt(x_0_guess_variance), size=NUM_STATES)]).reshape(-1), \
           P_n

def rover_ekf_simulations(rover_trajectories):

    monte_ekf_estimates = np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/(1/imu_hz)), NUM_STATES)) if imu_hz > ranging_hz else np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/(1/ranging_hz)), NUM_STATES)) 
    monte_covaraince_time_steps = np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/(1/imu_hz)), NUM_STATES, NUM_STATES)) if imu_hz > ranging_hz else np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/(1/ranging_hz)), NUM_STATES, NUM_STATES))

    imu_rate_idx = (1/imu_hz) / (1/simulation_hz)
    ranging_rate_idx = (1/ranging_hz) / (1/simulation_hz)

    for run_idx in range(NUM_MONTE_RUNS):

        run_states = rover_trajectories[run_idx]['state_sum']
        run_acc = rover_trajectories[run_idx]['acc_sum']

        x_n, P_n = state_init()
        estimate_counter = 0

        beacon_idx = 0

        for time_step_idx in range(int(simulation_time/(1/simulation_hz))):

            new_estimate = False

            if time_step_idx % imu_rate_idx == 0:

                # state prediction
                acc_measurement = run_acc[time_step_idx]
                x_n, P_n = ekf_predict_t(x_n, P_n, acc_measurement)

                new_estimate = True

                error = np.linalg.norm(x_n - run_states[time_step_idx])

            if time_step_idx % ranging_rate_idx == 0:
                # state update
                true_position = np.array([run_states[time_step_idx][0], run_states[time_step_idx][3], run_states[time_step_idx][6]])

                ranging_buffer = np.linalg.norm( true_position - beacons[beacon_idx] ) + np.random.normal(0, np.sqrt(measurement_noise_variance))
                x_n, P_n = ekf_update_t(x_n, P_n, ranging_buffer, beacon_idx)

                beacon_idx = ( beacon_idx + 1 ) % NUM_BEACONS
                new_estimate = True

                error = np.linalg.norm(x_n - run_states[time_step_idx])

            if new_estimate:

                monte_ekf_estimates[run_idx][estimate_counter] = x_n
                monte_covaraince_time_steps[run_idx][estimate_counter] = P_n

                estimate_counter += 1
            
    ekf_sim_sum = { 'rover_trajectories': rover_trajectories,
                    'monte_ekf_estimates': monte_ekf_estimates,
                    'monte_covariance_time_steps': monte_covaraince_time_steps }

    return ekf_sim_sum

def plot_rover_ekf_error(ekf_sim_sum, fig_num=1, save_as_png=False, dpi=300):

    plt.figure(fig_num)

    t = np.arange(0, simulation_time, (1/imu_hz)) if imu_hz > ranging_hz else np.arange(0, simulation_time, (1/ranging_hz))

    for run_idx in range(NUM_MONTE_RUNS):

        rover_truth_states = ekf_sim_sum['rover_trajectories'][run_idx]['state_sum']
        time_indices = np.arange(int(simulation_time/(1/simulation_hz))) % int((1/imu_hz)/(1/simulation_hz)) == 0 if imu_hz > ranging_hz else np.arange(int(simulation_time/(1/simulation_hz))) % int((1/ranging_hz)/(1/simulation_hz)) == 0
        ekf_estimates = ekf_sim_sum['monte_ekf_estimates'][run_idx]

        state_errors = rover_truth_states[time_indices] - ekf_estimates

        if run_idx == 0:
            plt.plot(t, np.linalg.norm(state_errors, axis=1), label=f'Total State Error', color='k')
            plt.xlabel('Time [s]')
            plt.ylabel('Error')
            plt.legend(loc='upper right')
            plt.grid(True)

        else:
            plt.plot(t, np.linalg.norm(state_errors, axis=1), color='k')

    # Save as PNG with higher DPI if requested
    if save_as_png:
        plt.savefig(f'rover_ekf_error.png', format='png', dpi=dpi)

def plot_rover_position_error(ekf_sim_sum, save_as_png=False, dpi=300):

    fig, axs = plt.subplots(3, figsize=(10, 12), sharex=True)

    state_labels = [ 'x', 'vx', 'ax', 'y', 'vy', 'ay', 'z', 'vz', 'az' ]

    t = np.arange(0, simulation_time, (1/imu_hz)) if imu_hz > ranging_hz else np.arange(0, simulation_time, (1/ranging_hz))

    for run_idx in range(NUM_MONTE_RUNS):

        rover_truth_states = ekf_sim_sum['rover_trajectories'][run_idx]['state_sum']
        time_indices = np.arange(int(simulation_time/(1/simulation_hz))) % int((1/imu_hz)/(1/simulation_hz)) == 0 if imu_hz > ranging_hz else np.arange(int(simulation_time/(1/simulation_hz))) % int((1/ranging_hz)/(1/simulation_hz)) == 0
        ekf_estimates = ekf_sim_sum['monte_ekf_estimates'][run_idx]
        covariance_time_steps = ekf_sim_sum['monte_covariance_time_steps'][run_idx]

        state_errors = rover_truth_states[time_indices] - ekf_estimates

        for state_idx in range(3):    

            position_idx = [0, 3, 6]

            confidence_interval_upper = state_errors[:, position_idx[state_idx]] + 3 * covariance_time_steps[:, position_idx[state_idx], position_idx[state_idx]]
            confidence_interval_lower = state_errors[:, position_idx[state_idx]] - 3 * covariance_time_steps[:, position_idx[state_idx], position_idx[state_idx]]

            if run_idx == 0:
                axs[state_idx].plot(t, state_errors[:, position_idx[state_idx]], label=f'{state_labels[position_idx[state_idx]]} Error', color='k')
                axs[state_idx].fill_between(t, confidence_interval_lower, confidence_interval_upper, color='y', alpha=0.5, label='3-σ Uncertainty in Theta')
                axs[state_idx].set_ylabel('Error')
                axs[state_idx].legend(loc='upper right')
                axs[state_idx].grid(True)

            else:
                axs[state_idx].plot(t, state_errors[:, position_idx[state_idx]], color='k')
                axs[state_idx].fill_between(t, confidence_interval_lower, confidence_interval_upper, color='y', alpha=0.5)

        if run_idx == 0:
            axs[-1].set_xlabel('Time')

        fig.suptitle('State Errors Over Time', fontsize=16)

    # Save as PNG with higher DPI if requested
    if save_as_png:
        plt.savefig('rover_position_error.png', format='png', dpi=dpi)


if __name__ == "__main__":

    rover_trajectories = generate_rover_trajectories()
    rover_trajectories = add_process_noise_to_rover_trajectories(rover_trajectories)

    ekf_sim_sum = rover_ekf_simulations(rover_trajectories)

    plot_rover_trajectory(rover_trajectories, fig_num=1, save_as_png=True)
    plot_rover_ekf_error(ekf_sim_sum, fig_num=2, save_as_png=True)
    plot_rover_position_error(ekf_sim_sum, save_as_png=True)

    plt.show()
