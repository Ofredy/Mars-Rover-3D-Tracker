# library imports
import numpy as np
import matplotlib.pyplot as plt

# our imports
from rover_ekf import *


######### monte constants #########
NUM_MONTE_RUNS = 50

initial_state_std = 0.1

simulation_time = 20 # s
simulation_hz = 20


def rover_x_dot(x_n, acceleration_n):

    # EXTRA -> Add attitude

    return np.concatenate(( 
                           x_n[3:6],
                           acceleration_n    
                         ))

def runge_kutta(get_x_dot, x_0, t_0, t_f, dt=0.1):

    # EXTRA -> Add attitude

    state_summary = np.zeros(shape=((int(simulation_time/(1/simulation_hz)), NUM_STATES))) 
    acceleration_summary = np.zeros(shape=((int(simulation_time/(1/simulation_hz)), 3)))

    t = t_0
    time_step_idx = 0
    x_n = x_0

    while t < t_f:

        state_summary[time_step_idx] = x_n

        # Generate random acceleration (constant during the whole time step)
        acceleration = np.array([np.random.normal(0, 0.1),  # Random acceleration in x,
                                 np.random.normal(0, 0.1),  # Random acceleration in y,
                                 0.0])                       # Zero acceleration in z direction

        acceleration_summary[time_step_idx] = acceleration

        time_step_idx += 1

        # Runge-Kutta stages using the same acceleration
        k1 = dt * get_x_dot(x_n, acceleration)
        k2 = dt * get_x_dot(x_n + 0.5 * k1, acceleration)
        k3 = dt * get_x_dot(x_n + 0.5 * k2, acceleration)
        k4 = dt * get_x_dot(x_n + k3, acceleration)

        # Update the state using the weighted average of the stages
        x_n = x_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        t += dt

    return state_summary, acceleration_summary

def generate_rover_trajectories():

    rover_trajectories = []

    for _ in range(NUM_MONTE_RUNS):

        initial_state = np.random.normal(0, initial_state_std, size=NUM_STATES)

        temp_state_sum, temp_acceleration_sum = runge_kutta(rover_x_dot, initial_state, 0, simulation_time, (1/simulation_hz))

        temp_monte_sum = { 'state_sum': temp_state_sum,
                           'acceleration_sum': temp_acceleration_sum }

        rover_trajectories.append(temp_monte_sum)

    return rover_trajectories

def add_process_noise_to_rover_trajectories(rover_trajectories):

    for monte_idx, run_hash in enumerate(rover_trajectories):

        run_hash['state_sum'] += np.random.normal(0, np.sqrt(process_noise_variance))
        run_hash['acceleration_sum'] += np.random.normal(0, np.sqrt(process_noise_variance))
        
        rover_trajectories[monte_idx]['state_sum'] = run_hash['state_sum']
        rover_trajectories[monte_idx]['acceleration_sum'] = run_hash['acceleration_sum']

    return rover_trajectories

def plot_rover_trajectory(rover_trajectories, fig_num=1):

    plt.figure(fig_num)

    ax = plt.axes(projection='3d')  # Create a 3D axis

    for run_hash in rover_trajectories:
        ax.plot(run_hash['state_sum'][:, 0], run_hash['state_sum'][:, 1], run_hash['state_sum'][:, 2])
    
    ax.set_title('Positions of Mars Rover Trajectories')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position') 

def state_init():

    return np.array([np.random.normal(0, np.sqrt(x_0_guess_variance), size=NUM_STATES)]).reshape(-1), \
           np.eye(NUM_STATES) * x_0_guess_variance

def rover_ekf_simulations(rover_trajectories):

    monte_ekf_estimates = np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/(1/imu_hz)), NUM_STATES))
    monte_covaraince_time_steps = np.zeros(shape=(NUM_MONTE_RUNS, int(simulation_time/(1/imu_hz)), NUM_STATES, NUM_STATES))

    for run_idx in range(NUM_MONTE_RUNS):

        run_states = rover_trajectories[run_idx]['state_sum']
        run_accelerations = rover_trajectories[run_idx]['acceleration_sum']

        x_n, P_n = state_init()

        estimate_counter = 0

        beacon_idx = 0
        ranging_buffer = np.zeros(3)

        for time_step_idx in range(int(simulation_time/(1/simulation_hz))):

            if time_step_idx % (1/imu_hz) == 0:

                # state prediction
                imu_measurements = np.concatenate((run_states[time_step_idx][3:6], run_accelerations[time_step_idx]))
                x_pred_n, P_pred_n = ekf_predict_t(x_n, P_n, imu_measurements)

                if time_step_idx % (1/ranging_hz) == 0:

                    # state update
                    ranging_buffer[beacon_idx] = np.linalg.norm(run_states[time_step_idx][0:3] - beacons[beacon_idx]) + np.random.normal(0, np.sqrt(measurement_noise_variance))

                    x_n, P_n, k_n = ekf_update_t(x_pred_n, P_pred_n, ranging_buffer)

                else:

                    x_n = x_pred_n
                    P_n = P_pred_n

                monte_ekf_estimates[run_idx][estimate_counter] = x_n
                monte_covaraince_time_steps[run_idx][estimate_counter] = P_n

                estimate_counter += 1
                beacon_idx = ( beacon_idx + 1 ) % NUM_RANGING_MEASUREMENTS
                ranging_buffer.fill(0.0)

    ekf_sim_sum = { 'rover_trajectories': rover_trajectories,
                    'monte_ekf_estimates': monte_ekf_estimates,
                    'monte_covariance_time_steps': monte_covaraince_time_steps }

    return ekf_sim_sum

def plot_rover_ekf_error(ekf_sim_sum):

    fig, axs = plt.subplots(3, figsize=(10, 12), sharex=True)

    state_labels = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']

    t = np.arange(0, simulation_time, (1/imu_hz))

    for run_idx in range(NUM_MONTE_RUNS):

        rover_truth_states = ekf_sim_sum['rover_trajectories'][run_idx]['state_sum']
        time_indices = np.arange(int(simulation_time/(1/simulation_hz))) % int((1/imu_hz)/(1/simulation_hz)) == 0
        ekf_estimates = ekf_sim_sum['monte_ekf_estimates'][run_idx]
        covariance_time_steps = ekf_sim_sum['monte_covariance_time_steps'][run_idx]

        state_errors = rover_truth_states[time_indices] - ekf_estimates

        for state_idx in range(NUM_STATES-3):    

            confidence_interval_upper = state_errors[:, state_idx] + 3 * covariance_time_steps[:, state_idx, state_idx]
            confidence_interval_lower = state_errors[:, state_idx] - 3 * covariance_time_steps[:, state_idx, state_idx]

            if run_idx == 0:
                axs[state_idx].plot(t, state_errors[:, state_idx], label=f'{state_labels[state_idx]} Error', color='k')
                axs[state_idx].fill_between(t, confidence_interval_lower, confidence_interval_upper, color='y', alpha=0.5, label='3-σ Uncertainty in Theta')
                axs[state_idx].set_ylabel('Error')
                axs[state_idx].legend(loc='upper right')
                axs[state_idx].grid(True)

            else:
                axs[state_idx].plot(t, state_errors[:, state_idx], color='k')
                axs[state_idx].fill_between(t, confidence_interval_lower, confidence_interval_upper, color='y', alpha=0.5)

        if run_idx == 0:
            axs[-1].set_xlabel('Time')

        fig.suptitle('State Errors Over Time', fontsize=16)
    

if __name__ == "__main__":

    rover_trajectories = generate_rover_trajectories()
    rover_trajectories = add_process_noise_to_rover_trajectories(rover_trajectories)

    ekf_sim_sum = rover_ekf_simulations(rover_trajectories)
    plot_rover_ekf_error(ekf_sim_sum)

    plt.show()
