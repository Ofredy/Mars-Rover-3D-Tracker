import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.backend_bases import MouseEvent

# Set up serial connection (assuming XBEE is connected)
ser = serial.Serial('COM4', 115200)  # Adjust COM port and baud rate as needed

# Initial positions
estimated_position = np.array([-10.0, 10.0])
actual_position = np.array([-8.0, 12.0])
beacon_positions = np.array([[0, 0], [10, -10], [-10, -10]])

# Initialize the plot
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_title("Mars Rover Rendering")

# Scatter plot elements
est_sc = ax.scatter([], [], c='blue', label='Estimated Position')
act_sc = ax.scatter([], [], c='green', label='Actual Position')
beacon_sc = ax.scatter(beacon_positions[:,0], beacon_positions[:,1], c='red', marker='x', label='Beacons')
error_line, = ax.plot([], [], 'r--', label='Error')

# Target point placeholder (initially set to None)
target_point = None

# Update function
def update_plot():
    est_sc.set_offsets(estimated_position)
    act_sc.set_offsets(actual_position)
    error_line.set_data([estimated_position[0], actual_position[0]], [estimated_position[1], actual_position[1]])
    plt.draw()

# Mouse click function
def on_click(event: MouseEvent):
    global target_point
    if event.xdata is not None and event.ydata is not None:
        destination = np.array([event.xdata, event.ydata])
        print("New destination:", destination)
        
        # Send destination to rover
        ser.write(f"{destination[0]},{destination[1]}\n".encode())
        
        # Remove previous target point, if it exists
        if target_point:
            target_point.remove()
        
        # Mark the new target point
        target_point = ax.scatter(destination[0], destination[1], c='purple', marker='+', label='Target')
        plt.legend()
        plt.draw()

# Attach the click event to the figure
fig.canvas.mpl_connect('button_press_event', on_click)

# Real-time loop with exception handling for plot closure
try:
    while plt.fignum_exists(fig.number):  # Check if the figure window is still open
        # Read incoming data from rover (estimated position)
        if ser.in_waiting > 0:
            data = ser.readline().decode().strip()
            x, y = map(float, data.split(','))
            estimated_position[:] = [x, y]  # Update estimated position
        
        # Simulate actual position update (for testing)
        actual_position += np.random.uniform(-0.1, 0.1, 2)
        
        # Update plot with new positions
        update_plot()
        plt.pause(0.1)

except KeyboardInterrupt:
    print("Exiting program.")

finally:
    # Turn off interactive mode and close the plot
    plt.ioff()
    plt.close(fig)
    # Close serial connection when finished
    ser.close()
    print("Plot closed and serial connection closed.")
