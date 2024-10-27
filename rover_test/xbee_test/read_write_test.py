import serial 

PORT = 'COM4'
BAUD_RATE = 115200

xbee = serial.Serial(PORT, BAUD_RATE)

while True:

    line = xbee.readline().decode('utf-8').strip()  # Decode and strip newlines
    if line:
        print(f"Received: {line}")
