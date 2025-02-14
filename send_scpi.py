import socket

# Define Red Pitaya connection settings
HOST = "rp-f0cbc6.local"  # Replace with your Red Pitaya's hostname or IP address
PORT = 5000  # SCPI server port

# Function to send SCPI commands
def send_scpi_command(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall((command + "\n").encode())
        response = s.recv(1024).decode().strip()
        return response

# Example SCPI commands to generate a sine wave
print(send_scpi_command("GEN:RST"))  # Reset generator
print(send_scpi_command("SOUR1:FUNC SINE"))  # Set waveform type to Sine
print(send_scpi_command("SOUR1:FREQ:FIX 1000"))  # Set frequency to 1000 Hz
print(send_scpi_command("SOUR1:VOLT 1"))  # Set amplitude to 1 V
print(send_scpi_command("SOUR1:VOLT:OFFS 0"))  # Set DC offset to 0
print(send_scpi_command("OUTPUT1:STATE ON"))  # Enable output on channel 1

