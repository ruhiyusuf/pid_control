import numpy as np
import matplotlib.pyplot as plt
from redpitaya_scpi import scpi
import time

### ENTER INPUTS HERE
# Input parameters for the waveform
v_start = 0 # In volts (V)
v_end = 1 # In volts (V)
v_down_end = 1 * v_end  # Voltage achieved at the end of the downward slope

# Input voltage duration in miliseconds 
start_v_ms = 20
slope_up_ms = 6 
slope_down_ms = 0 
end_v_ms = 500 

### DO NOT EDIT BELOW THIS LINE 
start_v_seconds = start_v_ms / 1000 
slope_up_seconds = slope_up_ms / 1000 
slope_down_seconds = slope_down_ms / 1000 
end_v_seconds = end_v_ms / 1000 

sampling_rate = 1000  # Number of samples per second
num_samples = 16384 # Length of one period of the custom signal
total_time = start_v_seconds + slope_up_seconds + slope_down_seconds + end_v_seconds 
frequency = 1 / (total_time)  # Calculate frequency based on one period duration
# Time vector for one period
t_waveform = np.linspace(0, total_time, num_samples)

# Generate full waveform with both slopes
waveform = np.piecewise(t_waveform, 
    [t_waveform < start_v_seconds, 
     (t_waveform >= start_v_seconds) & (t_waveform < start_v_seconds + slope_up_seconds), 
     (t_waveform >= start_v_seconds + slope_up_seconds) & (t_waveform < start_v_seconds + slope_up_seconds + slope_down_seconds),
     t_waveform >= start_v_seconds + slope_up_seconds + slope_down_seconds], 
    [lambda t: v_start,  # DC start duration
     lambda t: v_start + (v_end - v_start) * ((t - start_v_seconds) / slope_up_seconds),  # Upward slope
     lambda t: v_end + (v_down_end - v_end) * ((t - start_v_seconds - slope_up_seconds) / slope_down_seconds),  # Downward slope
     lambda t: v_down_end])  # DC end duration    lambda t: v_down_end])  # DC end

# Plot the waveform
plt.figure()
plt.plot(t_waveform, waveform, linewidth=1.5, label='Combined Slope Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Single Waveform with Upward and Downward Slope')
plt.legend()
plt.grid()
plt.show()

# Red Pitaya configuration
rp_ip = "rp-f0cbc6.local"  # Replace with your Red Pitaya's hostname
rp = scpi(rp_ip)

print("initialized")
# Reset generator
rp.tx_txt("GEN:RST")

# Deploy waveform to Red Pitaya
rp.tx_txt("SOUR1:FUNC ARBITRARY")
rp.tx_txt("SOUR1:TRAC:DATA:DATA " + ",".join(map(str, waveform)))
rp.tx_txt("SOUR1:FREQ:FIX {}".format(frequency))  # Set frequency
# rp.tx_txt("SOUR1:VOLT 1")  # Ensure amplitude is set
rp.tx_txt("SOUR1:VOLT:OFFS 0")  # Ensure offset is set correctly
rp.tx_txt("OUTPUT1:STATE ON")  # Explicitly turn on output
rp.tx_txt("SOUR1:TRIG:INT")
print("waveform")

# Close connection
rp.close()

# Indicate completion
print("Custom waveform deployed to Red Pitaya.")



