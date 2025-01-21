import numpy as np
import matplotlib.pyplot as plt
from redpitaya_scpi import scpi

# Parameters for the waveform
frequency = 10  # Frequency of the waveform
period = 10 # Period of the waveform
num_cycles = 1  # Number of cycles to generate
sampling_rate = 1000  # Number of samples per second
offset = 0
v_start = 0 
v_end = 1 
slope_percent = 0.2 
amplitude = v_end - v_start  # Peak amplitude of the waveform

# Time vector
t = np.arange(0, period, 1 / sampling_rate)


# ramp up, for laser, correct
# waveform = np.where(t < slope_percent/frequency, v_start + (v_end - v_start)*(t/(slope_percent/frequency)), v_end)

# ORIGINAL< IT WORKS
waveform = np.where(t < slope_percent*period, v_start + (v_end - v_start)*(t/(slope_percent*period)), v_end)

# Plot the waveform
plt.figure()
plt.plot(t, waveform, linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Half-Triangle Waveform (Descending Slope)')
plt.grid()
plt.show()

# Red Pitaya configuration
rp_ip = "rp-f0cbc6.local"  # Replace with your Red Pitaya's hostname
rp = scpi(rp_ip)

# Prepare waveform data for Red Pitaya
waveform_scaled = waveform  

# Send waveform to Red Pitaya
rp.tx_txt("GEN:RST")  # Reset generator
rp.tx_txt("SOUR1:FUNC ARBITRARY")  # Set channel 1 to arbitrary waveform
rp.tx_txt("SOUR1:TRAC:DATA:DATA " + ",".join(map(str, waveform_scaled)))  # Send waveform
rp.tx_txt("SOUR1:FREQ:FIX {}".format(frequency))  # Set frequency
rp.tx_txt("SOUR1:VOLT {}".format(amplitude))
rp.tx_txt("SOUR1:VOLT:OFFS {}".format(offset))
rp.tx_txt("OUTPUT1:STATE ON")  # Enable output channel 1
rp.tx_txt('SOUR1:Trig:INT')
rp.close();
