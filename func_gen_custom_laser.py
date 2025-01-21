import numpy as np
import matplotlib.pyplot as plt
from redpitaya_scpi import scpi

# Parameters for the waveform
frequency = 10  # Frequency of the waveform
# period = 1 / frequency # Period of the waveform
period = 10 # Period of the waveform
num_cycles = 1  # Number of cycles to generate
sampling_rate = 1000  # Number of samples per second
offset = 0
v_start = 0 
v_end = 1 
slope_percent = 0.2 
amplitude = v_end - v_start  # Peak amplitude of the waveform

# Time vector
# t = np.arange(0, num_cycles * period, 1 / sampling_rate)
# t = np.arange(0, 1/frequency, 1 / sampling_rate)
t = np.arange(0, period, 1 / sampling_rate)
# N = 16384
# t = np.linspace(0, 1, N) * 2 * np.pi
# t = np.linspace(0, 5/frequency, sampling_rate)

# Generate the half-triangle waveform
# waveform = (((t + 1) % period) - amplitude)
# triangle
#waveform = v_start + (v_end-v_start) * 2 * abs(((t*frequency)%period) - 0.5)

# ramp up, for laser, correct
# waveform = np.where(t < slope_percent/frequency, v_start + (v_end - v_start)*(t/(slope_percent/frequency)), v_end)

# ORIGINAL< IT WORKS
waveform = np.where(t < slope_percent*period, v_start + (v_end - v_start)*(t/(slope_percent*period)), v_end)
# waveform = 1 - (t % period) / (period / 2)  # Linear ramp down from 1 to 0
# waveform[waveform < 0] = 0  # Reset to 0 after reaching 0

# Scale by amplitude
# waveform = amplitude * waveform

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
# # waveform_scaled = (waveform * (2**14 - 1)).astype(int)  # Scale to 14-bit integer
# waveform_scaled = waveform * (2**14 - 1)
waveform_scaled = waveform  

# Send waveform to Red Pitaya
rp.tx_txt("GEN:RST")  # Reset generator
# rp.tx_txt("SOUR1:RESET")
# rp.tx_txt("SOUR1:START")
rp.tx_txt("SOUR1:FUNC ARBITRARY")  # Set channel 1 to arbitrary waveform
rp.tx_txt("SOUR1:TRAC:DATA:DATA " + ",".join(map(str, waveform_scaled)))  # Send waveform
rp.tx_txt("SOUR1:FREQ:FIX {}".format(frequency))  # Set frequency
rp.tx_txt("SOUR1:VOLT {}".format(amplitude))
rp.tx_txt("SOUR1:VOLT:OFFS {}".format(offset))
rp.tx_txt("OUTPUT1:STATE ON")  # Enable output channel 1
rp.tx_txt('SOUR1:Trig:INT')
rp.close();
