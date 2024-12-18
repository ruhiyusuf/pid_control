import numpy as np
import matplotlib.pyplot as plt
from redpitaya_scpi import scpi

# Parameters for the waveform
amplitude = 2  # Peak amplitude of the waveform
frequency = 1000  # Period of the waveform
# period = 1 / frequency
period = 2
num_cycles = 5  # Number of cycles to generate
sampling_rate = 1000  # Number of samples per second
offset = 0

# Time vector
t = np.arange(0, num_cycles * period, 1 / sampling_rate)

# Generate the half-triangle waveform
waveform = 1 - (t % period) / (period / 2)  # Linear ramp down from 1 to 0
waveform[waveform < 0] = 0  # Reset to 0 after reaching 0

# Scale by amplitude
waveform = amplitude * waveform

# Plot the waveform
# plt.figure()
# plt.plot(t, waveform, linewidth=1.5)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Half-Triangle Waveform (Descending Slope)')
# plt.grid()
# plt.show()

# Red Pitaya configuration
rp_ip = "rp-f0cbc6.local"  # Replace with your Red Pitaya's hostname
rp = scpi(rp_ip)

# Prepare waveform data for Red Pitaya
waveform_scaled = (waveform * (2**14 - 1)).astype(int)  # Scale to 14-bit integer

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
