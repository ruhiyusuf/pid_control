import numpy as np
import matplotlib.pyplot as plt
from redpitaya_scpi import scpi
import time

# Parameters
amplitude = 1
period = 10
num_cycles = 3
sampling_rate = 100  # Lowered sampling rate to reduce data size
offset = 0

# Time vector
t = np.arange(0, num_cycles * period, 1 / sampling_rate)

# Generate waveform
waveform = 1 - (t % period) / (period / 2)
waveform[waveform < 0] = 0
waveform = amplitude * waveform

# Plot waveform
plt.figure()
plt.plot(t, waveform, linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Extended DC Step (10 Seconds)')
plt.grid()
plt.show()

# Connect to Red Pitaya
rp_ip = "rp-f0cbc6.local"
rp = scpi(rp_ip)

# Prepare data
# waveform_scaled = (waveform * (2**13 - 1)).astype(int)
# max_samples = 16384
# waveform_scaled = waveform_scaled[:max_samples]

waveform_scaled = waveform
# Send data
rp.tx_txt("GEN:RST")
time.sleep(0.1)
rp.tx_txt("SOUR1:FUNC ARBITRARY")
time.sleep(0.1)
rp.tx_txt(f"SOUR1:TRAC:DATA:DATA {','.join(map(str, waveform_scaled))}")
time.sleep(0.1)
rp.tx_txt(f"SOUR1:FREQ:FIX {1 / period}")
rp.tx_txt("SOUR1:VOLT {}".format(amplitude))
rp.tx_txt("SOUR1:VOLT:OFFS {}".format(offset))
rp.tx_txt("OUTPUT1:STATE ON")
rp.tx_txt("SOUR1:TRIG:INT")
rp.close()

