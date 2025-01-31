
import numpy as np
import matplotlib.pyplot as plt
from redpitaya_scpi import scpi

# Parameters for the waveform
period = 10  # Total duration of the waveform
sampling_rate = 1000  # Number of samples per second
offset = 0
v_start = 0 
v_end = 1 
slope_percent = 0.02
short_down_slope_percent = 0.05
v_down_end = 0.9 * v_end  # Voltage achieved at the end of the short downward slope

# Time vector
t = np.arange(0, period, 1 / sampling_rate)

# Waveform generation
waveform = np.where(
    t < slope_percent * period,  # Upward slope condition
    v_start + (v_end - v_start) * (t / (slope_percent * period)),  # Value during the upward slope
    np.where(
        t < (slope_percent + short_down_slope_percent) * period,  # Downward slope condition
        v_end + (v_down_end - v_end) * ((t - slope_percent * period) / (short_down_slope_percent * period)),  # Value during the short downward slope
        v_down_end  # Final voltage (end of downward slope)
    )
)

# Plot the waveform
plt.figure()
plt.plot(t, waveform, linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform')
plt.grid()
plt.show()

# Red Pitaya configuration
rp_ip = "rp-f0cbc6.local"  # Replace with your Red Pitaya's hostname
rp = scpi(rp_ip)

# Prepare and send the initial waveform
waveform_scaled = waveform  # Scaled waveform for the Red Pitaya
rp.tx_txt("GEN:RST")  # Reset generator
rp.tx_txt("SOUR1:FUNC ARBITRARY")  # Set channel 1 to arbitrary waveform
rp.tx_txt("SOUR1:TRAC:DATA:DATA " + ",".join(map(str, waveform_scaled)))  # Send waveform
rp.tx_txt("SOUR1:FREQ:FIX {:.6f}".format(1 / period))  # Set frequency based on total duration
rp.tx_txt("SOUR1:VOLT {:.6f}".format(v_end - v_start))  # Set amplitude
rp.tx_txt("SOUR1:VOLT:OFFS 0")  # Set offset to 0
rp.tx_txt("OUTPUT1:STATE ON")  # Enable output channel 1
rp.tx_txt("SOUR1:TRIG:SOUR INT")  # Set internal trigger source
rp.tx_txt("SOUR1:TRIG:INT")  # Trigger waveform generation

# Wait for the waveform to finish
import time
waveform_duration = period  # Duration of the waveform in seconds
time.sleep(waveform_duration / 2)

# Switch to DC voltage
# rp.tx_txt("GEN:RST")  # Reset generator
rp.tx_txt("SOUR1:FUNC DC")  # Set channel 1 to DC
rp.tx_txt("SOUR1:VOLT {}".format(v_down_end))  # Set DC voltage to v_down_end
rp.tx_txt("SOUR1:FREQ:FIX {}".format(1 / period))  # Retain original frequency
rp.tx_txt("OUTPUT1:STATE ON")  # Enable output channel 1
rp.tx_txt("SOUR1:TRIG:INT")  # Trigger waveform generation
rp.close()

# Indicate DC state
print("Waveform completed. DC voltage applied.")
