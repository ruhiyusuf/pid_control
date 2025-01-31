import numpy as np
import matplotlib.pyplot as plt
from redpitaya_scpi import scpi

# Parameters for the waveform
frequency = 10  # Frequency of the waveform
period = 10 # Period of the waveform
sampling_rate = 1000  # Number of samples per second
offset = 0
v_start = 0 
v_end = 1 
pause_percent = 0.3
slope_percent = 0.2 
amplitude = v_end - v_start  # Peak amplitude of the waveform
v_pause = v_start

# Time vector
t = np.arange(0, period, 1 / sampling_rate)

waveform_option = 3
# ramp up, for laser, correct
# waveform = np.where(t < slope_percent/frequency, v_start + (v_end - v_start)*(t/(slope_percent/frequency)), v_end)

# ORIGINAL< IT WORKS
if (waveform_option == 1):
    waveform = np.where(t < slope_percent*period, v_start + (v_end - v_start)*(t/(slope_percent*period)), v_end)
    print("Half-trapezoid, no gap in between")

# adding pause at the start
if (waveform_option == 2):
    waveform = np.where(
        t < pause_percent * period,  # Condition for the pause
        v_pause,  # Value during the pause
        np.where(
            t < (pause_percent + slope_percent) * period,  # Condition for the slope
            v_start + (v_end - v_start) * ((t - pause_percent * period) / (slope_percent * period)),  # Value during the slope
            v_end  # Value after the slope
        )
    )
    print("Half-trapezoid with gap in between")

# adding short slope at front
if (waveform_option == 3):
    pause_percent = 0.3
    slope_percent = 0.1
    short_down_slope_percent = 0.05
    v_pause = v_start
    v_down_end = 0.9 * v_end

    waveform = np.where(
        t < pause_percent * period,  # Initial pause condition
        v_pause,  # Value before the slope
        np.where(
            t < (pause_percent + slope_percent) * period,  # Upward slope condition
            v_start + (v_end - v_start) * ((t - pause_percent * period) / (slope_percent * period)),  # Value during the upward slope
            np.where(
                t < (pause_percent + slope_percent + short_down_slope_percent) * period,  # Downward slope condition
                v_end + (v_down_end - v_end) * ((t - (pause_percent + slope_percent) * period) / (short_down_slope_percent * period)),  # Value during the short downward slope
                v_down_end  # Final voltage (end of downward slope)
            )
        )
    )
    print("Half-trapezoid with gap in between and short downward slope")

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
