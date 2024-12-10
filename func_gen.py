# import scpi probably works best with os2.0
import redpitaya_scpi as scpi
# Connect to the Red Pitaya
rp_ip = "rp-f0cbc6.local"  # Use localhost if running on the Red Pitaya
rp = scpi.scpi(rp_ip)
# Function Generator Settings
channel = "1"  # Output channel 1
frequency = 1000  # Frequency of the sine wave in Hz
amplitude = 2.0  # Amplitude of the sine wave (1.0 = full scale)
offset = 0.0  # DC offset in volts
wave_form = 'triangle'
# Configure the output signal
rp.tx_txt("GEN:RST")  # Reset generator settings
rp.tx_txt("SOUR{}:FUNC {}".format(channel, wave_form))  # Set waveform type to Sine
rp.tx_txt("SOUR{}:FREQ:FIX {}".format(channel, frequency))  # Set frequency
rp.tx_txt("SOUR{}:VOLT {}".format(channel, amplitude))  # Set amplitude
rp.tx_txt("SOUR{}:VOLT:OFFS {}".format(channel, offset))  # Set DC offset
# Enable output
rp.tx_txt("OUTPUT{}:STATE ON".format(channel))
print("Sine wave generated on Output {}: Frequency = {} Hz, Amplitude = {} V".format(channel, frequency, amplitude))
# Keep the sine wave running
rp.tx_txt('SOUR1:TRig:INT')
rp.close()


