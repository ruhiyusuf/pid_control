import time
import redpitaya_scpi as scpi

# Connect to the Red Pitaya
rp_ip = "rp-f0cbc6.local"  # Replace with your Red Pitaya's IP address
rp = scpi.scpi(rp_ip)

# PID parameters
Kp = 1.0  # Proportional gain
Ki = 0.5  # Integral gain
Kd = 0.1  # Derivative gain

# Setpoint and initial values
setpoint = 1.0  # Desired output
prev_error = 0
integral = 0

# Sampling interval
dt = 0.01  # 10 ms

# Configure ADC and DAC
rp.tx_txt("ACQ:DEC 64")  # Configure decimation factor for ADC
rp.tx_txt("ACQ:START")  # Start acquisition

try:
    while True:
        # Read current output from ADC
        rp.tx_txt("ACQ:TRIG NOW")  # Trigger acquisition
        time.sleep(dt)
        rp.tx_txt("ACQ:SOUR1:DATA:LAT:N? 1")  # Get the latest sample
        raw_adc_value = rp.rx_txt()  # Receive the response

        # Clean the response to extract the numeric value
        adc_value = float(raw_adc_value.strip('{}'))  # Remove curly braces and convert to float
        
        # Calculate error
        error = setpoint - adc_value
        
        # Calculate PID terms
        proportional = Kp * error
        integral += error * dt
        derivative = (error - prev_error) / dt
        
        # PID output
        output = proportional + Ki * integral + Kd * derivative
        
        # Limit output to DAC range (adjust as per your DAC specifications)
        output = max(min(output, 1.0), 0.0)
        
        # Write output to DAC
        rp.tx_txt(f"SOUR1:VOLT {output}")
        
        # Update previous error
        prev_error = error

        # Print debugging info
        print(f"Setpoint: {setpoint}, ADC: {adc_value}, Output: {output}")

except KeyboardInterrupt:
    print("PID control stopped.")
    rp.tx_txt("ACQ:STOP")  # Stop acquisition
    rp.tx_txt("SOUR1:VOLT 0")  # Reset DAC output
    rp.close()  # Close the connection

