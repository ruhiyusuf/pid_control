# pid_control

Using PID controller to lock the resonance on microheater chip. External factors that influence resonance include temperature and humidity. 

`func_gen`: deploys default waveforms (triangle, DC, square, etc.). Can customize amplitude, frequency, period, offset.
`func_gen_custom`: deploys custom waveforms (n number of options), constructs waveform using numpy then deploys as an arbitrary waveform

**Note**: save bash scripts to /etc/profile.d to run on startup. Make sure to name the bash script with "_" or something to make it alphabetically last (so that it's the last script to run on reboot.

Example bash script:
```sh
#!/bin/sh
python python_script.py
```

