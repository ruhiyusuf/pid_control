# pid_control

Using PID controller to lock the resonance on microheater chip. External factors that influence resonance include temperature and humidity. 

`func_gen`: deploys default waveforms (triangle, DC, square, etc.). Can customize amplitude, frequency, period, offset.
`func_gen_custom`: deploys custom waveforms (n number of options), constructs waveform using numpy then deploys as an arbitrary waveform

