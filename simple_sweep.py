import numpy as np
from redpitaya_scpi import scpi
import time

# === Configuration ===
RP_IP = 'rp-f0cbc6.local'
CHANNEL = 1
STEPS = 6
NUM_SAMPLES_PER_STEP = 2048  # Controls smoothness per segment

# === Parameter Sets (Start and End) ===
waveform_start = {
    "v_start": 0.0,
    "v_end": 1.0,
    "v_down_end": 0.8,
    "start_v_ms": 20.0,
    "slope_up_ms": 20.0,
    "slope_down_ms": 40.0,
    "end_v_ms": 500.0
}

waveform_end = {
    "v_start": 0.0,
    "v_end": 0.2,
    "v_down_end": 0.1,
    "start_v_ms": 20.0,
    "slope_up_ms": 20.0,
    "slope_down_ms": 40.0,
    "end_v_ms": 500.0
}

def generate_waveform(p, num_samples=16384):
    s = p["start_v_ms"] / 1000
    u = p["slope_up_ms"] / 1000
    d = p["slope_down_ms"] / 1000
    e = p["end_v_ms"] / 1000
    total_time = s + u + d + e
    t = np.linspace(0, total_time, num_samples)

    waveform = np.piecewise(t,
        [t < s,
         (t >= s) & (t < s + u),
         (t >= s + u) & (t < s + u + d),
         t >= s + u + d],
        [lambda t: p["v_start"],
         lambda t: p["v_start"] + (p["v_end"] - p["v_start"]) * (t - s) / u,
         lambda t: p["v_end"] + (p["v_down_end"] - p["v_end"]) * (t - s - u) / d,
         lambda t: p["v_down_end"]])
    return waveform, total_time

# === Build Full Sweep Waveform ===
full_waveform = []
total_time = 0

for i in range(STEPS):
    alpha = i / (STEPS - 1)
    interp_params = {
        key: (1 - alpha) * waveform_start[key] + alpha * waveform_end[key]
        for key in waveform_start
    }
    segment, segment_time = generate_waveform(interp_params, num_samples=NUM_SAMPLES_PER_STEP)
    full_waveform.extend(segment)
    total_time += segment_time
    print(f"[{i+1}/{STEPS}] v_end={interp_params['v_end']:.3f}, v_down_end={interp_params['v_down_end']:.3f}, duration={segment_time:.3f}s")

# === Compute Frequency for RP Playback
combined_freq = 1 / total_time
print(f"[DEBUG] Total samples: {len(full_waveform)}, Total Time: {total_time:.3f}s → Freq: {combined_freq:.4f} Hz")

# === Send to Red Pitaya ===
try:
    rp = scpi(RP_IP)
    rp.tx_txt("SOUR1:RESET")
    rp.tx_txt(f"SOUR{CHANNEL}:FUNC ARB")
    rp.tx_txt(f"SOUR{CHANNEL}:TRAC:DATA:DATA " + ",".join(map(str, full_waveform)))
    rp.tx_txt(f"SOUR{CHANNEL}:FREQ:FIX {combined_freq}")
    rp.tx_txt(f"SOUR{CHANNEL}:VOLT:OFFS 0")
    rp.tx_txt(f"OUTPUT{CHANNEL}:STATE ON")
    rp.tx_txt(f"SOUR{CHANNEL}:TRIG:INT")
    rp.close()
    print("[DEPLOY] Sweep waveform sent to Red Pitaya.")
except Exception as e:
    print(f"[ERROR] Could not send to Red Pitaya: {e}")

