#!/usr/bin/python3
import socket, struct, numpy as np, rp, array

HOST = "0.0.0.0"
PORT = 9000
CHANNEL = rp.RP_CH_1
DEFAULT_AMP = 1.0

rp.rp_Init()
rp.rp_GenReset()

print(f"[RP] Server ready on port {PORT}")

def program_waveform(duration_s, wave_i16):

    nsamp = len(wave_i16)
    wave_f32 = (wave_i16 / float(2**13 - 1)).astype(np.float32)

    # Load arbitrary waveform
    if hasattr(rp, "rp_GenArbWaveformNP"):
        rp.rp_GenArbWaveformNP(CHANNEL, wave_f32)
    else:
        c_arr = array.array('f', wave_f32.tolist())
        rp.rp_GenArbWaveform(CHANNEL, c_arr, len(c_arr))

    # Compute frequency = ONE cycle takes exactly duration_s
    freq = 1.0 / duration_s

    # Configure output like SCPI version
    rp.rp_GenWaveform(CHANNEL, rp.RP_WAVEFORM_ARBITRARY)
    rp.rp_GenFreqDirect(CHANNEL, freq)
    rp.rp_GenAmp(CHANNEL, DEFAULT_AMP)
    rp.rp_GenMode(CHANNEL, rp.RP_GEN_MODE_CONTINUOUS)
    rp.rp_GenOutEnable(CHANNEL)

    rp.rp_GenTriggerOnly(CHANNEL)

    print(f"[RP] Loaded {nsamp} samples → duration={duration_s:.3f}s → freq={freq:.6f} Hz")


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)

    while True:
        conn, addr = s.accept()
        print(f"[RP] Connection from {addr}")

        with conn:
            # Read header: total payload size
            hdr = conn.recv(4)
            if len(hdr) < 4:
                print("[RP] Bad header")
                continue

            (nbytes,) = struct.unpack(">I", hdr)

            # Read duration (float32)
            dur_bytes = conn.recv(4)
            (duration_s,) = struct.unpack(">f", dur_bytes)

            # Read waveform payload (int16)
            payload = b""
            while len(payload) < nbytes:
                chunk = conn.recv(min(4096, nbytes - len(payload)))
                if not chunk:
                    break
                payload += chunk

            waveform = np.frombuffer(payload, dtype=np.int16)

            try:
                program_waveform(duration_s, waveform)
            except Exception as e:
                print("[RP] Error:", e)
