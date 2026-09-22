import queue
import threading
import numpy as np
import sounddevice as sd
from rtlsdr import RtlSdr
from scipy.signal import butter, sosfilt, lfilter

# ================= PENGATURAN =================
FREQ      = 155.000000e6   # sesuai VFO aktif di HT (tanda '>')
GAIN      = 30
VOLUME    = 2.0
SQUELCH   = 0.02           # makin kecil makin sensitif, makin gampang kebuka noise
CH_BW     = 25e3           # HT-mu mode "W" (wide) = 25 kHz. Ganti 12.5e3 kalau salah
FS        = 1_200_000
# ===============================================

OFFSET   = 250_000
BLOCK    = 96_000
AUDIO_FS = 48_000
MID_FS   = 240_000
DECIM1   = FS // MID_FS
DECIM2   = MID_FS // AUDIO_FS

sdr = RtlSdr()
sdr.sample_rate = FS
sdr.center_freq = FREQ + OFFSET
sdr.gain = GAIN

sos1 = butter(6, CH_BW, fs=FS, output="sos")
sos2 = butter(6, 3e3, fs=MID_FS, output="sos")   # suara voice dibatasi ~3 kHz
z1 = np.zeros((sos1.shape[0], 2), dtype=complex)
z2 = np.zeros((sos2.shape[0], 2))
lo = np.exp(2j * np.pi * OFFSET * np.arange(BLOCK) / FS)
prev = 1 + 0j

raw_q = queue.Queue(maxsize=60)
audio_q = queue.Queue(maxsize=60)
cmd_q = queue.Queue()
flush = threading.Event()
running = threading.Event()
running.set()
underruns = 0
dropped = 0
leftover = np.zeros(0, dtype=np.float32)
cur_freq = FREQ / 1e6

def reader():
    global dropped
    while running.is_set():
        try:
            kind, val = cmd_q.get_nowait()
            if kind == "freq":
                sdr.center_freq = val * 1e6 + OFFSET
            sdr.read_samples(BLOCK)
            flush.set()
        except queue.Empty:
            pass
        x = sdr.read_samples(BLOCK)
        try:
            raw_q.put_nowait(x)
        except queue.Full:
            dropped += 1

def keyboard():
    global cur_freq
    print("Ketik frekuensi MHz lalu Enter buat pindah (contoh: 155.00625)")
    print("q = keluar")
    while running.is_set():
        try:
            s = input().strip().lower()
        except EOFError:
            return
        if s == "q":
            running.clear()
        else:
            try:
                f = float(s)
                cur_freq = f
                cmd_q.put(("freq", f))
                print(f">> pindah ke {f} MHz")
            except ValueError:
                print("Tidak dimengerti")

def callback(outdata, frames, time, status):
    global leftover, underruns
    while len(leftover) < frames:
        try:
            leftover = np.concatenate((leftover, audio_q.get_nowait()))
        except queue.Empty:
            underruns += 1
            break
    n = min(frames, len(leftover))
    outdata[:n, 0] = leftover[:n]
    outdata[n:, 0] = 0
    leftover = leftover[n:]

stream = sd.OutputStream(samplerate=AUDIO_FS, channels=1, dtype="float32",
                         latency="high", callback=callback)
t = threading.Thread(target=reader, daemon=True)
t.start()
threading.Thread(target=keyboard, daemon=True).start()

started = False
count = 0
try:
    print(f"Menyiapkan di {cur_freq} MHz... Ctrl+C atau 'q' untuk berhenti")
    while running.is_set():
        if flush.is_set():
            flush.clear()
            for qq in (raw_q, audio_q):
                while True:
                    try:
                        qq.get_nowait()
                    except queue.Empty:
                        break
            z1[:] = 0
            z2[:] = 0
            prev = 1 + 0j
            continue
        try:
            x = raw_q.get(timeout=1)
        except queue.Empty:
            continue

        x = x * lo
        x, z1 = sosfilt(sos1, x, zi=z1)
        x = x[::DECIM1]

        power = np.mean(np.abs(x) ** 2)

        y = np.concatenate(([prev], x))
        prev = x[-1]
        d = np.angle(y[1:] * np.conj(y[:-1])) / np.pi

        d, z2 = sosfilt(sos2, d, zi=z2)
        d = d[::DECIM2]

        if power < SQUELCH:       # tidak ada sinyal, diam saja
            d = np.zeros_like(d)

        out = np.clip(d * VOLUME, -1, 1).astype(np.float32)
        if audio_q.full():
            audio_q.get_nowait()
        audio_q.put_nowait(out)

        if not started and audio_q.qsize() >= 10:
            stream.start()
            started = True
            print("Jalan! (diam kalau tidak ada yang ngomong, itu normal)")

        count += 1
        if started and count % 50 == 0:
            print(f"[{cur_freq} MHz] daya={power:.4f} (squelch={SQUELCH}) "
                  f"putus={underruns} hilang={dropped}")
except KeyboardInterrupt:
    pass
finally:
    running.clear()
    t.join(timeout=3)
    stream.close()
    sdr.close()