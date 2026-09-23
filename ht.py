import queue
import threading
import numpy as np
import sounddevice as sd
from rtlsdr import RtlSdr
from scipy.signal import butter, sosfilt

# ================= PENGATURAN =================
FREQ        = 150.000000e6   # sesuai VFO aktif di HT (tanda '>')
GAIN        = 22             # diturunin dari 30 -> noise floor RF lebih rendah
FREQ_CORR   = 0              # PPM koreksi RTL-SDR kalau device kamu butuh (cek datasheet/tes)
VOLUME      = 15.0           # turunin/naikin sesuai hasil tes

# --- squelch daya RF (gerbang paling dasar, jangan terlalu ketat) ---
SQUELCH_MIN = 0.01           # ambang daya RF minimum biar tidak buka di keheningan total

# --- noise-squelch dengan HYSTERESIS (dua ambang, anti-chatter) ---
# Prinsip: kalau ada sinyal FM valid, noise di band atas suara (4-8kHz)
# akan tertekan (capture effect). Kalau cuma noise murni, energinya tinggi.
#
# Dengan SATU ambang tunggal, kalau nilai noise= goyang naik-turun persis
# di sekitar ambang itu, gate bisa buka-tutup berkali-kali per detik -->
# kedengeran sebagai "noise berulang" / klik-klik.
#
# Solusinya: dua ambang berbeda.
#   NOISE_SQUELCH_OPEN  = ambang KETAT, dipakai buat MEMBUKA gate (gate
#                          masih tertutup / baru mau kebuka).
#   NOISE_SQUELCH_CLOSE = ambang LONGGAR, dipakai buat MENUTUP gate
#                          (gate sudah kebuka, baru nutup lagi kalau
#                          noise_power naik jelas melewati ambang ini).
# Begitu gate kebuka, dia jadi "keras kepala" -- gak gampang balik nutup
# cuma gara-gara noise goyang tipis. Inilah yang disebut hysteresis.
#
# CARA TUNING (WAJIB dilakukan per lokasi/HT):
#   1. Jalankan script di kanal kosong (tidak ada yang ngomong), catat
#      rentang nilai 'noise=' yang muncul di log selama ~30 detik.
#   2. Suruh HT lawan ngomong di frekuensi sama, catat nilai 'noise='
#      pas dia ngomong.
#   3. Set NOISE_SQUELCH_OPEN sedikit di ATAS nilai tertinggi saat
#      "ada siaran" (biar gate yakin sebelum buka).
#   4. Set NOISE_SQUELCH_CLOSE di tengah rentang nilai "diam", tapi
#      masih jelas di bawah nilai noise pas benar-benar hening total.
NOISE_SQUELCH_OPEN  = 0.05   # ketat -> syarat MEMBUKA gate
NOISE_SQUELCH_CLOSE = 0.09   # longgar -> syarat MENUTUP gate (harus > OPEN)

HANG_MS     = 120            # diturunin dari 200 -> gate lebih cepat nutup lagi
FADE_MS     = 30             # durasi fade halus pas buka/tutup

# --- anti-thump: buang noise/klik di awal-awal PTT ditekan ---
# Pas PTT baru dipencet, carrier RF belum stabil sepersekian detik (relay TX,
# ramp-up daya, FM belum lock). Kalau langsung dibuka, kedengeran noise/klik.
# OPEN_CONFIRM_MS = sinyal harus konsisten "ada" selama ini dulu sebelum
#                    dianggap resmi ada siaran (anti false-trigger dari noise sesaat).
# OPEN_MUTE_MS    = setelah resmi terkonfirmasi, tetap dibisukan dulu selama ini
#                    (buang transient/klik key-up), baru audio beneran dibuka.
OPEN_CONFIRM_MS = 40
OPEN_MUTE_MS    = 80

CH_BW       = 25e3
FS          = 1_200_000

# --- band audio suara manusia (buang rumble & hiss di luar ini) ---
VOICE_LO = 300
VOICE_HI = 3000
# --- band untuk deteksi noise (di atas band suara) ---
NOISE_LO = 4000
NOISE_HI = 8000
# ===============================================

OFFSET   = 250_000
BLOCK    = 96_000
AUDIO_FS = 48_000
MID_FS   = 240_000
DECIM1   = FS // MID_FS
DECIM2   = MID_FS // AUDIO_FS
BLOCK_MS = (BLOCK / DECIM1 / DECIM2) / AUDIO_FS * 1000
HANG_BLOCKS = max(1, round(HANG_MS / BLOCK_MS))
FADE_SAMPLES = max(1, round(FADE_MS / 1000 * AUDIO_FS))
OPEN_CONFIRM_BLOCKS = max(1, round(OPEN_CONFIRM_MS / BLOCK_MS))
OPEN_MUTE_BLOCKS = max(1, round(OPEN_MUTE_MS / BLOCK_MS))

sdr = RtlSdr()
sdr.sample_rate = FS
if FREQ_CORR:
    sdr.freq_correction = FREQ_CORR
sdr.center_freq = FREQ + OFFSET
sdr.gain = GAIN

sos1       = butter(6, CH_BW, fs=FS, output="sos")                      # channel filter (RF)
sos_voice  = butter(4, [VOICE_LO, VOICE_HI], btype="bandpass", fs=MID_FS, output="sos")  # audio band
sos_noise  = butter(4, [NOISE_LO, NOISE_HI], btype="bandpass", fs=MID_FS, output="sos")  # noise detector

z1 = np.zeros((sos1.shape[0], 2), dtype=complex)
z_voice = np.zeros((sos_voice.shape[0], 2))
z_noise = np.zeros((sos_noise.shape[0], 2))
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
    print("Ketik frekuensi MHz lalu Enter buat pindah (contoh: 150.00625)")
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
peak = 0.0
gate_gain = 0.0
hang_left = 0
gate_state = "closed"   # closed -> confirming -> muted -> open -> (balik ke closed)
confirm_left = 0
mute_left = 0
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
            z_voice[:] = 0
            z_noise[:] = 0
            prev = 1 + 0j
            gate_gain = 0.0
            hang_left = 0
            gate_state = "closed"
            confirm_left = 0
            mute_left = 0
            continue
        try:
            x = raw_q.get(timeout=1)
        except queue.Empty:
            continue

        x = x * lo
        x, z1 = sosfilt(sos1, x, zi=z1)
        x = x[::DECIM1]

        power = np.mean(np.abs(x) ** 2)

        # --- limiter: normalisasi amplitudo sebelum diskriminator FM ---
        # Ini yang paling berpengaruh mengurangi noise impulsif/AM yang
        # ikut "kebawa" ke audio kalau langsung pakai amplitudo asli.
        mag = np.abs(x)
        x_norm = x / (mag + 1e-9)

        y = np.concatenate(([prev], x_norm))
        prev = x_norm[-1]
        d = np.angle(y[1:] * np.conj(y[:-1])) / np.pi

        # --- deteksi noise band atas (untuk noise-squelch) ---
        d_noise, z_noise = sosfilt(sos_noise, d, zi=z_noise)
        noise_power = np.mean(d_noise ** 2)

        # --- audio band 300-3000 Hz ---
        d_voice, z_voice = sosfilt(sos_voice, d, zi=z_voice)
        d_voice = d_voice[::DECIM2]

        # ---- gate: power squelch (dasar) + noise squelch hysteresis (utama) + anti-thump ----
        # Kalau gate masih "closed", pakai ambang KETAT (OPEN) supaya gak
        # gampang salah buka gara-gara noise sesaat.
        # Begitu sudah mulai proses membuka (confirming/muted/open), pakai
        # ambang LONGGAR (CLOSE) supaya gak gampang balik nutup gara-gara
        # noise yang goyang tipis -> ini yang mencegah "noise berulang".
        if gate_state == "closed":
            signal_present = (power >= SQUELCH_MIN) and (noise_power <= NOISE_SQUELCH_OPEN)
        else:
            signal_present = (power >= SQUELCH_MIN) and (noise_power <= NOISE_SQUELCH_CLOSE)

        if gate_state == "closed":
            target = 0.0
            if signal_present:
                gate_state = "confirming"
                confirm_left = OPEN_CONFIRM_BLOCKS

        elif gate_state == "confirming":
            # sinyal harus konsisten "ada" selama OPEN_CONFIRM_BLOCKS blok
            # berturut-turut, kalau sempat hilang di tengah jalan -> balik nutup
            target = 0.0
            if not signal_present:
                gate_state = "closed"
            else:
                confirm_left -= 1
                if confirm_left <= 0:
                    gate_state = "muted"
                    mute_left = OPEN_MUTE_BLOCKS

        elif gate_state == "muted":
            # sudah terkonfirmasi ada siaran, tapi masih dibisukan sebentar
            # buat buang klik/transient key-up sebelum audio beneran dibuka
            target = 0.0
            mute_left -= 1
            if mute_left <= 0:
                gate_state = "open"
                hang_left = HANG_BLOCKS

        else:  # gate_state == "open"
            if signal_present:
                hang_left = HANG_BLOCKS
                target = 1.0
            elif hang_left > 0:
                hang_left -= 1
                target = 1.0
            else:
                gate_state = "closed"
                target = 0.0

        n = len(d_voice)
        ramp = np.linspace(gate_gain, target, min(FADE_SAMPLES, n))
        if n > len(ramp):
            ramp = np.concatenate([ramp, np.full(n - len(ramp), target)])
        gate_gain = target
        d_voice = d_voice * ramp[:n]

        # --- soft clip (tanh) biar tidak distorsi kasar kalau kepotong ---
        out = np.tanh(d_voice * VOLUME).astype(np.float32)
        if len(out):
            peak = max(peak, float(np.max(np.abs(out))))
        if audio_q.full():
            audio_q.get_nowait()
        audio_q.put_nowait(out)

        if not started and audio_q.qsize() >= 10:
            stream.start()
            started = True
            print("Jalan! (diam kalau tidak ada yang ngomong, itu normal)")

        count += 1
        if started and count % 50 == 0:
            print(f"[{cur_freq} MHz] daya={power:.4f} noise={noise_power:.4f} "
                f"(squelch_min={SQUELCH_MIN}, open={NOISE_SQUELCH_OPEN}, close={NOISE_SQUELCH_CLOSE}) "
                f"state={gate_state} puncak={peak:.3f} putus={underruns} hilang={dropped}")
            peak = 0.0
except KeyboardInterrupt:
    pass
finally:
    running.clear()
    t.join(timeout=3)
    stream.close()
    sdr.close()