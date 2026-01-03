from pathlib import Path
import random
import numpy as np
import soundfile as sf
import librosa

# =========================
# Configuration
# =========================
TARGET_SR = 16000
SNR_DB = 10.0                 # Ziel-SNR in dB
SEGMENT_SECONDS = 30          # Länge pro Negativ-Datei
N_OUTPUTS = 120               # Anzahl Output-Dateien (120 × 30s = 60 Minuten)
RANDOM_SEED = 42

# Pfade (passen zu deinem Setup)
SPEECH_ROOT = Path("data/raw/librispeech/test-clean")
NOISE_ROOT  = Path("data/raw/demand")
OUT_ROOT    = Path("data/neg_wavs")

# =========================
# Helper functions
# =========================
def find_files(root: Path, exts):
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return [f for f in files if f.is_file()]

def load_audio_mono(path: Path, target_sr=TARGET_SR) -> np.ndarray:
    """
    Load audio as float32 mono in range [-1, 1], resampled to target_sr.
    Supports wav/flac via soundfile.
    """
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)

    x = x.astype(np.float32)

    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)

    # leichte Normalisierung (verhindert extreme Pegel)
    peak = np.max(np.abs(x)) if len(x) else 1.0
    if peak > 0:
        x = x / max(1.0, peak)

    return x

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)

def mix_at_snr_db(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mix speech + noise at given SNR (dB).
    """
    n = len(speech)

    # Noise ggf. loopen
    if len(noise) < n:
        reps = int(np.ceil(n / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:n]

    s_rms = rms(speech)
    n_rms = rms(noise)

    # Ziel-Noise-RMS für gewünschtes SNR
    target_noise_rms = s_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / n_rms)

    mixed = speech + noise_scaled

    # Clipping verhindern
    peak = np.max(np.abs(mixed)) if len(mixed) else 1.0
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)

    return mixed

# =========================
# Main
# =========================
def main():
    random.seed(RANDOM_SEED)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    speech_files = find_files(SPEECH_ROOT, [".flac", ".wav"])
    noise_files  = find_files(NOISE_ROOT,  [".wav", ".flac"])

    if not speech_files:
        raise RuntimeError(f"No speech files found under {SPEECH_ROOT}")
    if not noise_files:
        raise RuntimeError(f"No noise files found under {NOISE_ROOT}")

    segment_len = int(TARGET_SR * SEGMENT_SECONDS)

    print("======================================")
    print("Generating negative wakeword dataset")
    print("--------------------------------------")
    print(f"Speech root : {SPEECH_ROOT}")
    print(f"Noise root  : {NOISE_ROOT}")
    print(f"Speech files: {len(speech_files)}")
    print(f"Noise files : {len(noise_files)}")
    print(f"Outputs     : {N_OUTPUTS} × {SEGMENT_SECONDS}s")
    print(f"SNR         : {SNR_DB} dB")
    print("======================================")

    for i in range(N_OUTPUTS):
        sf_path = random.choice(speech_files)
        nf_path = random.choice(noise_files)

        speech = load_audio_mono(sf_path)
        noise  = load_audio_mono(nf_path)

        # Zufälligen Ausschnitt aus Speech wählen
        if len(speech) < segment_len:
            reps = int(np.ceil(segment_len / max(1, len(speech))))
            speech = np.tile(speech, reps)
        s_start = random.randint(0, len(speech) - segment_len)
        speech_seg = speech[s_start:s_start + segment_len]

        # Zufälligen Ausschnitt aus Noise wählen
        if len(noise) < segment_len:
            reps = int(np.ceil(segment_len / max(1, len(noise))))
            noise = np.tile(noise, reps)
        n_start = random.randint(0, len(noise) - segment_len)
        noise_seg = noise[n_start:n_start + segment_len]

        mixed = mix_at_snr_db(speech_seg, noise_seg, SNR_DB)

        out_path = OUT_ROOT / f"neg_{i:04d}_snr{int(SNR_DB)}.wav"
        sf.write(str(out_path), mixed, TARGET_SR, subtype="PCM_16")

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i + 1}/{N_OUTPUTS}] wrote {out_path.name}")

    print("======================================")
    print("Done.")
    print(f"Negative WAVs written to: {OUT_ROOT}")
    print("======================================")

if __name__ == "__main__":
    main()
