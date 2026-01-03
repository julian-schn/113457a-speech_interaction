from pathlib import Path
import random
import numpy as np
import soundfile as sf
import librosa
import shutil


# =========================
# Configuration
# =========================
TARGET_SR = 16000
SNR_DB = 10.0
RANDOM_SEED = 42

SPEECH_ROOT = Path("data/raw/librispeech/test-clean")
NOISE_ROOT  = Path("data/raw/demand")
OUT_ROOT    = Path("data/neg_wavs")

# Optional: auf ~5.0h kappen (None = alles, ~5.4h bei test-clean)
TARGET_HOURS = 5.0  # z.B. 5.0 oder None


# =========================
# Helper functions
# =========================
def find_files(root: Path, exts):
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return [f for f in files if f.is_file()]

def load_audio_mono(path: Path, target_sr=TARGET_SR) -> np.ndarray:
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)

    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)

    peak = np.max(np.abs(x)) if len(x) else 1.0
    if peak > 0:
        x = x / max(1.0, peak)
    return x

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)

def mix_at_snr_db(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    n = len(speech)

    # Noise ggf. loopen/abschneiden, Speech wird NICHT geloopt
    if len(noise) < n:
        reps = int(np.ceil(n / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:n]

    s_rms = rms(speech)
    n_rms = rms(noise)

    target_noise_rms = s_rms / (10 ** (snr_db / 20.0))
    noise_scaled = noise * (target_noise_rms / n_rms)

    mixed = speech + noise_scaled

    peak = np.max(np.abs(mixed)) if len(mixed) else 1.0
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)

    return mixed


# =========================
# Main
# =========================
def main():
    random.seed(RANDOM_SEED)
    # =========================
    # Clean output directory
    # =========================
    if OUT_ROOT.exists():
        print(f"Cleaning output directory: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    speech_files = find_files(SPEECH_ROOT, [".flac", ".wav"])
    noise_files  = find_files(NOISE_ROOT,  [".wav", ".flac"])

    if not speech_files:
        raise RuntimeError(f"No speech files found under {SPEECH_ROOT}")
    if not noise_files:
        raise RuntimeError(f"No noise files found under {NOISE_ROOT}")

    # Optional: deterministisch, aber trotzdem “zufällig”
    random.shuffle(speech_files)

    target_samples = None
    if TARGET_HOURS is not None:
        target_samples = int(TARGET_HOURS * 3600 * TARGET_SR)

    written = 0
    total_samples = 0

    print("======================================")
    print("Mixing speech files (original length) + noise")
    print("--------------------------------------")
    print(f"Speech root : {SPEECH_ROOT}")
    print(f"Noise root  : {NOISE_ROOT}")
    print(f"Speech files: {len(speech_files)}")
    print(f"Noise files : {len(noise_files)}")
    print(f"SNR         : {SNR_DB} dB")
    print(f"Target hours: {TARGET_HOURS if TARGET_HOURS is not None else 'ALL'}")
    print("======================================")

    for sf_path in speech_files:
        speech = load_audio_mono(sf_path)
        if len(speech) == 0:
            continue

        # ggf. auf Zielstunden kappen
        if target_samples is not None and total_samples >= target_samples:
            break

        # Noise wählen + passenden Ausschnitt ziehen
        nf_path = random.choice(noise_files)
        noise = load_audio_mono(nf_path)
        if len(noise) == 0:
            continue

        # Wenn wir auf Zielstunden kappen: Speech ggf. kürzen
        if target_samples is not None:
            remaining = target_samples - total_samples
            if remaining <= 0:
                break
            if len(speech) > remaining:
                speech = speech[:remaining]

        n = len(speech)

        # zufälliger Start im Noise (falls Noise lang genug)
        if len(noise) > n:
            start = random.randint(0, len(noise) - n)
            noise_seg = noise[start:start+n]
        else:
            noise_seg = noise  # mix_at_snr_db looped/trimmt noise auf n

        mixed = mix_at_snr_db(speech, noise_seg, SNR_DB)

        # Output-Dateiname: Relativpfad spiegeln (optional, aber praktisch)
        rel = sf_path.relative_to(SPEECH_ROOT)
        out_path = (OUT_ROOT / rel).with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(out_path), mixed, TARGET_SR, subtype="PCM_16")

        total_samples += n
        written += 1

        if written % 50 == 0:
            hours = total_samples / TARGET_SR / 3600
            print(f"[{written}] total written ≈ {hours:.2f} h")

    hours = total_samples / TARGET_SR / 3600
    print("======================================")
    print("Done.")
    print(f"Files written: {written}")
    print(f"Total duration: {hours:.2f} h")
    print(f"Output folder: {OUT_ROOT}")
    print("======================================")


if __name__ == "__main__":
    main()
