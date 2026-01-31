import os
import json
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, sosfiltfilt
import librosa   
import pywt   
import matplotlib.pyplot as plt

def evaluate_denoising(clean_path, noisy_path,
                       sigma=0.95, low=1, high=8800, order=2,
                       sg_n_fft=1024, sg_hop_length=512,
                       sg_noise_duration=0.5, sg_thresh_factor=1.5, sg_med_size=(3,3),
                       wav_wavelet='db8',
                       return_audio=False):

    clean, fs = sf.read(clean_path)
    noisy, fs2 = sf.read(noisy_path)
    if fs2 != fs:
        noisy = librosa.resample(noisy, orig_sr=fs2, target_sr=fs)

    if clean.ndim > 1: clean = clean.mean(1)
    if noisy.ndim > 1: noisy = noisy.mean(1)
    N = min(len(clean), len(noisy))
    clean, noisy = clean[:N], noisy[:N]

    clean /= np.max(np.abs(clean)) + 1e-12
    noisy /= np.max(np.abs(noisy)) + 1e-12

    def _snr(r, x):  return 10 * np.log10(np.sum(r*r) / (np.sum((x-r)**2) + 1e-12))
    def _rmse(r, x): return np.sqrt(np.mean((r-x)**2))

    snr_orig = _snr(clean, noisy)

    gauss     = gaussian_filter1d(noisy, sigma)
    snr_g     = _snr(clean, gauss)
    rmse_g    = _rmse(clean, gauss)

    nyq       = fs/2
    high_cut  = min(high, nyq*0.999)
    sos       = butter(order, [low/nyq, high_cut/nyq], btype='band', output='sos')
    butter_bp = sosfiltfilt(sos, noisy)
    snr_b     = _snr(clean, butter_bp)
    rmse_b    = _rmse(clean, butter_bp)

    D         = librosa.stft(noisy, n_fft=sg_n_fft, hop_length=sg_hop_length)
    mag, ph   = np.abs(D), np.angle(D)
    n_noise   = int(sg_noise_duration * fs / sg_hop_length)
    n_noise   = max(1, min(n_noise, mag.shape[1]))  # guard
    noise_pr  = np.median(mag[:, :n_noise], axis=1, keepdims=True)
    thresh    = noise_pr * sg_thresh_factor
    mask      = median_filter((mag >= thresh).astype(float), size=sg_med_size)
    mag_den   = mag * mask
    D_den     = mag_den * np.exp(1j * ph)
    spec_gate = librosa.istft(D_den, hop_length=sg_hop_length, length=N)
    snr_sg    = _snr(clean, spec_gate)
    rmse_sg   = _rmse(clean, spec_gate)

    max_lev   = pywt.dwt_max_level(N, pywt.Wavelet(wav_wavelet).dec_len)
    coeffs    = pywt.wavedec(noisy, wav_wavelet, level=max_lev)
    sigma_n   = np.median(np.abs(coeffs[-1])) / 0.6745
    thr       = sigma_n * np.sqrt(2 * np.log(N))
    coeffs[1:] = [pywt.threshold(c, thr, 'soft') for c in coeffs[1:]]
    wav_den   = pywt.waverec(coeffs, wav_wavelet)[:N]
    snr_w     = _snr(clean, wav_den)
    rmse_w    = _rmse(clean, wav_den)

    stats = {
        'snr_original': snr_orig,
        'gaussian':        {'snr': snr_g,  'improvement': snr_g - snr_orig,  'rmse': rmse_g},
        'butterworth':     {'snr': snr_b,  'improvement': snr_b - snr_orig,  'rmse': rmse_b},
        'spectral_gating': {'snr': snr_sg, 'improvement': snr_sg - snr_orig, 'rmse': rmse_sg},
        'wavelet':         {'snr': snr_w,  'improvement': snr_w - snr_orig,  'rmse': rmse_w},
        'fs': fs
    }

    if return_audio:
        audio = {
            'gaussian': gauss,
            'butterworth': butter_bp,
            'spectral_gating': spec_gate,
            'wavelet_transform': wav_den,  
        }
        return stats, audio

    return stats

if __name__ == "__main__":
    clean_files = [
        "./assets/speech/Speech1.wav",
        "./assets/speech/Speech2.wav",
        "./assets/speech/Speech3.wav",
        "./assets/speech/Speech4.wav",
        "./assets/speech/Speech5.wav",
        "./assets/speech/Speech6.wav"
    ]
    noise_files = [
        "./assets/noise/Sprinkler.wav",
        "./assets/noise/Keyboard.wav",
        "./assets/noise/WaterDrop.wav",
        "./assets/noise/FireCrackle.wav",
        "./assets/noise/Chainsaw.wav"
    ]
    mixed_dir = "./assets/merged/"

    all_stats = []

    denoised_base = "./assets/denoised"
    out_dirs = {
        "gaussian": os.path.join(denoised_base, "gaussian"),
        "butterworth": os.path.join(denoised_base, "butterworth"),
        "spectral_gating": os.path.join(denoised_base, "spectral_gating"),
        "wavelet_transform": os.path.join(denoised_base, "wavelet_transform"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    for clean_file in clean_files:
        clean_name = os.path.splitext(os.path.basename(clean_file))[0]

        for noise_file in noise_files:
            noise_name = os.path.splitext(os.path.basename(noise_file))[0]
            mfp = os.path.join(mixed_dir, f"{clean_name},{noise_name}.wav")

            if not os.path.isfile(mfp):
                print(f"Missing {mfp}, skipping")
                continue

            stats, audio_out = evaluate_denoising(clean_file, mfp, return_audio=True)
            stats["clean_file"] = clean_name
            stats["noise_file"] = noise_name
            all_stats.append(stats)

            out_name = f"{clean_name},{noise_name}.wav"
            fs = stats["fs"]

            print(f"Writing denoised files for {out_name}")
            for method, y in audio_out.items():
                y = np.asarray(y, dtype=np.float32)

                peak = np.max(np.abs(y)) + 1e-12
                if peak > 1.0:
                    y = y / peak

                sf.write(os.path.join(out_dirs[method], out_name), y, fs)

    with open("./assets/batch_denoise_results.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Wrote batch_denoise_results.json with {len(all_stats)} entries")

    rows = []
    for e in all_stats:
        rows.append({
            "clean_file": e["clean_file"],
            "noise_file": e["noise_file"],
            "snr_original": e["snr_original"],

            "snr_gauss": e["gaussian"]["snr"],
            "imp_gauss": e["gaussian"]["improvement"],
            "rmse_gauss": e["gaussian"]["rmse"],

            "snr_butter": e["butterworth"]["snr"],
            "imp_butter": e["butterworth"]["improvement"],
            "rmse_butter": e["butterworth"]["rmse"],

            "snr_specgate": e["spectral_gating"]["snr"],
            "imp_specgate": e["spectral_gating"]["improvement"],
            "rmse_specgate": e["spectral_gating"]["rmse"],

            "snr_wavelet": e["wavelet"]["snr"],
            "imp_wavelet": e["wavelet"]["improvement"],
            "rmse_wavelet": e["wavelet"]["rmse"],
        })

    df = pd.DataFrame(rows)
    print(df.head())

    df.to_csv("./assets/denoise_results.csv", index=False)
    print("Wrote denoise_results.csv")

    df.to_excel("./assets/denoise_results.xlsx", index=False)
    print("Wrote denoise_results.xlsx")


df = pd.read_csv("./assets/denoise_results.csv")
    
# Histogram for SNR Improvement (Gaussian)
plt.figure()
df['imp_gauss'].hist(bins=20)
plt.title("Histogram of SNR Improvement (Gaussian)")
plt.xlabel("SNR Improvement (dB)")
plt.ylabel("Count")
plt.grid(False)
plt.show()

# Histogram for SNR Improvement (Butterworth)
plt.figure()
df['imp_butter'].hist(bins=20)
plt.title("Histogram of SNR Improvement (Butterworth)")
plt.xlabel("SNR Improvement (dB)")
plt.ylabel("Count")
plt.show()

# Histogram for SNR Improvement (Spectral Gating)
plt.figure()
df['imp_specgate'].hist(bins=20)
plt.title("Histogram of SNR Improvement (Spectral Gating)")
plt.xlabel("SNR Improvement (dB)")
plt.ylabel("Count")
plt.show()

# Histogram for SNR Improvement (Wavelet)
plt.figure()
df['imp_wavelet'].hist(bins=20)
plt.title("Histogram of SNR Improvement (Wavelet)")
plt.xlabel("SNR Improvement (dB)")
plt.ylabel("Count")
plt.show()

# RMSE for (Gaussian)
plt.figure()
df['rmse_gauss'].hist(bins=20)
plt.title("Histogram of RMSE (Gaussian)")
plt.xlabel("RMSE")
plt.ylabel("Count")
plt.show()

# RMSE for (Butterworth)
plt.figure()
df['rmse_butter'].hist(bins=20)
plt.title("Histogram of RMSE (Butterworth)")
plt.xlabel("RMSE")
plt.ylabel("Count")
plt.show()

# RMSE for (Spectral Gating)
plt.figure()
df['rmse_specgate'].hist(bins=20)
plt.title("Histogram of RMSE (Spectral Gating)")
plt.xlabel("RMSE")
plt.ylabel("Count")
plt.show()

# RMSE for (Wavelet)
plt.figure()
df['rmse_wavelet'].hist(bins=20)
plt.title("Histogram of RMSE (Wavelet)")
plt.xlabel("RMSE")
plt.ylabel("Count")
plt.show()

# Scatterplot of SNR Improvement vs RMSE (Gaussian)
plt.figure()
plt.scatter(df['imp_gauss'], df['rmse_gauss'])
plt.title("SNR Improvement vs RMSE (Gaussian)")
plt.xlabel("SNR Improvement (dB)")
plt.ylabel("RMSE")
plt.show()

# Scatterplot of SNR Improvement vs RMSE (Butterworth)
plt.figure()
plt.scatter(df['imp_butter'], df['rmse_butter'])
plt.title("SNR Improvement vs RMSE (Butterworth)")
plt.xlabel("SNR Improvement (dB)")
plt.ylabel("RMSE")
plt.show()

# Scatterplot of SNR Improvement vs RMSE (Spectral Gating)
plt.figure()
plt.scatter(df['imp_specgate'], df['rmse_specgate'])
plt.title("SNR Improvement vs RMSE (Spectral Gating)")
plt.xlabel("SNR Improvement (dB)")
plt.ylabel("RMSE")
plt.show()

# Scatterplot of SNR Improvement vs RMSE (Wavelet)
plt.figure()
plt.scatter(df['imp_wavelet'], df['rmse_wavelet'])
plt.title("SNR Improvement vs RMSE (Wavelet)")
plt.xlabel("SNR Improvement (dB)")
plt.ylabel("RMSE")
plt.show()

# Boxplot for different methods RMSE
plt.figure()
plt.boxplot(
    [df['rmse_gauss'], df['rmse_butter'], df['rmse_specgate'], df['rmse_wavelet']],
    labels=['Gaussian','Butterworth','Spectral Gate','Wavelet']
)
plt.title("RMSE Comparison Across Methods")
plt.ylabel("RMSE")
plt.show()

# Boxplot for different methods SNR Improvement
plt.figure()
plt.boxplot(
    [df['imp_gauss'], df['imp_butter'], df['imp_specgate'], df['imp_wavelet']],
    labels=['Gaussian','Butterworth','Spectral Gate','Wavelet']
)
plt.title("SNR Improvement Comparison Across Methods")
plt.ylabel("SNR Improvement (dB)")
plt.show()
