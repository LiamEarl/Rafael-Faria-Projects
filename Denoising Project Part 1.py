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
                       wav_wavelet='db8'):
    clean, fs = sf.read(clean_path)
    noisy, fs2 = sf.read(noisy_path)
    if fs2 != fs:
        noisy = librosa.resample(noisy, orig_sr=fs2, target_sr=fs)

    if clean.ndim > 1: clean = clean.mean(1)
    if noisy.ndim > 1: noisy = noisy.mean(1)
    N = min(len(clean), len(noisy))
    clean, noisy = clean[:N], noisy[:N]
    clean /= np.max(np.abs(clean))
    noisy /= np.max(np.abs(noisy))

    def _snr(r, x):  return 10 * np.log10(np.sum(r*r) / np.sum((x-r)**2))
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
    noise_pr  = np.median(mag[:, :n_noise], axis=1, keepdims=True)
    thresh    = noise_pr * sg_thresh_factor
    mask      = median_filter((mag>=thresh).astype(float), size=sg_med_size)
    mag_den   = mag * mask
    D_den     = mag_den * np.exp(1j*ph)
    spec_gate = librosa.istft(D_den, hop_length=sg_hop_length, length=N)
    snr_sg    = _snr(clean, spec_gate)
    rmse_sg   = _rmse(clean, spec_gate)

    max_lev   = pywt.dwt_max_level(N, pywt.Wavelet(wav_wavelet).dec_len)
    coeffs    = pywt.wavedec(noisy, wav_wavelet, level=max_lev)
    sigma_n   = np.median(np.abs(coeffs[-1]))/0.6745
    thr       = sigma_n * np.sqrt(2 * np.log(N))
    coeffs[1:]= [pywt.threshold(c, thr, 'soft') for c in coeffs[1:]]
    wav_den   = pywt.waverec(coeffs, wav_wavelet)[:N]
    snr_w     = _snr(clean, wav_den)
    rmse_w    = _rmse(clean, wav_den)

    return {
        'snr_original': snr_orig,
        'gaussian':      {'snr': snr_g,  'improvement': snr_g - snr_orig,  'rmse': rmse_g},
        'butterworth':   {'snr': snr_b,  'improvement': snr_b - snr_orig,  'rmse': rmse_b},
        'spectral_gating': {'snr': snr_sg,'improvement': snr_sg - snr_orig,'rmse': rmse_sg},
        'wavelet':       {'snr': snr_w,  'improvement': snr_w - snr_orig,  'rmse': rmse_w},
    }

if __name__ == "__main__":
    clean_files = [
        r"C:\Users\faria\AppData\Local\Programs\Python\Python312\Speech 1.wav",
        r"C:\Users\faria\AppData\Local\Programs\Python\Python312\Speech 2.wav",
        r"C:\Users\faria\AppData\Local\Programs\Python\Python312\Speech 3.wav",
        r"C:\Users\faria\AppData\Local\Programs\Python\Python312\Speech 4.wav",
        r"C:\Users\faria\AppData\Local\Programs\Python\Python312\Speech 5.wav",
        r"C:\Users\faria\AppData\Local\Programs\Python\Python312\Speech 6.wav"
    ]
    noise_files = [
        r"C:\Users\faria\AppData\Local\Programs\Python\Python312\Sprinkler.wav",
        r"C:\Users\faria\AppData\Local\Programs\Python\Python312\Keyboard.wav",
        r"C:\Users\faria\AppData\Local\Programs\Python\Python312\Water Drop.wav",
        r"C:\Users\faria\AppData\Local\Programs\Python\Python312\Fire Crackle.wav",
        r"C:\Users\faria\AppData\Local\Programs\Python\Python312\Chainsaw.wav"
    ]
    mixed_dir = r"C:\Users\faria\Downloads\Merged"

    all_stats = []
    for cfp in clean_files:
        cname = os.path.splitext(os.path.basename(cfp))[0]
        for nfp in noise_files:
            nname = os.path.splitext(os.path.basename(nfp))[0]
            mfp   = os.path.join(mixed_dir, f"{cname}, {nname}.wav")
            if not os.path.isfile(mfp):
                print(f"Missing {mfp}, skipping")
                continue
            stats = evaluate_denoising(cfp, mfp)
            stats['clean_file'] = cname
            stats['noise_file'] = nname
            all_stats.append(stats)

    with open("batch_denoise_results.json","w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Wrote batch_denoise_results.json with {len(all_stats)} entries")

    rows = []
    for e in all_stats:
        rows.append({
            'clean_file':     e['clean_file'],
            'noise_file':     e['noise_file'],
            'snr_original':   e['snr_original'],

            'snr_gauss':      e['gaussian']['snr'],
            'imp_gauss':      e['gaussian']['improvement'],
            'rmse_gauss':     e['gaussian']['rmse'],

            'snr_butter':     e['butterworth']['snr'],
            'imp_butter':     e['butterworth']['improvement'],
            'rmse_butter':    e['butterworth']['rmse'],

            'snr_specgate':   e['spectral_gating']['snr'],
            'imp_specgate':   e['spectral_gating']['improvement'],
            'rmse_specgate':  e['spectral_gating']['rmse'],

            'snr_wavelet':    e['wavelet']['snr'],
            'imp_wavelet':    e['wavelet']['improvement'],
            'rmse_wavelet':   e['wavelet']['rmse'],
        })

    df = pd.DataFrame(rows)
    print(df.head())   

    df.to_csv("denoise_results.csv", index=False)
    print("Wrote denoise_results.csv")

    df.to_excel("denoise_results.xlsx", index=False)
    print("Wrote denoise_results.xlsx ")



df = pd.read_csv("denoise_results.csv")
    
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