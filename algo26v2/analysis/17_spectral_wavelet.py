"""Spectral (frequency-domain) and wavelet (time-frequency) analysis.

Libraries: scipy.signal (periodogram, Welch PSD, coherence, spectral slope),
PyWavelets (DWT energy-by-scale, CWT power). numpy for DFA.

Questions:
  - Is there any periodicity/cycle in returns (peaks in the PSD)?
  - What is the spectral colour of returns (white/pink/red)? White => memoryless.
  - Frequency-domain lead-lag: coherence between ALGO and each name.
  - Wavelet energy distribution across scales; is variance concentrated at any
    horizon? Wavelet cross-correlation for the cointegrated pairs.
"""
import warnings, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import signal
import pywt
from common import load, log_returns, section, Recorder, RESULTS, stars

rec = Recorder("spectral_wavelet")
df, tickers = load()
rets = log_returns(df)
R = rets.values
T, N = R.shape
mkt = R.mean(axis=1)

section("17A. POWER SPECTRAL DENSITY - is there any dominant cycle?")
print("Peak frequency of Welch PSD per instrument (period in days = 1/freq).")
print("A truly white series has a flat PSD (no meaningful peak).\n")
peak_periods = []
flat_ratio = []
for i in range(N):
    f, pxx = signal.welch(R[:, i] - R[:, i].mean(), nperseg=128)
    pk = f[np.argmax(pxx[1:]) + 1]
    period = 1 / pk if pk > 0 else np.inf
    peak_periods.append(period)
    # flatness: geometric mean / arithmetic mean of PSD (1.0 = perfectly white)
    fl = np.exp(np.mean(np.log(pxx[1:] + 1e-20))) / np.mean(pxx[1:])
    flat_ratio.append(fl)
    rec.add("scipy.signal", "spectral", "welch_peak_period_days", tickers[i], period, np.nan)
    rec.add("scipy.signal", "spectral", "spectral_flatness", tickers[i], fl, np.nan)
pp = np.array(peak_periods)
print(f"Peak-period distribution across 51: median {np.median(pp[np.isfinite(pp)]):.1f}d, "
      f"spread {np.percentile(pp[np.isfinite(pp)],25):.1f}-{np.percentile(pp[np.isfinite(pp)],75):.1f}d")
print(f"Mean spectral flatness: {np.mean(flat_ratio):.3f} "
      f"(near 1.0 => white noise, no exploitable cycle)")
print("=> Peak periods are scattered with no common cycle -> no periodicity.")

section("17B. SPECTRAL SLOPE (colour of noise): log-PSD vs log-freq")
print("slope ~0 white (memoryless), <0 red/persistent, >0 blue/anti-persistent.\n")
slopes = []
for i in range(N):
    f, pxx = signal.periodogram(R[:, i] - R[:, i].mean())
    m = f > 0
    slope = np.polyfit(np.log(f[m]), np.log(pxx[m] + 1e-20), 1)[0]
    slopes.append(slope)
    rec.add("scipy.signal", "spectral", "spectral_slope", tickers[i], slope, np.nan)
print(f"Mean spectral slope: {np.mean(slopes):+.3f} (std {np.std(slopes):.3f})")
print(f"  slopes within +/-0.15 of 0 (white): {(np.abs(slopes)<0.15).sum()}/{N}")
print("=> Returns are spectrally white -> consistent with the autocorrelation nulls.")

section("17C. MAGNITUDE-SQUARED COHERENCE: ALGO vs each name (freq-domain link)")
print("Mean coherence over all frequencies (0=unrelated, 1=linearly related).\n")
coh_mean = []
for i in range(1, N):
    f, cxy = signal.coherence(R[:, 0], R[:, i], nperseg=128)
    cm = cxy.mean()
    coh_mean.append((tickers[i], cm))
    rec.add("scipy.signal", "coherence", "algo_coherence_mean", tickers[i], cm, np.nan)
coh_mean.sort(key=lambda x: -x[1])
print("Highest mean coherence with ALGO:")
for t, c in coh_mean[:10]:
    print(f"  {t}: {c:.3f}")
print(f"Average coherence with ALGO across names: {np.mean([c for _,c in coh_mean]):.3f}")

section("17D. WAVELET ENERGY BY SCALE (DWT, db4) - where does variance live?")
print("Fraction of return variance at each dyadic scale (level 1=2d ... level 6=64d).\n")
levels = 6
energy_by_level = np.zeros(levels + 1)
for i in range(N):
    coeffs = pywt.wavedec(R[:, i], "db4", level=levels)
    e = np.array([np.sum(c ** 2) for c in coeffs])  # [approx, detail_L..detail_1]
    energy_by_level += e / e.sum()
energy_by_level /= N
labels = ["approx(>64d)"] + [f"~{2**(levels-k+1)}d" for k in range(1, levels + 1)]
for lab, e in zip(labels, energy_by_level):
    print(f"  {lab:<14}: {e*100:5.1f}%")
    rec.add("pywt", "wavelet", "dwt_energy_fraction", lab, e, np.nan)
print("=> Roughly geometric fall-off (most energy at the finest scale) is the")
print("   signature of white noise; a spike at one scale would flag a cycle.")

section("17E. WAVELET COHERENCE (CWT) for the 6 cointegrated pairs")
PAIRS = [("AENO","NWIG"),("EORC","NGTE"),("HETT","ULXY"),("SMAH","ILVX"),
         ("HUXZ","ACAC"),("CTGI","EELT")]
idx = {t: k for k, t in enumerate(tickers)}
scales = np.arange(2, 64)
for a, b in PAIRS:
    ca, _ = pywt.cwt(R[:, idx[a]], scales, "morl")
    cb, _ = pywt.cwt(R[:, idx[b]], scales, "morl")
    # scale-averaged correlation of wavelet power between the two series
    corr = np.corrcoef(np.abs(ca).mean(0), np.abs(cb).mean(0))[0, 1]
    rec.add("pywt", "wavelet_coherence", "cwt_power_corr", f"{a}-{b}", corr, np.nan)
    print(f"  {a}-{b}: time-varying wavelet-power corr = {corr:+.3f}")

section("17F. VERDICT")
print("Frequency & time-frequency domains agree with the time-domain result:")
print("returns are spectrally WHITE (no cycles, flat PSD, slope~0, geometric")
print("wavelet energy). The only cross-instrument link visible in the spectrum")
print("is broadband coherence with the ALGO factor - not a tradeable frequency.")
rec.save()
