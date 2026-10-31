"""Coherent (analytic-signal) deconvolution of a pulse-echo A-scan with the model wavelet.

When two echoes overlap in time they add as signals, not as envelopes, so the envelope
model I = g (o * h_theta) + n no longer holds.  Here the analytic signal of the
band-passed A-scan is modelled as a sparse train of complex-weighted copies of the
analytic model wavelet w_a(t) of Eq. (3.3):

    z(t) = sum_k x_k w_a(t - t_k) + n(t),   x_k complex,

and x is estimated by complex L1 (FISTA with modulus soft-thresholding).  The only
calibration is the carrier frequency of w_a, taken from the spectrum of one isolated
reference echo (D2): the model wavelet is evaluated on a frequency scale such that its
carrier equals the measured carrier.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

import csh0_model as M

FS_HZ = 20e6
BAND_HZ = (200e3, 480e3)
TAU0_US = 6.25          # burst-centre delay; aligns the D2 reference with the packaged C-scan
GAIN_B = 0.013080802295779018
WINDOW_DEG = (72.0, 136.0)
GRID_DEG = np.arange(72.0, 137.0, 1.0)

_B, _A = butter(2, [BAND_HZ[0] / (FS_HZ / 2), BAND_HZ[1] / (FS_HZ / 2)], "bandpass")


def bandpass(x):
    x = np.asarray(x, float)
    return filtfilt(_B, _A, x - x[:100].mean())


def analytic_signal(trace):
    """Analytic signal of the band-passed A-scan. A complex input is taken as already processed
    (the public data set stores this signal, computed on the full records, within the analysis window)."""
    x = np.asarray(trace)
    return x if np.iscomplexobj(x) else hilbert(bandpass(x))


def theta_deg(time_s):
    t_us = np.asarray(time_s) * 1e6
    return np.degrees((M.GROUP_VELOCITY_M_S * 1e-3 * (t_us - TAU0_US) + 75.0) / 150.0)


def model_wavelet(scale=1.0, f0_hz=355.9e3, cycles=5, fs_hz=FS_HZ):
    """Analytic model pulse-echo wavelet (Eq. 3.3), optionally time-scaled by `scale`.

    scale < 1 compresses the wavelet (raises every frequency by 1/scale).
    Returns (w_a normalised to unit peak modulus, index of the peak).
    """
    fs_gen = fs_hz * scale
    T = cycles / f0_hz
    t = np.arange(0.0, T, 1.0 / fs_gen)
    burst = np.sin(np.pi * t / T) ** 2 * np.sin(2 * np.pi * f0_hz * t)
    pitch = M.SOURCE_PITCH_MM * 1e-3
    c = M.GROUP_VELOCITY_M_S
    arr = np.zeros(int(round((M.SOURCE_COUNT - 1) * pitch / c * fs_gen)) + 1)
    for i in range(M.SOURCE_COUNT):
        arr[int(round(i * pitch / c * fs_gen))] += (-1) ** i
    w = np.pad(np.convolve(np.convolve(burst, arr), arr), 400)
    b, a = butter(2, [BAND_HZ[0] / (fs_gen / 2), BAND_HZ[1] / (fs_gen / 2)], "bandpass")
    wa = hilbert(filtfilt(b, a, w))
    return wa / np.abs(wa).max(), int(np.abs(wa).argmax())


def carrier_hz(z, fs_hz=FS_HZ, level=0.5):
    """Mean instantaneous frequency of an analytic signal within its -6 dB support."""
    fi = np.diff(np.unwrap(np.angle(z))) * fs_hz / (2 * np.pi)
    m = np.abs(z)[1:] > level * np.abs(z).max()
    return float(fi[m].mean())


def calibrated_scale(reference_trace, time_s, ref_window_deg=(88.0, 116.0)):
    """Time scale that matches the model carrier to the carrier of a reference echo."""
    th = theta_deg(time_s)
    z = analytic_signal(reference_trace)
    g = (th > ref_window_deg[0]) & (th < ref_window_deg[1])
    w1, _ = model_wavelet(1.0)
    return carrier_hz(w1) / carrier_hz(z[g])


def deconvolve_coherent(trace_V, time_s, scale, kappa, iterations=1000, tol=1e-7):
    """Return the restored profile (mV, 1-degree bins on GRID_DEG) of one A-scan."""
    th = theta_deg(time_s)
    z_full = analytic_signal(trace_V) * 1e3
    idx = np.where((th >= WINDOW_DEG[0] - 0.5) & (th < WINDOW_DEG[1] + 0.5))[0]
    thw = th[idx]
    tp = thw[np.abs(z_full[idx]).argmax()]
    gain = np.exp(GAIN_B * (thw - tp))                       # Eq. (3.6)
    z = z_full[idx] * gain
    wa, k = model_wavelet(scale)
    n, L = len(z), len(wa)
    N = 1 << int(np.ceil(np.log2(n + 2 * L)))
    Hf = np.fft.fft(wa, N)

    def A(x):
        return np.fft.ifft(np.fft.fft(x, N) * Hf)[k:k + n]

    def AH(r):
        rp = np.zeros(N, complex)
        rp[k:k + n] = r
        return np.fft.ifft(np.fft.fft(rp) * np.conj(Hf))[:n]

    step = 1.0 / np.max(np.abs(Hf)) ** 2
    lam = kappa * np.abs(AH(z)).max()
    x = np.zeros(n, complex)
    y = x.copy()
    q = 1.0
    for _ in range(iterations):
        v = y - step * AH(A(y) - z)
        mag = np.abs(v)
        xn = np.where(mag > lam * step, (1 - lam * step / np.maximum(mag, 1e-30)) * v, 0)
        qn = 0.5 * (1 + np.sqrt(1 + 4 * q * q))
        y = xn + (q - 1) / qn * (xn - x)
        done = np.linalg.norm(xn - x) <= tol * (np.linalg.norm(x) + 1e-12)
        x, q = xn, qn
        if done:
            break
    o = np.abs(x) / gain
    out = np.zeros(len(GRID_DEG))
    for j, c in enumerate(GRID_DEG):
        m = (thw >= c - 0.5) & (thw < c + 0.5)
        if m.any():
            out[j] = o[m].max()
    return out


def envelope_profile(trace_V, time_s):
    """Raw C-scan pixel values of one A-scan: envelope maximum in each 1-degree bin (mV)."""
    th = theta_deg(time_s)
    e = np.abs(analytic_signal(trace_V)) * 1e3
    return np.array([e[(th >= c - 0.5) & (th < c + 0.5)].max() for c in GRID_DEG])
