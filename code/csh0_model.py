"""Analytical CSH0 PPM-EMAT model: directivity, axial PSF and circumferential PSF."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

K0_MM_INV = 0.663547            # CSH0 design wavenumber at 340 kHz
WAVELENGTH_MM = 2.0 * np.pi / K0_MM_INV
SOURCE_COUNT = 6                # magnet columns
SOURCE_PITCH_MM = np.pi / K0_MM_INV
OBSERVATION_RADIUS_MM = 80.0
GROUP_VELOCITY_M_S = 3160.0
OUTER_RADIUS_MM = 75.0
APERTURES_MM = {"original": (2.0, 8.0), "optimized": (1.0, 15.0)}


def _sinc(x: np.ndarray) -> np.ndarray:
    """Return sin(x)/x with the limiting value one at x = 0."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = np.abs(x) > 1e-12
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def directivity(
    alpha_rad: np.ndarray,
    rc_mm: float,
    Rc_mm: float,
    k0_mm_inv: float = K0_MM_INV,
    observation_radius_mm: float = OBSERVATION_RADIUS_MM,
    source_count: int = SOURCE_COUNT,
    source_pitch_mm: float = SOURCE_PITCH_MM,
) -> np.ndarray:
    """Calculate the normalized finite-source single-pass directivity."""
    alpha_rad = np.asarray(alpha_rad, dtype=float)
    source_index = np.arange(1, source_count + 1)
    source_x = (source_index - (source_count + 1) / 2.0) * source_pitch_mm
    signs = (-1.0) ** (source_index + 1)
    response = np.zeros_like(alpha_rad, dtype=complex)

    for x_i, sign_i in zip(source_x, signs):
        rho_i = np.sqrt(
            (observation_radius_mm * np.cos(alpha_rad) - x_i) ** 2
            + (observation_radius_mm * np.sin(alpha_rad)) ** 2
        )
        alpha_i = np.arctan2(
            observation_radius_mm * np.sin(alpha_rad),
            observation_radius_mm * np.cos(alpha_rad) - x_i,
        )
        aperture_weight = (
            2.0 * Rc_mm * _sinc(k0_mm_inv * Rc_mm * np.sin(alpha_i))
            - 2.0 * rc_mm * _sinc(k0_mm_inv * rc_mm * np.sin(alpha_i))
        )
        propagator = np.sqrt(2.0 / (np.pi * k0_mm_inv * rho_i)) * np.exp(
            -1j * k0_mm_inv * rho_i
        )
        response += sign_i * propagator * aperture_weight

    magnitude = np.abs(response)
    return magnitude / magnitude.max()


def axial_psf(alpha_deg: np.ndarray, aperture: str = "optimized") -> np.ndarray:
    """Axial PSF h_z = D^2(alpha), alpha = arctan(dz / s), Eq. (3.4)."""
    rc, Rc = APERTURES_MM[aperture]
    h = directivity(np.deg2rad(np.asarray(alpha_deg, float)), rc, Rc) ** 2
    return h / h.max()


def circumferential_psf(f0_hz: float = 355.9e3, cycles: int = 5, fs_hz: float = 100e6,
                        band_hz=(200e3, 480e3)):
    """Circumferential PSF h_theta, Eq. (3.3).

    Envelope of the pulse-echo wavelet: Hann-windowed burst, PPM array response
    applied in transmission and reception, and the band-pass filter used for the
    C-scan. Returns (angle_deg, h) with the angle from Eq. (3.1).
    """
    c = GROUP_VELOCITY_M_S
    T = cycles / f0_hz
    t = np.arange(0.0, T, 1.0 / fs_hz)
    burst = np.sin(np.pi * t / T) ** 2 * np.sin(2.0 * np.pi * f0_hz * t)
    pitch_m = SOURCE_PITCH_MM * 1e-3
    array = np.zeros(int(round((SOURCE_COUNT - 1) * pitch_m / c * fs_hz)) + 1)
    for i in range(SOURCE_COUNT):
        array[int(round(i * pitch_m / c * fs_hz))] += (-1) ** i
    w = np.pad(np.convolve(np.convolve(burst, array), array), 20000)
    b, a = butter(2, [band_hz[0] / (fs_hz / 2), band_hz[1] / (fs_hz / 2)], "bandpass")
    env = np.abs(hilbert(filtfilt(b, a, w)))
    env /= env.max()
    angle = np.degrees(c * (np.arange(env.size) - np.argmax(env)) / fs_hz / 2.0
                       / (OUTER_RADIUS_MM * 1e-3))
    return angle, env


def fwhm_3db(x: np.ndarray, y: np.ndarray) -> float:
    """Linearly interpolated -3 dB amplitude full width around the global peak."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    peak = int(np.argmax(y))
    threshold = y[peak] * 10.0 ** (-3.0 / 20.0)

    left_index = peak
    while left_index > 0 and y[left_index] >= threshold:
        left_index -= 1
    left = np.interp(
        threshold,
        [y[left_index], y[left_index + 1]],
        [x[left_index], x[left_index + 1]],
    )

    right_index = peak
    while right_index < len(y) - 1 and y[right_index] >= threshold:
        right_index += 1
    right = np.interp(
        threshold,
        [y[right_index], y[right_index - 1]],
        [x[right_index], x[right_index - 1]],
    )
    return float(right - left)
