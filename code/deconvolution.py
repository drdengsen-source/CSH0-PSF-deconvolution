"""Core non-negative L1 circular-deconvolution routine."""

from __future__ import annotations

import numpy as np


def centre_kernel(kernel: np.ndarray) -> np.ndarray:
    """Move the centre of a sampled PSF to index zero for circular convolution."""
    return np.fft.ifftshift(np.asarray(kernel, dtype=float))


def circular_convolution(signal: np.ndarray, centred_kernel: np.ndarray) -> np.ndarray:
    """Circular convolution on a uniform closed coordinate."""
    signal = np.asarray(signal, dtype=float)
    centred_kernel = np.asarray(centred_kernel, dtype=float)
    return np.real(
        np.fft.ifft(np.fft.fft(signal) * np.fft.fft(centred_kernel))
    )


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.clip(np.abs(values) - threshold, 0.0, None)


def deconvolve_l1(
    observation: np.ndarray,
    psf: np.ndarray,
    regularization: float,
    iterations: int = 500,
    tolerance: float = 1e-7,
) -> np.ndarray:
    """Solve a non-negative L1 circular inverse problem with FISTA.

    The objective is

        0.5 ||h circconv o - I||_2^2 + regularization ||o||_1,
        subject to o >= 0.
    """
    observation = np.asarray(observation, dtype=float)
    H = np.fft.fft(centre_kernel(psf))
    lipschitz = float(np.max(np.abs(H) ** 2)) + 1e-12
    step = 1.0 / lipschitz

    def forward(values: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft(H * np.fft.fft(values)))

    def adjoint(values: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft(np.conj(H) * np.fft.fft(values)))

    solution = np.zeros_like(observation)
    accelerated = solution.copy()
    momentum = 1.0

    for _ in range(int(iterations)):
        gradient = adjoint(forward(accelerated) - observation)
        updated = _soft_threshold(
            accelerated - step * gradient, regularization * step
        )
        updated = np.clip(updated, 0.0, None)
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        accelerated = updated + (momentum - 1.0) / next_momentum * (
            updated - solution
        )
        if np.linalg.norm(updated - solution) <= tolerance * (
            np.linalg.norm(solution) + 1e-12
        ):
            solution = updated
            break
        solution = updated
        momentum = next_momentum

    return solution
