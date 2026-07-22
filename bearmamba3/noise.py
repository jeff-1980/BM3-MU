"""
bearmamba3/noise.py — additive noise generators shared across dataset pipelines.

AWGN injection has always lived inline in each Dataset's __getitem__ (see
data_cwru.py, data_xjtu.py) — this module adds a second, spectrally-shaped
noise type (pink, 1/f power spectral density) for E5's noise-model
generalisation check, without touching the existing AWGN code paths.

Pink noise generation (Voss-ish FFT filtering method, standard technique):
  1. Generate white Gaussian noise w[n] of the target length.
  2. rFFT to the frequency domain: W[k].
  3. Scale each bin's AMPLITUDE by 1/sqrt(f[k]) (f[0]=DC handled separately —
     scaling DC would blow up to infinity, so DC is left untouched: pink
     noise's power spectral density S(f) ~ 1/f is undefined at f=0 anyway).
     Amplitude ~ 1/sqrt(f) <=> power spectral density |W|^2 ~ 1/f, which is
     the definition of pink/flicker noise.
  4. irFFT back to the time domain.
  5. Rescale to match a target power (so the caller can hit an exact SNR the
     same way the existing AWGN code does: noise_power = sig_power / 10**(snr_db/10)).
"""
import numpy as np


def generate_pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance-ish pink noise, length n_samples. Caller rescales power."""
    white = rng.standard_normal(n_samples).astype(np.float64)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)  # freqs[0] == 0 (DC)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * scale, n=n_samples)
    # normalize to unit variance before the caller applies its own target power,
    # so this function's output variance doesn't depend on n_samples/scale drift
    pink = pink / (pink.std() + 1e-12)
    return pink.astype(np.float32)


def add_noise_at_snr(x: np.ndarray, snr_db: float, rng: np.random.Generator,
                      noise_type: str = "awgn") -> np.ndarray:
    """Add noise to 1D signal x at the given SNR (dB). noise_type: 'awgn' | 'pink'.
    Mirrors the existing inline AWGN formula (data_cwru.py/data_xjtu.py) exactly
    for noise_type='awgn', so switching noise_type doesn't change AWGN behaviour."""
    sig_power = np.mean(x.astype(np.float64) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    if noise_type == "awgn":
        noise = rng.standard_normal(x.shape).astype(np.float32)
    elif noise_type == "pink":
        noise = generate_pink_noise(x.shape[-1], rng)
        if x.ndim > 1:
            noise = np.broadcast_to(noise, x.shape).copy()
    else:
        raise ValueError(f"unknown noise_type={noise_type!r}, expected 'awgn' or 'pink'")
    noise = noise * np.sqrt(noise_power)
    return (x + noise).astype(np.float32)
