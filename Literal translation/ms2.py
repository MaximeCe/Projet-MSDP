# ms2.py
"""
Traduction Python de ms2.f (routines numériques / profils / interpolation).
But :
 - Fournir des profils usuels (gaussien, lorentzien, voigt), convolution, interpolation 1D
 - Offrir des utilitaires pour normalisation, intégration discrète, et fit de pics (optionnel)
 - Être défensif : fallback si scipy absent
 - Conserver une API simple pour être appelée depuis ms1.py

Dépendances : numpy. scipy est utilisé si présent (pour fftconvolve, special.wofz, optimize.curve_fit)
"""

from __future__ import annotations
from typing import Tuple, Optional, Callable
import math
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("ms2")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Try to import scipy components optionally
_have_scipy = True
try:
    from scipy.signal import fftconvolve
    from scipy.special import wofz  # for Voigt
    from scipy.optimize import curve_fit
except Exception:
    _have_scipy = False
    fftconvolve = None
    wofz = None
    curve_fit = None
    logger.info(
        "scipy non disponible : certaines fonctions utiliseront des versions de repli plus lentes.")


# ---------------------------
# Profiles: Gaussian, Lorentzian, Voigt
# ---------------------------
def gaussian(x: np.ndarray, center: float = 0.0, sigma: float = 1.0, amplitude: float = 1.0) -> np.ndarray:
    """Gaussien normalisé (amplitude paramétrable)."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    arg = (x - center) / sigma
    return amplitude * np.exp(-0.5 * arg * arg)


def lorentzian(x: np.ndarray, center: float = 0.0, gamma: float = 1.0, amplitude: float = 1.0) -> np.ndarray:
    """Lignes de Lorentz (demi-largeur à mi-hauteur = gamma)."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    denom = (x - center) ** 2 + gamma * gamma
    return amplitude * (gamma * gamma) / denom


def voigt(x: np.ndarray, center: float = 0.0, sigma: float = 1.0, gamma: float = 1.0, amplitude: float = 1.0) -> np.ndarray:
    """
    Profils de Voigt (convolution d'un gaussien et d'un lorentzien).
    Utilise la fonction wofz (Faddeeva) si disponible, sinon approx par convolution.
    """
    if sigma <= 0 or gamma < 0:
        raise ValueError("sigma must be >0 and gamma >=0")
    if _have_scipy and wofz is not None:
        z = ((x - center) + 1j * gamma) / (sigma * math.sqrt(2))
        return amplitude * np.real(wofz(z)) / (sigma * math.sqrt(2 * math.pi))
    else:
        # fallback: approx via numeric convolution of gaussian & lorentzian (slower)
        logger.debug(
            "voigt: fallback numeric convolution (lent) car scipy.special.wofz absent.")
        xs = np.linspace(x.min() - 5 * sigma, x.max() +
                         5 * sigma, x.size * 4 + 1)
        g = gaussian(xs, center=0.0, sigma=sigma, amplitude=1.0)
        l = lorentzian(xs, center=0.0, gamma=gamma, amplitude=1.0)
        kern = np.convolve(g, l, mode="same")
        # interpolate kern back to x positions (centered)
        # recentrer
        center_idx = len(xs) // 2
        kern_centered = kern[center_idx -
                             (x.size // 2): center_idx - (x.size // 2) + x.size]
        # normaliser amplitude ~1 then appliquer amplitude parameter
        kern_centered /= kern_centered.max() if kern_centered.max() != 0 else 1.0
        return amplitude * kern_centered


# ---------------------------
# Convolution utilities
# ---------------------------
def convolve_1d(signal: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """
    Convolution 1D with optional fft acceleration (scipy) or numpy fft fallback.
    mode: 'same' or 'full' or 'valid'
    """
    if signal.ndim != 1 or kernel.ndim != 1:
        raise ValueError("signal and kernel must be 1D arrays")
    if _have_scipy and fftconvolve is not None:
        return fftconvolve(signal, kernel, mode=mode)
    else:
        # numpy FFT-based convolution (works but careful with precision)
        n = signal.size + kernel.size - 1
        # next power of two for efficiency
        nfft = 1 << (n - 1).bit_length()
        S = np.fft.rfft(signal, nfft)
        K = np.fft.rfft(kernel, nfft)
        conv = np.fft.irfft(S * K, nfft)[:n]
        if mode == "full":
            return conv
        elif mode == "same":
            start = (kernel.size - 1) // 2
            return conv[start:start + signal.size]
        elif mode == "valid":
            start = kernel.size - 1
            end = start + signal.size - kernel.size + 1
            return conv[start:end]
        else:
            raise ValueError("mode must be 'same', 'full' or 'valid'")


def convolve_2d_rows(array2d: np.ndarray, kernel_1d: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Convolve each row (or column) of a 2D array by a 1D kernel.
    axis=1 => convolve along x (rows), axis=0 => along y (columns)
    """
    if array2d.ndim != 2:
        raise ValueError("array2d must be 2D")
    out = np.zeros_like(array2d, dtype=float)
    if axis == 1:
        for j in range(array2d.shape[0]):
            out[j, :] = convolve_1d(array2d[j, :], kernel_1d, mode="same")
    elif axis == 0:
        for i in range(array2d.shape[1]):
            out[:, i] = convolve_1d(array2d[:, i], kernel_1d, mode="same")
    else:
        raise ValueError("axis must be 0 or 1")
    return out


# ---------------------------
# Interpolation / resampling
# ---------------------------
def linear_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, left: Optional[float] = None, right: Optional[float] = None) -> np.ndarray:
    """
    Simple wrapper around numpy.interp for 1D linear interpolation.
    xp must be increasing.
    """
    return np.interp(x, xp, fp, left=left, right=right)


def resample_even(x_old: np.ndarray, y_old: np.ndarray, n_new: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample (x_old,y_old) onto an evenly spaced grid of n_new points between x_old.min() and x_old.max().
    Uses linear interpolation.
    """
    x_new = np.linspace(float(x_old.min()), float(x_old.max()), int(n_new))
    y_new = linear_interp(x_new, x_old, y_old)
    return x_new, y_new


# ---------------------------
# Normalisation / integration
# ---------------------------
def normalize_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Normalise y so that integral(y dx) == 1 using trapezoidal rule.
    """
    area = np.trapz(y, x)
    if area == 0:
        logger.warning("normalize_area: area == 0, returning original y")
        return y
    return y / area


def integrate_trapz(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal integral."""
    return float(np.trapz(y, x))


# ---------------------------
# Peak finding / fitting (optionnel)
# ---------------------------
def find_peaks_simple(y: np.ndarray, threshold: float = 0.0, min_distance: int = 1) -> np.ndarray:
    """
    Un algorithme simple de détection de pics: points où y[i] > neighbors and > threshold.
    Renvoie indices des pics.
    """
    n = y.size
    peaks = []
    for i in range(1, n - 1):
        if y[i] > threshold and y[i] > y[i - 1] and y[i] > y[i + 1]:
            # vérifier distance minimale
            if len(peaks) == 0 or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def fit_peak_gaussian(x: np.ndarray, y: np.ndarray, p0: Optional[Tuple[float, float, float]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fit a single gaussian to (x,y) using scipy.optimize.curve_fit if disponible.
    p0 = (center, sigma, amplitude)
    Retourne (popt, pcov) where popt = [center, sigma, amplitude]
    """
    def model(xi, center, sigma, amplitude):
        return gaussian(xi, center=center, sigma=sigma, amplitude=amplitude)

    if curve_fit is None:
        logger.warning(
            "curve_fit absent (scipy). fit_peak_gaussian returns approximate moment-based estimates.")
        # approx: moments
        area = np.trapz(y, x)
        if area <= 0:
            return np.array([0.0, 1.0, 0.0]), None
        center_est = np.trapz(x * y, x) / area
        sigma_est = math.sqrt(
            abs(np.trapz(((x - center_est) ** 2) * y, x) / area))
        amp_est = y.max()
        return np.array([center_est, sigma_est, amp_est]), None

    # provide p0 if not given
    if p0 is None:
        amp0 = y.max()
        center0 = x[np.argmax(y)]
        # simple sigma guess: width where y drops to half (rough)
        half = amp0 / 2.0
        # find indices around max where y < half
        idx = np.where(y >= half)[0]
        if idx.size > 1:
            sigma0 = (x[idx[-1]] - x[idx[0]]) / \
                (2.0 * math.sqrt(2 * math.log(2)))  # FWHM->sigma
            if sigma0 <= 0:
                sigma0 = (x.max() - x.min()) / 10.0
        else:   
            sigma0 = (x.max() - x.min()) / 10.0
        p0 = (center0, sigma0, amp0) # type: ignore

    try:
        popt, pcov = curve_fit(model, x, y, p0=p0)
        return popt, pcov
    except Exception as e:
        logger.warning(
            f"fit_peak_gaussian: curve_fit failed ({e}). Returning moment estimates.")
        # fallback to moment estimates
        area = np.trapz(y, x)
        if area <= 0:
            return np.array([0.0, 1.0, 0.0]), None
        center_est = np.trapz(x * y, x) / area
        sigma_est = math.sqrt(
            abs(np.trapz(((x - center_est) ** 2) * y, x) / area))
        amp_est = y.max()
        return np.array([center_est, sigma_est, amp_est]), None


# ---------------------------
# Utility: read a numeric table from file (simple Fortran-ish parser)
# ---------------------------
def read_numeric_table(path: str, skip_comments: bool = True) -> np.ndarray:
    """
    Lire un fichier texte avec colonnes de nombres ; retourne un ndarray NxM.
    Ignore les lignes vides et celles commençant par 'c' ou '#'.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    data = []
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if skip_comments and (line.lower().startswith("c") or line.startswith("#")):
                continue
            # split by whitespace
            parts = line.split()
            # attempt to parse floats, ignore token parse errors
            nums = []
            for t in parts:
                try:
                    # Fortran D->E
                    nums.append(float(t.replace("d", "e").replace("D", "E")))
                except Exception:
                    # ignore non-numeric tokens
                    pass
            if nums:
                data.append(nums)
    if not data:
        return np.empty((0, 0), dtype=float)
    # pad rows to equal length
    maxlen = max(len(r) for r in data)
    arr = np.full((len(data), maxlen), np.nan)
    for i, r in enumerate(data):
        arr[i, : len(r)] = r
    return arr


# ---------------------------
# If used as script, quick demo
# ---------------------------
if __name__ == "__main__":
    import sys

    # petit test / demo
    x = np.linspace(-1.0, 1.0, 201)
    g = gaussian(x, center=0.0, sigma=0.1, amplitude=1.0)
    l = lorentzian(x, center=0.0, gamma=0.05, amplitude=1.0)
    v = voigt(x, center=0.0, sigma=0.1, gamma=0.05, amplitude=1.0)

    print("gauss max:", g.max(), "lorentz max:",
          l.max(), "voigt max:", v.max())

    # demo convolution if scipy present
    s = np.zeros_like(x)
    s[len(x) // 2] = 1.0
    kern = gaussian(x, 0.0, 0.05, amplitude=1.0)
    conv = convolve_1d(s, kern, mode="same")
    print("conv sum:", conv.sum())
