"""
Diagnostic plots for the beam-orientation validation experiment.

Each function takes a single fully-resolved ``(Nt, Nf)`` complex NumPy array,
plots its real part via matplotlib's Agg backend, and writes one PNG. Profiles
average with ``nanmean`` and treat fully-flagged bins (exact complex ``0``) as
missing, so they are excluded from the mean and not drawn. Nothing is displayed
interactively and nothing is normalised.
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _profile(data: np.ndarray, axis: int) -> np.ndarray:
    """``Re(data)`` averaged along ``axis``, ignoring NaN and fully-flagged bins.

    Fully-flagged samples arrive as exact complex ``0`` (the fill used when a
    bin has no unflagged data); treat them as missing so they neither bias the
    mean nor get plotted. A slice with no surviving bins reduces to ``NaN``.
    """
    val = np.real(data).astype(float)
    val[data == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slice -> NaN
        return np.nanmean(val, axis=axis)


def dyn_spectrum(
    times: np.ndarray,
    freq: np.ndarray,
    data: np.ndarray,
    out_path: Path,
    *,
    title: str,
    cbar_label: str,
) -> None:
    """Dynamic spectrum (time × freq) of ``Re(data)`` for one ``(Nt, Nf)`` array."""
    val = np.real(data)
    t0 = times[0]
    extent = [freq[0] * 1e-9, freq[-1] * 1e-9, times[-1] - t0, times[0] - t0]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(val, aspect="auto", extent=extent, interpolation="nearest")
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("time (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def time_profile(
    times: np.ndarray,
    data: np.ndarray,
    out_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    """``Re(data)`` averaged over frequency, plotted as a function of time."""
    prof = _profile(data, axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times - times[0], prof, marker=".", linestyle="none")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def freq_profile(
    freq: np.ndarray,
    data: np.ndarray,
    out_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    """``Re(data)`` averaged over time, plotted as a function of frequency."""
    prof = _profile(data, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freq * 1e-9, prof, marker=".", linestyle="none")
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
