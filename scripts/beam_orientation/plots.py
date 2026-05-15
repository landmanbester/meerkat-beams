"""
Diagnostic plots for the beam-orientation validation experiment.

All functions take fully-resolved NumPy arrays (no xarray/zarr), apply
``cond > threshold`` masking, and write a single PNG via matplotlib's
Agg backend. They do not display anything interactively.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from beam_orientation.calibrator import evaluate as catalog_spectrum

STOKES_INDEX = {"I": 0, "Q": 1, "U": 2, "V": 3}
DEFAULT_COND_THRESHOLD = 1e6


def _mask(arr: np.ndarray, cond: np.ndarray, threshold: float) -> np.ndarray:
    """Return arr with cond>threshold bins set to NaN (broadcast over trailing axes)."""
    bad = cond > threshold
    bad = np.broadcast_to(bad[..., None], arr.shape) if arr.ndim > cond.ndim else bad
    return np.where(bad, np.nan, arr)


def waterfall(
    times: np.ndarray,
    freq: np.ndarray,
    B: np.ndarray,  # noqa: N803
    cond: np.ndarray,
    stokes: str,
    out_path: Path,
    cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> None:
    """Waterfall (time × freq) plot of one Stokes component of B."""
    idx = STOKES_INDEX[stokes]
    val = _mask(B[..., idx].real, cond, cond_threshold)
    fig, ax = plt.subplots(figsize=(8, 4))
    extent = [freq[0] * 1e-9, freq[-1] * 1e-9, times[-1], times[0]]
    im = ax.imshow(val, aspect="auto", extent=extent, interpolation="nearest")
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"Stokes {stokes}")
    fig.colorbar(im, ax=ax, label="Jy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def mean_spectrum(
    freq: np.ndarray,
    B: np.ndarray,  # noqa: N803
    cond: np.ndarray,
    out_path: Path,
    cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> None:
    """Time-averaged recovered I(ν) with catalog polynomial overlaid; residuals subplot."""
    val = _mask(B[..., 0].real, cond, cond_threshold)
    mean_I = np.nanmean(val, axis=0)
    cat = catalog_spectrum(freq)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax_top.plot(freq * 1e-9, mean_I, label="recovered ⟨I⟩_t")
    ax_top.plot(freq * 1e-9, cat, "--", label="PKS 1934-638 catalog")
    ax_top.set_ylabel("Jy")
    ax_top.legend()
    ax_bot.plot(freq * 1e-9, mean_I - cat)
    ax_bot.axhline(0, color="k", linewidth=0.5)
    ax_bot.set_xlabel("frequency (GHz)")
    ax_bot.set_ylabel("residual (Jy)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def time_variation(
    freq: np.ndarray,
    B: np.ndarray,  # noqa: N803
    cond: np.ndarray,
    out_path: Path,
    cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> None:
    """Per-channel std_t(B) / median_t(|B|) for each Stokes."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, idx in STOKES_INDEX.items():
        val = _mask(B[..., idx], cond, cond_threshold)
        std = np.nanstd(val.real, axis=0)
        med = np.nanmedian(np.abs(val), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            metric = np.where(med > 0, std / med, np.nan)
        ax.plot(freq * 1e-9, metric, label=f"Stokes {label}")
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("std_t(B) / median_t(|B|)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def control_overlay(
    freq: np.ndarray,
    runs: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> None:
    """Overlay the Stokes-I time-variation metric across multiple runs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, (B, cond) in runs.items():
        val = _mask(B[..., 0], cond, cond_threshold)
        std = np.nanstd(val.real, axis=0)
        med = np.nanmedian(np.abs(val), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            metric = np.where(med > 0, std / med, np.nan)
        ax.plot(freq * 1e-9, metric, label=name)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("std_t(I) / median_t(|I|)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Residual time variation: unperturbed vs controls")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
