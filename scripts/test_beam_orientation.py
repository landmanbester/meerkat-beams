#!/usr/bin/env python
"""
Beam-orientation validation experiment.

End-to-end pipeline:
  1. Download the calibrator MS (cached) if not provided via --ms.
  2. Read visibilities, weights, UVW, freq, phase centre.
  3. Phase-rotate to PKS 1934-638 using a SIN-projection (Δl, Δm) offset
     computed from the (ra, dec) in tests/conftest.py.
  4. Noise-weighted average over baselines → coherency visibility V̄(t, ν).
  5. For each perturbation in {"none", "flip_x", "flip_y", "swap_xy"}:
       - assemble the complex coherency Mueller M_C(t, ν) from the cached L-band BDS
       - solve the coherency dynamic spectrum B_C(t, ν) = M_C⁻¹ V̄
       - convert to Stokes B = (coherency→Stokes) · B_C
       - write dynamic_spectrum.zarr + the PNG plots
  6. Write a control_overlay.png across the four runs.

Working in the coherency frame (rather than converting the visibilities to
Stokes up front) keeps the data in its native basis and uses the complex
Mueller directly; the recovered Stokes spectrum is identical either way.

The spec for this script is in
docs/superpowers/specs/2026-05-15-beam-orientation-test-design.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr
from astropy.coordinates import SkyCoord
from astropy.time import Time

# Ensure scripts/ and project root are on sys.path when the script is run directly.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from beam_orientation import mueller, phase_rotate, plots  # noqa: E402
from beam_orientation.download import ensure_ms  # noqa: E402
from beam_orientation.ms_io import read_ms  # noqa: E402

from meerkat_beams.utils import BeamWizard, log  # noqa: E402
from tests.conftest import dec as DEC_STR  # noqa: E402, N812
from tests.conftest import ra as RA_STR  # noqa: E402, N812

PERTURBATIONS: dict[str, tuple[tuple[int, int], bool]] = {
    "none": ((1, 1), False),
    "flip_x": ((-1, 1), False),
    "flip_y": ((1, -1), False),
    "swap_xy": ((1, 1), True),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scratch/orientation_test"),
        help="output directory; one subdir per perturbation will be created here.",
    )
    p.add_argument(
        "--ms",
        type=Path,
        default=None,
        help="override path to the calibrator MS; defaults to the cached download.",
    )
    p.add_argument(
        "--field-id",
        type=int,
        default=0,
        help="FIELD_ID to select; also chooses the original pointing direction.",
    )
    p.add_argument(
        "--perturbations",
        nargs="+",
        choices=list(PERTURBATIONS),
        default=list(PERTURBATIONS),
        help="which perturbation runs to execute (default: all four).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ms_path = args.ms or ensure_ms()
    log.info(f"reading MS from {ms_path}")
    bundle = read_ms(ms_path, field_id=args.field_id)
    log.info(
        f"MS shape Nb={bundle.vis.shape[0]} Nt={bundle.vis.shape[1]} "
        f"Nf={bundle.vis.shape[2]} corr={bundle.vis.shape[3]}"
    )

    # Source position from conftest constants (HMS/DMS strings).
    srcpos = SkyCoord(RA_STR, DEC_STR.replace(".", ":", 2), unit=("hourangle", "deg"))
    ra_src_rad = float(srcpos.ra.rad)
    dec_src_rad = float(srcpos.dec.rad)

    # SIN-projection direction-cosine offset from the MS phase centre. For the
    # pre-rephased MS this centre is already the source, so (dl, dm) ~= 0 and
    # phase_rotate is a no-op; for a non-rephased MS it rephases correctly.
    ra_pc, dec_pc = bundle.phase_centre
    dl = np.cos(dec_src_rad) * np.sin(ra_src_rad - ra_pc)
    dm = np.sin(dec_src_rad) * np.cos(dec_pc) - np.cos(dec_src_rad) * np.sin(dec_pc) * np.cos(ra_src_rad - ra_pc)
    log.info(f"phase-rotating to (dl, dm) = ({dl:.6e}, {dm:.6e}) rad")

    vis_rot = phase_rotate.phase_rotate(bundle.vis, bundle.uvw, bundle.freq, dl=dl, dm=dm)

    # Mask flagged samples by zeroing their weight.
    w = bundle.weight_spectrum.astype(float).copy()
    w[bundle.flag] = 0.0

    # Noise-weighted average over baselines: V̄_coh(t, ν, corr), the observed
    # coherency visibility (XX, XY, YX, YY) at the source.
    num = np.einsum("btfc,btfc->tfc", w, vis_rot)
    den = np.einsum("btfc->tfc", w)
    with np.errstate(invalid="ignore", divide="ignore"):
        V_coh = np.where(den > 0, num / den, 0.0 + 0.0j)

    # Astropy Time vector for the Mueller assembly. The observer location comes
    # from BeamWizard.default_location (EarthLocation.of_site("MeerKAT")), so we
    # don't pass loc= below and avoid duplicating the site coordinates here.
    times = Time(bundle.time / 86400.0, format="mjd", scale="utc")

    # Coherency (XX,XY,YX,YY) → Stokes (I,Q,U,V), matching the BDS convention.
    coh_to_stokes = mueller.coherency_to_stokes_matrix()

    bw = BeamWizard(band="L")
    # Beam pointing centre = original dish pointing for this field (radians).
    bw.set_field_centre(SkyCoord(*bundle.pointing_centre, unit="rad", frame="icrs"))
    runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name in args.perturbations:
        signs, swap = PERTURBATIONS[name]
        run_dir = args.out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"=== perturbation '{name}': signs={signs}, swap={swap} ===")

        # Solve in the coherency frame, then convert the recovered spectrum to Stokes.
        M_C = mueller.assemble_mueller(bw, srcpos, times, bundle.freq, signs=signs, swap=swap)
        B_C, cond = mueller.solve_per_bin(M_C, V_coh)
        B = np.einsum("ij,tfj->tfi", coh_to_stokes, B_C)

        _write_zarr(run_dir / "dynamic_spectrum.zarr", B, cond, bundle.time, bundle.freq, name)
        plots.waterfall(bundle.time, bundle.freq, B, cond, "I", run_dir / "dyn_spec_I.png")
        plots.waterfall(bundle.time, bundle.freq, B, cond, "Q", run_dir / "dyn_spec_Q.png")
        plots.waterfall(bundle.time, bundle.freq, B, cond, "U", run_dir / "dyn_spec_U.png")
        plots.waterfall(bundle.time, bundle.freq, B, cond, "V", run_dir / "dyn_spec_V.png")
        plots.mean_spectrum(bundle.freq, B, cond, run_dir / "mean_I_spectrum.png")
        plots.time_variation(bundle.freq, B, cond, run_dir / "time_variation.png")
        runs[name] = (B, cond)

    if len(runs) > 1:
        plots.control_overlay(bundle.freq, runs, args.out_dir / "control_overlay.png")
        log.info(f"control overlay → {args.out_dir / 'control_overlay.png'}")


def _write_zarr(
    path: Path,
    B: np.ndarray,  # noqa: N803
    cond: np.ndarray,
    times_sec: np.ndarray,
    freq: np.ndarray,
    perturbation: str,
) -> None:
    root = zarr.open(str(path), mode="w")
    root.create_dataset("B", data=B.astype(np.complex64), chunks=False)
    root.create_dataset("cond_M", data=cond.astype(np.float32), chunks=False)
    root.create_dataset("time", data=times_sec.astype(np.float64), chunks=False)
    root.create_dataset("frequency", data=freq.astype(np.float64), chunks=False)
    root.attrs["source"] = "PKS 1934-638"
    root.attrs["polarization"] = ["I", "Q", "U", "V"]
    root.attrs["band"] = "L"
    root.attrs["perturbation"] = perturbation


if __name__ == "__main__":
    main()
