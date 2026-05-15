"""Round-trip test: cli/*.py -> cabs/*.yml -> cli/*.py should be the identity.

If this fails, do NOT hand-fix cli/*.py. Either:
  * edit the cab YAML and run `bash scripts/genfuncs.sh`, or
  * file a hip-cargo bug if the regen is non-idempotent.

Modelled on pfb-imaging/tests/test_roundtrip.py.
"""

import tempfile
from pathlib import Path

import pytest
from hip_cargo.core.generate_cabs import generate_cabs
from hip_cargo.core.generate_function import generate_function

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_DIR = REPO_ROOT / "src" / "meerkat_beams" / "cli"
PYPROJECT = REPO_ROOT / "pyproject.toml"


@pytest.mark.unit
def test_cli_cab_cli_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cab_dir = tmpdir / "cabs"
        cab_dir.mkdir()

        generate_cabs(
            module=[CLI_DIR / "*.py"],
            output_dir=cab_dir,
            image=None,
        )

        cabs = sorted(cab_dir.glob("*.yml"))
        assert cabs, "generate_cabs produced no cab files"

        for cab_file in cabs:
            generated_file = tmpdir / f"{cab_file.stem}_roundtrip.py"
            generate_function(cab_file, generated_file, config_file=PYPROJECT)

            assert generated_file.exists(), f"generate_function produced no output for {cab_file.name}"
            generated_code = generated_file.read_text()
            compile(generated_code, str(generated_file), "exec")

            module_path = CLI_DIR / f"{cab_file.stem}.py"
            original_code = module_path.read_text()

            original_lines = original_code.splitlines()
            generated_lines = generated_code.splitlines()

            assert len(original_lines) == len(generated_lines), (
                f"Line count mismatch for {cab_file.stem}: "
                f"original has {len(original_lines)} lines, generated has {len(generated_lines)}"
            )

            for i, (orig, gen) in enumerate(zip(original_lines, generated_lines), 1):
                assert orig == gen, f"{cab_file.stem}.py line {i} differs:\n  Original:  {orig}\n  Generated: {gen}"
