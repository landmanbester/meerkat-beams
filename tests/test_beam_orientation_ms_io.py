"""Unit tests for scripts/beam_orientation/ms_io.py."""

import pytest
from beam_orientation import ms_io


@pytest.mark.unit
def test_original_pointing_table_values():
    """ORIGINAL_POINTING carries the five fields with the documented coords."""
    expected = {
        0: (5.146178203219011, -1.1119958085589738),
        1: (5.146178203219011, -1.1364304180868943),
        2: (5.146178203219011, -1.0875611990310532),
        3: (5.201372059151767, -1.1119958085589738),
        4: (5.090979983963126, -1.1119958085589738),
    }
    assert ms_io.ORIGINAL_POINTING == expected
