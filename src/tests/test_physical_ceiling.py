import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from tunnel_score import compute_delta_r_max, ALPHA_H

def test_aadh_ceiling_value():
    """AADH ceiling should be ≈ 0.036 Å → 2.5× KIE fold per mutation."""
    dr = compute_delta_r_max(90.0, 6.857, 300.0)
    assert 0.030 < dr < 0.045, f"AADH delta_r_max={dr:.4f} Å, expected ~0.036"
    fold = np.exp(ALPHA_H * dr)
    assert 2.0 < fold < 3.5, f"AADH KIE fold={fold:.2f}, expected ~2.5×"

def test_dhfr_ceiling_self_consistent():
    """DHFR ceiling exp(ALPHA_H * delta_r_max) should be < 6.0 (I14G exp fold ~2.6×)."""
    dr = compute_delta_r_max(50.0, 6.000, 300.0)
    assert 0.050 < dr < 0.080, f"DHFR delta_r_max={dr:.4f} Å, expected ~0.062"
    fold = np.exp(ALPHA_H * dr)
    assert fold < 6.0, f"DHFR ceiling fold={fold:.2f}, should be < 6.0"

def test_scales_with_frequency():
    """Lower frequency → larger thermal amplitude (softer mode)."""
    dr_90 = compute_delta_r_max(90.0, 6.857, 300.0)
    dr_40 = compute_delta_r_max(40.0, 6.857, 300.0)
    assert dr_40 > dr_90, "SLO (40 cm⁻¹) should have larger ceiling than AADH (90 cm⁻¹)"

def test_scales_with_mass():
    """Larger reduced mass → smaller thermal amplitude."""
    dr_light = compute_delta_r_max(90.0, 6.000, 300.0)
    dr_heavy = compute_delta_r_max(90.0, 6.857, 300.0)
    assert dr_light > dr_heavy, "C-C (6.0 u) should have larger ceiling than C-O (6.857 u)"

def test_temperature_dependence():
    """Higher temperature → larger thermal amplitude."""
    dr_300 = compute_delta_r_max(90.0, 6.857, 300.0)
    dr_350 = compute_delta_r_max(90.0, 6.857, 350.0)
    assert dr_350 > dr_300
