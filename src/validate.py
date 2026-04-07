"""
validate.py
-----------
Validates the tunnelling model against published experimental data.

This is your first scientific checkpoint. Before trusting any mutant
predictions, you need to reproduce the wild-type KIE for a known system.

Benchmark: Aromatic Amine Dehydrogenase (AADH) from Alcaligenes faecalis
  - Substrate: tryptamine
  - Experimental KIE (kH/kD): 55 ± 4 at 25°C (Scrutton group, 2006)
  - Source: Masgrau et al., Science 2006, 312, 237-241
  - This is one of the most deeply tunnelling enzymes known
  - Makes it an ideal stress test for the model

Also tests Alcohol Dehydrogenase (ADH) from Bacillus stearothermophilus:
  - Experimental KIE ≈ 6.8 at 25°C (near-classical — much less tunnelling)
  - Useful as a "low tunnelling" control
  - Source: Klinman et al., multiple papers 1980s-2000s

Run with:  python validate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tunnelling_model import bell_correction, rank_mutations


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_aadh_wildtype():
    """
    AADH wild-type benchmark.
    
    Input parameters from Johannissen et al., ACS Catalysis 2020:
    - Barrier height: 13.4 kcal/mol (from QM/MM)
    - Imaginary frequency: 1184 cm⁻¹ (from TS frequency calculation)
    - D-A distance: 2.87 Å (from crystal structure / MD)
    - Experimental KIE: 55 at 298K
    
    The Bell correction will underpredict (gives ~15-20) because AADH
    tunnels so deeply that higher-order terms matter. But it should give
    a KIE >> 7, confirming significant tunnelling is detected.
    """
    print_section("TEST 1: AADH wild-type (deeply tunnelling)")
    print("  Literature: Masgrau et al., Science 2006")
    print("  Expected KIE ≈ 55 (Bell will underpredict — this is expected)")
    print("  Key test: is predicted KIE >> 7? (classical limit = 6.9)")

    result = bell_correction(
        barrier_height_kcal=13.4,
        imaginary_freq_cm1=1184.0,
        da_distance_angstrom=2.87,
        temperature=298.15,
        experimental_KIE=55.0
    )
    print(result)

    # Assertions
    assert result.predicted_KIE > 7.0, \
        f"FAIL: KIE {result.predicted_KIE:.1f} should be >> 7 (classical limit)"
    assert result.tunnelling_fraction > 0.1, \
        f"FAIL: tunnelling fraction {result.tunnelling_fraction:.1%} too low"
    assert result.tunnelling_regime in ('deep', 'moderate'), \
        f"FAIL: regime should not be 'minimal' for AADH"

    print(f"  ✓ KIE > 7 (classical limit): PASS")
    print(f"  ✓ Tunnelling detected: PASS")
    print(f"  ✓ Regime: {result.tunnelling_regime}")
    print(f"  Note: Bell underpredict expected. Full PIMD needed for quantitative match.")

    return result


def test_adh_wildtype():
    """
    ADH wild-type benchmark.
    
    Bacillus stearothermophilus ADH — much less tunnelling than AADH.
    Experimental KIE ≈ 6.8 at 25°C (near-classical).
    Lower imaginary frequency → less tunnelling.
    
    Key test: model should predict KIE close to classical limit (~7),
    not a large tunnelling enhancement.
    """
    print_section("TEST 2: ADH wild-type (near-classical)")
    print("  Literature: Klinman group, multiple papers")
    print("  Expected KIE ≈ 6.8 (near-classical limit of 6.9)")

    result = bell_correction(
        barrier_height_kcal=15.2,
        imaginary_freq_cm1=650.0,   # lower frequency → less tunnelling
        da_distance_angstrom=3.15,  # longer D-A → less tunnelling
        temperature=298.15,
        experimental_KIE=6.8
    )
    print(result)

    assert result.predicted_KIE < 15.0, \
        f"FAIL: KIE {result.predicted_KIE:.1f} should be near-classical for ADH"

    print(f"  ✓ KIE near classical limit: PASS")

    return result


def test_temperature_dependence():
    """
    KIE should decrease with increasing temperature.
    
    This is a fundamental physical prediction: higher temperature means
    more classical over-barrier passage relative to tunnelling.
    At very high T, KIE → classical Swain-Schaad limit.
    """
    print_section("TEST 3: Temperature dependence")
    print("  Prediction: KIE decreases as T increases (tunnelling less dominant)")

    temps = [278.15, 298.15, 318.15, 338.15]  # 5, 25, 45, 65°C
    kies  = []

    for T in temps:
        r = bell_correction(
            barrier_height_kcal=13.4,
            imaginary_freq_cm1=1184.0,
            da_distance_angstrom=2.87,
            temperature=T
        )
        kies.append(r.predicted_KIE)
        print(f"  T = {T-273.15:.0f}°C → KIE = {r.predicted_KIE:.2f}")

    # KIE should decrease monotonically with temperature
    for i in range(len(kies) - 1):
        assert kies[i] > kies[i+1], \
            f"FAIL: KIE should decrease with temperature"

    print(f"  ✓ KIE decreases monotonically with T: PASS")


def test_mutation_ranking():
    """
    Test mutation ranking against known AADH mutant data.
    
    Known mutations from Scrutton group literature:
    - T172A: shortens D-A distance → enhanced tunnelling (KIE increases)
    - T172V: smaller compression than T172A
    - N198A: less effect (further from D-A axis)
    
    Key test: T172A should rank higher than N198A.
    """
    print_section("TEST 4: Mutation ranking (AADH residues)")
    print("  Testing known literature mutations from Scrutton group")
    print("  T172A should rank higher than N198A (closer to D-A axis)")

    wt = bell_correction(
        barrier_height_kcal=13.4,
        imaginary_freq_cm1=1184.0,
        da_distance_angstrom=2.87,
        experimental_KIE=55.0
    )

    # Known AADH mutation targets from literature
    candidates = [
        (172, 'THR', 'ALA', 'donor_side'),    # T172A — Scrutton group benchmark
        (172, 'THR', 'VAL', 'donor_side'),    # T172V
        (172, 'THR', 'GLY', 'donor_side'),    # T172G — very small
        (198, 'ASN', 'ALA', 'flanking'),      # N198A — less direct
        (198, 'ASN', 'GLY', 'flanking'),      # N198G
        (169, 'PHE', 'ALA', 'acceptor_side'), # F169A
        (169, 'PHE', 'LEU', 'acceptor_side'), # F169L
    ]

    ranked = rank_mutations(wt, candidates)

    print(f"\n  Wild-type KIE baseline: {wt.predicted_KIE:.1f}")
    print(f"\n  {'Mutation':<12} {'ΔD-A (Å)':<12} {'KIE':<8} {'Enhancement':<14} Priority")
    print(f"  {'─'*60}")
    for pred in ranked:
        print(f"  {pred.name:<12} {pred.da_change:>+8.3f} Å   "
              f"{pred.predicted_KIE:>6.1f}   "
              f"{pred.fold_enhancement:>6.2f}x          "
              f"{pred.priority}")

    # T172A should outrank N198A
    t172a = next(p for p in ranked if p.name == 'T172A')
    n198a = next(p for p in ranked if p.name == 'N198A')

    assert t172a.predicted_KIE > n198a.predicted_KIE, \
        "FAIL: T172A should rank higher than N198A"

    print(f"\n  ✓ T172A ranks higher than N198A: PASS")
    print(f"  ✓ Top mutation: {ranked[0].name} "
          f"(predicted KIE {ranked[0].predicted_KIE:.1f}, "
          f"{ranked[0].fold_enhancement:.2f}x enhancement)")


def test_physical_limits():
    """
    Sanity checks: physical constraints that must always hold.
    """
    print_section("TEST 5: Physical limits and sanity checks")

    # Qt must always be ≥ 1 (tunnelling can only increase rate, never decrease)
    result = bell_correction(10.0, 800.0, 3.0)
    assert result.Qt_H >= 1.0, "FAIL: Qt_H < 1 (unphysical)"
    assert result.Qt_D >= 1.0, "FAIL: Qt_D < 1 (unphysical)"
    print("  ✓ Qt ≥ 1 always: PASS")

    # KIE must be ≥ classical KIE (tunnelling inflates KIE)
    assert result.predicted_KIE >= result.classical_KIE * 0.95, \
        "FAIL: tunnelling KIE < classical KIE (unphysical)"
    print("  ✓ KIE ≥ classical KIE: PASS")

    # Higher imaginary frequency → more tunnelling
    r_low  = bell_correction(13.4, 500.0,  2.87)
    r_high = bell_correction(13.4, 1500.0, 2.87)
    assert r_high.predicted_KIE > r_low.predicted_KIE, \
        "FAIL: higher imaginary freq should give more tunnelling"
    print("  ✓ Higher ν‡ → more tunnelling: PASS")

    # Tunnelling fraction must be between 0 and 1
    assert 0.0 <= result.tunnelling_fraction <= 1.0, \
        f"FAIL: tunnelling fraction {result.tunnelling_fraction} out of range"
    print("  ✓ Tunnelling fraction ∈ [0,1]: PASS")


def run_all_tests():
    print("\n" + "█"*60)
    print("  TUNNELASE — VALIDATION SUITE")
    print("  Testing Bell correction against published enzyme data")
    print("█"*60)

    passed = 0
    failed = 0

    tests = [
        test_aadh_wildtype,
        test_adh_wildtype,
        test_temperature_dependence,
        test_mutation_ranking,
        test_physical_limits,
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n  ✗ {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ✗ Unexpected error in {test_fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  ✓ All tests passed. Physics is consistent.")
        print("  Next step: get QM/MM inputs from ORCA for real barrier heights.")
    else:
        print("  ✗ Some tests failed. Check the physics before proceeding.")
    print(f"{'='*60}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
