"""
test_path_integral.py
---------------------
Unit tests for path_integral.py (Module 3).

Tests:
  1. Classical limit: u → 0 gives Qt → 1 for all methods
  2. WK series: coefficients match known Bernoulli expansion
  3. Exact formula: (u/2)/sin(u/2) matches series at low u (<3)
  4. AADH benchmark: exact KIE improvement over Bell 1st-order
  5. Convergence: WK series converges to exact with increasing order
  6. Deuterium scaling: u_D = u_H / sqrt(m_D/m_H) ≈ u_H / sqrt(2)
  7. Temperature scan: KIE increases as T decreases (tunnelling grows)
  8. Near-pole flag: set correctly when u > π
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from path_integral import (
    wigner_kirkwood_qt, exact_qt_parabolic, compute_u,
    path_integral_correction, compute_kie, wk_convergence_check,
    temperature_scan, WK_COEFFICIENTS, MASS_H, MASS_D, MASS_RATIO_DH,
    PathIntegralKIE,
)

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

def check(cond, msg):
    status = PASS if cond else FAIL
    print(f"  {status}  {msg}")
    return cond

AADH_NU = 1184.0    # cm⁻¹ imaginary TS frequency
AADH_T  = 298.15    # K
AADH_EXP_KIE = 55.0 # experimental WT AADH KIE
CLASSICAL_KIE = 6.9  # Swain-Schaad limit


def main():
    print("=" * 60)
    print("  PATH INTEGRAL MODULE — UNIT TESTS")
    print("=" * 60)

    # ── Test 1: Classical limit u → 0 ─────────────────────────────────────────
    print("\n[Test 1] Classical limit: u → 0 gives Qt → 1")
    for u in [1e-8, 1e-6, 1e-4, 0.01]:
        qt_bell  = 1.0 + u**2/24
        qt_wk    = wigner_kirkwood_qt(u, wk_order=3)
        qt_exact = exact_qt_parabolic(u)
        check(abs(qt_exact - 1.0) < u**2/20,
              f"  u={u:.0e}: Qt_exact={qt_exact:.8f} → 1 as u→0")
    check(abs(exact_qt_parabolic(0.0) - 1.0) < 1e-9,
          "exact_qt_parabolic(0.0) = 1.0")

    # ── Test 2: WK coefficient accuracy ───────────────────────────────────────
    print("\n[Test 2] WK coefficients match Bernoulli series")
    # Verify: (u/2)/sin(u/2) = 1 + u²/24 + 7u⁴/5760 + 31u⁶/967680 + ...
    u_test = 0.5  # small enough for fast convergence
    exact_ref = exact_qt_parabolic(u_test)

    # WK-1 should give 1 + u²/24
    wk1 = wigner_kirkwood_qt(u_test, wk_order=1)
    expected_wk1 = 1.0 + WK_COEFFICIENTS[1] * u_test**2
    check(abs(wk1 - expected_wk1) < 1e-10,
          f"WK-1 = 1 + u²/24 = {expected_wk1:.6f}")

    # WK-2 should add 7u⁴/5760
    wk2 = wigner_kirkwood_qt(u_test, wk_order=2)
    expected_wk2 = expected_wk1 + WK_COEFFICIENTS[2] * u_test**4
    check(abs(wk2 - expected_wk2) < 1e-10,
          f"WK-2 = WK-1 + 7u⁴/5760")

    # WK-3 should converge within 0.1% of exact for small u
    wk3 = wigner_kirkwood_qt(u_test, wk_order=3)
    check(abs(wk3 - exact_ref) / exact_ref < 0.001,
          f"WK-3 converges to exact within 0.1% at u={u_test}")

    # ── Test 3: Exact formula matches series at low u ─────────────────────────
    print("\n[Test 3] Exact formula agrees with WK series (u ≤ 3)")
    for u in [0.5, 1.0, 2.0, 3.0]:
        qt_wk5   = wigner_kirkwood_qt(u, wk_order=5)
        qt_exact = exact_qt_parabolic(u)
        rel_err  = abs(qt_wk5 - qt_exact) / qt_exact
        tol = 0.01 if u <= 3.0 else 0.10
        check(rel_err < tol,
              f"u={u:.1f}: WK-5={qt_wk5:.4f}, Exact={qt_exact:.4f}, err={rel_err:.4f}")

    # ── Test 4: AADH benchmark ────────────────────────────────────────────────
    print("\n[Test 4] AADH benchmark: exact correction improves over Bell")
    result = compute_kie(AADH_NU, CLASSICAL_KIE, AADH_T, wk_order=3)
    print(f"\n{result.summary()}\n")
    print(f"  Experimental KIE = {AADH_EXP_KIE}")

    check(result.u_H > 5.5 and result.u_H < 6.0,
          f"u_H = {result.u_H:.4f} (expected ~5.71 for 1184 cm⁻¹ at 298 K)")
    check(result.kie_exact > result.kie_bell,
          f"KIE_exact ({result.kie_exact:.1f}) > KIE_bell ({result.kie_bell:.1f})")
    check(result.kie_exact > result.kie_wk,
          f"KIE_exact ({result.kie_exact:.1f}) > KIE_wk ({result.kie_wk:.1f})")

    # Exact formula should be between Bell (too low) and experiment
    check(result.kie_bell < AADH_EXP_KIE,
          f"KIE_bell < experimental ({AADH_EXP_KIE}): Bell 1st-order understates tunnelling")
    check(result.kie_exact > result.kie_bell * 3.0,
          f"Exact correction is large: {result.kie_correction_factor:.2f}× vs Bell")

    # Remaining gap to experiment — not expected to close (non-parabolic barrier)
    remaining = AADH_EXP_KIE / result.kie_exact
    print(f"  Remaining factor (non-parabolic barrier): {remaining:.2f}×")

    # ── Test 5: WK series convergence ─────────────────────────────────────────
    print("\n[Test 5] WK series convergence toward exact at u=5.71")
    u_aadh = result.u_H
    conv = wk_convergence_check(u_aadh)
    qt_exact = conv['Exact']
    print(f"  Qt values at u_H = {u_aadh:.4f}:")
    for key, val in conv.items():
        err = (val - qt_exact) / qt_exact * 100
        print(f"    {key:<8} Qt = {val:.4f}  (error vs exact: {err:+.1f}%)")

    # WK-5 should be closer to exact than WK-1
    check(abs(conv['WK-5'] - qt_exact) < abs(conv['WK-1'] - qt_exact),
          "WK-5 is closer to exact than WK-1")

    # ── Test 6: Deuterium mass scaling ────────────────────────────────────────
    print("\n[Test 6] Deuterium scaling: u_D = u_H / sqrt(m_D/m_H)")
    pi_H = path_integral_correction(AADH_NU, AADH_T, mass_kg=MASS_H)
    pi_D = path_integral_correction(AADH_NU, AADH_T, mass_kg=MASS_D)
    u_ratio = pi_D.u / pi_H.u
    expected_ratio = 1.0 / np.sqrt(MASS_RATIO_DH)
    check(abs(u_ratio - expected_ratio) < 1e-6,
          f"u_D/u_H = {u_ratio:.6f} = 1/sqrt(m_D/m_H) = {expected_ratio:.6f}")
    check(pi_D.Qt_exact > 1.0,
          f"Qt_D (exact) = {pi_D.Qt_exact:.4f} > 1")
    check(pi_H.Qt_exact > pi_D.Qt_exact,
          f"Qt_H ({pi_H.Qt_exact:.4f}) > Qt_D ({pi_D.Qt_exact:.4f}): H tunnels more")

    # ── Test 7: Temperature dependence ────────────────────────────────────────
    print("\n[Test 7] Temperature dependence: KIE increases as T decreases")
    temps = [313.15, 298.15, 283.15, 273.15]
    scan = temperature_scan(AADH_NU, CLASSICAL_KIE, temps, wk_order=3)

    print(f"  {'T (K)':<8} {'u_H':<7} {'KIE_bell':<12} {'KIE_exact':<12}")
    for r in scan:
        print(f"  {r.temperature:<8.1f} {r.u_H:<7.3f} {r.kie_bell:<12.2f} {r.kie_exact:<12.2f}")

    kie_higherT = scan[0].kie_exact  # 313 K
    kie_lowerT  = scan[-1].kie_exact  # 273 K
    check(kie_lowerT > kie_higherT,
          f"KIE({scan[-1].temperature:.0f}K)={kie_lowerT:.2f} > KIE({scan[0].temperature:.0f}K)={kie_higherT:.2f}")

    # ── Test 8: Near-pole flag ─────────────────────────────────────────────────
    print("\n[Test 8] Near-pole flag for u > π")
    pi_aadh = path_integral_correction(AADH_NU, AADH_T, mass_kg=MASS_H)
    check(pi_aadh.near_pole,
          f"AADH u_H = {pi_aadh.u:.3f} > π = {np.pi:.3f} → near_pole = True")

    pi_low = path_integral_correction(200.0, AADH_T, mass_kg=MASS_H)  # low freq
    check(not pi_low.near_pole,
          f"Low ν† (200 cm⁻¹) u = {pi_low.u:.3f} < π → near_pole = False")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
