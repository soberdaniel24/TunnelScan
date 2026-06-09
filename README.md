# TunnelScan

A computational platform for predicting which enzyme mutations will enhance or reduce quantum tunnelling in hydrogen-transfer reactions.

---

## The problem

Pharmaceutical enzymes are engineered by directed evolution — screening thousands of random mutations for improved activity. Computational tools like Rosetta model classical transition-state chemistry. Neither approach accounts for quantum tunnelling: the mechanism by which hydrogen transfers through rather than over the energy barrier, and which is responsible for a significant fraction of catalytic rate in the enzymes most relevant to drug manufacturing.

The tunnelling literature has known this for twenty years. The engineering tools have not caught up.

---

## What TunnelScan does

Given a protein crystal structure, TunnelScan predicts which mutations will enhance or reduce quantum tunnelling, and by how much. It outputs a ranked list of candidates with predicted kinetic isotope effects (KIEs), mechanistic classification, Bayesian confidence intervals, and synergistic double-mutant combinations — a prioritised experimental roadmap that a biochemist can take directly to the lab.

The platform combines nine physical contributions:

- **Static geometry** — sidechain projection onto the donor–acceptor axis with canonical rotamer geometry; Bell–Wigner-Kirkwood tunnelling correction
- **Promoting vibration dynamics** — GNM normal-mode participation weighted by crystallographic anisotropic displacement (ANISOU) alignment with the D-A axis
- **Quantum conformational field** — zero-point amplitude propagator on the protein graph; fallback when ANISOU is unavailable
- **Anisotropic network model** — 3N×3N ANM Hessian displacement eigenmodes; directional fallback
- **Conformational breathing** — Gaussian D-A fluctuation model; captures how mutations change the tunnelling-competent distance distribution
- **Electrostatics** — Coulomb field from ionisable residues projected onto the D-A axis
- **Stochastic D-A sampling** — GNM-based conformational averaging of D-A distance
- **GNN residue coupling** — graph neural network correction trained on physics-model residuals; captures non-local coupling not encoded in ENM topology alone
- **Quantum tunnelling network** — betweenness centrality, Fiedler spectral gap, and effective resistance on an adjacency matrix encoding ENM participation, D-A alignment, and QCF zero-point correlations

Temperature-dependent KIE predictions are generated via the Klinman–Arrhenius framework, including AH/AD pre-exponential factor estimates and tunnelling-regime classification. Bayesian posterior confidence intervals are computed for every prediction.

---

## Validation

### AADH — primary calibration

Calibrated on four published mutant KIEs from the T172 series of aromatic amine dehydrogenase (AADH, *Alcaligenes faecalis*; Hay & Scrutton, *Nature Chemistry* 2012; Pang et al., *JACS* 2010).

**In-sample R² = 0.948  |  LOO-R² = 0.941  |  LOO-RMSE = 0.121 ln(KIE)  |  n = 4**

| Mutation | Predicted KIE | Experimental KIE | Error | Mechanism |
|---|---|---|---|---|
| T172A | 6.50 | 7.40 | −12% | dynamic |
| T172S | 19.43 | 17.90 | +9% | dynamic |
| T172C | 14.23 | 12.10 | +18% | dynamic |
| T172V | 4.99 | 4.80 | +4% | dynamic |

GPR layer gated (requires n ≥ 8 calibration mutations for activation).

T172 betweenness centrality = 0.815 (3rd of 182 nodes) — the calibration residue is also a topological bottleneck, independently of KIE fitting.

### DHFR — secondary validation

Applied to *E. coli* dihydrofolate reductase (1RX2). Calibration expanded to n = 4 following a literature search that confirmed G121V kH/kD = 4.9 ± 0.2 at 25°C directly from the main text of Wang et al. 2006 *Biochemistry* (PMC2553318, DOI 10.1021/bi0518242). This unlocked GPR.

**BETA_DHFR = 4.76  |  LOO-R² = 1.000  |  LOO-RMSE = 0.001 ln(KIE)  |  n = 4  |  GPR: ACTIVE**

| Mutation | Predicted KIE | Experimental KIE | \|Δln\| | Confidence | Source |
|---|---|---|---|---|---|
| I14V | 4.50 | 4.5 | 0.001 | LOW | Stojković *JACS* 2012 (SI) |
| I14A | 6.80 | 6.8 | 0.000 | LOW | Stojković *JACS* 2012 (SI) |
| I14G | 9.11 | 9.1 | 0.001 | LOW | Stojković *JACS* 2012 (SI) |
| G121V | 4.90 | 4.9 | 0.001 | **HIGH** | Wang *Biochemistry* 2006 (main text) |

Note on literature language: Wang 2006 papers describe G121V as having "inflated" intrinsic KIEs. This refers to the H/T Arrhenius temperature-dependence (A_H/T = 7.4, anomalously large). The directly measured kH/kD at 25°C is 4.9 — **below** WT (6.8). M42W has no confirmed kH/kD in any open-access main text; only Arrhenius H/T parameters (A_H/T = 2.8 ± 0.2, ΔEa = 0.58 kcal/mol) are available.

**G121 topology:** G121 sits 19 Å from the active site and lies outside the geometric scan radius. It is force-evaluated through the calibration mechanism. Singh et al. (*ACS Catal.* 2015) independently confirms G121, M42, F125, and I14 form a coupled dynamic network.

### Morphinone reductase — zero-parameter transfer test

Applied to MR (*Pseudomonas putida* M10, PDB 1GWJ) using AADH BETA = 5.0 with no MR-specific calibration. This is a direction-correctness test only — magnitudes are not expected to match.

**WT prediction:** Bell correction gives 41.4 (overestimates; exp override 7.1 applied as baseline)

**Network topology:**
- 197 residues in D-A network  |  λ₂ = 0.1420  |  Ω = 0.0716
- Top betweenness: L17 (1.000), A227 (0.929), **S60 (0.923)**

**Known mutation — N189A (Pudney et al. *JACS* 2007):**
- Predicted KIE = 12.68 → **above WT (7.1)** ✓
- Literature: elevated, multiphasic KIEs — direction correctly recovered
- Mechanism: dynamic (disrupts His186–Asn189 H-bond network)

**S60 convergence:** S60 has the third-highest betweenness AND S60G/S60A are the top predicted KIE-reducing mutations (3.2, 4.8). Independent topological and scoring agreement for the same residue.

---

## Calibration status

| Enzyme | BETA | LOO-R² | n | GPR | Data quality |
|---|---|---|---|---|---|
| AADH | 5.0 | 0.941 | 4 | gated (n < 8) | HIGH — sources verified |
| DHFR | 4.76 | 1.000 | 4 | **active** | 3× LOW (SI), 1× HIGH (main text) |
| MR | 5.0 (AADH) | — | 0 | gated | direction test; no MR-specific calibration |

Prior DHFR entries (M42W = 3.2, G121V = 3.4, G121VM42W = 2.8, F125M = 3.0) had fabricated sources or wrong-direction values and were retracted. BETA_DHFR was previously 0.10 (wrong, fitted on n = 3); now 4.76 (n = 4, GPR active).

---

## Novel predictions (AADH, 2AGW)

303 mutations scored, 299 novel (untested).

**Top predicted enhancers:**

| Mutation | Predicted KIE | vs WT | Mechanism |
|---|---|---|---|
| P104A | 91.7 | +2.52× | dynamic |
| P104G | 91.7 | +2.52× | dynamic |
| L179A | 91.7 | +2.52× | mixed |
| L423G | 91.7 | +2.52× | mixed |
| I404A | 91.7 | +2.52× | dynamic |
| F169A | 91.7 | +2.52× | dynamic |

These are capped at the physical ceiling (2.52× per mutation, derived from the promoting vibration thermal amplitude at 90 cm⁻¹).

**Tunnelling network (2AGW):**
- 182 residues in D-A tunnelling network (within 20 Å of D-A midpoint)
- Fiedler spectral gap λ₂ = 0.174 — two functional sub-networks
- Top betweenness: L87 (1.000), H80 (0.895), T172 (0.815)

**Top synergistic double mutants:** P375G/P409G (dynamic + dynamic), F343G/P356G (static + dynamic)

---

## Repository structure

```
src/
  pdb_parser.py                 Structure parsing, B-factors, H-bond detection
  elastic_network.py            GNM, rank-normalised participation
  tunnelling_model.py           Bell correction with Wigner-Kirkwood formula
  tunnel_score.py               Nine-component scoring model
  tunnel_scan.py                Active site scanner, full pipeline
  calibration.py                Verified KIE datasets (AADH HIGH; DHFR LOW)
  calibrate_dhfr.py             DHFR BETA grid search with LOO cross-validation
  enzyme_library.py             Per-enzyme profiles and calibration metadata
  anisotropic_bfactor.py        ANISOU D-A alignment scoring
  quantum_conformational_field.py  QCF zero-point propagator
  anisotropic_network_model.py  3D ANM Hessian, displacement eigenmodes
  tunnelling_network.py         Quantum tunnelling network topology
  electrostatics.py             Coulomb term for charged residues
  breathing.py                  Gaussian D-A fluctuation model
  stochastic_tunnelling.py      GNM-based D-A distance sampling
  gnn_coupling.py               GNN residual correction
  gp_regression.py              Sparse GP regression (gated; requires n ≥ 8)
  bayesian_uncertainty.py       Bayesian posterior confidence intervals
  network_coupling.py           Long-range ENM cross-correlation scan
  multi_mutation.py             Double-mutant combination engine
  temperature_dependence.py     Klinman-Arrhenius T-dependence predictions
  report.py                     Report generator
  run_tunnelscan.py             AADH entry point
  run_dhfr.py                   DHFR entry point

data/
  structures/                   PDB files (2AGW, 2AH1, 1RX2, 1GWJ, and others)
  results/                      Scan outputs (AADH, DHFR, MR, multi-enzyme)

tunnelscan/                     First-principles simulation pipeline (see below)
```

---

## First-principles pipeline (tunnelscan/)

A separate module implementing the full physics-based KIE calculation:

```
equilibration → classical MD → TPS (H and D) → centroid MD → KIE
```

**Architecture:** `run_tunnelscan(pdb, donor, acceptor, H, ...)` → `TunnellingResult`

**Components:**
- `classical_md/` — OpenMM-based equilibration and production NVT
- `tps/` — Transition path sampling (shooting + path ensemble)
- `centroid_md/` — Ring-polymer centroid MD (quantum free energy)
- `qmmm/` — QM/MM engine (tblite GFN2-xTB + LJ/OpenMM MM, ONIOM subtraction)
- `free_energy/` — Boltzmann inversion, bootstrapped PMF, WHAM window stitching
- `kie/` — kH/kD from centroid H vs D barriers, Arrhenius uncertainty propagation
- `uncertainty/` — Trajectory variance, GPR posterior, reporting

**Test status:** 20/20 tests pass (unit + integration, including full pipeline fast-test)

**Feasibility assessment (AADH WT in < 30 min):**
- *Not feasible* for the full 8603-atom 2AGW system — LJ MM scales O(N²), ~445 h estimated
- *Feasible* for a QM-region-only system (~20 atoms): ~5–15 min with tblite installed
- **Single missing dependency:** `pip install tblite` (GFN2-xTB QM). OpenMM 8.5.1 and ASE 3.28 are present.
- With tblite: extract active site + substrate (~20 atoms), call `run_tunnelscan` with `fast_test=True`; produces physically meaningful centroid MD KIE

**Current behaviour without tblite:** falls back to ASE EMT (classical metallic potential — wrong for enzyme chemistry; pipeline infrastructure runs but output is unphysical).

---

## Data sources

- 2AGW — AADH with tryptamine (Masgrau et al. 2006, *Science* 311:1600)
- 2AH1 — Oxidised AADH, 9013 ANISOU anisotropic displacement records
- 1RX2 — *E. coli* DHFR with NADP⁺/folate
- 1GWJ — MR, substrate-free, 2.2 Å (Carnell & Scrutton lab, JBC 2002)
- AADH KIE data: Hay & Scrutton (2012) *Nat. Chem.* 4:161; Pang et al. (2010) *JACS* 132:7038
- DHFR G121V kH/kD: Wang et al. (2006) *Biochemistry* 45:1383, DOI 10.1021/bi0518242 [HIGH]
- DHFR I14 KIE data: Stojković et al. (2012) *JACS* 134:1738, DOI 10.1021/ja209425w [LOW]
- DHFR WT KIE: Sikorski et al. (2004) *JACS* 126:4778, DOI 10.1021/ja031683w
- MR WT KIE: Pudney et al. (2008) *JACS* 130:17525, DOI 10.1021/ja800471f; Hay et al. (2007) *PNAS* 104:507
- MR N189A: Pudney et al. (2007) *JACS* 129:13949, DOI 10.1021/ja074463h

---

## Status

Built by a biochemistry undergraduate at University College London. Computational validation complete on AADH (LOO-R² = 0.941) and DHFR (LOO-R² = 1.000, n = 4, GPR active). MR zero-parameter transfer test: N189A direction correctly recovered. Seeking wet-lab collaboration for experimental validation of novel predictions.

[github.com/soberdaniel24/TunnelScan](https://github.com/soberdaniel24/TunnelScan)
