# TunnelScan

A computational platform for predicting enzyme mutations that exploit quantum tunnelling.

---

## The problem

Pharmaceutical enzymes are engineered by trial and error. Directed evolution screens thousands of random mutations. Computational tools like Rosetta model classical transition state chemistry. Neither approach accounts for quantum tunnelling — the mechanism by which hydrogen transfer occurs through rather than over the energy barrier, and which is responsible for a significant fraction of catalytic rate in the enzymes most relevant to drug manufacturing.

The tunnelling literature has known this for twenty years. The engineering tools have not caught up.

---

## What TunnelScan does

Given a protein crystal structure, TunnelScan predicts which mutations will enhance or reduce quantum tunnelling, and by how much. It outputs a ranked list of candidates with predicted kinetic isotope effects (KIEs), mechanistic classification, Bayesian confidence intervals, and synergistic double mutant combinations — a prioritised experimental roadmap that a biochemist can take directly to the lab.

The platform models nine physical contributions:

- **Static geometry** — per-atom sidechain projection onto the D-A axis with canonical rotamer geometry; Bell correction with Wigner-Kirkwood exact path-integral formula
- **Promoting vibration dynamics** — GNM normal mode participation weighted by crystallographic anisotropic displacement alignment (ANISOU) with the D-A axis
- **Quantum conformational field** — zero-point amplitude propagator from a scalar quantum field on the protein graph; fills coverage gaps where ANISOU data is unavailable
- **Anisotropic network model** — 3D displacement eigenmodes from the 3N×3N ANM Hessian; magnitude-based fallback when directional alignment data is unavailable
- **Conformational breathing** — Gaussian D-A fluctuation model capturing how mutations change the width of the tunnelling-competent distance distribution
- **Electrostatics** — Coulomb field from ionisable residues projected onto the D-A axis
- **Stochastic D-A sampling** — GNM-based conformational averaging; accounts for how stiffer or more flexible mutations shift the sampled D-A distance distribution
- **GNN residue coupling** — graph neural network correction trained on residual errors from the physics model; captures non-local coupling effects not encoded in ENM topology alone
- **Quantum tunnelling network** — novel topological analysis: adjacency matrix W_ij = √(P_i P_j) × A_i A_j × Q_ij encoding ENM participation, D-A alignment, and QCF zero-point correlations; per-residue betweenness centrality, Fiedler spectral gap, effective resistance, and spectral community assignment

Temperature-dependent KIE predictions are generated via the Klinman-Arrhenius framework, including AH/AD pre-exponential factor estimates and tunnelling regime classification. Bayesian posterior confidence intervals are computed for every prediction.

---

## Validation

Calibrated against published AADH (aromatic amine dehydrogenase) mutant KIE data (Hay & Scrutton, Nature Chemistry 2012). Calibration R² = **0.985** on the T172 series.

| Mutation | Predicted | Experimental | Error |
|---|---|---|---|
| T172A | 7.4 | 7.4 | <1% |
| T172S | 17.9 | 17.9 | <1% |
| T172C | 12.1 | 12.1 | <1% |
| T172V | 4.8 | 4.8 | <1% |

Cross-validated on DHFR (E. coli dihydrofolate reductase) with no parameter changes:

- I14A correctly predicted as reducing tunnelling — donor-side backstop, sign derived from D-A axis geometry
- F125 mutations correctly predicted as reducing tunnelling
- G121 identified as tunnelling-relevant at 19Å from the active site via ENM network coupling — outside the geometric scan radius, recovered through collective mode cross-correlation

G121 identification is significant: it is the canonical example of a distal tunnelling-network residue in the enzyme kinetics literature, and no existing enzyme engineering tool finds it.

---

## Novel predictions

220 mutations scored on AADH (2AGW). 216 novel (untested).

**Tunnelling network highlights (2AGW):**
- 182 residues in the D-A tunnelling network (within 20Å of D-A midpoint)
- Fiedler spectral gap λ₂ = 0.174 — moderately clustered, two functional sub-networks
- T172 betweenness = 0.815 (3rd highest of 182 nodes) — confirms topological centrality of the calibration residue

**Top static enhancers:** L380G, L423G, F343G, I374G, F169G

**Top dynamic enhancers:** P356G, P409G, P375G

**Top synergistic double mutants:** P375G/P409G (interaction +0.090), F343G/P356G (static + dynamic, interaction +0.052)

---

## Repository structure

```
src/
  pdb_parser.py                 Structure parsing, B-factors, H-bond detection
  elastic_network.py            GNM, rank-normalised participation
  tunnelling_model.py           Bell correction with Wigner-Kirkwood exact formula
  tunnel_score.py               Nine-component scoring model
  tunnel_scan.py                Active site scanner, full pipeline
  anisotropic_bfactor.py        ANISOU record parsing, D-A alignment scoring
  quantum_conformational_field.py  QCF zero-point propagator (scalar field on protein graph)
  anisotropic_network_model.py  3D ANM Hessian, displacement eigenmodes
  tunnelling_network.py         Quantum tunnelling network topology (Module 9)
  electrostatics.py             Coulomb term for charged residues
  breathing.py                  Gaussian D-A fluctuation model
  stochastic_tunnelling.py      GNM-based D-A distance sampling
  gnn_coupling.py               Graph neural network residual correction
  gp_regression.py              Sparse GP regression (gated; requires n≥8)
  bayesian_uncertainty.py       Bayesian posterior confidence intervals
  network_coupling.py           Long-range ENM cross-correlation scan
  multi_mutation.py             Double mutant combination engine
  temperature_dependence.py     Klinman-Arrhenius T-dependence predictions
  calibration.py                Published KIE dataset
  report.py                     Report generator
  run_tunnelscan.py             AADH entry point
  run_dhfr.py                   DHFR validation entry point

data/
  structures/                   PDB files (2AGW, 2AH1, 1RX2)
  results/                      Scan outputs
```

---

## Data sources

- 2AGW — AADH with tryptamine (Masgrau et al. 2006, Science 311:1600)
- 2AH1 — Oxidised AADH with 9013 ANISOU anisotropic displacement records
- 1RX2 — E. coli DHFR with NADP+/folate
- Experimental KIE data: Hay & Scrutton (2012) Nat. Chem. 4:161; Johannissen et al. (2007) FEBS J 278:1701

---

## Status

Built by a biochemistry undergraduate at University College London. Computational validation complete. Seeking wet lab collaboration for experimental confirmation of novel predictions.

[github.com/soberdaniel24/TunnelScan](https://github.com/soberdaniel24/TunnelScan)
