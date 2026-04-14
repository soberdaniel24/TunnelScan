# TunnelScan

A computational platform for predicting enzyme mutations that exploit quantum tunnelling.

---

## The problem

Pharmaceutical enzymes are engineered by trial and error. Directed evolution screens thousands of random mutations. Computational tools like Rosetta model classical transition state chemistry. Neither approach accounts for quantum tunnelling — the mechanism by which hydrogen transfer occurs through rather than over the energy barrier, and which is responsible for a significant fraction of catalytic rate in the enzymes most relevant to drug manufacturing.

The tunnelling literature has known this for twenty years. The engineering tools have not caught up.

---

## What TunnelScan does

Given a protein crystal structure, TunnelScan predicts which mutations will enhance or reduce quantum tunnelling, and by how much. It outputs a ranked list of candidates with predicted kinetic isotope effects (KIEs), mechanistic classification, confidence scores, and synergistic double mutant combinations — a prioritised experimental roadmap that a biochemist can take directly to the lab.

The platform models five physical contributions to KIE change:

- **Static geometry** — residue volume change and D-A axis compression, Bell correction with Marcus exponential
- **Promoting vibration dynamics** — ENM normal mode participation weighted by crystallographic anisotropic displacement alignment with the D-A axis
- **Conformational breathing** — Gaussian D-A fluctuation model with H-bond disruption scoring
- **Electrostatics** — Coulomb field from charged residues projected onto the D-A axis
- **Long-range network coupling** — ENM cross-correlation scan for distal residues coupled to the active site through collective protein motion

Temperature-dependent KIE predictions are generated via the Klinman-Arrhenius framework, including AH/AD pre-exponential factor estimates and tunnelling regime classification.

---

## Validation

Calibrated against published AADH (aromatic amine dehydrogenase) mutant KIE data from the Scrutton group (Hay & Scrutton, Nature Chemistry 2012):

| Mutation | Predicted | Experimental | Error |
|---|---|---|---|
| T172A | 7.2 | 7.4 | 3% |
| T172S | 18.2 | 17.9 | 2% |
| T172C | 10.8 | 12.1 | 11% |
| T172V | 2.4 | 4.8 | 50%* |

*T172V is a documented outlier. Val introduces beta-branched backbone conformational constraints not yet captured by the volume-proxy static model. Flagged as a known limitation.

Cross-validated on DHFR (E. coli dihydrofolate reductase) with no parameter changes:

- I14A correctly predicted as reducing tunnelling — donor-side backstop effect, sign derived from D-A axis geometry
- F125 mutations correctly predicted as reducing tunnelling
- G121 identified as tunnelling-relevant at 19Å from the active site via ENM network coupling — outside the geometric scan radius, found through collective mode cross-correlation with Met20 loop anchors

G121 identification is significant: it is the canonical example of a distal tunnelling-network residue in the enzyme kinetics literature, and no existing enzyme engineering tool finds it.

---

## Novel predictions

220 mutations scored on AADH (2AGW). 216 novel (untested).

Top static predictions: L380G, L423G, F343G, I374G, F169G

Top dynamic predictions: P356G, P409G, P375G

Top synergistic double mutants: P375G/P409G (interaction +0.090), F343G/P356G (static + dynamic, interaction +0.052)

---

## Repository structure

```
src/
  pdb_parser.py             Structure parsing, B-factors, H-bond detection
  elastic_network.py        GNM, rank-normalised participation
  tunnelling_model.py       Bell correction, KIE baseline
  tunnel_score.py           Five-component scoring model
  tunnel_scan.py            Active site scanner
  anisotropic_bfactor.py    ANISOU record parsing, D-A alignment scoring
  electrostatics.py         Coulomb term for charged residues
  breathing.py              Gaussian D-A fluctuation model
  network_coupling.py       Long-range ENM cross-correlation scan
  multi_mutation.py         Double mutant combination engine
  temperature_dependence.py Klinman-Arrhenius T-dependence predictions
  calibration.py            Published KIE dataset
  report.py                 Report generator
  run_tunnelscan.py         AADH entry point
  run_dhfr.py               DHFR validation entry point

data/
  structures/               PDB files (2AGW, 2AH1, 1RX2)
  results/                  Scan outputs
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
