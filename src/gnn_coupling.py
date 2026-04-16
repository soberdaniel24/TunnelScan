"""
gnn_coupling.py
---------------
Graph Neural Network residue coupling correction.

Replaces the scalar ENM participation score with a 3-layer message-passing
network that propagates the mutation signal through the protein contact graph
to the D-A pair.  The key insight: a mutation's dynamic effect on tunnelling
is NOT purely local — it propagates through the protein contact network and
is amplified or attenuated depending on the network topology.

Architecture
------------
A mutation at residue m introduces a perturbation signal s_m.  This signal
propagates through the ENM contact graph to the donor (D) and acceptor (A):

  Layer 0 (initialisation):
    h_i^0 = phi_i   — node feature vector (physics-derived, F=5 dimensions)
    Mutated node m gets an additional signal: h_m^0 += delta * s_m

  Layer l → l+1 (message passing):
    message_i^l = Σ_{j ∈ N(i)}  A_{ij} × h_j^l
    h_i^{l+1}   = σ( h_i^l + w_l × message_i^l )

  where  A_{ij} = contact_strength(i,j) / Σ_k contact_strength(i,k)
  (row-normalised adjacency matrix from GNM Kirchhoff matrix)
  and σ(x) = x / (1 + |x|)  (smooth, bounded activation; avoids exp overflow)
  and w_l ∈ R is a scalar weight per layer  (3 parameters total)

  Readout (after L=3 layers):
    r_DA = mean(h_D^L, h_A^L)                (signal at D-A nodes)
    gnn_delta = w_out × r_DA · h_m^L         (inner product: how much m
                                               reaches the D-A pair)

Training
--------
With n=4 calibration mutations (T172 series), the GNN has 4 parameters
(w_1, w_2, w_3, w_out) and we fit them by minimising:

  L = Σ_i (residual_i - gnn_delta_i)² + λ Σ_k w_k²

where residual_i = ln(exp_KIE_i) - ln(physics_KIE_i) (from the existing scorer).
This is an L2-regularised least-squares problem; we solve via grid search
over (w_out) after fixing (w_1, w_2, w_3) via NNLS on the residuals.

Physics interpretation
----------------------
- w_l > 0: the mutation signal is amplified by passing through the network
- w_l < 0: network coupling reduces the signal (damping)
- w_out > 0: reaching the D-A pair more strongly → larger KIE correction

The 3-layer depth corresponds to up to 3-hop signal propagation (~12-18 Å
in a typical protein), covering the active site neighbourhood completely.

References
----------
Kipf TN, Welling M (2017) ICLR. — Graph Convolutional Networks
Zheng W et al. (2009) Biophys J 97:2485. — ENM allosteric communication
Benkovic & Hammes-Schiffer (2003) Science 301:1196. — DHFR dynamic network
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elastic_network import ENMResult
from pdb_parser import Structure


# ── Physical constants ────────────────────────────────────────────────────────

N_LAYERS     = 3          # number of message-passing layers
N_FEATURES   = 5          # node feature dimension
GNM_CUTOFF   = 7.5        # Å, must match ENM build
LAMBDA_REG   = 0.10       # L2 regularisation weight
MIN_CONTACTS = 1          # skip isolated nodes


# ── Output dataclasses ────────────────────────────────────────────────────────

@dataclass
class GNNResult:
    """
    GNN coupling correction for one mutation.

    Fields
    ------
    gnn_delta       : learned correction to add to ln(KIE)
    layer_signals   : [signal after each layer] at the mutated node
    da_reach        : how much of the mutation signal reached the D-A pair
    n_hops_covered  : number of network hops explored (= N_LAYERS)
    """
    gnn_delta:      float   # correction to ln(KIE)
    layer_signals:  List[float]   # propagated signal at each layer
    da_reach:       float   # fraction of signal reaching D-A nodes
    n_hops_covered: int = N_LAYERS


@dataclass
class GNNModel:
    """
    Trained GNN model: weights + adjacency matrix + feature matrix.

    Do not construct directly — use GNNCoupling.fit().
    """
    weights:  np.ndarray   # (N_LAYERS+1,) = [w1, w2, w3, w_out]
    train_r2: float
    residuals_before: List[float]
    residuals_after:  List[float]


# ── Node feature builder ──────────────────────────────────────────────────────

def _build_node_features(
    structure: Structure,
    enm:       ENMResult,
    donor_key:    Tuple[str, int],
    acceptor_key: Tuple[str, int],
    donor_coords: np.ndarray,
    acceptor_coords: np.ndarray,
    substrate_hbond_keys: Optional[set] = None,
) -> np.ndarray:
    """
    Build the (n, F) node feature matrix for the ENM contact graph.

    Features (F=5):
      0: ENM participation score (rank-normalised, 0-1)
      1: B-factor (normalised by mean, clipped 0-2)
      2: Axis proximity = exp(−dist_to_axis/3.0)   (0-1)
      3: Rigidity proxy = bfactor (inverse flexibility)
      4: Substrate H-bond flag (0 or 1)
    """
    from breathing import AA_RIGIDITY

    keys = enm.residue_keys
    n    = len(keys)

    # Precompute axis geometry
    da_vec  = acceptor_coords - donor_coords
    da_len  = float(np.linalg.norm(da_vec))
    da_unit = da_vec / da_len if da_len > 1e-6 else np.array([0., 0., 1.])
    da_mid  = (donor_coords + acceptor_coords) / 2.0

    # ENM participation (mode-weighted)
    participations = np.array([
        enm.get_participation(c, r) for (c, r) in keys
    ])

    # B-factor normalisation
    b_mean = structure.mean_bfactor if structure.mean_bfactor > 1.0 else 20.0

    phi = np.zeros((n, N_FEATURES), dtype=np.float64)

    for idx, (chain, resnum) in enumerate(keys):
        res = structure.get_residue(chain, resnum)
        if res is None:
            continue

        # Feature 0: ENM participation
        phi[idx, 0] = float(participations[idx])

        # Feature 1: B-factor normalised
        bf = structure.normalised_bfactor(res)
        phi[idx, 1] = float(np.clip(bf, 0.0, 2.0)) / 2.0

        # Feature 2: axis proximity
        ca = res.ca_coords
        if ca is not None:
            v     = ca - donor_coords
            t     = float(np.dot(v, da_unit))
            proj  = donor_coords + t * da_unit
            dist  = float(np.linalg.norm(ca - proj))
            phi[idx, 2] = float(np.exp(-dist / 3.0))
        else:
            phi[idx, 2] = 0.0

        # Feature 3: rigidity
        phi[idx, 3] = AA_RIGIDITY.get(res.name, 0.4)

        # Feature 4: substrate H-bond flag
        if substrate_hbond_keys and (chain, resnum) in substrate_hbond_keys:
            phi[idx, 4] = 1.0

    return phi


# ── Row-normalised adjacency from GNM Kirchhoff matrix ───────────────────────

def _build_adjacency(enm: ENMResult, gnm_cutoff: float) -> np.ndarray:
    """
    Build row-normalised adjacency A from the GNM contact graph.

    A[i,j] = (1 if |r_i - r_j| < cutoff) / degree(i)

    Returns dense (n, n) float64 matrix.  For n=300: 720 KB — acceptable.
    """
    n    = len(enm.residue_keys)
    adj  = np.zeros((n, n), dtype=np.float64)

    # The Kirchhoff matrix Γ has −1 for each contact and +deg on diagonal.
    # We recover the contact pattern from its off-diagonal negative entries.
    # eigenvalues/vectors are available but we need raw contacts.
    # Use the fact that the ENM was built with the same structure object —
    # approximate contacts from the ENM by thresholding the off-diagonal
    # of the Kirchhoff matrix.  Kirchhoff[i,j] = -1 for contacts, 0 otherwise.
    # We reconstruct Γ from eigenvalues and eigenvectors:
    #   Γ = U diag(λ) Uᵀ   (note: mode 0 is the trivial mode with λ=0)
    evals = enm.eigenvalues   # (n,)
    evecs = enm.eigenvectors  # (n, n)

    kirchhoff = evecs @ np.diag(evals) @ evecs.T   # reconstruct Γ

    # Contact: Γ[i,j] < -0.5 (should be close to -1 for true contacts)
    contact_mask = kirchhoff < -0.5
    np.fill_diagonal(contact_mask, False)

    adj[contact_mask] = 1.0

    # Row-normalise
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-10] = 1.0   # avoid division by zero for isolated nodes
    adj /= row_sums

    return adj


# ── Core GNN class ────────────────────────────────────────────────────────────

class GNNCoupling:
    """
    GNN residue coupling model for one enzyme–substrate system.

    Build once per structure, fit on calibration data, then call predict().

    Parameters
    ----------
    structure : Structure
    enm : ENMResult
    donor_key, acceptor_key : (chain, resnum)
    donor_coords, acceptor_coords : np.ndarray
    substrate_hbond_keys : set of (chain, resnum), optional
    """

    def __init__(
        self,
        structure:       Structure,
        enm:             ENMResult,
        donor_key:       Tuple[str, int],
        acceptor_key:    Tuple[str, int],
        donor_coords:    np.ndarray,
        acceptor_coords: np.ndarray,
        substrate_hbond_keys: Optional[set] = None,
    ):
        self.enm             = enm
        self.donor_key       = donor_key
        self.acceptor_key    = acceptor_key
        self.donor_coords    = donor_coords
        self.acceptor_coords = acceptor_coords

        keys = enm.residue_keys
        self.donor_idx    = keys.index(donor_key)    if donor_key    in keys else None
        self.acceptor_idx = keys.index(acceptor_key) if acceptor_key in keys else None

        # If substrate HETATM, fall back to nearest protein Cα
        if self.donor_idx is None or self.acceptor_idx is None:
            ca_list = []
            for (c, r) in keys:
                res = structure.get_residue(c, r)
                if res and res.ca:
                    ca_list.append(res.ca.coords)
                else:
                    ca_list.append(np.zeros(3))
            ca_arr = np.array(ca_list)
            if self.donor_idx is None:
                self.donor_idx = int(np.argmin(
                    np.linalg.norm(ca_arr - donor_coords, axis=1)))
            if self.acceptor_idx is None:
                self.acceptor_idx = int(np.argmin(
                    np.linalg.norm(ca_arr - acceptor_coords, axis=1)))

        # Node features (n, F)
        self._phi = _build_node_features(
            structure, enm, donor_key, acceptor_key,
            donor_coords, acceptor_coords, substrate_hbond_keys)

        # Row-normalised adjacency (n, n)
        self._adj = _build_adjacency(enm, GNM_CUTOFF)

        # Learned weights — initialised to near-zero (no correction before fitting)
        self._weights: Optional[np.ndarray] = None
        self._model:   Optional[GNNModel]   = None

    # ── Message passing ───────────────────────────────────────────────────────

    @staticmethod
    def _activation(x: np.ndarray) -> np.ndarray:
        """Smooth bounded activation: x / (1 + |x|).  No overflow risk."""
        return x / (1.0 + np.abs(x))

    def _forward(
        self,
        mut_idx:        int,
        mutation_signal: float,   # scalar encoding the mutation's dynamic effect
        weights:         np.ndarray,  # (N_LAYERS,) message-passing weights
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Run L=3 layers of message passing and return the final node state.

        The mutation signal is added to h_mut^0 as a perturbation.
        Only the mutated node carries the initial signal; the GNN propagates
        it to all other nodes through the contact adjacency.

        Returns
        -------
        h_final : (n, F) final node states
        layer_signals : list of signal magnitudes at mutated node per layer
        """
        n, F  = self._phi.shape
        h     = self._phi.copy()

        # Inject mutation signal at the mutated node
        # The signal has the same direction as the existing feature vector
        # (mutation is a perturbation of the node's natural state)
        h[mut_idx] = h[mut_idx] + mutation_signal * h[mut_idx]

        layer_signals = []
        for l in range(N_LAYERS):
            # Aggregate neighbours: (n, F) ← A × (n, F)
            agg   = self._adj @ h           # (n, F)
            h_new = self._activation(h + weights[l] * agg)
            layer_signals.append(float(np.linalg.norm(h_new[mut_idx])))
            h = h_new

        return h, layer_signals

    # ── Mutation signal encoding ──────────────────────────────────────────────

    @staticmethod
    def _mutation_signal(orig_aa: str, new_aa: str) -> float:
        """
        Scalar encoding of the mutation's dynamic perturbation.

        Rigidity change: f = rigidity_new / rigidity_orig − 1
        Negative for ALA/GLY replacements (more flexible), positive for PRO.
        """
        from breathing import AA_RIGIDITY
        r_orig = AA_RIGIDITY.get(orig_aa, 0.4)
        r_new  = AA_RIGIDITY.get(new_aa,  0.4)
        return float(r_new / max(r_orig, 0.05) - 1.0)

    # ── GNN output (single scalar correction) ────────────────────────────────

    def _gnn_output(
        self,
        mut_idx:  int,
        orig_aa:  str,
        new_aa:   str,
        weights:  np.ndarray,   # (N_LAYERS+1,) = [w1, w2, w3, w_out]
    ) -> Tuple[float, float, List[float]]:
        """
        Compute GNN correction to ln(KIE) for one mutation.

        Returns
        -------
        gnn_delta : float — correction to ln(KIE)
        da_reach  : float — |signal at D-A nodes| / |signal at mut node|
        layer_sigs: list  — signal norms per layer
        """
        ms        = self._mutation_signal(orig_aa, new_aa)
        h_final, layer_sigs = self._forward(mut_idx, ms, weights[:N_LAYERS])

        # Readout: how much of the mutated node's final state aligns with
        # the D-A pair's final state?
        D = self.donor_idx
        A = self.acceptor_idx
        if D is None or A is None:
            return 0.0, 0.0, layer_sigs

        h_mut   = h_final[mut_idx]         # (F,)
        h_DA    = (h_final[D] + h_final[A]) / 2.0  # mean of D and A states (F,)

        # Inner product: how aligned is the mutation's influence with the D-A state?
        dot     = float(np.dot(h_mut, h_DA))
        norm    = float(np.linalg.norm(h_mut) * np.linalg.norm(h_DA) + 1e-10)
        da_reach = dot / norm              # cosine similarity [-1, 1]

        gnn_delta = float(weights[N_LAYERS] * da_reach * ms)

        return gnn_delta, da_reach, layer_sigs

    # ── Fitting on calibration data ───────────────────────────────────────────

    def fit(
        self,
        calibration_residuals: List[Tuple[Tuple[str, int], str, str, float]],
        lambda_reg: float = LAMBDA_REG,
        verbose:    bool  = True,
    ) -> GNNModel:
        """
        Fit the GNN weights (w_1, w_2, w_3, w_out) on residual corrections.

        Parameters
        ----------
        calibration_residuals : list of (residue_key, orig_aa, new_aa, residual)
            residual = ln(exp_KIE) − ln(physics_KIE) for each calibration mutation
        lambda_reg : float
            L2 regularisation on weights (default 0.10).
        verbose : bool

        Returns
        -------
        GNNModel with fitted weights and training R².
        """
        keys = self.enm.residue_keys
        n_cal = len(calibration_residuals)
        if n_cal == 0:
            # No data — use zero weights
            self._weights = np.zeros(N_LAYERS + 1)
            self._model   = GNNModel(self._weights, 0.0, [], [])
            return self._model

        # Grid search over w_out (main scaling weight) with fixed message weights
        # found by a coarse 4D search.  This is efficient for n_cal ≤ 10.
        # We use a 2-stage approach:
        #   Stage 1: coarse grid over (w_mp, w_out) where w_mp is shared across layers
        #   Stage 2: refine around the best point with per-layer weights

        best_loss = np.inf
        best_w    = np.zeros(N_LAYERS + 1)

        residuals = np.array([r for _, _, _, r in calibration_residuals])

        # Stage 1: coarse grid (w_mp shared, w_out varies)
        for w_mp in np.linspace(-2.0, 2.0, 20):
            for w_out in np.linspace(-5.0, 5.0, 40):
                w = np.array([w_mp] * N_LAYERS + [w_out])
                preds = []
                for (chain, resnum), orig_aa, new_aa, _ in calibration_residuals:
                    idx = keys.index((chain, resnum)) if (chain, resnum) in keys else None
                    if idx is None:
                        preds.append(0.0)
                        continue
                    delta, _, _ = self._gnn_output(idx, orig_aa, new_aa, w)
                    preds.append(delta)
                preds = np.array(preds)
                fit_loss  = float(np.sum((preds - residuals)**2))
                reg_loss  = lambda_reg * float(np.sum(w**2))
                loss      = fit_loss + reg_loss
                if loss < best_loss:
                    best_loss = loss
                    best_w    = w.copy()

        # Stage 2: fine grid around best point
        wm_star  = best_w[:N_LAYERS].mean()
        wo_star  = best_w[N_LAYERS]
        for dw_mp in np.linspace(-0.3, 0.3, 15):
            for dw_out in np.linspace(-0.5, 0.5, 15):
                w = np.array([wm_star + dw_mp] * N_LAYERS + [wo_star + dw_out])
                preds = []
                for (chain, resnum), orig_aa, new_aa, _ in calibration_residuals:
                    idx = keys.index((chain, resnum)) if (chain, resnum) in keys else None
                    if idx is None:
                        preds.append(0.0)
                        continue
                    delta, _, _ = self._gnn_output(idx, orig_aa, new_aa, w)
                    preds.append(delta)
                preds = np.array(preds)
                fit_loss  = float(np.sum((preds - residuals)**2))
                reg_loss  = lambda_reg * float(np.sum(w**2))
                loss      = fit_loss + reg_loss
                if loss < best_loss:
                    best_loss = loss
                    best_w    = w.copy()

        self._weights = best_w

        # Training R²: how well does GNN explain the residuals?
        preds_final = []
        for (chain, resnum), orig_aa, new_aa, _ in calibration_residuals:
            idx = keys.index((chain, resnum)) if (chain, resnum) in keys else None
            if idx is None:
                preds_final.append(0.0)
                continue
            delta, _, _ = self._gnn_output(idx, orig_aa, new_aa, best_w)
            preds_final.append(delta)

        preds_arr = np.array(preds_final)
        ss_res = float(np.sum((residuals - preds_arr)**2))
        ss_tot = float(np.sum((residuals - residuals.mean())**2))
        train_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

        self._model = GNNModel(
            weights          = best_w,
            train_r2         = train_r2,
            residuals_before = residuals.tolist(),
            residuals_after  = (residuals - preds_arr).tolist(),
        )

        if verbose:
            print(f"  GNN fitted: w_mp={best_w[:N_LAYERS].mean():.3f}  "
                  f"w_out={best_w[N_LAYERS]:.3f}  "
                  f"train R²(residuals)={train_r2:.3f}")

        return self._model

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        residue_key: Tuple[str, int],
        orig_aa:     str,
        new_aa:      str,
    ) -> GNNResult:
        """
        Predict the GNN coupling correction for one mutation.

        Parameters
        ----------
        residue_key : (chain, resnum)
        orig_aa, new_aa : three-letter amino acid codes

        Returns
        -------
        GNNResult with gnn_delta and diagnostics.
        """
        w = self._weights if self._weights is not None else np.zeros(N_LAYERS + 1)
        keys = self.enm.residue_keys
        idx  = keys.index(residue_key) if residue_key in keys else None

        if idx is None:
            return GNNResult(
                gnn_delta=0.0, layer_signals=[0.0]*N_LAYERS,
                da_reach=0.0, n_hops_covered=N_LAYERS)

        delta, da_reach, layer_sigs = self._gnn_output(idx, orig_aa, new_aa, w)

        return GNNResult(
            gnn_delta     = float(delta),
            layer_signals = layer_sigs,
            da_reach      = float(da_reach),
            n_hops_covered = N_LAYERS,
        )

    def is_fitted(self) -> bool:
        return self._weights is not None


# ── Factory and integration helpers ──────────────────────────────────────────

def build_gnn_model(
    structure:       Structure,
    enm:             ENMResult,
    donor_key:       Tuple[str, int],
    acceptor_key:    Tuple[str, int],
    donor_coords:    np.ndarray,
    acceptor_coords: np.ndarray,
    calibration_residuals: Optional[List[Tuple[Tuple[str,int], str, str, float]]] = None,
    substrate_hbond_keys: Optional[set] = None,
    lambda_reg:      float = LAMBDA_REG,
    verbose:         bool  = True,
) -> GNNCoupling:
    """
    Build and optionally fit a GNNCoupling model for one enzyme system.

    Parameters
    ----------
    calibration_residuals : list of (residue_key, orig_aa, new_aa, residual)
        If provided, fits the GNN weights on these data.  If None, GNN is
        initialised but not fitted (all corrections will be zero).
    """
    model = GNNCoupling(
        structure, enm, donor_key, acceptor_key,
        donor_coords, acceptor_coords, substrate_hbond_keys)

    if calibration_residuals:
        model.fit(calibration_residuals, lambda_reg=lambda_reg, verbose=verbose)
    else:
        model._weights = np.zeros(N_LAYERS + 1)

    return model


def compute_gnn_residuals_from_scan(
    scan_scores:    list,   # list of MutationScore
    calibration:    list,   # list of CalibrationDatapoint
) -> List[Tuple[Tuple[str, int], str, str, float]]:
    """
    Extract (residue_key, orig_aa, new_aa, ln_residual) tuples from a scan.

    Used to build the calibration_residuals argument to build_gnn_model().
    """
    from calibration import get_known_kie

    result = []
    for sc in scan_scores:
        if sc.experimental_kie is None:
            continue
        ln_residual = np.log(sc.experimental_kie) - np.log(sc.predicted_kie)
        result.append(
            ((sc.chain, sc.residue_number), sc.orig_aa, sc.new_aa, float(ln_residual))
        )
    return result
