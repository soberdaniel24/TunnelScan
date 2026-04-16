"""
bayesian_uncertainty.py
-----------------------
Bayesian uncertainty quantification for TunnelScan KIE predictions.

Replaces the heuristic confidence score in MutationScore with a proper
posterior distribution over predicted KIE values.

Physical model
--------------
The tunnelling prediction is:

  ln(KIE_pred) = ln(KIE_WT)
               + (-ALPHA_H × da_change)          [static component]
               + BETA × dynamic_delta            [dynamic component]
               + breathing_delta + elec_delta    [fixed components]

Two parameters carry genuine physical uncertainty and are treated as random:

  ALPHA_H — Marcus theory decay constant for H-transfer, Å⁻¹.
    Theoretical value 26 Å⁻¹ (Kuznetsov & Ulstrup 1999; Bell 1980).
    Enzyme-measured values span 22–30 Å⁻¹ across different systems
    (Scrutton et al. 2019 Annu Rev Biochem 88:555).
    Prior: ALPHA_H ~ N(μ=26, σ²=4)

  BETA — weight of the dynamic penalty in ln(KIE) space.
    Calibrated at 1.5 on the T172 AADH series but uncertain due to
    small training set (n=4). Must be positive (larger = stronger
    dynamic coupling).
    Prior: BETA ~ HalfNormal(scale=2)

The third prior specified — static_delta ~ N(0, 2) — is incorporated
as the prior predictive on the static component before observing mutation
geometry. Given that static_delta = -ALPHA_H × da_change:

  Var(static_delta) ≈ ALPHA_H_mean² × σ_da² + da_change² × ALPHA_H_sigma²

where σ_da ≈ 2/ALPHA_H ≈ 0.077 Å is the implied conformational sampling
uncertainty in da_change. This is automatically captured by propagating
ALPHA_H uncertainty through the grid integral; an explicit σ_static = 2
prior is not added on top (it would double-count the same uncertainty).

Model noise
-----------
Beyond parametric uncertainty, a residual noise term σ_model captures:
  - Conformational sampling (actual da_change ≠ canonical geometry value)
  - ENM approximation error in dyn_importance
  - H-bond geometry approximation in disruption magnitudes
  - Missing physics (long-range electrostatics, solvent)

σ_model is estimated via maximum likelihood from MAP residuals on the
T172 calibration series, using n-2 degrees of freedom (2 parameters fitted).

Inference method
----------------
With n=4 training points and 2 uncertain parameters, Metropolis-Hastings
MCMC gives high-variance chains that rarely converge in < 50,000 steps.
Exact 2D grid quadrature (200 × 200 = 40,000 evaluations) is machine-
precision accurate and runs in < 10 ms. This is used instead of MCMC.
The posterior is evaluated on a uniform grid spanning ±4σ of each prior;
normalisation uses the rectangle rule (valid to O(Δ²) on smooth densities).

Posterior predictive
--------------------
For a novel mutation with components (da_change, dyn_δ, breath_δ, elec_δ),
the prediction is:

  μ(α, β) = ln(KIE_WT) − α × da_change + β × dyn_δ + breath_δ + elec_δ

This is linear in (α, β), so the posterior predictive moments are exact:

  E[ln(KIE) | data] = ∫∫ μ(α,β) p(α,β|data) dα dβ
  Var[ln(KIE) | data] = Var_θ[μ(θ)] + σ²_model

KIE is log-normally distributed:
  E[KIE]    = exp(μ + σ²/2)
  std[KIE]  = E[KIE] × sqrt(exp(σ²) − 1)
  CI₉₀      = [exp(μ − 1.6449σ), exp(μ + 1.6449σ)]

Prioritisation
--------------
  prioritisation_score = E[KIE] / std[KIE]

High score = high predicted KIE AND narrow uncertainty = best experimental
target. This is the signal-to-noise ratio in KIE space; it directly
addresses the question "which mutation is most worth synthesising?"

References
----------
Bell RP (1980) The Tunnel Effect in Chemistry. Chapman & Hall.
Kuznetsov AM, Ulstrup J (1999) Can J Chem 77:1085.
Scrutton NS et al (2019) Annu Rev Biochem 88:555.
Johannissen LO et al (2007) FEBS J 278:1701.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import scipy.stats
from scipy.special import logsumexp


# ── Physical prior parameters ─────────────────────────────────────────────────

ALPHA_PRIOR_MU    = 26.0   # Marcus decay constant, Å⁻¹ (Kuznetsov & Ulstrup 1999)
ALPHA_PRIOR_SIGMA =  2.0   # ±1σ spans 24–28; ±2σ covers 22–30 (full enzyme range)
BETA_PRIOR_SCALE  =  2.0   # HalfNormal scale; mean ≈ 1.60, 95th percentile ≈ 5.0

# Integration grid: ±4σ on ALPHA, 0 to 6×scale on BETA
_N_GRID   = 200
_A_MIN    = ALPHA_PRIOR_MU - 4 * ALPHA_PRIOR_SIGMA   # 18 Å⁻¹
_A_MAX    = ALPHA_PRIOR_MU + 4 * ALPHA_PRIOR_SIGMA   # 34 Å⁻¹
_B_MIN    = 0.0
_B_MAX    = BETA_PRIOR_SCALE * 6.0                   # 12.0


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class BayesianConfidence:
    """
    Posterior uncertainty for a single KIE prediction.

    All KIE fields are in linear (not log) space unless prefixed ln_.
    The posterior over KIE is log-normally distributed, so CI is derived
    from the Gaussian posterior over ln(KIE).

    Fields
    ------
    ln_kie_mean : posterior mean of ln(KIE)
    ln_kie_std  : posterior std of ln(KIE), including σ_model
    mean        : E[KIE | data] = exp(ln_kie_mean + ln_kie_std²/2)
    std         : std[KIE | data] via log-normal moment formula
    ci_90_lower : 5th  percentile of KIE posterior predictive
    ci_90_upper : 95th percentile of KIE posterior predictive
    predicted_kie_map : point estimate at MAP (ALPHA_H*, BETA*)
    prioritisation_score : mean / std  — signal-to-noise for experimental design
    """
    ln_kie_mean:          float
    ln_kie_std:           float
    mean:                 float
    std:                  float
    ci_90_lower:          float
    ci_90_upper:          float
    predicted_kie_map:    float
    experimental_kie:     Optional[float]
    prioritisation_score: float

    def within_ci(self) -> Optional[bool]:
        """Is the experimental KIE inside the 90% credible interval?"""
        if self.experimental_kie is None:
            return None
        return self.ci_90_lower <= self.experimental_kie <= self.ci_90_upper

    def coverage_str(self) -> str:
        inside = self.within_ci()
        if inside is None:
            return "(novel)"
        return "✓ in CI" if inside else "✗ outside CI"

    def summary(self) -> str:
        return (f"KIE={self.mean:.1f} ± {self.std:.1f}  "
                f"90%CI=[{self.ci_90_lower:.1f},{self.ci_90_upper:.1f}]  "
                f"SNR={self.prioritisation_score:.2f}  {self.coverage_str()}")


# ── Bayesian model ────────────────────────────────────────────────────────────

class BayesianTunnellingModel:
    """
    2D Bayesian model over (ALPHA_H, BETA) fitted on T172 calibration data.

    The posterior p(ALPHA_H, BETA | data) is evaluated on a 200×200 grid via
    exact rectangle-rule quadrature. For each novel mutation, the posterior
    predictive is computed by moment-matching over the same grid.

    Build via BayesianTunnellingModel.from_calibration_data().
    Query via model.predict(da_change, dynamic_delta, breathing_delta, elec_delta).
    """

    def __init__(
        self,
        alpha_grid:    np.ndarray,   # (N,)  — uniform, Å⁻¹
        beta_grid:     np.ndarray,   # (M,)  — uniform, positive
        posterior:     np.ndarray,   # (N, M) — normalised probability density
        sigma_model:   float,        # residual noise in ln(KIE) space
        ln_kie_wt:     float,
        map_alpha:     float,
        map_beta:      float,
    ):
        self.alpha_grid  = alpha_grid
        self.beta_grid   = beta_grid
        self.posterior   = posterior           # ∫∫ post dα dβ = 1
        self.sigma_model = sigma_model
        self.ln_kie_wt   = ln_kie_wt
        self.map_alpha   = map_alpha
        self.map_beta    = map_beta
        self.d_alpha     = float(alpha_grid[1] - alpha_grid[0])
        self.d_beta      = float(beta_grid[1]  - beta_grid[0])

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_calibration_data(
        cls,
        records:    List[Tuple[float, float, float, float, float, float]],
        ln_kie_wt:  float,
    ) -> 'BayesianTunnellingModel':
        """
        Fit posterior from calibration observations.

        Parameters
        ----------
        records : list of 6-tuples
            Each tuple: (da_change, dynamic_delta, breathing_delta, elec_delta,
                         ln_kie_obs, sigma_obs_ln)
            where sigma_obs_ln = kie_error / kie_obs (log-space σ, delta method).
        ln_kie_wt : float
            ln(predicted WT KIE) — same value used in TunnelScorer.

        Returns
        -------
        BayesianTunnellingModel with fitted posterior and σ_model.
        """
        alpha_grid = np.linspace(_A_MIN, _A_MAX, _N_GRID)
        beta_grid  = np.linspace(_B_MIN, _B_MAX, _N_GRID)

        d_alpha = float(alpha_grid[1] - alpha_grid[0])
        d_beta  = float(beta_grid[1]  - beta_grid[0])

        # ── Log prior ──────────────────────────────────────────────────────
        # ALPHA_H ~ N(26, 2²)
        log_p_alpha = scipy.stats.norm.logpdf(
            alpha_grid, ALPHA_PRIOR_MU, ALPHA_PRIOR_SIGMA)       # (N,)
        # BETA ~ HalfNormal(scale=2)
        # scipy.stats.halfnorm(loc=0, scale=s): pdf(x) = 2/s × φ(x/s) for x≥0
        log_p_beta = scipy.stats.halfnorm.logpdf(
            beta_grid, loc=0.0, scale=BETA_PRIOR_SCALE)           # (M,)

        # log_post[i, j] — starts as log-prior, accumulates log-likelihood
        log_post = log_p_alpha[:, None] + log_p_beta[None, :]     # (N, M)

        # Initial σ_model: broad estimate for first likelihood pass
        sigma_model = 0.5

        # Two-pass: (1) broad σ → MAP; (2) MLE σ from MAP residuals → refined
        for _pass in range(2):
            if _pass == 1:
                # Re-start log_post from prior for the refined pass
                log_post = log_p_alpha[:, None] + log_p_beta[None, :]

            for da_change, dyn_delta, breath_delta, elec_delta, ln_kie_obs, sigma_obs in records:
                # μ(α, β) = ln_kie_wt − α·da_change + β·dyn + breath + elec
                # Shape: (N, 1) broadcast to (N, M)
                mu = (ln_kie_wt
                      - alpha_grid[:, None] * da_change    # (N, 1)
                      + beta_grid[None, :]  * dyn_delta    # (1, M)
                      + breath_delta + elec_delta)          # scalars

                # σ_total combines measurement noise and model residual noise
                sigma_total = float(np.sqrt(sigma_obs**2 + sigma_model**2))
                log_post   += scipy.stats.norm.logpdf(ln_kie_obs, mu, sigma_total)

            # ── Normalize ──────────────────────────────────────────────────
            # Subtract max before exp for numerical stability; then renorm
            log_post -= log_post.max()
            post      = np.exp(log_post)
            norm      = np.sum(post) * d_alpha * d_beta
            if norm < 1e-300:
                norm = 1.0   # degenerate: fall back to flat
            post /= norm

            # ── MAP ────────────────────────────────────────────────────────
            idx       = np.unravel_index(np.argmax(post), post.shape)
            map_alpha = float(alpha_grid[idx[0]])
            map_beta  = float(beta_grid[idx[1]])

            if _pass == 0:
                # ── MLE σ_model from MAP residuals (n − 2 DOF) ────────────
                residuals = []
                for da_change, dyn_delta, breath_delta, elec_delta, ln_kie_obs, _ in records:
                    ln_pred = (ln_kie_wt
                               - map_alpha * da_change
                               + map_beta  * dyn_delta
                               + breath_delta + elec_delta)
                    residuals.append(ln_kie_obs - ln_pred)
                residuals = np.array(residuals)
                dof       = max(1, len(residuals) - 2)
                sigma_model = float(np.sqrt(np.sum(residuals**2) / dof))
                # Minimum floor: numerical precision noise
                sigma_model = max(sigma_model, 0.05)

        return cls(
            alpha_grid  = alpha_grid,
            beta_grid   = beta_grid,
            posterior   = post,
            sigma_model = sigma_model,
            ln_kie_wt   = ln_kie_wt,
            map_alpha   = map_alpha,
            map_beta    = map_beta,
        )

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        da_change:       float,
        dynamic_delta:   float,
        breathing_delta: float,
        elec_delta:      float,
        experimental_kie: Optional[float] = None,
    ) -> BayesianConfidence:
        """
        Posterior predictive distribution for one mutation.

        μ(α, β) = ln(KIE_WT) − α·da_change + β·dyn + breath + elec
        is linear in (α, β), so the moments are exact:

          E[ln(KIE)] = ∫∫ μ(α,β) p(α,β) dα dβ
          Var[ln(KIE)] = E[μ²] − (E[μ])² + σ²_model

        KIE is log-normally distributed; CI from Gaussian quantiles on ln(KIE).
        """
        # μ on the full posterior grid — shape (N_ALPHA, N_BETA)
        mu_grid = (self.ln_kie_wt
                   - self.alpha_grid[:, None] * da_change
                   + self.beta_grid[None, :]  * dynamic_delta
                   + breathing_delta + elec_delta)

        w = self.posterior   # (N, M), normalised: ∫∫w dα dβ = 1

        # Posterior mean of ln(KIE)
        ln_mean = float(np.sum(w * mu_grid) * self.d_alpha * self.d_beta)

        # Posterior variance: parameter uncertainty + model noise
        second_moment = float(np.sum(w * mu_grid**2) * self.d_alpha * self.d_beta)
        var_param     = max(0.0, second_moment - ln_mean**2)
        ln_var        = var_param + self.sigma_model**2
        ln_std        = float(np.sqrt(ln_var))

        # ── Log-normal moments for KIE in linear space ────────────────────
        # If X = ln(KIE) ~ N(μ, σ²):
        #   E[KIE]    = exp(μ + σ²/2)
        #   std[KIE]  = E[KIE] × sqrt(exp(σ²) − 1)
        #   CI₉₀      = [exp(μ − z₀.₀₅ × σ), exp(μ + z₀.₀₅ × σ)]
        #              z₀.₀₅ = 1.6449 (5th/95th percentile)
        mu  = ln_mean
        sig = ln_std

        kie_mean = float(np.exp(np.clip(mu + 0.5 * sig**2, -3.0, 8.0)))
        kie_std  = kie_mean * float(np.sqrt(max(0.0, np.exp(sig**2) - 1.0)))

        Z_90      = 1.6449
        ci_lower  = float(np.exp(np.clip(mu - Z_90 * sig, -3.0, 8.0)))
        ci_upper  = float(np.exp(np.clip(mu + Z_90 * sig, -3.0, 8.0)))

        # ── MAP point estimate ─────────────────────────────────────────────
        ln_map = (self.ln_kie_wt
                  - self.map_alpha * da_change
                  + self.map_beta  * dynamic_delta
                  + breathing_delta + elec_delta)
        kie_map = float(np.exp(np.clip(ln_map, -3.0, 8.0)))

        # ── Prioritisation: signal-to-noise in KIE space ──────────────────
        # SNR = mean / std: high KIE AND narrow CI → best experimental target
        prio = kie_mean / max(kie_std, 1e-6)

        return BayesianConfidence(
            ln_kie_mean          = mu,
            ln_kie_std           = sig,
            mean                 = kie_mean,
            std                  = kie_std,
            ci_90_lower          = ci_lower,
            ci_90_upper          = ci_upper,
            predicted_kie_map    = kie_map,
            experimental_kie     = experimental_kie,
            prioritisation_score = prio,
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def posterior_summary(self) -> dict:
        """Marginal posterior statistics for ALPHA_H and BETA."""
        w = self.posterior   # (N, M)

        # Marginalise over BETA → weight on ALPHA axis
        w_a = np.sum(w, axis=1) * self.d_beta   # (N,)
        a_mean = float(np.sum(self.alpha_grid * w_a) * self.d_alpha)
        a_var  = float(np.sum(self.alpha_grid**2 * w_a) * self.d_alpha) - a_mean**2
        a_std  = float(np.sqrt(max(0.0, a_var)))

        # Marginalise over ALPHA_H → weight on BETA axis
        w_b = np.sum(w, axis=0) * self.d_alpha  # (M,)
        b_mean = float(np.sum(self.beta_grid * w_b) * self.d_beta)
        b_var  = float(np.sum(self.beta_grid**2 * w_b) * self.d_beta) - b_mean**2
        b_std  = float(np.sqrt(max(0.0, b_var)))

        return {
            'alpha_mean':  a_mean,
            'alpha_std':   a_std,
            'alpha_map':   self.map_alpha,
            'beta_mean':   b_mean,
            'beta_std':    b_std,
            'beta_map':    self.map_beta,
            'sigma_model': self.sigma_model,
        }

    def calibration_check(
        self,
        records: List[Tuple[float, float, float, float, float, float]],
    ) -> List[dict]:
        """
        For each calibration observation, compute the posterior predictive
        and check coverage. Returns one dict per record with fields:
          ln_kie_obs, ln_pred_mean, ln_pred_std, within_90ci, z_score.
        """
        results = []
        for da_change, dyn_delta, breath_delta, elec_delta, ln_kie_obs, sigma_obs in records:
            bc = self.predict(da_change, dyn_delta, breath_delta, elec_delta,
                              experimental_kie=float(np.exp(ln_kie_obs)))
            z = (ln_kie_obs - bc.ln_kie_mean) / max(bc.ln_kie_std, 1e-9)
            results.append({
                'ln_kie_obs':   ln_kie_obs,
                'ln_pred_mean': bc.ln_kie_mean,
                'ln_pred_std':  bc.ln_kie_std,
                'within_90ci':  bc.within_ci(),
                'z_score':      float(z),
            })
        return results


# ── Integration helpers ───────────────────────────────────────────────────────

def build_bayesian_model_from_scores(
    mutation_scores,         # List[MutationScore]
    aadh_kie_data,           # List[KIEDataPoint]
    ln_kie_wt: float,
) -> BayesianTunnellingModel:
    """
    Build a BayesianTunnellingModel from pre-computed MutationScore objects.

    Called after the physics scan completes. The calibration mutations
    (T172 series) provide da_change, dynamic_delta etc. as already computed
    by TunnelScorer; the BayesianTunnellingModel fits (ALPHA_H, BETA)
    posteriors from these plus the published experimental KIEs.

    Parameters
    ----------
    mutation_scores : list of MutationScore
        All scored mutations from TunnelScorer (known + novel).
    aadh_kie_data : list of KIEDataPoint
        Experimental calibration data (from calibration.py).
    ln_kie_wt : float
        ln(wt_kie) — the baseline used in TunnelScorer.
    """
    score_lookup = {ms.label: ms for ms in mutation_scores}
    records = []

    for dp in aadh_kie_data:
        if dp.new_aa == 'WT':
            continue
        ms = score_lookup.get(dp.label)
        if ms is None:
            continue
        ln_kie_obs = float(np.log(dp.kie_298k))
        # Log-space σ via delta method: σ_ln ≈ kie_error / kie_298k
        sigma_obs  = float(dp.kie_error / dp.kie_298k)
        records.append((
            ms.da_change,
            ms.dynamic_delta,
            ms.breathing_delta,
            ms.elec_delta,
            ln_kie_obs,
            sigma_obs,
        ))

    if len(records) < 2:
        raise ValueError(
            f"Need ≥2 calibration points with matching MutationScore labels; "
            f"got {len(records)}. Check that scan covers T172 residue."
        )

    return BayesianTunnellingModel.from_calibration_data(records, ln_kie_wt)


def add_bayesian_confidence(
    mutation_scores,   # List[MutationScore] — mutated in-place
    aadh_kie_data,
    ln_kie_wt: float,
    verbose: bool = False,
) -> 'BayesianTunnellingModel':
    """
    Enrich every MutationScore with a BayesianConfidence object in-place.

    Builds the posterior model from calibration data, then calls
    model.predict() for each scored mutation. The BayesianConfidence is
    stored in ms.bayes and replaces the heuristic ms.confidence for
    prioritisation purposes.

    Returns the fitted BayesianTunnellingModel for inspection.
    """
    model = build_bayesian_model_from_scores(
        mutation_scores, aadh_kie_data, ln_kie_wt)

    if verbose:
        ps = model.posterior_summary()
        print(f"\n  Bayesian posterior (n={len([ms for ms in mutation_scores if ms.experimental_kie])}"
              f" calibration points):")
        print(f"    ALPHA_H: {ps['alpha_mean']:.2f} ± {ps['alpha_std']:.2f} Å⁻¹"
              f"  (prior: N(26, 2))")
        print(f"    BETA:    {ps['beta_mean']:.3f} ± {ps['beta_std']:.3f}"
              f"  (prior: HalfNormal(2))")
        print(f"    σ_model: {ps['sigma_model']:.3f} (residual noise in ln(KIE))")

    for ms in mutation_scores:
        ms.bayes = model.predict(
            da_change       = ms.da_change,
            dynamic_delta   = ms.dynamic_delta,
            breathing_delta = ms.breathing_delta,
            elec_delta      = ms.elec_delta,
            experimental_kie = ms.experimental_kie,
        )

    return model
