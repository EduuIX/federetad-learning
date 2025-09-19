from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import copy, json, os
from collections import deque

from fedalert.trigger import AlertRateHoeffdingTrigger, AlertRateCUSUMTrigger
from fedalert.verify_df import DFVerifier, KLVerifier

# --- local KL helpers (null bootstrap) ---
def _fit_diag_gaussian(X, eps: float = 1e-6):
    X = np.asarray(X)
    mu = X.mean(axis=0)
    var = X.var(axis=0) + eps
    return mu, var

def _kl_diag_gaussians(mu0, var0, mu1, var1):
    term = np.log(var1 / var0) + (var0 + (mu0 - mu1) ** 2) / var1 - 1.0
    return 0.5 * float(np.sum(term))

def _sym_kl_diag_gaussians(mu0, var0, mu1, var1):
    return _kl_diag_gaussians(mu0, var0, mu1, var1) + _kl_diag_gaussians(mu1, var1, mu0, var0)


class FedAlertStrategy:
    # Add multi-round tracking variables
    def __init__(self, cfg: Dict[str, Any], init_weights: Dict[str, np.ndarray], cohorts: Dict[int, List[str]] | None = None, baseline_mode: bool = False):
        self.cfg = cfg
        self.weights = copy.deepcopy(init_weights)
        self.baseline_mode = baseline_mode

        # -------- Stage-1 trigger factories (global & cohort) --------
        trig_cfg = cfg.get("trigger", {}) or {}
        kind = trig_cfg.get("kind", "hoeffding").lower()
        alpha_global = float(trig_cfg.get("alpha_FAR", 0.01))
        alpha_cohort = float(cfg.get("alpha_FAR_cohort", alpha_global))

        def mk_trigger(alpha):
            if kind == "hoeffding":
                return AlertRateHoeffdingTrigger(alpha=alpha, warmup=trig_cfg.get("warmup_B", 30))
            c = trig_cfg.get("cusum", {}) or {}
            return AlertRateCUSUMTrigger(
                warmup=trig_cfg.get("warmup_B", 30),
                p1_guess=c.get("p1_guess", 0.2),
                h=c.get("h", 5.0),
                reset_on_trigger=c.get("reset_on_trigger", True),
            )

        self.trigger = mk_trigger(alpha_global)

        # -------- Cooldowns --------
        self.cooldown = int(cfg.get("cooldown_rounds", 8))
        self.confirm_cooldown = int(cfg.get("confirm_cooldown", 12))
        self.cooldown_ctr = 0
        self.confirm_cooldown_ctr = 0

        # -------- Stage-2 settings --------
        self.stage2_method = str(cfg.get("stage2_method", "ed")).lower()  # "ed" | "kl"
        self.stage2_ed_thresh = float(cfg.get("stage2_ed_thresh", 0.35))
        self.stage2_kl_thresh_cfg = cfg.get("stage2_kl_thresh", None)  # may be None if auto
        self.stage2_sketch = int(cfg.get("stage2_sketch", 12))
        self.update_dim = int(cfg.get("update_dim", 32))
        self.baseline_updates: List[np.ndarray] = []
        self.baseline_max = int(cfg.get("baseline_max", 300))

        # Participation gates
        self.min_part_global = int(cfg.get("min_part_global", 0))
        self.min_part_cohort = int(cfg.get("min_part_cohort", 0))

        # Auto calibration + confirmation policy
        self.auto_on = bool(cfg.get("stage2_calibrate", False))
        self.calib = cfg.get("stage2_calib", {}) or {}
        self.calib_mode = str(self.calib.get("mode", "percentile")).lower()  # percentile | sigma
        self.low_pct = float(self.calib.get("low_pct", 99.0))
        self.high_pct = float(self.calib.get("high_pct", 99.7))
        self.k_sigma_low = float(self.calib.get("k_sigma_low", 2.5))
        self.k_sigma_high = float(self.calib.get("k_sigma_high", 3.0))
        self.bootstrap_B = int(self.calib.get("bootstrap_B", 200))

        self.policy = cfg.get("stage2_confirm_policy", {}) or {}
        self.policy_mode = str(self.policy.get("mode", "1x_high_or_2x_low"))
        self.window = int(self.policy.get("window", 3))  # multi-round drift

        # Per-cohort structures
        self.cohorts = cohorts or {}
        self.cohort_ids = sorted(list(self.cohorts.keys()))
        self.triggers_c = {c: mk_trigger(alpha_cohort) for c in self.cohort_ids}
        self.cooldown_c = {c: int(self.cooldown) for c in self.cohort_ids}
        self.confirm_cooldown_c = {c: int(self.confirm_cooldown) for c in self.cohort_ids}
        self.baseline_c = {c: [] for c in self.cohort_ids}

        # Rolling low-hit windows for policy
        self.low_hits_global = deque(maxlen=self.window)  # track hits across rounds for global confirmation
        self.low_hits_c = {c: deque(maxlen=self.window) for c in self.cohort_ids}  # track cohort-specific hits

        # Auto thresholds (computed after warm-up from baseline)
        self.auto_thr_global = None  # (low, high)
        self.auto_thr_c = {c: None for c in self.cohort_ids}

        # Logging
        self.round = 0
        os.makedirs("outputs", exist_ok=True)
        if self.baseline_mode:
            print("--- RUNNING IN BASELINE MODE ---")
            log_name = "outputs/run_log_baseline.jsonl"
            log_coh_name = "outputs/run_log_cohorts_baseline.jsonl"
        else:
            log_name = "outputs/run_log.jsonl"
            log_coh_name = "outputs/run_log_cohorts.jsonl"
        
        self.log_fp = open(log_name, "w", encoding="utf-8")
        self.log_coh_fp = open(log_coh_name, "w", encoding="utf-8")

    # -------- helpers --------
    def _flatten_update(self, old: Dict[str, np.ndarray], new: Dict[str, np.ndarray]) -> np.ndarray:
        vecs = []
        for k in old.keys():
            delta = new[k].ravel() - old[k].ravel()
            vecs.append(delta.astype(np.float32))
        v = np.concatenate(vecs)
        if v.size <= self.update_dim:
            return v
        rng = np.random.RandomState(1234 + self.round)
        R = rng.normal(size=(self.update_dim, v.size)).astype(np.float32)
        proj = (R @ v) / np.sqrt(self.update_dim)
        return proj

    def _verifier(self):
        if self.stage2_method == "kl":
            return KLVerifier, float(self.stage2_kl_thresh_cfg) if self.stage2_kl_thresh_cfg is not None else None
        return DFVerifier, float(self.stage2_ed_thresh)

    def _bootstrap_null_thresholds(self, baseline: List[np.ndarray]) -> tuple[float, float]:
        """Estimate null KL distribution from baseline sketches; return (low_thr, high_thr)."""
        X = np.asarray(baseline)
        k = min(self.stage2_sketch, len(X))
        if k < 2:
            return None, None
        stats = []
        rng = np.random.RandomState(2025 + self.round)
        for _ in range(self.bootstrap_B):
            idx_a = rng.choice(len(X), size=k, replace=True)
            idx_b = rng.choice(len(X), size=k, replace=True)
            A = X[idx_a]; B = X[idx_b]
            mu_a, var_a = _fit_diag_gaussian(A)
            mu_b, var_b = _fit_diag_gaussian(B)
            stats.append(_sym_kl_diag_gaussians(mu_a, var_a, mu_b, var_b))
        stats = np.asarray(stats, dtype=float)
        if self.calib_mode == "sigma":
            mu = float(np.mean(stats)); sd = float(np.std(stats, ddof=1))
            low = mu + self.k_sigma_low * sd
            high = mu + self.k_sigma_high * sd
        else:
            low = float(np.percentile(stats, self.low_pct))
            high = float(np.percentile(stats, self.high_pct))
        return low, high

    # -------- FedAvg --------
    def aggregate_fit(self, results: List[Tuple[str, Dict[str, np.ndarray], Dict[str, float]]]) -> Dict[str, np.ndarray]:
        total = 0
        agg = {k: np.zeros_like(v) for k, v in self.weights.items()}
        for _, w_new, _ in results:
            total += 1
            for k in agg.keys():
                agg[k] += w_new[k]
        for k in agg.keys():
            agg[k] /= max(1, total)
        self.weights = agg
        return agg

    # -------- Round end --------
    def on_round_end(self, results: List[Tuple[str, Dict[str, np.ndarray], Dict[str, float]]]):
        self.round += 1
        warmup_B = int(self.cfg.get("trigger", {}).get("warmup_B", 30))

        # GLOBAL alerts
        a_bits = [int(m.get("a_bit", 0)) for _, _, m in results]
        alert_sum = int(np.sum(a_bits))
        m_part = len(a_bits)

        # Baselines during warm-up
        if self.round <= warmup_B and len(self.baseline_updates) < self.baseline_max:
            for _, w_new, _ in results[: self.stage2_sketch]:
                self.baseline_updates.append(self._flatten_update(self.weights, w_new))
        for c in self.cohort_ids:
            # collect per-cohort baseline
            cohort_res = [(cid, w_new, m) for cid, w_new, m in results if int(m.get("cohort", -1)) == c]
            if self.round <= warmup_B and len(self.baseline_c[c]) < self.baseline_max:
                for _, w_new, _ in cohort_res[: self.stage2_sketch]:
                    self.baseline_c[c].append(self._flatten_update(self.weights, w_new))

        # One-time auto calibration right after warm-up
        if self.auto_on and self.stage2_method == "kl" and self.round == warmup_B + 1:
            if len(self.baseline_updates) >= self.stage2_sketch:
                self.auto_thr_global = self._bootstrap_null_thresholds(self.baseline_updates)
            for c in self.cohort_ids:
                if len(self.baseline_c[c]) >= self.stage2_sketch:
                    self.auto_thr_c[c] = self._bootstrap_null_thresholds(self.baseline_c[c])

        # Update global trigger
        raw_trigger, theta, p0_est_internal = self.trigger.update(alert_sum, max(1, m_part))

        fired = False
        stage2_ran = False
        confirmed = False
        stage2_stat = None
        low_hits_in_window = None
        thr_low_used = None
        thr_high_used = None

        # ---- Stage-2 (global) ----
        can_eval_global = (m_part >= max(1, self.min_part_global))
        
        run_stage2 = False
        if self.baseline_mode:
            # For baseline, run Stage-2 every round after warmup
            if self.round > warmup_B and can_eval_global and len(self.baseline_updates) > 0:
                run_stage2 = True
        else:
            # For normal FedAlert, use the trigger
            if raw_trigger and can_eval_global and self.cooldown_ctr <= 0 and self.round > warmup_B and len(self.baseline_updates) > 0:
                run_stage2 = True
        
        if run_stage2:
            fired = True
            self.cooldown_ctr = self.cooldown

            curr = []
            for _, w_new, _ in results[: self.stage2_sketch]:
                curr.append(self._flatten_update(self.weights, w_new))

            if len(curr) > 0:
                stage2_ran = True
                Verifier, thresh_cfg = self._verifier()
                ok, stat = Verifier(self.baseline_updates, thresh_cfg if thresh_cfg is not None else 0.0).confirm(np.stack(curr))
                stage2_stat = float(stat)

                # Dual-threshold policy
                if self.stage2_method == "kl" and self.auto_thr_global is not None:
                    thr_low_used, thr_high_used = self.auto_thr_global
                elif self.stage2_method == "kl" and thresh_cfg is not None:
                    # no auto: treat cfg threshold as "high", and set low=0.9*high
                    thr_high_used = float(thresh_cfg)
                    thr_low_used = 0.9 * float(thresh_cfg)
                else:
                    # ED path: single threshold
                    thr_high_used = float(self.stage2_ed_thresh)
                    thr_low_used = 0.9 * float(self.stage2_ed_thresh)

                is_high = stage2_stat > thr_high_used
                is_low = stage2_stat > thr_low_used
                self.low_hits_global.append(1 if is_low else 0)
                low_hits_in_window = int(sum(self.low_hits_global))

                if self.policy_mode == "1x_high_or_2x_low":
                    ok_policy = is_high or (low_hits_in_window >= 2)
                else:
                    ok_policy = is_high

                if ok_policy and self.confirm_cooldown_ctr <= 0:
                    confirmed = True
                    self.confirm_cooldown_ctr = self.confirm_cooldown

        # cooldown bookkeeping
        if self.cooldown_ctr > 0:
            self.cooldown_ctr -= 1
        if self.confirm_cooldown_ctr > 0:
            self.confirm_cooldown_ctr -= 1

        p0_est_out = float(p0_est_internal) if theta is not None else None

        # ---- PER-COHORT ----
        alerts_c = {c: 0 for c in self.cohort_ids}
        m_c = {c: 0 for c in self.cohort_ids}
        by_cohort_results = {c: [] for c in self.cohort_ids}
        for cid, w_new, m in results:
            c = int(m.get("cohort", -1))
            if c in alerts_c:
                alerts_c[c] += int(m.get("a_bit", 0))
                m_c[c] += 1
                by_cohort_results[c].append((cid, w_new, m))

        trigger_cohorts, confirmed_cohorts = [], []
        for c in self.cohort_ids:
            raw_c, theta_c, p0_c = self.triggers_c[c].update(alerts_c[c], max(1, m_c[c]))
            trig_c = False
            ran_c = False
            conf_c = False
            stat_c = None
            low_hits_c = None
            thr_low_c = None
            thr_high_c = None

            can_eval_c = (m_c[c] >= max(1, self.min_part_cohort))
            if raw_c and can_eval_c and self.cooldown_c[c] <= 0 and self.round > warmup_B and len(self.baseline_c[c]) > 0 and m_c[c] > 0:
                trig_c = True
                self.cooldown_c[c] = self.cooldown

                curr_c = []
                for _, w_new, _ in by_cohort_results[c][: self.stage2_sketch]:
                    curr_c.append(self._flatten_update(self.weights, w_new))

                if len(curr_c) > 0:
                    ran_c = True
                    Verifier, thresh_cfg = self._verifier()
                    ok_c, stat_val = Verifier(self.baseline_c[c], thresh_cfg if thresh_cfg is not None else 0.0).confirm(np.stack(curr_c))
                    stat_c = float(stat_val)

                    # thresholds (per-cohort auto if available)
                    auto_thr = self.auto_thr_c.get(c)
                    if self.stage2_method == "kl" and auto_thr is not None:
                        thr_low_c, thr_high_c = auto_thr
                    elif self.stage2_method == "kl" and thresh_cfg is not None:
                        thr_high_c = float(thresh_cfg); thr_low_c = 0.9 * float(thresh_cfg)
                    else:
                        thr_high_c = float(self.stage2_ed_thresh); thr_low_c = 0.9 * float(self.stage2_ed_thresh)

                    is_high_c = stat_c > thr_high_c
                    is_low_c = stat_c > thr_low_c
                    self.low_hits_c[c].append(1 if is_low_c else 0)
                    low_hits_c = int(sum(self.low_hits_c[c]))

                    if self.policy_mode == "1x_high_or_2x_low":
                        ok_policy_c = is_high_c or (low_hits_c >= 2)
                    else:
                        ok_policy_c = is_high_c

                    if ok_policy_c and self.confirm_cooldown_c[c] <= 0:
                        conf_c = True
                        self.confirm_cooldown_c[c] = self.confirm_cooldown

            if self.cooldown_c[c] > 0:
                self.cooldown_c[c] -= 1
            if self.confirm_cooldown_c[c] > 0:
                self.confirm_cooldown_c[c] -= 1

            # per-cohort log
            rec_c = {
                "round": self.round,
                "cohort": int(c),
                "m_c": int(m_c[c]),
                "alert_sum_c": int(alerts_c[c]),
                "rhat_c": (alerts_c[c] / max(1, m_c[c])) if m_c[c] > 0 else 0.0,
                "raw_trigger_c": bool(raw_c),
                "trigger_c": bool(trig_c),
                "theta_c": float(theta_c) if theta_c is not None else None,
                "p0_est_c": float(p0_c) if (theta_c is not None and p0_c is not None) else None,
                "stage2_ran_c": bool(ran_c),
                "stage2_stat_c": stat_c,
                "confirmed_c": bool(conf_c),
                "thr_low_c": thr_low_c,
                "thr_high_c": thr_high_c,
                "low_hits_c": low_hits_c,
                "can_eval_c": bool(can_eval_c),
            }
            self.log_coh_fp.write(json.dumps(rec_c) + "\n")

            if trig_c:
                trigger_cohorts.append(int(c))
            if conf_c:
                confirmed_cohorts.append(int(c))

        # global log
        rec = {
            "round": self.round,
            "m_part": m_part,
            "alert_sum": alert_sum,
            "rhat": alert_sum / max(1, m_part),
            "trigger": bool(fired),
            "raw_trigger": bool(raw_trigger),
            "stage2_ran": bool(stage2_ran),
            "stage2_stat": stage2_stat,
            "confirmed": bool(confirmed),
            "theta": float(theta) if theta is not None else None,
            "p0_est": float(p0_est_internal) if theta is not None else None,
            "stage2_method": self.stage2_method,
            "can_eval_global": bool(m_part >= max(1, self.min_part_global)),
            "thr_low": thr_low_used,
            "thr_high": thr_high_used,
            "low_hits_window": low_hits_in_window,
            "policy_mode": self.policy_mode,
            "trigger_cohorts": trigger_cohorts,
            "confirmed_cohorts": confirmed_cohorts,
        }
        self.log_fp.write(json.dumps(rec) + "\n")
        self.log_fp.flush()
        self.log_coh_fp.flush()