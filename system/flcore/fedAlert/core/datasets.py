from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
from typing import Dict, Tuple

def make_synthetic_partitions(
    num_clients: int,
    n_features: int = 20,
    samples_per_client: int = 400,
    class_sep: float = 1.0,
    random_state: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    rng = np.random.RandomState(random_state)
    data = {}
    for cid in range(num_clients):
        X, y = make_classification(
            n_samples=samples_per_client,
            n_features=n_features,
            n_informative=int(n_features*0.6),
            n_redundant=int(n_features*0.2),
            n_classes=2,
            class_sep=class_sep,
            flip_y=0.01,
            random_state=rng.randint(0, 10_000),
        )
        data[str(cid)] = {"x": X.astype(np.float32), "y": y.astype(np.int64)}
    return data

def apply_cohort_drift(
    data: Dict[str, Dict[str, np.ndarray]],
    cohorts: Dict[int, list],
    round_idx: int,
    drift_schedule: Dict[int, Dict],
    drift_active_tracker: Dict[int, bool],
    shift_mag_key: str = "feature_shift",
) -> None:
    """
    Apply a mean shift to features for cohorts whose drift starts at this round.
    drift_schedule: { start_round: { 'cohorts': [ids], 'feature_shift': 0.8, 'label_flip_p': 0.0 } }
    """
    if round_idx not in drift_schedule:
        return
    event = drift_schedule[round_idx]
    sh = float(event.get(shift_mag_key, 0.6))
    flip = float(event.get("label_flip_p", 0.0))
    targets = event.get("cohorts", [])
    for c in targets:
        drift_active_tracker[c] = True
    # Apply in-place shift to all clients in target cohorts (we shift *future* batches by toggling a flag)
    for c in targets:
        for cid in cohorts.get(c, []):
            # To simulate drift, we translate the second half of features
            X = data[str(cid)]["x"]
            d = X.shape[1]
            X[:, : d//2] += sh
            data[str(cid)]["x"] = X
            if flip > 0.0:
                y = data[str(cid)]["y"]
                mask = np.random.rand(y.size) < flip
                y[mask] = 1 - y[mask]
                data[str(cid)]["y"] = y
