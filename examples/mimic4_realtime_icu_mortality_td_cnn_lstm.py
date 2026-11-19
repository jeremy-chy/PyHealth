# Authors: Hanyang Chen (hc81), Siqi Song (siqis3)
# Paper: Frost, Li & Harris. Robust Real-Time Mortality Prediction in the Intensive
#   Care Unit using Temporal Difference Learning. ML4H 2024, PMLR 259:350-363.
# Paper link: https://proceedings.mlr.press/v259/frost25a.html
"""Ablation study: TD vs supervised CNN-LSTM for real-time ICU mortality.

This script runs the full pipeline of the PyHealth reproduction of Frost et al.
(ML4H 2024):

    MIMIC4EHRDataset -> RealTimeICUMortalityMIMIC4 (task) -> TDCNNLSTM (model)
    -> pyhealth.trainer.Trainer -> AUROC at 1/3/7/14/28 days, AUPRC, Brier, ECE

and an ablation study over model and task settings.

Data (any of):
    * ``--synthetic`` (default when no ``--root``): writes a small synthetic
      MIMIC-IV-like folder (latent-severity patients) to a temporary directory.
      Runs anywhere, no credentials needed.
    * ``--root /path/to/mimic-iv-demo/2.2``: the open MIMIC-IV demo (100
      patients; https://physionet.org/content/mimic-iv-demo/2.2/).
    * ``--root /path/to/mimiciv/3.0``: full MIMIC-IV (credentialed). Use
      ``--state-sample-prob`` (e.g. 0.05) to keep the sample cache manageable.

Ablations (``--ablations``, comma-separated; all by default):
    * ``target``: TD vs supervised baselines (28-day and 1-day labels, both
      class-balanced), the paper's main comparison (Tables 2-3).
    * ``tau``: target-network soft-update rate tau in {0.005, 0.01, 0.05}
      (the paper fixes tau = 0.01, i.e. alpha = 0.99). Model hyperparameter.
    * ``lambda``: our extension, targets ``lambda * label + (1 - lambda) *
      TD target`` for lambda in {0.0, 0.5}, interpolating TD and supervision.
    * ``hidden``: embedding size in {16, 32} (LSTM is 8x). Model
      hyperparameter.
    * ``delay``: next-state delay in {4, 24, 48} hours (paper Fig. 3). Task
      configuration.
    * ``features``: labs + drugs vs labs only. Task configuration.

Setup: patients are split 60/20/20 (train/val/test) with ``PatientSplit``;
the processors and the model's standardisation statistics are fitted on the
training split only. Every model uses the authors' CNN-LSTM, AdamW with the
paper's learning rate 1/sqrt(n_params), batch size 64, and the epoch with
the best validation AUROC on the ``mortality`` label. Test metrics are
computed on every test state marker, against each horizon label.

Usage:
    python examples/mimic4_realtime_icu_mortality_td_cnn_lstm.py --synthetic
    python examples/mimic4_realtime_icu_mortality_td_cnn_lstm.py \\
        --root /path/to/mimic-iv-demo/2.2 --ablations target,tau

Results (``--synthetic --seeds 0,1,2 --epochs 8``; 400 synthetic patients,
mean over 3 seeds; test-split AUROC by horizon, plus 28-day AUPRC, Brier and
ECE; "+" = class-balanced):

    ============== ===== ===== ===== ===== ===== ====== ===== =====
    config          1d    3d    7d    14d   28d  AUPRC  Brier  ECE
    ============== ===== ===== ===== ===== ===== ====== ===== =====
    TD (tau=0.01)  0.921 0.884 0.900 0.891 0.909 0.908  0.128 0.071
    supervised 28d+ 0.918 0.887 0.902 0.891 0.908 0.908 0.139 0.108
    supervised 1d+ 0.937 0.874 0.876 0.858 0.870 0.888  0.214 0.216
    TD tau=0.005   0.932 0.888 0.898 0.888 0.906 0.907  0.136 0.089
    TD tau=0.05    0.930 0.893 0.899 0.889 0.906 0.906  0.131 0.079
    TD lambda=0.5  0.935 0.887 0.898 0.889 0.906 0.910  0.135 0.089
    TD embedding=16 0.929 0.897 0.901 0.891 0.908 0.907 0.136 0.096
    TD delay=4h    0.916 0.856 0.867 0.857 0.875 0.877  0.166 0.126
    TD delay=48h   0.923 0.888 0.898 0.888 0.905 0.906  0.141 0.115
    TD labs only   0.944 0.889 0.882 0.874 0.888 0.841  0.142 0.096
    ============== ===== ===== ===== ===== ===== ====== ===== =====

Findings on synthetic data (they check the pipeline, not the paper's claims,
which need full MIMIC-IV):
    * As in the paper, the 1-day supervised model is best at 1 day but worst
      at 7-28 days; TD matches the 28-day supervised model in AUROC at long
      horizons and is better calibrated (ECE 0.071 vs 0.108).
    * A short next-state delay (4 h) hurts every horizon (28-day AUROC 0.875
      vs 0.909), in line with the paper's delay sweep (Fig. 3); 48 h is close
      to 24 h.
    * TD is insensitive to the target-update rate in [0.005, 0.05], to
      lambda = 0.5 mixing, and to halving the embedding size.
    * Dropping the drug features costs about 0.02 AUROC at 28 days.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from pyhealth.datasets import MIMIC4EHRDataset, PatientSplit, get_dataloader
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import TDCNNLSTM
from pyhealth.tasks import RealTimeICUMortalityMIMIC4
from pyhealth.trainer import Trainer

HORIZONS = (1, 3, 7, 14, 28)
FMT = "%Y-%m-%d %H:%M:%S"


# ----------------------------------------------------------------------
# Synthetic MIMIC-IV-like data
# ----------------------------------------------------------------------
def _write_csv_gz(path: str, header: List[str], rows: List[List[Any]]) -> None:
    """Writes rows to a gzipped CSV file (empty values for ``None``)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join("" if v is None else str(v) for v in row) + "\n")


def make_synthetic_mimic4(root: str, n_patients: int = 400, seed: int = 0) -> str:
    """Creates a small synthetic MIMIC-IV-like dataset.

    Each patient has a latent severity that follows a random walk with a
    patient-specific drift. Labs (lactate, creatinine, WBC, sodium) are noisy
    functions of severity, noradrenaline is infused while severity is high,
    and the patient dies when severity crosses a threshold, either in the ICU
    (``deathtime``) or within 28 days after discharge (``dod``). Distant
    outcomes therefore depend on the future trajectory, the setting the paper
    targets.

    Args:
        root: Output folder (``hosp/`` and ``icu/`` are created inside).
        n_patients: Number of patients.
        seed: Random seed.

    Returns:
        ``root``.
    """
    rng = np.random.default_rng(seed)
    patients, admissions, stays, charts, inputs = [], [], [], [], []
    base = datetime(2150, 1, 1)
    for i in range(n_patients):
        sid, hadm, stay = 10_000_000 + i, 20_000_000 + i, 30_000_000 + i
        age = int(rng.integers(20, 90))
        gender = "F" if rng.random() < 0.45 else "M"
        start = base + timedelta(days=int(rng.integers(0, 300)), hours=8)
        drift = rng.normal(0.0, 0.03)
        severity = rng.normal(0.0, 0.8) + 0.02 * (age - 60)
        planned_los = 24.0 * min(1.0 + rng.exponential(3.0), 14.0)
        weight = float(rng.normal(80.0, 15.0))
        death, hour, rate_on = None, 0.0, 0.0
        while hour < planned_los + 28 * 24:
            severity += drift + rng.normal(0.0, 0.12)
            if severity > 3.0:
                death = start + timedelta(hours=hour)
                break
            in_icu = hour < planned_los
            if in_icu and rng.random() < 0.3:  # measurements every ~3 h
                when = (start + timedelta(hours=hour)).strftime(FMT)
                for itemid, value in (
                    (225668, math.exp(0.4 + 0.35 * severity + rng.normal(0, 0.2))),
                    (220615, max(0.3, 1.0 + 0.4 * severity + rng.normal(0, 0.2))),
                    (220546, max(1.0, 9.0 + 2.5 * severity + rng.normal(0, 1.5))),
                    (220645, 140.0 + rng.normal(0, 3.0)),
                ):
                    if rng.random() < 0.8:
                        charts.append([sid, hadm, stay, when, itemid, round(value, 3)])
            if in_icu and abs(hour % 2) < 1e-9:  # infusion decisions every 2 h
                target_rate = 0.02 * max(severity - 1.0, 0.0)
                if target_rate > 0 or rate_on > 0:
                    begin = start + timedelta(hours=hour)
                    end = begin + timedelta(hours=2)
                    if target_rate > 0:
                        inputs.append(
                            [
                                sid,
                                hadm,
                                stay,
                                begin.strftime(FMT),
                                end.strftime(FMT),
                                221906,
                                round(target_rate * 120, 4),
                                "mg",
                                round(weight, 1),
                                "Continuous Med",
                            ]
                        )
                    rate_on = target_rate
            if in_icu and rng.random() < 0.02:  # occasional antibiotic dose
                begin = start + timedelta(hours=hour)
                inputs.append(
                    [
                        sid,
                        hadm,
                        stay,
                        begin.strftime(FMT),
                        (begin + timedelta(minutes=30)).strftime(FMT),
                        225798,
                        1,
                        "dose",
                        round(weight, 1),
                        "Intermittent",
                    ]
                )
            hour += 1.0
        died_in_icu = death is not None and death <= start + timedelta(
            hours=planned_los
        )
        outtime = death if died_in_icu else start + timedelta(hours=planned_los)
        dischtime = outtime + timedelta(days=0 if died_in_icu else 2)
        patients.append(
            [
                sid,
                gender,
                age,
                2150,
                "2017 - 2019",
                death.strftime("%Y-%m-%d") if death is not None else None,
            ]
        )
        admissions.append(
            [
                sid,
                hadm,
                start.strftime(FMT),
                dischtime.strftime(FMT),
                death.strftime(FMT) if died_in_icu else None,
                "EW EMER.",
                "EMERGENCY ROOM",
                "HOME",
                "Medicare",
                "ENGLISH",
                "SINGLE",
                "WHITE",
                int(died_in_icu),
            ]
        )
        stays.append(
            [
                sid,
                hadm,
                stay,
                "MICU",
                "MICU",
                start.strftime(FMT),
                outtime.strftime(FMT),
                round(planned_los / 24, 2),
            ]
        )
    _write_csv_gz(
        f"{root}/hosp/patients.csv.gz",
        [
            "subject_id",
            "gender",
            "anchor_age",
            "anchor_year",
            "anchor_year_group",
            "dod",
        ],
        patients,
    )
    _write_csv_gz(
        f"{root}/hosp/admissions.csv.gz",
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "hospital_expire_flag",
        ],
        admissions,
    )
    _write_csv_gz(
        f"{root}/icu/icustays.csv.gz",
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "first_careunit",
            "last_careunit",
            "intime",
            "outtime",
            "los",
        ],
        stays,
    )
    _write_csv_gz(
        f"{root}/icu/chartevents.csv.gz",
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "charttime",
            "itemid",
            "value",
            "valuenum",
            "valueuom",
        ],
        [[*r[:5], r[5], r[5], ""] for r in charts],
    )
    _write_csv_gz(
        f"{root}/icu/inputevents.csv.gz",
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "starttime",
            "endtime",
            "itemid",
            "amount",
            "amountuom",
            "rate",
            "rateuom",
            "patientweight",
            "orderid",
            "linkorderid",
            "ordercategoryname",
            "statusdescription",
        ],
        [
            [*r[:8], None, None, r[8], k, k, r[9], "FinishedRunning"]
            for k, r in enumerate(inputs)
        ],
    )
    return root


# ----------------------------------------------------------------------
# Experiment helpers
# ----------------------------------------------------------------------
def build_splits(
    root: str, cache_dir: str, task: RealTimeICUMortalityMIMIC4, workers: int
) -> Tuple[Any, Any, Any]:
    """Loads MIMIC-IV, applies the task and splits patients 60/20/20.

    Args:
        root: MIMIC-IV root (with ``hosp/`` and ``icu/``).
        cache_dir: PyHealth cache folder.
        task: The configured task.
        workers: Number of worker processes for ``set_task``.

    Returns:
        ``(train, val, test)`` sample datasets; processors are fitted on
        ``train`` only.
    """
    dataset = MIMIC4EHRDataset(
        root=root,
        tables=["patients", "admissions", "chartevents", "inputevents"],
        cache_dir=cache_dir,
        num_workers=workers,
    )
    return dataset.set_task(
        task, split=PatientSplit(ratios=(0.6, 0.2, 0.2), seed=0), num_workers=workers
    )


def evaluate(trainer: Trainer, loader: Any) -> Dict[str, float]:
    """Computes multi-horizon AUROC plus AUPRC, Brier and ECE (28-day label).

    Args:
        trainer: A trained :class:`~pyhealth.trainer.Trainer`.
        loader: Test dataloader.

    Returns:
        A flat dict of metric name -> value (NaN when a label has one class).
    """
    _, y_prob, _, extra = trainer.inference(
        loader, additional_outputs=["mortality_horizons"]
    )
    horizons = extra["mortality_horizons"]
    if isinstance(horizons, list):
        horizons = np.concatenate(horizons, axis=0)
    y_prob = np.asarray(y_prob).reshape(-1)
    results: Dict[str, float] = {}
    for j, days in enumerate(HORIZONS):
        y = horizons[:, j]
        both = 0 < y.sum() < len(y)
        results[f"auroc_{days}d"] = (
            binary_metrics_fn(y, y_prob, metrics=["roc_auc"])["roc_auc"]
            if both
            else math.nan
        )
    y28 = horizons[:, HORIZONS.index(28)]
    if 0 < y28.sum() < len(y28):
        extra_metrics = binary_metrics_fn(
            y28, y_prob, metrics=["pr_auc", "brier", "ECE"]
        )
        results.update({f"{k.lower()}_28d": v for k, v in extra_metrics.items()})
    results["n_test_states"] = float(len(y_prob))
    return results


def run_config(
    name: str,
    splits: Tuple[Any, Any, Any],
    model_kwargs: Dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, float]:
    """Trains and evaluates one configuration.

    Args:
        name: Configuration name.
        splits: ``(train, val, test)`` sample datasets.
        model_kwargs: Keyword arguments for :class:`TDCNNLSTM`.
        args: Parsed command-line arguments.
        seed: Random seed.

    Returns:
        Test metrics of the best-validation epoch.
    """
    train, val, test = splits
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = TDCNNLSTM(train, seed=seed, **model_kwargs)
    train_loader = get_dataloader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val, batch_size=256)
    test_loader = get_dataloader(test, batch_size=256)
    paper = model.paper_optimizer_params(len(train_loader))
    weight_decay = (
        paper["weight_decay"] if args.paper_weight_decay else args.weight_decay
    )
    trainer = Trainer(
        model=model, metrics=["roc_auc"], device=args.device, enable_logging=False
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": paper["lr"]},
        weight_decay=weight_decay,
        monitor="roc_auc",
        monitor_criterion="max",
    )
    results = evaluate(trainer, test_loader)
    print(
        f"[{name} | seed {seed}] "
        + ", ".join(f"{k}={v:.3f}" for k, v in results.items() if k.startswith("auroc"))
    )
    return results


def ablation_grid(
    selected: List[str],
) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Lists ``(name, task_kwargs, model_kwargs)`` for the chosen ablations."""
    td = {"training_target": "td"}
    grid: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    if "target" in selected:
        grid += [
            ("TD (tau=0.01)", {}, td),
            (
                "supervised 28d+",
                {"label_horizon_days": 28},
                {"training_target": "supervised", "class_balanced": True},
            ),
            (
                "supervised 1d+",
                {"label_horizon_days": 1},
                {"training_target": "supervised", "class_balanced": True},
            ),
        ]
    if "tau" in selected:
        grid += [
            (f"TD tau={t}", {}, {**td, "target_update_rate": t}) for t in (0.005, 0.05)
        ]
    if "lambda" in selected:
        grid += [("TD lambda=0.5", {}, {**td, "td_lambda": 0.5})]
    if "hidden" in selected:
        grid += [("TD embedding=16", {}, {**td, "embedding_dim": 16})]
    if "delay" in selected:
        grid += [(f"TD delay={d}h", {"next_state_delay_hours": d}, td) for d in (4, 48)]
    if "features" in selected:
        grid += [("TD labs only", {"feature_set": "labs"}, td)]
    return grid


def main() -> None:
    """Parses arguments, runs the ablation grid and prints a results table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=None, help="MIMIC-IV root folder")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use generated synthetic data (default if no --root)",
    )
    parser.add_argument("--synthetic-patients", type=int, default=400)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--ablations", default="target,tau,lambda,hidden,delay,features"
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--paper-weight-decay",
        action="store_true",
        help="weight decay = 1/(lr * batches per epoch), as in the "
        "paper; meant for full MIMIC-IV, too strong for small data",
    )
    parser.add_argument("--state-sample-prob", type=float, default=1.0)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default=None, help="optional JSON results file")
    args = parser.parse_args()

    tmp = tempfile.TemporaryDirectory()
    root = args.root
    if root is None or args.synthetic:
        root = make_synthetic_mimic4(
            os.path.join(tmp.name, "mimic4"), n_patients=args.synthetic_patients
        )
        print(f"Synthetic MIMIC-IV-like data written to {root}")
    cache_dir = args.cache_dir or os.path.join(tmp.name, "cache")
    seeds = [int(s) for s in args.seeds.split(",")]

    split_cache: Dict[str, Tuple[Any, Any, Any]] = {}
    results: Dict[str, Dict[str, float]] = {}
    for name, task_kwargs, model_kwargs in ablation_grid(args.ablations.split(",")):
        task = RealTimeICUMortalityMIMIC4(
            state_sample_prob=args.state_sample_prob, **task_kwargs
        )
        key = json.dumps(task_kwargs, sort_keys=True)
        if key not in split_cache:
            split_cache[key] = build_splits(root, cache_dir, task, args.workers)
        runs = [
            run_config(
                name,
                split_cache[key],
                {"num_features": task.num_features, **model_kwargs},
                args,
                s,
            )
            for s in seeds
        ]
        results[name] = {k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]}

    columns = [f"auroc_{d}d" for d in HORIZONS] + ["pr_auc_28d", "brier_28d", "ece_28d"]
    print("\n| config | " + " | ".join(columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    for name, res in results.items():
        cells = [f"{res.get(c, math.nan):.3f}" for c in columns]
        print(f"| {name} | " + " | ".join(cells) + " |")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"args": vars(args), "results": results}, handle, indent=2)
    tmp.cleanup()


if __name__ == "__main__":
    main()
