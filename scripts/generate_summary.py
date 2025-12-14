import os
import json
import glob
import datetime
import re
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RESULTS_DIR = "/home/denis/projects/DSSFN-BCA/results/configurations"
OUTPUT_FILE = "/home/denis/projects/DSSFN-BCA/results/COMPREHENSIVE_RESULTS_SUMMARY.md"

# Metrics where "higher is better" (others default to higher unless listed in COST_METRICS)
BENEFIT_METRICS = {
    "OA",
    "AA",
    "Kappa",
}

# Metrics where "lower is better"
COST_METRICS = {
    "TrainTime",
    "InferenceMs",
    "FLOPs",
}


@dataclass(frozen=True)
class ResultKey:
    config_id: str
    dataset: str
    train_ratio: float
    val_ratio: float


def _safe_float_from_token(token: str) -> Optional[float]:
    """Converts tokens like '0p05' or '0.05' to float."""
    try:
        return float(token.replace("p", "."))
    except Exception:
        return None


def _parse_results_filename(filename: str) -> Optional[Tuple[str, float, float, datetime.datetime]]:
    """Parses `results_{dataset}_tr0p05_val0p15_YYYYMMDD_HHMMSS.json`."""
    m = re.match(
        r"^results_(?P<dataset>.+?)_tr(?P<tr>[0-9p.]+)_val(?P<val>[0-9p.]+)_(?P<ts>\d{8}_\d{6})\.json$",
        filename,
    )
    if not m:
        return None
    dataset = m.group("dataset")
    tr = _safe_float_from_token(m.group("tr"))
    vr = _safe_float_from_token(m.group("val"))
    if tr is None or vr is None:
        return None
    ts = datetime.datetime.strptime(m.group("ts"), "%Y%m%d_%H%M%S")
    return dataset, tr, vr, ts


def _pretty_model_name(config_id: str) -> str:
    # Keep your existing canonical names where possible.
    mapping = {
        "base": "DSSFN (Base)",
        "bca": "DSSFN-BCA",
        "bca_sc_optimized": "DSSFN-BCA (Optimized)",
        "adaptive_sc_optimized": "Adaptive DSSFN (SC)",
        "adaptive_mc_optimized": "Adaptive DSSFN (MC)",
        "adaptive": "Adaptive DSSFN",
    }
    return mapping.get(config_id, config_id)


def _discover_latest_jsons() -> Dict[ResultKey, str]:
    """Find latest evaluation JSON per (config, dataset, train_ratio, val_ratio)."""
    pattern = os.path.join(RESULTS_DIR, "*", "evaluation", "results_*.json")
    candidates = glob.glob(pattern)
    latest: Dict[ResultKey, Tuple[datetime.datetime, str]] = {}
    for path in candidates:
        fname = os.path.basename(path)
        parsed = _parse_results_filename(fname)
        if not parsed:
            continue
        dataset, tr, vr, ts = parsed
        # config_id = .../results/configurations/{config_id}/evaluation/...
        config_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        key = ResultKey(config_id=config_id, dataset=dataset, train_ratio=tr, val_ratio=vr)
        prev = latest.get(key)
        if prev is None or ts > prev[0]:
            latest[key] = (ts, path)

    return {k: v[1] for k, v in latest.items()}

def get_log_file(json_file):
    # Convert .../evaluation/results_X.json to .../logs/eval_X.log
    # Note: json filename is results_Dataset_trRatio_valRatio_Timestamp.json
    # Log filename is eval_Dataset_trRatio_valRatio_Timestamp.log
    
    dir_name = os.path.dirname(json_file) # .../evaluation
    base_name = os.path.basename(json_file) # results_...
    
    # Replace 'evaluation' with 'logs' in dir path
    log_dir = dir_name.replace('evaluation', 'logs')
    
    # Replace 'results_' with 'eval_' and '.json' with '.log' in filename
    log_name = base_name.replace('results_', 'eval_').replace('.json', '.log')
    
    log_path = os.path.join(log_dir, log_name)
    return log_path

def get_bands_from_log(log_path):
    if not os.path.exists(log_path):
        return "N/A"
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            
        # 1) E-FDPC explicit message
        match = re.search(r"Automatically selected (\d+) bands", content)
        if match:
            return int(match.group(1))

        # 2) Band count in padded shape line, e.g. "New shape: (159, 159, 13)"
        shape_matches = re.findall(r"New shape: \(\d+, \d+, (\d+)\)", content)
        if shape_matches:
            return int(shape_matches[-1])

        # 3) Band count in patch creation line, e.g. "15x15x13."
        patch_matches = re.findall(r"patches of size \d+x\d+x(\d+)\.", content)
        if patch_matches:
            return int(patch_matches[-1])

        return "N/A"
    except Exception:
        return "N/A"


def get_band_selection_from_log(log_path: str) -> str:
    """Best-effort inference of band selection method from evaluation logs."""
    if not os.path.exists(log_path):
        return "N/A"
    try:
        with open(log_path, "r") as f:
            content = f.read()
        if "Starting E-FDPC Band Selection" in content:
            return "E-FDPC"
        if "SWGMF" in content:
            return "SWGMF"
        return "N/A"
    except Exception:
        return "N/A"

def _json_compact(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float, str, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _as_number(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _format_metric(metric_name: str, value: Any) -> str:
    if value is None:
        return "N/A"
    if metric_name == "FLOPs":
        n = _as_number(value)
        if n is None:
            return _json_compact(value)
        return f"{int(n)} ({n / 1e6:.1f}M)"
    if metric_name in {"TrainTime"}:
        n = _as_number(value)
        if n is None:
            return _json_compact(value)
        return f"{n:.2f}"
    if metric_name in {"InferenceMs"}:
        n = _as_number(value)
        if n is None:
            return _json_compact(value)
        return f"{n:.3f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return _json_compact(value)


def _extract_input_bands(filepath: str, params: Dict[str, Any]) -> Any:
    # Prefer explicit params
    if "input_bands" in params:
        return params.get("input_bands")
    if "swgmf_bands" in params and params.get("band_selection") == "SWGMF":
        return params.get("swgmf_bands")

    # For E-FDPC, parse log output
    log_path = get_log_file(filepath)
    bands = get_bands_from_log(log_path)
    return bands


def parse_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r") as f:
        data = json.load(f)

    metrics = data.get("metrics", {}) or {}
    params = data.get("params", {}) or {}

    # Add derived/implicit hyperparams that are critical for comparison
    params = dict(params)
    params.setdefault("input_bands", _extract_input_bands(filepath, params))

    # If band_selection wasn't saved in params (some optimized configs), infer from logs.
    if not params.get("band_selection"):
        params.setdefault("band_selection_effective", get_band_selection_from_log(get_log_file(filepath)))

    return {
        "metrics": metrics,
        "params": params,
        "raw": data,
    }


def _metric_is_cost(metric_name: str) -> bool:
    return metric_name in COST_METRICS


def _metric_is_benefit(metric_name: str) -> bool:
    return metric_name in BENEFIT_METRICS


def _best_value(values: List[Optional[float]], is_cost: bool) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return min(nums) if is_cost else max(nums)


def _is_best(value: Optional[float], best: Optional[float]) -> bool:
    if value is None or best is None:
        return False
    return math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-12)

def generate_markdown():
    latest = _discover_latest_jsons()

    # Organize selected results
    records: List[Dict[str, Any]] = []
    for key, path in sorted(latest.items(), key=lambda kv: (kv[0].train_ratio, kv[0].val_ratio, kv[0].dataset, kv[0].config_id)):
        parsed = parse_json(path)
        rel_source = os.path.relpath(path, start=os.path.dirname(OUTPUT_FILE))
        records.append(
            {
                "config_id": key.config_id,
                "model": _pretty_model_name(key.config_id),
                "dataset": key.dataset,
                "train_ratio": key.train_ratio,
                "val_ratio": key.val_ratio,
                "metrics": parsed["metrics"],
                "params": parsed["params"],
                "source": rel_source.replace(os.sep, "/"),
            }
        )

    # Group by (train_ratio, val_ratio)
    groups: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["train_ratio"], r["val_ratio"]), []).append(r)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md: List[str] = []
    md.append("# Comprehensive DSSFN Model Comparison Results\n")
    md.append(f"**Last Updated:** {now}\n")
    md.append("## Experiment Summary\n")
    md.append(
        "This document is auto-generated from the latest evaluation JSON per (configuration, dataset, train ratio, val ratio) found in `results/configurations/*/evaluation/`.\n"
    )
    md.append(
        "For each dataset and training ratio, the best value per metric column is highlighted in **bold** (maximized for accuracy metrics, minimized for cost metrics like time/FLOPs).\n"
    )

    for (train_ratio, val_ratio), group_records in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        md.append(f"## Results (Train: {train_ratio:.2f}, Val: {val_ratio:.2f})\n")

        # Per dataset tables
        datasets = sorted({r["dataset"] for r in group_records})
        for dataset in datasets:
            ds_rows = [r for r in group_records if r["dataset"] == dataset]
            ds_rows.sort(key=lambda r: r["model"])

            # Collect all metric/param keys for this dataset
            metric_keys = sorted({k for r in ds_rows for k in (r["metrics"] or {}).keys()})
            param_keys = sorted({k for r in ds_rows for k in (r["params"] or {}).keys()})

            md.append(f"### {dataset}\n")

            # ---- Metrics table (with bolding) ----
            md.append("#### Metrics\n")
            header = ["Model"] + metric_keys + ["Source"]
            md.append("| " + " | ".join(header) + " |")
            md.append("|" + "|".join(["---"] * len(header)) + "|")

            # Pre-compute best per metric column
            best_by_metric: Dict[str, Optional[float]] = {}
            for mk in metric_keys:
                vals = [_as_number((r["metrics"] or {}).get(mk)) for r in ds_rows]
                is_cost = _metric_is_cost(mk)
                # If explicitly benefit, override cost
                if _metric_is_benefit(mk):
                    is_cost = False
                best_by_metric[mk] = _best_value(vals, is_cost=is_cost)

            for r in ds_rows:
                row = [r["model"]]
                for mk in metric_keys:
                    raw_val = (r["metrics"] or {}).get(mk)
                    num_val = _as_number(raw_val)
                    cell = _format_metric(mk, raw_val)
                    if _is_best(num_val, best_by_metric.get(mk)):
                        cell = f"**{cell}**"
                    row.append(cell)
                row.append(f"[{os.path.basename(r['source'])}]({r['source']})")
                md.append("| " + " | ".join(row) + " |")

            # ---- Hyperparameters table ----
            md.append("\n#### Hyperparameters\n")
            p_header = ["Model"] + param_keys + ["Source"]
            md.append("| " + " | ".join(p_header) + " |")
            md.append("|" + "|".join(["---"] * len(p_header)) + "|")
            for r in ds_rows:
                prow = [r["model"]]
                for pk in param_keys:
                    prow.append(_json_compact((r["params"] or {}).get(pk)))
                prow.append(f"[{os.path.basename(r['source'])}]({r['source']})")
                md.append("| " + " | ".join(prow) + " |")
            md.append("")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(md).rstrip() + "\n")
    print(f"Summary generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_markdown()
