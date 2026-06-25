"""Orchestrator: run all modeling scripts and write combined summary."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modeling.common import RESULTS_DIR, load_all_results

SCRIPTS = [
    ("modeling.run_mlr", "MLR"),
    ("modeling.run_arima", "ARIMA"),
    ("modeling.run_garch", "GARCH"),
    ("modeling.run_mean_reversion", "Mean Reversion"),
    ("modeling.run_gbm", "GBM"),
]


def build_summary_table(all_results: dict) -> list[dict]:
    rows = []
    for model_name, payload in all_results.items():
        for run in payload.get("runs", []):
            symbol = run.get("symbol", "")
            target = run.get("target", "")
            if "garch" in run and "rolling_20d" in run:
                for variant in ("garch", "rolling_20d"):
                    wf = run[variant]["walk_forward"]
                    rows.append({
                        "model": f"{model_name}_{variant}",
                        "symbol": symbol,
                        "target": target,
                        "rmse": wf.get("rmse"),
                        "oos_r2": wf.get("oos_r2"),
                        "qlike": wf.get("qlike"),
                        "accuracy": wf.get("accuracy"),
                    })
            else:
                wf = run.get("walk_forward", {})
                rows.append({
                    "model": model_name,
                    "symbol": symbol,
                    "target": target,
                    "series": run.get("series", ""),
                    "rmse": wf.get("rmse"),
                    "oos_r2": wf.get("oos_r2"),
                    "qlike": wf.get("qlike"),
                    "accuracy": wf.get("accuracy"),
                })
    return rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  Multi-Model Trading Analysis")
    print("=" * 60)

    for module_path, label in SCRIPTS:
        print(f"\n>>> Running {label}...")
        mod = importlib.import_module(module_path)
        mod.main()

    all_results = load_all_results()
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison_table": build_summary_table(all_results),
        "models": list(all_results.keys()),
    }
    out = RESULTS_DIR / "combined_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"  Combined summary → {out}")
    print("=" * 60)
    for row in summary["comparison_table"]:
        r2 = row.get("oos_r2")
        r2s = f"{r2:.4f}" if r2 is not None else "n/a"
        print(f"  {row['model']:20s} {row['symbol']:8s} {row['target']:18s} RMSE={row['rmse']:.6f} R²={r2s}")


if __name__ == "__main__":
    main()
