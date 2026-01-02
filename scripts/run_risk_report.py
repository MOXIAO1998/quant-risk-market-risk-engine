from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# allow running from repo root: python scripts/run_risk_report.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from risk_engine.report import generate_risk_report  # noqa: E402


def load_returns_from_csv(path: Path) -> pd.Series:
    """
    CSV expected columns:
      - date (or Date)
      - return (or Return)
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "return" not in cols:
        raise ValueError("CSV must contain columns: date, return (case-insensitive)")

    df[cols["date"]] = pd.to_datetime(df[cols["date"]])
    df = df.sort_values(cols["date"]).set_index(cols["date"])
    r = df[cols["return"]].astype(float)
    r.name = "return"
    return r


def make_demo_returns(n: int = 800, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    r = pd.Series(rng.normal(0.0002, 0.01, size=n), index=idx, name="return")
    return r


def main():
    p = argparse.ArgumentParser(description="Run end-to-end Quant Risk report (VaR/ES + backtest + Basel + stress).")
    p.add_argument("--csv", type=str, default="", help="Path to CSV with columns date, return (optional)")
    p.add_argument("--alpha", type=float, default=0.99, help="Confidence level (e.g., 0.99)")
    p.add_argument("--window", type=int, default=252, help="Rolling window for VaR/ES")
    p.add_argument("--var_model", type=str, default="HS", choices=["HS", "GAUSSIAN", "EWMA"], help="VaR model")
    p.add_argument("--es_model", type=str, default="HS", choices=["HS", "GAUSSIAN", "EWMA"], help="ES model")
    p.add_argument("--basel_window", type=int, default=250, help="Basel traffic light window length")
    p.add_argument("--seed", type=int, default=0, help="Seed for demo data (if no CSV)")

    args = p.parse_args()

    if args.csv:
        returns = load_returns_from_csv(Path(args.csv))
    else:
        returns = make_demo_returns(seed=args.seed)

    report = generate_risk_report(
        returns,
        alpha=args.alpha,
        window=args.window,
        var_model=args.var_model,
        es_model=args.es_model,
        basel_window=args.basel_window,
        loss_positive=True,
        historical_stress_window=None,  # you can set later with real dates
    )

    s = report.summary()
    print("\n=== Risk Report Summary ===")
    print(f"Model: VaR={report.var_model}, ES={report.es_model} | alpha={report.alpha} | window={report.window}")
    print(f"Latest VaR (loss+): {s['VaR']:.6f}")
    print(f"Latest ES  (loss+): {s['ES']:.6f}")
    print("\n--- Backtest (VaR) ---")
    print(f"Evaluated days: {s['n_eval']}")
    print(f"Violations:     {s['n_viol']}  (rate={s['viol_rate']:.4f}, expected={s['expected_rate']:.4f})")
    print(f"Kupiec LR:      {s['kupiec_lr']:.6f}  p={s['kupiec_pvalue']}")
    print(f"Christoff LR:   {s['christoffersen_lr_ind']:.6f}  p={s['christoffersen_pvalue_ind']}")
    print(f"CC LR:          {s['lr_cc']:.6f}  p={s['pvalue_cc']}")
    print(f"Basel Light:    {s['basel_traffic_light']}")

    print("\n--- Stress Summary ---")
    print(report.stress_table)

if __name__ == "__main__":
    main()
