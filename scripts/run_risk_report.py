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


def parse_portfolio(s: str) -> dict[str, float]:
    """
    Parse portfolio string like: "SPY=0.5,QQQ=0.3,TLT=0.2"
    """
    weights: dict[str, float] = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid portfolio item '{item}'. Expected TICKER=WEIGHT.")
        ticker, w = item.split("=", 1)
        ticker = ticker.strip()
        w = w.strip()
        if not ticker:
            raise ValueError(f"Invalid portfolio item '{item}': empty ticker.")
        weights[ticker] = float(w)
    if not weights:
        raise ValueError("Parsed empty portfolio weights.")
    return weights


def _normalize_weights(weights: dict[str, float]) -> tuple[dict[str, float], bool]:
    """
    Normalize weights to sum to 1 if needed.
    Returns (weights_normed, did_normalize).
    """
    s = float(sum(weights.values()))
    if s == 0.0:
        raise ValueError("Portfolio weights sum to 0; cannot normalize.")
    did = not np.isclose(s, 1.0, atol=1e-8)
    if did:
        weights = {k: v / s for k, v in weights.items()}
    return weights, did


def load_returns_from_csv(path: Path) -> pd.Series:
    """
    Single series CSV expected columns:
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


def load_returns_wide_from_csv(path: Path) -> pd.DataFrame:
    """
    Wide returns CSV expected columns:
      - date (or Date)
      - then one column per ticker, containing returns
    Example:
      date,SPY,QQQ,TLT
      2020-01-02,0.001,-0.002,0.0005
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols:
        raise ValueError("Wide returns CSV must contain a date column (case-insensitive).")

    date_col = cols["date"]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    # everything except date becomes asset return columns
    ret_cols = [c for c in df.columns if c != date_col]
    if not ret_cols:
        raise ValueError("Wide returns CSV must contain at least one ticker column besides date.")

    returns = df[ret_cols].astype(float)
    returns.index.name = "date"
    return returns


def detect_csv_mode(path: Path) -> str:
    """
    Return "series" if (date, return) exists; else "wide" if date + >=1 other column;
    """
    df = pd.read_csv(path, nrows=5)
    cols_lower = {c.lower() for c in df.columns}
    if "date" in cols_lower and "return" in cols_lower:
        return "series"
    if "date" in cols_lower:
        return "wide"
    raise ValueError("CSV must have a 'date' column. For series mode also include 'return'.")


def make_demo_returns(n: int = 800, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    r = pd.Series(rng.normal(0.0002, 0.01, size=n), index=idx, name="return")
    return r


def make_demo_returns_wide(tickers: list[str], n: int = 800, seed: int = 0) -> pd.DataFrame:
    """
    Create correlated demo returns for multiple tickers.
    """
    rng = np.random.default_rng(seed)
    k = len(tickers)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")

    # random positive-definite covariance via A A^T, then scale to typical daily vols (~1%)
    A = rng.normal(size=(k, k))
    cov = A @ A.T
    # normalize to have reasonable vol scale
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    vols = rng.uniform(0.008, 0.015, size=k)  # 0.8% to 1.5% daily vol
    cov_scaled = corr * np.outer(vols, vols)

    mu = np.full(k, 0.0002)  # small positive drift
    x = rng.multivariate_normal(mean=mu, cov=cov_scaled, size=n)

    df = pd.DataFrame(x, index=idx, columns=tickers)
    df.index.name = "date"
    return df


def portfolio_return_from_wide(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    missing = [t for t in weights.keys() if t not in returns.columns]
    if missing:
        raise ValueError(
            f"Portfolio tickers not found in returns data: {missing}. "
            f"Available columns: {list(returns.columns)}"
        )
    # keep only tickers in weights, preserve order
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers], dtype=float)
    r = returns[tickers].to_numpy(dtype=float)
    port = r @ w
    out = pd.Series(port, index=returns.index, name="return")
    return out


def main():
    p = argparse.ArgumentParser(
        description="Run end-to-end Quant Risk report (VaR/ES + backtest + Basel + stress)."
    )
    p.add_argument("--csv", type=str, default="", help="Path to CSV with returns. Either (date,return) or wide (date,ticker1,ticker2,...)")
    p.add_argument("--alpha", type=float, default=0.99, help="Confidence level (e.g., 0.99)")
    p.add_argument("--window", type=int, default=252, help="Rolling window for VaR/ES")
    p.add_argument("--var_model", type=str, default="HS", choices=["HS", "GAUSSIAN", "EWMA"], help="VaR model")
    p.add_argument("--es_model", type=str, default="HS", choices=["HS", "GAUSSIAN", "EWMA"], help="ES model")
    p.add_argument("--basel_window", type=int, default=250, help="Basel traffic light window length")
    p.add_argument("--seed", type=int, default=0, help="Seed for demo data (if no CSV)")
    p.add_argument("--portfolio", type=str, default=None, help="Portfolio weights, e.g. SPY=0.5,QQQ=0.3,TLT=0.2")

    args = p.parse_args()

    weights = None
    did_norm = False
    if args.portfolio is not None:
        weights = parse_portfolio(args.portfolio)
        weights, did_norm = _normalize_weights(weights)

    # Build the return series that will be fed into the risk engine
    if args.csv:
        csv_path = Path(args.csv)
        mode = detect_csv_mode(csv_path)

        if mode == "series":
            if weights is not None:
                raise ValueError(
                    "You provided --portfolio but CSV is (date,return) series mode. "
                    "If you want to build a portfolio from weights, provide a wide returns CSV: date,SPY,QQQ,TLT,..."
                )
            returns = load_returns_from_csv(csv_path)

        else:  # wide
            wide = load_returns_wide_from_csv(csv_path)
            if weights is None:
                raise ValueError(
                    "Wide returns CSV detected (date + tickers columns), but you didn't provide --portfolio. "
                    "Provide --portfolio like 'SPY=0.5,QQQ=0.3,TLT=0.2'."
                )
            returns = portfolio_return_from_wide(wide, weights)

    else:
        # demo mode
        if weights is None:
            returns = make_demo_returns(seed=args.seed)
        else:
            tickers = list(weights.keys())
            wide = make_demo_returns_wide(tickers, seed=args.seed)
            returns = portfolio_return_from_wide(wide, weights)

    if weights is not None:
        print("\n=== Portfolio ===")
        if did_norm:
            print("Note: weights were normalized to sum to 1.")
        for k, v in weights.items():
            print(f"{k}: {v:.6f}")
        print(f"Sum: {sum(weights.values()):.6f}")

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
