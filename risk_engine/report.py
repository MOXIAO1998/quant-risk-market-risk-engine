from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal, Any

import numpy as np
import pandas as pd

from .var_models import historical_var, gaussian_var, ewma_var
from .es_models import historical_es, gaussian_es, ewma_es
from .backtest import backtest_var, compute_var_hits, BacktestResult
from .basel import traffic_light
from .stress import apply_parametric_shock, summarize_stress, historical_window_stress, StressResult


VarModel = Literal["HS", "GAUSSIAN", "EWMA"]
EsModel = Literal["HS", "GAUSSIAN", "EWMA"]


def _validate_returns(returns: pd.Series) -> None:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    if returns.isna().any():
        raise ValueError("returns contains NaNs")
    if len(returns) < 5:
        raise ValueError("returns too short for report")


def _pick_var_fn(model: VarModel):
    m = model.upper()
    if m == "HS":
        return historical_var
    if m == "GAUSSIAN":
        return gaussian_var
    if m == "EWMA":
        return ewma_var
    raise ValueError("var_model must be one of: HS, GAUSSIAN, EWMA")


def _pick_es_fn(model: EsModel):
    m = model.upper()
    if m == "HS":
        return historical_es
    if m == "GAUSSIAN":
        return gaussian_es
    if m == "EWMA":
        return ewma_es
    raise ValueError("es_model must be one of: HS, GAUSSIAN, EWMA")


@dataclass(frozen=True)
class RiskReport:
    alpha: float
    window: int
    var_model: str
    es_model: str

    returns: pd.Series
    var: pd.Series
    es: pd.Series

    backtest: BacktestResult
    basel_light: pd.Series  # rolling traffic light over hits
    stress_table: pd.DataFrame
    historical_stress: Optional[StressResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "window": self.window,
            "var_model": self.var_model,
            "es_model": self.es_model,
            "var": self.var.copy(),
            "es": self.es.copy(),
            "backtest": self.backtest,
            "basel_light": self.basel_light.copy(),
            "stress_table": self.stress_table.copy(),
            "historical_stress": self.historical_stress,
        }

    def summary(self) -> Dict[str, Any]:
        """
        Small summary block for printing / logging.
        """
        last_var = self.var.dropna().iloc[-1] if self.var.notna().any() else np.nan
        last_es = self.es.dropna().iloc[-1] if self.es.notna().any() else np.nan
        last_light = self.basel_light.dropna().iloc[-1] if self.basel_light.notna().any() else np.nan

        return {
            "VaR": float(last_var),
            "ES": float(last_es),
            "n_eval": self.backtest.n,
            "n_viol": self.backtest.n_viol,
            "viol_rate": float(self.backtest.viol_rate),
            "expected_rate": float(self.backtest.expected_rate),
            "kupiec_lr": float(self.backtest.kupiec_lr),
            "kupiec_pvalue": self.backtest.kupiec_pvalue,
            "christoffersen_lr_ind": float(self.backtest.christoffersen_lr_ind),
            "christoffersen_pvalue_ind": self.backtest.christoffersen_pvalue_ind,
            "lr_cc": float(self.backtest.lr_cc),
            "pvalue_cc": self.backtest.pvalue_cc,
            "basel_traffic_light": last_light,
        }


def generate_risk_report(
    returns: pd.Series,
    *,
    alpha: float = 0.99,
    window: int = 252,
    var_model: VarModel = "HS",
    es_model: EsModel = "HS",
    basel_window: int = 250,
    loss_positive: bool = True,
    stress_shocks: Optional[Dict[str, Dict[str, Any]]] = None,
    historical_stress_window: Optional[Dict[str, str]] = None,
) -> RiskReport:
    """
    End-to-end risk report generator (single source of truth):
      returns -> VaR/ES -> backtest -> Basel traffic light -> stress summary

    stress_shocks format (optional):
      {
        "eq_-5_add": {"shock": -0.05, "mode": "additive"},
        "eq_-10_mult": {"shock": -0.10, "mode": "multiplicative"},
      }

    historical_stress_window format (optional):
      {"start": "2020-02-15", "end": "2020-04-15", "name": "covid"}
    """
    _validate_returns(returns)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if window < 2:
        raise ValueError("window must be >= 2")

    var_fn = _pick_var_fn(var_model)
    es_fn = _pick_es_fn(es_model)

    var = var_fn(returns, alpha=alpha, window=window, loss_positive=loss_positive)
    es = es_fn(returns, alpha=alpha, window=window, loss_positive=loss_positive)

    bt = backtest_var(returns, var, alpha=alpha, loss_positive=loss_positive)

    hits_df = compute_var_hits(returns, var, loss_positive=loss_positive)
    tl = traffic_light(hits_df["hit"], window=basel_window)

    # Stress scenarios (parametric)
    if stress_shocks is None:
        stress_shocks = {
            "eq_-5_add": {"shock": -0.05, "mode": "additive"},
            "eq_-10_mult": {"shock": -0.10, "mode": "multiplicative"},
        }

    scenarios: Dict[str, pd.Series] = {}
    for name, cfg in stress_shocks.items():
        shock = float(cfg["shock"])
        mode = str(cfg.get("mode", "additive"))
        scenarios[name] = apply_parametric_shock(returns, shock=shock, mode=mode)

    stress_tbl = summarize_stress(returns, scenarios)

    # Optional historical stress window
    hist_res: Optional[StressResult] = None
    if historical_stress_window is not None:
        start = historical_stress_window["start"]
        end = historical_stress_window["end"]
        nm = historical_stress_window.get("name", "historical_window")
        hist_res = historical_window_stress(returns, start=start, end=end, name=nm)

    return RiskReport(
        alpha=float(alpha),
        window=int(window),
        var_model=str(var_model).upper(),
        es_model=str(es_model).upper(),
        returns=returns.copy(),
        var=var,
        es=es,
        backtest=bt,
        basel_light=tl,
        stress_table=stress_tbl,
        historical_stress=hist_res,
    )
