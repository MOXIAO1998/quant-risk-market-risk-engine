## 📊 Quant Risk Market Risk Engine

An end-to-end market risk engine implementing industry-standard VaR / ES modeling, statistical backtesting, Basel traffic-light, and stress testing, with a one-command reproducible risk report.

## 🚀 Features

Portfolio Returns 
 - Clean return pipeline (no look-ahead)

VaR Models
 - Historical Simulation (HS)
 - Gaussian (Normal)

EWMA
 - Expected Shortfall (ES)
 - Consistent tail-mean definition
 - Unified loss-positive convention

Backtesting
 - Kupiec Proportion of Failures (POF)
 - Christoffersen Independence Test 
 - Conditional Coverage (CC)

Basel Traffic Light
 - Rolling 250-day violation counts 
 - Green / Yellow / Red classification

Stress Testing 
 - Parametric shocks (additive / multiplicative)
 - Historical window stress scenarios

End-to-End Risk Report 
 - One command to generate a full risk summary 
 - Modular, testable, production-style design

## 🧠 Design Principles

Single Source of Truth 
 - Risk models produce VaR / ES once 
 - Backtesting and Basel modules consume model outputs only

No Look-Ahead Bias 
 - All rolling statistics strictly use information available at time t−1 

Clear Separation of Concerns
 - Modeling $\neq$ Validation $\neq$ Reporting

Test-Driven 
 - Full pytest coverage for all modules

## 📁 Project Structure

```
quant-risk-market-risk-engine/
├── risk_engine/
│   ├── data.py            # data & returns
│   ├── portfolio.py       # portfolio returns
│   ├── var_models.py      # VaR models
│   ├── es_models.py       # ES models
│   ├── backtest.py        # Kupiec / Christoffersen
│   ├── basel.py           # Basel traffic light
│   ├── stress.py          # stress testing
│   └── report.py          # end-to-end report
│
├── scripts/
│   └── run_risk_report.py # one-command CLI entry
│
├── tests/                 # pytest test suite
└── README.md
```
## ▶️ Quick Start (One-Command Demo)
### 1️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 2️⃣ Run a full risk report (demo data)
```
python scripts/run_risk_report.py \
  --var_model EWMA \
  --es_model EWMA \
  --alpha 0.99 \
  --window 252
```

### Example Output
```
             cumulative_return  max_drawdown
scenario                                    
eq_-10_mult               -1.0           1.0
eq_-5_add                 -1.0           1.0
StressResult(name='covid', start=Timestamp('2020-02-18 00:00:00'), end=Timestamp('2020-04-15 00:00:00'), cumulative_return=-0.014372818654846853, max_drawdown=0.16082893855825964)

=== Risk Report Summary ===
Model: VaR=EWMA, ES=EWMA | alpha=0.99 | window=252
Latest VaR (loss+): 0.023556
Latest ES  (loss+): 0.026678

--- Backtest (VaR) ---
Evaluated days: 548
Violations:     9  (rate=0.0164, expected=0.0100)
Kupiec LR:      1.913039  p=0.16662610143099335
Christoff LR:   0.301129  p=0.5831753235669301
CC LR:          2.214168  p=0.33052135380344844
Basel Light:    GREEN

--- Stress Summary ---
             cumulative_return  max_drawdown
scenario                                    
eq_-10_mult               -1.0           1.0
eq_-5_add                 -1.0           1.0
```

## 📊 Stress Testing Examples
```
from risk_engine.stress import apply_parametric_shock, summarize_stress

scenarios = {
    "equity_-5%": apply_parametric_shock(returns, shock=-0.05),
    "equity_-10%": apply_parametric_shock(returns, shock=-0.10, mode="multiplicative"),
}

summarize_stress(returns, scenarios)
```

## 🧪 Testing
All components are fully tested.

```
pytest -q
```

## 🎯 Intended Use

This project is designed to reflect real-world Quant Risk / Market Risk workflows, including:

 - Daily VaR / ES production 
 - Regulatory backtesting 
 - Model validation 
 - Stress and scenario analysis 
 - Risk reporting and communication
📌 Keywords (for Recruiters)

## 📌 Keywords 
`Quant Risk` · `Market Risk` · `VaR` · `Expected Shortfall` · `Backtesting` ·  
`Kupiec Test` · `Christoffersen Test` · `Basel Traffic Light` · `Stress Testing` ·  
`Python` · `Pandas` · `Statistical Modeling`

## 🧩 Future Extensions

 - ES backtesting (Acerbi–Szekely)
 - Multi-asset portfolios 
 - Factor-based risk decomposition 
 - Report export (Markdown / PDF)