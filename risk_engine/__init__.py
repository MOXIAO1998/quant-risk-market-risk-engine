from risk_engine.data import load_market_data_default
from risk_engine.portfolio import make_equal_weight_portfolio, portfolio_returns
from risk_engine.stress import apply_parametric_shock, summarize_stress, historical_window_stress

md = load_market_data_default(start="2016-01-01")
p = make_equal_weight_portfolio(md.returns.columns.tolist())
pr = portfolio_returns(md.returns, p)

scenarios = {
  "eq_-5_add": apply_parametric_shock(pr, shock=-0.05, mode="additive"),
  "eq_-10_mult": apply_parametric_shock(pr, shock=-0.10, mode="multiplicative"),
}
print(summarize_stress(pr, scenarios))

print(historical_window_stress(pr, start="2020-02-15", end="2020-04-15", name="covid"))
