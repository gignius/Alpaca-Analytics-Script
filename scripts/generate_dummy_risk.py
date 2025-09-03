import os
import sys

# Ensure project root is on path when running from scripts/
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.chart_generator import ChartGenerator


class Metrics:
    sharpe_ratio = 3.28
    sortino_ratio = 8.0
    max_drawdown = 1.65
    volatility = 19.0


class Account:
    def __init__(self, account_number: str):
        self.account_number = account_number


if __name__ == "__main__":
    metrics = Metrics()
    account = Account("TEST-ACCT")
    cg = ChartGenerator(chart_dir="charts")
    fn = cg.generate_risk_dashboard(metrics, account)
    print(fn)
