import numpy as np, backtester as bt, strategy as st
prc, names = bt.load_prices("prices.txt"), None
prcAll = bt.loadPrices_np if hasattr(bt,'loadPrices_np') else None
# load prices via backtester helper
import pandas as pd
df = pd.read_csv("prices.txt", sep=r"\s+")
prcAll = df.values.T
names = list(df.columns)
comm, lim = bt.make_grading_params(prcAll.shape[0])

print(f"{'window':>6} {'scale':>5} | {'Score':>7} {'Sharpe':>7} {'Sortino':>7} {'meanPL':>7} {'maxDD':>8} {'turnover':>9}")
best=None
for w in (10,20,30,45,60):
    for s in (1.0,1.5,2.0,3.0):
        res = bt.run_backtest(prcAll, st.make_get_position(w,s), 250, comm_rate=comm, dlr_pos_limit=lim, inst_names=names)
        row=(res.score,res.ann_sharpe,res.sortino,res.mean_pl,res.max_drawdown,res.avg_daily_turnover)
        print(f"{w:>6} {s:>5.1f} | {res.score:>7.2f} {res.ann_sharpe:>7.2f} {res.sortino:>7.2f} {res.mean_pl:>7.1f} {res.max_drawdown:>8.0f} {res.avg_daily_turnover:>9.0f}")
        if best is None or res.score>best[0]: best=(res.score,w,s)
print(f"\nbest by Score: window={best[1]} scale={best[2]} -> Score {best[0]:.2f}")
