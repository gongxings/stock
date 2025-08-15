import itertools
import pandas as pd


def grid_search(parameters, build_strategy_fn, run_backtest_fn):
    keys = list(parameters.keys())
    rows = []
    for values in itertools.product(*[parameters[k] for k in keys]):
        p = {k: v for k, v in zip(keys, values)}
        strat = build_strategy_fn(p)
        metrics = run_backtest_fn(strat)
        row = {**p, **metrics}
        rows.append(row)
    return pd.DataFrame(rows)
