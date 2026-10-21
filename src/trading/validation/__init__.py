"""
NOCTURNA Trading System - Strategy validation.

Everything a candidate strategy must survive before it may paper trade:

* ``stats``: performance statistics, the Probabilistic and Deflated Sharpe
  ratios and the number of effectively independent trials,
* ``overfitting``: the probability of backtest overfitting (CSCV) and
  combinatorial purged cross-validation,
* ``experiment``: the parameter grid run through the real engine, walk-forward
  selection traded through the engine, cost stress, parameter stability and
  regime splits,
* ``gates`` and ``report``: the promotion gates and the tear sheet,
* ``pipeline``: all of the above for one candidate.
"""
