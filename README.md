# LSTM Attention Forecasting Benchmark

An LSTM pipeline for stock price forecasting in Python, covering data download through training, walk-forward retraining, and a transaction-cost-aware backtest.

## What This Notebook Covers

| Section                         | Description                                                                  |
| ------------------------------- | ---------------------------------------------------------------------------- |
| 1. Libraries                    | Install and import all dependencies                                          |
| 2. Data Download                | SPY price data via yfinance (2018 to 2024)                                   |
| 3. Feature Engineering          | 15 technical and statistical features                                        |
| 4. Train/Val/Test Split         | Temporal 70/15/15 split with leak prevention                                 |
| 5. LSTM Architecture            | 3-layer stacked LSTM with dropout                                            |
| 6. Training                     | EarlyStopping and ReduceLROnPlateau callbacks                                |
| 7. Evaluation                   | RMSE, MAE, MAPE, R-squared on held-out test set                              |
| 8. Benchmark Comparison         | Naive random walk and GARCH(1,1), static fit                                 |
| 9. Directional Accuracy         | Up/down prediction analysis                                                  |
| 10. Walk-Forward Retraining     | LSTM fine-tuned and GARCH refit through the test period                      |
| 11. Backtest                    | Long/flat strategy, net of a per-trade transaction cost                      |
| 12. Out-of-Distribution Monitor | Flags when current volatility is outside the training range                  |
| 13. Attention-Augmented LSTM    | Attention pooling over the lookback window, benchmarked against the baseline |
| 14. Interpretability (SHAP)     | Which features and which days actually drive predictions                     |

## Features Engineered

| Group      | Features                                       |
| ---------- | ---------------------------------------------- |
| Price      | Log return, high low spread, open close return |
| Trend      | SMA 5/20/60 (deviation), EMA 12/26, MACD       |
| Volatility | Realised volatility at 5/20/60 day windows     |
| Volume     | 30 day rolling z-score normalised volume       |
| Momentum   | RSI normalised, lagged returns at 1/5/22 days  |

## Model Architecture

```
Input (60 days x 15 features)
    -> LSTM (128 units, return_sequences=True)
    -> Dropout (0.2)
    -> LSTM (64 units, return_sequences=True)
    -> Dropout (0.2)
    -> LSTM (32 units, return_sequences=False)
    -> Dropout (0.2)
    -> Dense (16, ReLU)
    -> Dropout (0.1)
    -> Dense (1, Linear)
```

Total parameters: ~170,000

## Performance Notes

- Sequence generation (turning scaled features into overlapping 60-day windows) uses a vectorised `numpy.lib.stride_tricks.sliding_window_view` instead of a per-row Python loop, verified to produce identical output.
- Feature and target arrays are cast to `float32` once, up front, instead of TensorFlow re-casting from `float64` on every training batch.
- `tf.keras.backend.clear_session()` runs before the model is built, so re-running that cell during experimentation does not accumulate graph state across a long session.

## Benchmark Comparison (Static Fit)

Section 8 scores the LSTM against two baselines, each fit once on the training window and then held fixed for the whole test period:

- **Naive random walk**: tomorrow's price = today's price.
- **GARCH(1,1)**: fitted on training log returns; its constant mean return is applied as a drift on the previous close for a price forecast, and its conditional volatility forecast is reported separately since that, not the price point forecast, is what GARCH is actually good at.

| Metric               | LSTM (Ours)         | GARCH(1,1)          | Naive Baseline      |
| -------------------- | ------------------- | ------------------- | ------------------- |
| RMSE ($)             | see notebook output | see notebook output | see notebook output |
| MAE ($)              | see notebook output | see notebook output | see notebook output |
| MAPE (%)             | see notebook output | see notebook output | see notebook output |
| R-squared            | see notebook output | see notebook output | see notebook output |
| Directional Accuracy | 38.7%               | not computed        | ~50% (theoretical)  |

## Walk-Forward Retraining (Section 10)

Section 10 adds:

- **Walk-forward LSTM**: every `WF_STEP` trading days (21 by default, about a month), the model is fine-tuned on the block of test data that just became available, warm-started from its current weights with a lower learning rate, then used to predict the next block. No block is ever predicted with data from later in time.
- **Rolling GARCH(1,1)**: refit on the same cadence, using an expanding window of returns (training data plus every test return revealed so far).

This produces a second, walk-forward set of predictions for both models, printed alongside the static-fit numbers from Section 8. Set `RUN_WALK_FORWARD = False` at the top of that cell to skip it and use the static-fit results only.

## Transaction-Cost-Aware Backtest (Section 11)

Section 11 turns each model's predictions (naive, both GARCH variants, both LSTM variants) into a long/flat position, long when the model expects tomorrow's close above today's, flat otherwise, and charges a transaction cost (`TRANSACTION_COST_BPS`, 5 bps by default) on every position change. It reports total return, annualised Sharpe ratio, max drawdown, and trade count, net of that cost, plus an equity curve plot.

This backtest is long/flat, with no shorting or leverage, using a flat per-trade cost.

## Out-of-Distribution Monitor (Section 12)

Computes a rolling 20-day realised volatility z-score against the training-period distribution and flags test-period days where that z-score exceeds `OOD_THRESHOLD` (3.0 by default).

## Attention-Augmented LSTM (Section 13)

The stacked LSTM in Section 5 pools the whole 60-day window down to whatever the final LSTM layer's last hidden state carries forward. This section adds a second architecture that instead scores every day in the window, softmaxes those scores into weights, and takes a weighted sum as the context vector, so the model can learn which days matter most for a given prediction rather than defaulting to "the most recent step wins." It is trained and evaluated the same way as the baseline model and the two are compared directly on RMSE, MAE, MAPE, R-squared, and directional accuracy, alongside a plot of where the model's attention actually lands on average across the test set.

## Model Interpretability with SHAP (Section 14)

SHAP's `DeepExplainer` does not reliably support recurrent layers in current TensorFlow, it fails on the LSTM gradient graph with a `LookupError`. This section uses `GradientExplainer` instead, confirmed to work against the full 3-layer stacked LSTM, and runs it against the baseline model from Section 5 so its output is directly comparable to that model's own metrics. It produces two views: mean absolute SHAP value per engineered feature (which inputs move the prediction most), and mean absolute SHAP value per day in the lookback window (whether the model actually leans on recent days more than distant ones, independent of the attention mechanism in Section 13).

## Data Download: yfinance, Retries, and Caching

Price data comes directly from yfinance, real OHLCV bars, never fabricated or simulated. The download cell wraps `yf.download()` with a few production-minded additions:

- **Flat columns by default**: passes `multi_level_index=False` so single-ticker downloads come back without the MultiIndex column structure that recent yfinance versions default to. A defensive check still flattens columns if an older yfinance version, or an older cache file, hands back a MultiIndex.
- **Retries with backoff**: transient failures (network blips, Yahoo rate limiting) are retried up to 3 times before raising. A failed download raises an error rather than silently continuing with empty or fabricated data.
- **Local caching**: the first successful download is saved to `{TICKER}_{START_DATE}_{END_DATE}_{INTERVAL}.csv` in the working directory. Re-running the notebook loads from that file instead of hitting Yahoo again. Delete the CSV to force a fresh download, for example when you want the latest trading day.
- **Configurable interval**: `INTERVAL` defaults to `'1d'`. yfinance also supports intraday bars (`'1h'`, `'15m'`, `'5m'`, `'1m'`); Yahoo's intraday history depth is roughly 730 days for hourly bars and shorter for minute bars. The feature engineering, `LOOKBACK` window, and annualisation (`sqrt(252)`) throughout the notebook are built for daily bars.

If your feature names ever appear as `Log_Return-`, `HL_Spread-` (trailing dash) and the correlation matrix is blank, that is the MultiIndex issue above; the defensive check already handles it, so this most likely means you are on a very old yfinance version.

## Requirements

```
yfinance>=1.0.0
tensorflow>=2.14.0
scikit-learn>=1.3.0
arch>=6.0.0
matplotlib>=3.7.0
seaborn>=0.13.0
pandas>=2.0.0
numpy>=1.24.0
shap>=0.44.0
```

yfinance graduated to a stable 1.x line with no breaking changes from the 0.2.x series, and that is what this notebook targets. It has been checked against yfinance 1.6.0. shap's `GradientExplainer` (used for interpretability in Section 14) has been checked against shap 0.52.0; `DeepExplainer` is not used because it does not work with the LSTM layers here.

Install all at once:

```bash
pip install yfinance tensorflow scikit-learn arch matplotlib seaborn pandas numpy shap
```

## Quick Start

```bash
git clone https://github.com/quantsingularity/LSTM-Attention-Forecasting-Benchmark
cd LSTM-Attention-Forecasting-Benchmark
jupyter notebook LSTM-Attention-Forecasting-Benchmark.ipynb
```

To change the ticker, edit the CONFIG block at the top of Cell 2:

```python
TICKER     = 'SPY'          # change to AAPL, MSFT, GLD, etc.
START_DATE = '2018-01-01'
END_DATE   = '2024-12-31'
LOOKBACK   = 60
INTERVAL   = '1d'           # see the intraday note above before changing this
```

## License

MIT License. Free to use, modify, and distribute with attribution.
