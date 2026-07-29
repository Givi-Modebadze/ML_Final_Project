# Walmart Store Sales Forecasting

Time-series forecasting for the
[Walmart Recruiting — Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)
Kaggle competition: predicting weekly sales (`Weekly_Sales`) per department across **45 stores**.

The competition metric is **WMAE** (weighted mean absolute error), which weights holiday weeks 5×
— so most of the effort went into making models aware of holiday timing and yearly seasonality.
**Eleven** model configurations were compared, spanning gradient boosting, classical statistical
models and deep sequence architectures.

**Best result: 2695.34 WMAE public / 2777.38 private**, using **PatchTST**.

> 🇬🇪 The original, more detailed Georgian notes are preserved in [README.ka.md](README.ka.md).

---

## Results

Lower WMAE is better. "Valid WMAE" is from cross-validation; Kaggle scores are leaderboard submissions.

| Model | Valid WMAE | Kaggle | Notes |
|---|---:|---:|---|
| **PatchTST** | **2581** | **2695 pub / 2777 priv** | **Best model.** Transformer, via NeuralForecast |
| NBEATS | 2683 | 2763 pub / 2848 priv | Trend + seasonality stacks |
| XGBoost v3 | 2167 | ~3280 | + Fourier features, holiday windows, tuned |
| LightGBM | 1972 | 3477 pub / 3544 priv | Fixed underfitting from the earlier run |
| XGBoost v2 | 2448 | ~3745 | + calendar features, `PrevYearSales` |
| Random Forest | 1905 | 4365 | Simple baseline, held up surprisingly well |
| DLinear | 5414 | — | Common baseline; not sophisticated enough here |
| XGBoost v1 | 3659 | — | Basic features, minimal tuning |
| ARIMA / SARIMA | 15231 | — | One model per store; failed (see below) |
| TFT | 20367 | — | Store 1 only — too slow to train fully |
| LightGBM (early) | 1530 | 19820 | ⚠️ Leaky — see note below |

### Two negative results worth reading

**ARIMA/SARIMA didn't work.** A wrapper class fits a separate model per store (per-department was
attempted too, but training crashed on time). These models expect *gradual* change — more people
buy bikinis as summer approaches — but the sales spikes that matter here are **sharp holiday
events** like Thanksgiving and New Year. The models could not track them, giving WMAE ~15200.

**The early LightGBM run was leaking.** It scored better on *validation* than on *training*
(1530 vs. 1776 WMAE), which is a red flag on its own — and Kaggle returned an absurd 19820,
confirming it. The cause was doing feature engineering before settling the train/validation
split. Fixed in the later `lightGBM` experiment.

---

## Approach

### Data analysis

Plotting mean `Weekly_Sales` over time showed clear **yearly cyclicality**. Analysing all 45
stores individually revealed a nearly identical trend across most of them, with a handful of
genuine outliers (stores 28, 30, 33, 36, 42, 43 and a few more). Because the shape repeats
year over year, lag features became the obvious lever.

### Feature engineering

- **Holiday windows** — `IsHoliday` alone is not enough: a model cannot learn the run-up and
  wind-down from a single-week flag. Added explicit `isWeekBefore{Holiday}` /
  `isWeekAfter{Holiday}` columns per holiday.
- **Lag features** — sales from 1 and 2 years prior (`PrevYearSales`), exploiting the
  year-over-year stability.
- **Fourier features** — `sin`/`cos` terms, which helped the models represent seasonality
  more smoothly than raw calendar dummies.
- **Calendar decomposition** — `Year`, `Month`, `Week`, `Day`, `DayOfWeek`.

Adding these moved XGBoost from ~3745 → ~3280 on the leaderboard.

### Best model

```python
PatchTST(
    h=60,
    input_size=52,      # one year of weekly history
    patch_len=16,
    stride=4,
    n_heads=8,
    dropout=0.1,
    batch_size=64,
    max_steps=5000,
    learning_rate=0.0005,
    activation="relu",
)
```

Notably, PatchTST scored well **before** any preprocessing or feature engineering — it was run
through `NeuralForecast` with a thin wrapper, then tuned from there.

---

## Experiment tracking

Runs are tracked with **MLflow** on DagsHub:

- [Final_Project_ML — XGBoost experiments](https://dagshub.com/Givi-Modebadze/Final_Project_ML.mlflow/#/experiments/0)
- [PatchTST](https://dagshub.com/Givi-Modebadze/Final_Project_ML.mlflow/#/experiments/1) ·
  [NBEATS](https://dagshub.com/Givi-Modebadze/Final_Project_ML.mlflow/#/experiments/2)
- [Remaining six models](https://dagshub.com/gbend22/ML_Final_Project.mlflow/#/experiments/)

---

## Repository structure

```
statistics.ipynb        Exploratory analysis — seasonality, per-store trends
experiments/            One notebook per model configuration (see results table)
model_inference.ipynb   Generates the submission using the best model (PatchTST)
walmart-recruiting-store-sales-forecasting/
                        Competition data
```

## Running it

```bash
pip install pandas numpy scikit-learn xgboost lightgbm statsmodels neuralforecast mlflow
```

Run `statistics.ipynb` for the analysis, any notebook under `experiments/` to reproduce a single
model, or `model_inference.ipynb` to regenerate the submission.
