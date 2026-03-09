
# Data Mining Pipeline

A general-purpose time series forecasting pipeline that covers the full workflow from raw data to evaluated models in a single Python script.

---

## Overview

This project started as a data mining course assignment at Universidad Politécnica de Madrid (UPM). The original dataset was the Beijing PM2.5 multivariate pollution series, and the goal was to compare classical and machine learning forecasting approaches on a real-world time series. Credits to **Nicolai Stein**, my classmate, for being a part of the original project.

After the course, I rewrote and generalized the pipeline so it works on any time series dataset — not just pollution data. You point it at a CSV with a datetime column and a numeric target, edit a single config block, and get a full model comparison with exported results.

The pipeline runs four model families in sequence: naive baselines, ARIMA/SARIMA, Gradient Boosting, and an optional LSTM. All models are evaluated on the same chronological train/val/test split and results are exported to a single comparison CSV.

---

## Project Structure

```
data_mining_pipeline.py   # The entire pipeline — one self-contained script
outputs/
    model_metrics.csv         # Flat table: model × split × MAE/RMSE/R²
    timeseries_overview.png   # Time series plot of all numeric features
    model_comparison.png      # Bar chart comparing Test MAE and RMSE across models
data/
    your_dataset.csv          # Place your input CSV here
```

All logic lives in `data_mining_pipeline.py`. There are no cross-file imports and no module dependencies beyond the standard data science stack.

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn statsmodels matplotlib
   # Optional — only needed if RUN_LSTM = True
   pip install torch
   ```

2. **Place your CSV file** in the `data/` folder. The file needs at least a datetime column and a numeric target column.

3. **Edit `PipelineConfig`** at the top of the script to match your dataset:
   ```python
   data_path        = "data/your_file.csv"
   date_column      = "date"
   target_column    = "your_target"
   train_end        = "2022-12-31 23:00:00"
   val_end          = "2023-12-31 23:00:00"
   ```
   Toggle models on or off with `RUN_BASELINES`, `RUN_ARIMA`, `RUN_GRADIENT_BOOSTING`, `RUN_LSTM`.

4. **Run the script:**
   ```bash
   python data_mining_pipeline.py
   ```

5. **Check the `outputs/` folder** for the metrics CSV and plots.

---

## Notes

- **ARIMA speed**: The rolling one-step-ahead forecast calls `results.extend()` at every timestamp. This is correct and avoids data leakage, but it is slow for long test periods with a seasonal order of `s=24`. Expect several minutes of runtime.
- **LSTM is off by default**: Set `RUN_LSTM = True` in `PipelineConfig` and make sure PyTorch is installed. If PyTorch is not found, the script skips LSTM gracefully without crashing.
- **Dataset assumptions**: The script expects hourly data with a clean datetime column. It automatically reindexes to a full hourly grid and forward-fills any gaps. If your data has a different frequency, update the `freq="H"` argument inside `load_data()`.
- **Exogenous columns**: If your dataset has no exogenous variables, set `exogenous_columns = ()` in the config. The Gradient Boosting model will fall back to lag features and calendar features only.