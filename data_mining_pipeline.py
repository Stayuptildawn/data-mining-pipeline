"""
data_mining_pipeline.py

General-purpose time series data mining pipeline.

Covers the full workflow from raw CSV to evaluated models:
  1. Data loading and inspection
  2. Preprocessing (cleaning, imputation, feature engineering)
  3. Train / validation / test splitting
  4. Baseline models (naive, seasonal naive, moving average)
  5. ARIMA / SARIMA (requires statsmodels)
  6. Gradient Boosting (requires scikit-learn)
  7. LSTM (requires PyTorch) — set RUN_LSTM = True to enable
  8. Model comparison and CSV export

Designed to work on any time series dataset with a datetime column
and a numeric target column. Edit PipelineConfig to adapt to your data.

Usage:
    python data_mining_pipeline.py
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# --- Configuration ---

@dataclass
class PipelineConfig:
    """All settings for the pipeline. Edit this block to adapt to your dataset."""

    # Paths
    data_path: str = "data/LSTM-Multivariate_pollution.csv"
    output_dir: str = "outputs"

    # Dataset schema
    date_column: str = "date"
    target_column: str = "pollution"
    exogenous_columns: Tuple[str, ...] = ("dew", "temp", "press", "wnd_spd", "snow", "rain")
    categorical_column: str = "wnd_dir"   # set to "" to skip one-hot encoding

    # Chronological split boundaries (inclusive)
    train_end: str = "2012-12-31 23:00:00"
    val_end: str = "2013-12-31 23:00:00"

    # Feature engineering
    lags: Tuple[int, ...] = (1, 2, 3, 24)
    add_calendar_features: bool = True

    # Baseline model settings
    seasonal_period: int = 24
    ma_window: int = 3

    # ARIMA / SARIMA
    arima_order: Tuple[int, int, int] = (1, 1, 1)
    arima_seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24)
    arima_use_log: bool = False

    # Gradient Boosting
    gb_n_estimators: int = 300
    gb_learning_rate: float = 0.03
    gb_max_depth: int = 5
    gb_min_samples_leaf: int = 20
    random_state: int = 42

    # LSTM
    lstm_input_length: int = 24
    lstm_units: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    lstm_epochs: int = 50
    lstm_batch_size: int = 128
    lstm_patience: int = 5
    lstm_learning_rate: float = 1e-4

    # Toggle models on/off
    RUN_BASELINES: bool = True
    RUN_ARIMA: bool = True
    RUN_GRADIENT_BOOSTING: bool = True
    RUN_LSTM: bool = False  # requires PyTorch


# --- Data Loading & Inspection ---

def load_data(data_path: str, date_column: str) -> pd.DataFrame:
    """Load CSV, parse dates, sort, dedup, and enforce a full hourly time grid."""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if date_column not in df.columns:
        raise ValueError(f"Expected date column '{date_column}' not found in CSV.")

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).drop_duplicates(subset=date_column).set_index(date_column)

    # Reindex to full hourly grid; gaps become NaN and are handled in imputation
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="H")
    df = df.reindex(full_range)

    print(f"Loaded {len(df)} records spanning {df.index.min()} to {df.index.max()}.")
    return df


def inspect_dataset(df: pd.DataFrame, target_column: str) -> None:
    """Print a structured data quality summary."""
    print("\n--- Dataset Overview ---")
    print(f"  Shape       : {df.shape}")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Date range  : {df.index.min()} → {df.index.max()}")

    missing_counts = df.isnull().sum()
    missing = missing_counts[missing_counts > 0]
    if missing.empty:
        print("  Missing     : none")
    else:
        print(f"  Missing values:\n{missing.to_string()}")

    if target_column in df.columns:
        target = df[target_column].dropna()
        print(f"\n  Target '{target_column}' — "
              f"Mean: {target.mean():.2f}, Std: {target.std():.2f}, "
              f"Min: {target.min():.2f}, Max: {target.max():.2f}")


# --- Preprocessing ---

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then back-fill all columns to handle gaps from reindexing."""
    return df.ffill().bfill()


def one_hot_encode_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """One-hot encode a categorical column and drop the original."""
    if column not in df.columns:
        print(f"  Warning: column '{column}' not found — skipping one-hot encoding.")
        return df
    dummies = pd.get_dummies(df[column].astype(str), prefix=column, dtype=float)
    return pd.concat([df.drop(columns=[column]), dummies], axis=1)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week, month, and sin/cos cyclical time encodings."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("add_calendar_features requires a DatetimeIndex.")
    df = df.copy()
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    df["hour_sin"]   = np.sin(2 * np.pi * df.index.hour / 24.0)
    df["hour_cos"]   = np.cos(2 * np.pi * df.index.hour / 24.0)
    return df


def build_lag_features(
    df: pd.DataFrame,
    target_column: str,
    lags: Tuple[int, ...],
) -> pd.DataFrame:
    """Append lagged versions of the target series as new feature columns."""
    df = df.copy()
    for lag in lags:
        df[f"{target_column}_lag{lag}"] = df[target_column].shift(lag)
    return df


# --- Splitting ---

def split_chronologically(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DatetimeIndex DataFrame into train/val/test with no timestamp overlap."""
    train = df.loc[:train_end].copy()
    val   = df.loc[train_end:val_end].copy()
    test  = df.loc[val_end:].copy()

    if not val.empty and not train.empty and val.index[0] == train.index[-1]:
        val = val.iloc[1:]
    if not test.empty and not val.empty and test.index[0] == val.index[-1]:
        test = test.iloc[1:]

    print(f"  Split → Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    return train, val, test


# --- Metrics ---

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute MAE, RMSE, and R² after dropping NaN pairs."""
    mask = (~y_true.isna()) & (~y_pred.isna())
    y_t, y_p = y_true[mask], y_pred[mask]
    if len(y_t) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    return {
        "MAE":  round(float(mean_absolute_error(y_t, y_p)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_t, y_p))), 4),
        "R2":   round(float(r2_score(y_t, y_p)), 4),
    }


# --- Baseline Models ---

def run_baselines(
    y_full: pd.Series,
    val_index: pd.Index,
    test_index: pd.Index,
    config: PipelineConfig,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute naive, seasonal naive, and moving-average baselines."""
    print("\n--- Baseline Models ---")

    predictions = {
        "naive":
            y_full.shift(1),
        f"seasonal_{config.seasonal_period}":
            y_full.shift(config.seasonal_period),
        f"ma_{config.ma_window}":
            y_full.rolling(window=config.ma_window, min_periods=config.ma_window).mean().shift(1),
    }

    all_metrics = {}
    for model_name, pred_series in predictions.items():
        all_metrics[model_name] = {
            "val":  compute_metrics(y_full.loc[val_index],  pred_series.loc[val_index]),
            "test": compute_metrics(y_full.loc[test_index], pred_series.loc[test_index]),
        }
        val_mae  = all_metrics[model_name]["val"]["MAE"]
        test_mae = all_metrics[model_name]["test"]["MAE"]
        print(f"  {model_name:<22} Val MAE: {val_mae:.2f}  Test MAE: {test_mae:.2f}")

    return all_metrics


# --- ARIMA / SARIMA ---

def run_arima(
    y_full: pd.Series,
    config: PipelineConfig,
) -> Dict[str, Dict[str, float]]:
    """Fit SARIMA on training data and roll forward one step at a time over val+test."""
    print("\n--- ARIMA / SARIMA ---")

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("  statsmodels not installed — skipping ARIMA.")
        return {}

    y_model = np.log1p(y_full.clip(lower=0.0)) if config.arima_use_log else y_full.copy()

    train_series = y_model.loc[:config.train_end]
    val_series   = y_model.loc[config.train_end:config.val_end]
    test_series  = y_model.loc[config.val_end:]

    if not val_series.empty and val_series.index[0] == train_series.index[-1]:
        val_series = val_series.iloc[1:]
    if not test_series.empty and test_series.index[0] == val_series.index[-1]:
        test_series = test_series.iloc[1:]

    model = SARIMAX(
        train_series,
        order=config.arima_order,
        seasonal_order=config.arima_seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=True,
    )

    t0 = time.perf_counter()
    fitted = model.fit(disp=False)
    print(f"  Training time: {time.perf_counter() - t0:.1f}s")

    # Rolling one-step-ahead forecast: update model state with each true observation
    forecast_index = val_series.index.append(test_series.index)
    preds_list = []
    current_fit = fitted

    for t in forecast_index:
        preds_list.append(float(current_fit.get_forecast(steps=1).predicted_mean.iloc[0]))
        current_fit = current_fit.extend(endog=[y_model.loc[t]])

    preds_series = pd.Series(preds_list, index=forecast_index)
    if config.arima_use_log:
        preds_series = np.expm1(preds_series)

    y_true_on_val  = y_full.loc[val_series.index]
    y_true_on_test = y_full.loc[test_series.index]

    metrics = {
        "val":  compute_metrics(y_true_on_val,  preds_series.loc[val_series.index]),
        "test": compute_metrics(y_true_on_test, preds_series.loc[test_series.index]),
    }
    print(f"  arima                  Val MAE: {metrics['val']['MAE']:.2f}  Test MAE: {metrics['test']['MAE']:.2f}")
    return metrics


# --- Gradient Boosting ---

def build_supervised_features(
    df: pd.DataFrame,
    target_column: str,
    lags: Tuple[int, ...],
    exog_columns: Tuple[str, ...],
    add_calendar: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct a tabular feature matrix from a time series for tree-based models."""
    feat_df = pd.DataFrame(index=df.index)

    for lag in lags:
        feat_df[f"{target_column}_lag{lag}"] = df[target_column].shift(lag)

    available_exog = [col for col in exog_columns if col in df.columns]
    if available_exog:
        feat_df[available_exog] = df[available_exog]

    if add_calendar:
        feat_df["hour"]      = feat_df.index.hour
        feat_df["dayofweek"] = feat_df.index.dayofweek
        feat_df["month"]     = feat_df.index.month

    feat_df[target_column] = df[target_column]
    feat_df = feat_df.dropna()

    feature_cols = [col for col in feat_df.columns if col != target_column]
    return feat_df, feature_cols


def run_gradient_boosting(
    df_clean: pd.DataFrame,
    config: PipelineConfig,
) -> Dict[str, Dict[str, float]]:
    """Fit a GradientBoostingRegressor on lag features and evaluate on each split."""
    print("\n--- Gradient Boosting ---")

    if config.target_column not in df_clean.columns:
        raise ValueError(f"Target column '{config.target_column}' not found in dataframe.")

    feat_df, feature_cols = build_supervised_features(
        df_clean,
        target_column=config.target_column,
        lags=config.lags,
        exog_columns=config.exogenous_columns,
        add_calendar=config.add_calendar_features,
    )
    print(f"  {len(feature_cols)} features: {feature_cols}")

    train_df, val_df, test_df = split_chronologically(feat_df, config.train_end, config.val_end)

    X_train, y_train = train_df[feature_cols], train_df[config.target_column]
    X_val,   y_val   = val_df[feature_cols],   val_df[config.target_column]
    X_test,  y_test  = test_df[feature_cols],  test_df[config.target_column]

    model = GradientBoostingRegressor(
        n_estimators=config.gb_n_estimators,
        learning_rate=config.gb_learning_rate,
        max_depth=config.gb_max_depth,
        min_samples_leaf=config.gb_min_samples_leaf,
        random_state=config.random_state,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"  Training time: {time.perf_counter() - t0:.1f}s")

    metrics = {
        "train": compute_metrics(y_train, pd.Series(model.predict(X_train), index=y_train.index)),
        "val":   compute_metrics(y_val,   pd.Series(model.predict(X_val),   index=y_val.index)),
        "test":  compute_metrics(y_test,  pd.Series(model.predict(X_test),  index=y_test.index)),
    }
    print(f"  gradient_boosting      Val MAE: {metrics['val']['MAE']:.2f}  Test MAE: {metrics['test']['MAE']:.2f}")

    importance_df = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    print(f"  Top 5 features by importance:\n{importance_df.head(5).to_string(index=False)}")

    return metrics


# --- LSTM ---

def run_lstm(
    df_clean: pd.DataFrame,
    config: PipelineConfig,
) -> Dict[str, Dict[str, float]]:
    """Build and train a PyTorch LSTM for one-step-ahead forecasting."""
    print("\n--- LSTM ---")

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  PyTorch not installed — skipping LSTM.")
        return {}

    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.target_column not in df_clean.columns:
        raise ValueError(f"Target column '{config.target_column}' not found in dataframe.")

    # Build feature matrix: target + optional exogenous + calendar
    feat_df = pd.DataFrame(index=df_clean.index)
    feat_df[config.target_column] = df_clean[config.target_column].astype(float)

    available_exog = [col for col in config.exogenous_columns if col in df_clean.columns]
    if available_exog:
        feat_df[available_exog] = df_clean[available_exog]

    if config.add_calendar_features:
        feat_df["hour"]      = feat_df.index.hour
        feat_df["dayofweek"] = feat_df.index.dayofweek
        feat_df["month"]     = feat_df.index.month

    feat_df = feat_df.dropna()
    feature_cols = list(feat_df.columns)
    n_features   = len(feature_cols)

    X_raw  = feat_df.values.astype(np.float32)
    y_raw  = feat_df[config.target_column].values.astype(np.float32)
    times  = feat_df.index

    seq_len   = config.lstm_input_length
    n_samples = len(feat_df) - seq_len

    X_seq = np.stack([X_raw[i: i + seq_len] for i in range(n_samples)])
    y_seq = y_raw[seq_len:]
    label_times = times[seq_len:]

    train_end_ts = pd.to_datetime(config.train_end)
    val_end_ts   = pd.to_datetime(config.val_end)

    train_mask = label_times <= train_end_ts
    val_mask   = (label_times > train_end_ts) & (label_times <= val_end_ts)
    test_mask  = label_times > val_end_ts

    # Fit scalers on training split only
    feat_scaler = StandardScaler().fit(X_seq[train_mask].reshape(-1, n_features))
    X_seq_scaled = feat_scaler.transform(X_seq.reshape(-1, n_features)).reshape(X_seq.shape)

    tgt_scaler = StandardScaler().fit(y_seq[train_mask].reshape(-1, 1))
    y_scaled   = tgt_scaler.transform(y_seq.reshape(-1, 1)).reshape(-1)

    def as_tensors(X, y):
        return (
            torch.from_numpy(X).float().to(device),
            torch.from_numpy(y).float().to(device),
        )

    X_tr_t, y_tr_t = as_tensors(X_seq_scaled[train_mask], y_scaled[train_mask])
    X_va_t, y_va_t = as_tensors(X_seq_scaled[val_mask],   y_scaled[val_mask])
    X_te_t, _      = as_tensors(X_seq_scaled[test_mask],  y_scaled[test_mask])

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=config.lstm_batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=config.lstm_batch_size)

    # Define LSTM model inside the function to avoid top-level torch dependency
    class _LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                n_features, config.lstm_units,
                num_layers=config.lstm_layers,
                batch_first=True,
                dropout=config.lstm_dropout if config.lstm_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(config.lstm_dropout)
            self.fc      = nn.Linear(config.lstm_units, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)

    lstm_model = _LSTMModel().to(device)
    optimizer  = torch.optim.Adam(lstm_model.parameters(), lr=config.lstm_learning_rate)
    criterion  = nn.MSELoss()

    best_val_loss, best_state, patience_counter = float("inf"), None, 0
    t0 = time.perf_counter()

    for epoch in range(config.lstm_epochs):
        lstm_model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            criterion(lstm_model(X_batch), y_batch).backward()
            optimizer.step()

        lstm_model.eval()
        val_loss = sum(
            criterion(lstm_model(X_b), y_b).item()
            for X_b, y_b in val_loader
        ) / len(val_loader)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss, best_state, patience_counter = val_loss, lstm_model.state_dict(), 0
        else:
            patience_counter += 1
            if patience_counter >= config.lstm_patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_state:
        lstm_model.load_state_dict(best_state)
    print(f"  Training time: {time.perf_counter() - t0:.1f}s")

    def predict_inverse(X_tensor):
        lstm_model.eval()
        with torch.no_grad():
            preds_scaled = lstm_model(X_tensor).cpu().numpy()
        return tgt_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)

    metrics = {
        "val":  compute_metrics(pd.Series(y_seq[val_mask]),  pd.Series(predict_inverse(X_va_t))),
        "test": compute_metrics(pd.Series(y_seq[test_mask]), pd.Series(predict_inverse(X_te_t))),
    }
    print(f"  lstm                   Val MAE: {metrics['val']['MAE']:.2f}  Test MAE: {metrics['test']['MAE']:.2f}")
    return metrics


# --- Output ---

def save_metrics_table(
    all_metrics: Dict[str, Dict],
    output_dir: str,
) -> None:
    """Flatten all model metrics into a single CSV for easy comparison."""
    rows = []
    for model_name, split_metrics in all_metrics.items():
        for split_name, metric_values in split_metrics.items():
            rows.append({"model": model_name, "split": split_name, **metric_values})

    metrics_df = pd.DataFrame(rows)
    output_path = Path(output_dir) / "model_metrics.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print("\n" + metrics_df.to_string(index=False))


def plot_timeseries_overview(
    df: pd.DataFrame,
    target_column: str,
    output_dir: str,
) -> None:
    """Save a time series plot of each numeric column for visual inspection."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_plots = min(len(numeric_cols), 8)

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols[:n_plots]):
        ax.plot(df.index, df[col].values, linewidth=0.5, alpha=0.85)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Feature Time Series Overview", fontsize=13, y=1.01)
    plt.tight_layout()

    output_path = Path(output_dir) / "timeseries_overview.png"
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison(
    all_metrics: Dict[str, Dict],
    output_dir: str,
) -> None:
    """Save a bar chart comparing Test MAE across all evaluated models."""
    rows = []
    for model_name, split_metrics in all_metrics.items():
        if "test" in split_metrics:
            rows.append({
                "model": model_name,
                "Test MAE": split_metrics["test"].get("MAE", np.nan),
                "Test RMSE": split_metrics["test"].get("RMSE", np.nan),
            })

    if not rows:
        return

    comp_df = pd.DataFrame(rows).sort_values("Test MAE")
    x = np.arange(len(comp_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, comp_df["Test MAE"],  width, label="Test MAE",  alpha=0.85)
    ax.bar(x + width / 2, comp_df["Test RMSE"], width, label="Test RMSE", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["model"], rotation=30, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Model Comparison — Test MAE and RMSE")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / "model_comparison.png"
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Saved: {output_path}")


# --- Main ---

def main() -> None:
    config = PipelineConfig()
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Data Mining Pipeline")
    print("=" * 60)

    # --- Data Loading ---
    df_raw = load_data(config.data_path, config.date_column)
    inspect_dataset(df_raw, config.target_column)

    # --- Preprocessing ---
    print("\n--- Preprocessing ---")
    df_clean = impute_missing_values(df_raw)

    if config.categorical_column:
        df_clean = one_hot_encode_column(df_clean, config.categorical_column)

    print(f"  Columns after encoding: {list(df_clean.columns)}")

    plot_timeseries_overview(df_clean, config.target_column, config.output_dir)

    # --- Splitting (for baselines operating on the raw target series) ---
    print("\n--- Data Splits ---")
    y_full = df_clean[config.target_column].astype(float)
    _, val_df_split, test_df_split = split_chronologically(df_clean, config.train_end, config.val_end)
    val_index  = val_df_split.index
    test_index = test_df_split.index

    all_metrics: Dict[str, Dict] = {}

    # --- Baselines ---
    if config.RUN_BASELINES:
        baseline_metrics = run_baselines(y_full, val_index, test_index, config)
        all_metrics.update(baseline_metrics)

    # --- ARIMA ---
    if config.RUN_ARIMA:
        arima_metrics = run_arima(y_full, config)
        if arima_metrics:
            all_metrics["arima"] = arima_metrics

    # --- Gradient Boosting ---
    if config.RUN_GRADIENT_BOOSTING:
        gb_metrics = run_gradient_boosting(df_clean, config)
        all_metrics["gradient_boosting"] = gb_metrics

    # --- LSTM ---
    if config.RUN_LSTM:
        lstm_metrics = run_lstm(df_clean, config)
        if lstm_metrics:
            all_metrics["lstm"] = lstm_metrics

    # --- Results ---
    print("\n--- Results ---")
    save_metrics_table(all_metrics, config.output_dir)
    plot_model_comparison(all_metrics, config.output_dir)

    print("\n" + "=" * 60)
    print(f"  Done. Outputs saved to: {config.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
