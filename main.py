import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore") 

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dataset import get_dataset

RANDOM_SEED   = 42      
TEST_SIZE     = 0.20    
RESULTS_DIR   = "results"  

os.makedirs(RESULTS_DIR, exist_ok=True) 

plt.rcParams.update({
    "figure.dpi":     120,
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.spines.top":    False,  
    "axes.spines.right":  False,  
})

print("=" * 60)
print("  Delhi Temperature Forecasting — ML Project")
print("=" * 60)

if os.path.exists("delhi_weather.csv"):
    print("\nLoading existing dataset from delhi_weather.csv ...")
    df = pd.read_csv("delhi_weather.csv", parse_dates=["date"])
else:
    print("\nGenerating dataset ...")
    df = get_dataset()

print("\n--- Dataset Overview ---")
print(f"  Total rows (days): {len(df)}")
print(f"  Date range       : {df['date'].min().date()}  to  {df['date'].max().date()}")
print(f"  Columns          : {list(df.columns)}")

print("\n--- Basic Statistics ---")
summary_cols = ["temp_max", "temp_min", "rainfall", "windspeed"]
print(df[summary_cols].describe().round(2).to_string())

rainy_days = (df["rainfall"] > 1).sum()
print(f"\n  Rainy days (rainfall > 1mm): {rainy_days} out of {len(df)} ({100*rainy_days/len(df):.1f}%)")

hottest = df.loc[df["temp_max"].idxmax()]
coldest = df.loc[df["temp_max"].idxmin()]
print(f"  Hottest day: {hottest['date'].date()}  ({hottest['temp_max']}°C)")
print(f"  Coldest day: {coldest['date'].date()}  ({coldest['temp_max']}°C)")

TARGET = "temp_max"

FEATURES = [
    "temp_max_lag1",   
    "temp_max_lag2",    
    "temp_max_lag3",   
    "temp_min_lag1",    
    "temp_rolling_7d", 
    "temp_range_lag1",  
    "rainfall_lag1",    
    "windspeed",        
    "month",           
    "day_of_year",      
]

X = df[FEATURES]
y = df[TARGET]

print(f"\n--- Features & Target ---")
print(f"  Features used : {FEATURES}")
print(f"  Target        : {TARGET}")
print(f"  X shape       : {X.shape}  (rows=days, cols=features)")
print(f"  y shape       : {y.shape}")

split_idx = int(len(df) * (1 - TEST_SIZE))  

X_train = X.iloc[:split_idx]   
X_test  = X.iloc[split_idx:]   
y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]
dates_test = df["date"].iloc[split_idx:]  

print(f"\n--- Train / Test Split ---")
print(f"  Training : {len(X_train)} days  ({df['date'].iloc[0].date()} to {df['date'].iloc[split_idx-1].date()})")
print(f"  Testing  : {len(X_test)} days   ({df['date'].iloc[split_idx].date()} to {df['date'].iloc[-1].date()})")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   
X_test_scaled  = scaler.transform(X_test)        

print("\n--- Training Models ---")

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
print("  [1/3] Linear Regression       ... trained")

ridge = Ridge(alpha=1.0, random_state=RANDOM_SEED)
ridge.fit(X_train_scaled, y_train)
print("  [2/3] Ridge Regression        ... trained")

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    random_state=RANDOM_SEED,
    n_jobs=-1   
)
rf.fit(X_train, y_train)  
print("  [3/3] Random Forest           ... trained")

pred_lr    = lr.predict(X_test_scaled)
pred_ridge = ridge.predict(X_test_scaled)
pred_rf    = rf.predict(X_test)   


def evaluate(y_true, y_pred, model_name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE (°C)": round(mae, 3),
            "RMSE (°C)": round(rmse, 3), "R² Score": round(r2, 4)}


results = [
    evaluate(y_test, pred_lr,    "Linear Regression"),
    evaluate(y_test, pred_ridge, "Ridge Regression"),
    evaluate(y_test, pred_rf,    "Random Forest"),
]

results_df = pd.DataFrame(results)
results_df["Rank"] = results_df["RMSE (°C)"].rank().astype(int)
results_df = results_df.sort_values("Rank")

print("\n--- Model Evaluation Results (Test Set) ---")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
print(f"\n  Best model: {best_model_name}")

importances = rf.feature_importances_ 
sorted_idx  = np.argsort(importances)[::-1]

print("\n--- Feature Importance (Random Forest) ---")
print(f"  {'Feature':<20}  Importance")
print(f"  {'-'*35}")
for i in sorted_idx:
    bar = "█" * int(importances[i] * 50) 
    print(f"  {FEATURES[i]:<20}  {importances[i]:.4f}  {bar}")

print("\n--- Generating Plots ---")
fig, ax = plt.subplots(figsize=(12, 5))

monthly_mean = df.groupby("month")["temp_max"].mean()
monthly_std  = df.groupby("month")["temp_max"].std()
months = monthly_mean.index
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

ax.plot(months, monthly_mean.values, "o-", color="firebrick",
        linewidth=2.5, markersize=7, label="Monthly Avg Max Temp")

ax.fill_between(months,
                monthly_mean - monthly_std,
                monthly_mean + monthly_std,
                alpha=0.2, color="firebrick", label="±1 Std Dev")

ax.set_xticks(months)
ax.set_xticklabels(month_names)
ax.set_xlabel("Month")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Delhi Monthly Maximum Temperature (Seasonal Pattern)", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/01_seasonal_pattern.png")
plt.close()
print("  [1/6] Seasonal pattern plot saved.")

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

axes[0].plot(df["date"], df["temp_max"], color="firebrick", alpha=0.7,
             linewidth=0.8, label="Daily Max Temp")
axes[0].plot(df["date"], df["temp_min"], color="steelblue", alpha=0.7,
             linewidth=0.8, label="Daily Min Temp")
axes[0].set_ylabel("Temperature (°C)")
axes[0].set_title("Delhi Daily Temperature (2020–2023)", fontsize=13, fontweight="bold")
axes[0].legend(loc="upper right")
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(df["date"], df["rainfall"], color="royalblue", alpha=0.6,
            width=1, label="Daily Rainfall")
axes[1].set_ylabel("Rainfall (mm)")
axes[1].set_xlabel("Date")
axes[1].set_title("Daily Rainfall", fontsize=11)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/02_temperature_timeseries.png")
plt.close()
print("  [2/6] Time series plot saved.")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

models_info = [
    ("Linear Regression", pred_lr,    "steelblue"),
    ("Ridge Regression",  pred_ridge, "darkorange"),
    ("Random Forest",     pred_rf,    "green"),
]

for ax, (name, preds, color) in zip(axes, models_info):
    ax.plot(dates_test.values, y_test.values, color="black",
            linewidth=1.5, label="Actual Temp", zorder=3)
    ax.plot(dates_test.values, preds, color=color,
            linewidth=1.2, alpha=0.85, label=f"Predicted ({name})", zorder=2)
    ax.fill_between(dates_test.values, y_test.values, preds,
                    alpha=0.15, color=color)   
    ax.set_ylabel("Temp Max (°C)")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

axes[-1].set_xlabel("Date")
fig.suptitle("Actual vs Predicted Temperature — Test Set", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/03_actual_vs_predicted.png", bbox_inches="tight")
plt.close()
print("  [3/6] Actual vs predicted plot saved.")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, preds, color) in zip(axes, models_info):
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    ax.scatter(y_test, preds, alpha=0.35, s=12, color=color)

    min_val = min(y_test.min(), preds.min()) - 2
    max_val = max(y_test.max(), preds.max()) + 2
    ax.plot([min_val, max_val], [min_val, max_val],
            "k--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual Temperature (°C)")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.set_title(f"{name}\nMAE={mae:.2f}°C  R²={r2:.3f}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

fig.suptitle("Predicted vs Actual Temperature Scatter Plot", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/04_scatter_predicted_vs_actual.png")
plt.close()
print("  [4/6] Scatter plot saved.")

fig, ax = plt.subplots(figsize=(9, 5))

sorted_features    = [FEATURES[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

colors = ["#1a5276" if i == 0 else "#2e86c1" if i < 3 else "#7fb3d3"
          for i in range(len(sorted_features))]

bars = ax.barh(sorted_features[::-1], sorted_importances[::-1],
               color=colors[::-1], edgecolor="white", height=0.6)

for bar, val in zip(bars, sorted_importances[::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance — Random Forest\n(higher = more useful for prediction)",
             fontsize=12, fontweight="bold")
ax.set_xlim(0, sorted_importances[0] * 1.18)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/05_feature_importance.png")
plt.close()
print("  [5/6] Feature importance plot saved.")

fig, axes = plt.subplots(1, 3, figsize=(13, 5))

model_names  = results_df["Model"].tolist()
mae_vals     = results_df["MAE (°C)"].tolist()
rmse_vals    = results_df["RMSE (°C)"].tolist()
r2_vals      = results_df["R² Score"].tolist()
bar_colors   = ["#2e86c1", "#e67e22", "#27ae60"]

metrics = [
    ("MAE (°C)",   mae_vals,  True,  "Lower is better"),  
    ("RMSE (°C)",  rmse_vals, True,  "Lower is better"),
    ("R² Score",   r2_vals,   False, "Higher is better"),
]

for ax, (metric_name, values, lower_better, note) in zip(axes, metrics):
    best_val = min(values) if lower_better else max(values)
    bar_color_list = ["#e74c3c" if v == best_val else "#95a5a6" for v in values]

    ax.bar(model_names, values, color=bar_color_list, width=0.5, edgecolor="white")

    for i, v in enumerate(values):
        ax.text(i, v + (max(values) * 0.01), f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(f"{metric_name}\n({note})", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Model Performance Comparison — Test Set",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/06_model_comparison.png")
plt.close()
print("  [6/6] Model comparison plot saved.")

print("\n")
print("=" * 60)
print("         FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\n  Task    : Predict next-day max temperature for Delhi")
print(f"  Data    : {len(df)} days  ({df['date'].min().date()} — {df['date'].max().date()})")
print(f"  Train   : {len(X_train)} days  |  Test : {len(X_test)} days")
print(f"  Features: {len(FEATURES)}")
print()

header = f"  {'Model':<22} {'MAE (°C)':>10} {'RMSE (°C)':>11} {'R²':>8}  {'Rank':>5}"
print(header)
print("  " + "-" * 58)
for _, row in results_df.iterrows():
    star = " ← BEST" if row["Model"] == best_model_name else ""
    print(f"  {row['Model']:<22} {row['MAE (°C)']:>10.3f} {row['RMSE (°C)']:>11.3f} "
          f"{row['R² Score']:>8.4f}  {int(row['Rank']):>5}{star}")

print()
best_row = results_df.iloc[0]
print(f"  Best Model : {best_model_name}")
print(f"  MAE        : {best_row['MAE (°C)']:.2f}°C  "
      f"(on average, predictions are off by this many degrees)")
print(f"  RMSE       : {best_row['RMSE (°C)']:.2f}°C")
print(f"  R² Score   : {best_row['R² Score']:.4f}  "
      f"(model explains {best_row['R² Score']*100:.1f}% of temperature variance)")

print()
print(f"  Top Feature: {FEATURES[sorted_idx[0]]}  "
      f"(importance = {importances[sorted_idx[0]]:.3f})")

print()
print(f"  Plots saved in: {RESULTS_DIR}/")
for i in range(1, 7):
    names = {1: "Seasonal pattern", 2: "Time series",
             3: "Actual vs predicted", 4: "Scatter plot",
             5: "Feature importance", 6: "Model comparison"}
    print(f"    0{i}_{list(names.values())[i-1].lower().replace(' ', '_')}.png")

print()
print("=" * 60)
print("  Done! Open the results/ folder to see your plots.")
print("=" * 60)