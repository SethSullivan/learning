import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import polars as pl
import data_visualization as dv
from datetime import datetime
import seaborn as sns
from sklearn.metrics import mean_squared_error

plt.ion()
dv.set_plot_style("cashaback_dark.mplstyle")
# %% import energy data
df = pl.read_csv("./machine_learning/data/AEP_hourly.csv")
df = df.with_columns(Datetime=pl.col("Datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"))
print(df.head())

# %% Plot data
fig, ax = plt.subplots()
ax.plot(df["Datetime"], df["AEP_MW"], label="AEP_MW")
ax.set_xlabel("Datetime")
ax.set_ylabel("MegaWatts")
ax.set_title("AEP Hourly Data")
ax.legend()
plt.show()

# %% Feature creation
df = df.with_columns(
    hour=pl.col("Datetime").dt.hour(),
    week=pl.col("Datetime").dt.week(),
    day_of_week=pl.col("Datetime").dt.weekday(),
    month=pl.col("Datetime").dt.month(),
    year=pl.col("Datetime").dt.year(),
    day_of_year=pl.col("Datetime").dt.ordinal_day(),
)

# %% Plot a week of data
start = datetime(2010, 1, 1)
end = datetime(2010, 1, 8)
week = df.filter(
    pl.col("Datetime") > start,
    pl.col("Datetime") < end,
).sort(by="Datetime")
fig, ax = plt.subplots()
ax.plot(week["Datetime"], week["AEP_MW"])
plt.show()

# %% Visualize Feature/Target Relationship
fig, ax = plt.subplots()
sns.boxplot(ax=ax, data=df, x="hour", y="AEP_MW")
ax.set_title("Hourly Usage")
plt.show()

fig, ax = plt.subplots()
sns.boxplot(ax=ax, data=df, x="month", y="AEP_MW")
ax.set_title("Monthly Usage")
plt.show()

# %% Train/Test Split
date = datetime(2015, 1, 1)
train = df.filter(pl.col("Datetime") < date)
test = df.filter(pl.col("Datetime") >= date)

fig, ax = plt.subplots()
ax.plot(train["Datetime"], train["AEP_MW"], label="train")
ax.plot(test["Datetime"], test["AEP_MW"], label="test")
# ax.axvline(date)
ax.legend()
ax.set_title("Train/Test Split")
plt.show()


# %% Model
# Define features
TARGET = "AEP_MW"
FEATURES = [c for c in df.columns if c not in [TARGET, "Datetime"]]

# Define input and output
X_TRAIN = train.select(pl.col(FEATURES)).to_pandas()
Y_TRAIN = train.select(pl.col(TARGET)).to_pandas()
X_TEST = test.select(pl.col(FEATURES)).to_pandas()
Y_TEST = test.select(pl.col(TARGET)).to_pandas()

# Create xgb regression model
# n_estimators is the number of trees
reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.001)
reg.fit(
    X_TRAIN,
    Y_TRAIN,
    eval_set=[(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST)],
    verbose=100,
)

# %% Feature Importance
fig, ax = plt.subplots()
xlocs = np.arange(0, len(FEATURES))
ax.bar(xlocs, -np.sort(-reg.feature_importances_))
ax.set_xticks(xlocs, labels=FEATURES)
ax.set_ylabel("Importance")

# %% Forecast on Test
# Predict
predictions = reg.predict(X_TEST)
# Add predictions to OG df
test = test.with_columns(prediction=predictions)
dff = df.clone().sort(by="Datetime")
dff = dff.join(test, on="Datetime", how="full")

# Plot real vs predictions
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dff["Datetime"], dff["AEP_MW"])
ax.plot(dff["Datetime"], dff["prediction"])

plt.show()

# %% Plot one week of predictions
start_date = datetime(2018, 4, 2)
end_date = datetime(2018, 4, 9)
weekdf = dff.filter(pl.col("Datetime").is_between(start_date, end_date))
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(weekdf["Datetime"], weekdf["AEP_MW"])
ax.plot(weekdf["Datetime"], weekdf["prediction"], marker="o")

plt.show()
