import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import data_visualization as dv
from IPython.display import display
wheel = dv.ColorWheel()
dv.set_plot_style("cashaback_dark.mplstyle")

# %% Load the data
dtypes = {
    "id": pl.Int64,
    "SeriousDlqin2yrs": pl.Int64,
    "RevolvingUtilizationOfUnsecuredLines": pl.Float64,
    "age": pl.Int64,
    "NumberOfTime30-59DaysPastDueNotWorse": pl.Int64,
    "DebtRatio": pl.Float64,
    "MonthlyIncome": pl.Float64,
    "NumberOfOpenCreditLinesAndLoans": pl.Int64,
    "NumberOfTimes90DaysLate": pl.Int64,
    "NumberOfTime60-89DaysPastDueNotWorse": pl.Int64,
    "NumberRealEstateLoansOrLines": pl.Int64,
    "NumberOfDependents": pl.Float64,
}

train_pd = pd.read_csv("data/GiveMeSomeCredit/cs-training.csv")
train = pl.DataFrame(train_pd)
test_pd = pd.read_csv("data/GiveMeSomeCredit/cs-test.csv")
test = pl.DataFrame(test_pd)
cols = train.columns[1:]
continuous_cols = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
]
categorical_cols = [
    "SeriousDlqin2yrs",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
# %% Summary tables
display(train_pd.sample(5))
display(test_pd.info())

# %% Explore the data
# Boxplots of continuous data
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

xlocs = np.arange(0, len(continuous_cols), 1)
for i, col in enumerate(continuous_cols):
    ax = axes.flatten()[i]
    data = train.select(pl.col(col))
    dv.boxplot(ax, data=data, x=xlocs[i])
    ax.set_title(col, fontsize=5)

plt.tight_layout()
plt.show()

# %%
# Boxplots of categorial data
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
xlocs = [0, 1]
for i, col in enumerate(categorical_cols):
    ax = axes.flatten()[i]
    # Plot 1's
    df = train.get_column(col).value_counts().sort(by=col, descending=True)
    print(df)
    ax.bar(
        df.get_column(col),
        df.get_column("count"),
        alpha=0.5,
        label="1",
        color=wheel.rak_blue,
    )
    ax.set_title(col, fontsize=5)
plt.tight_layout()
plt.show()
# %%
# Correlation plots of continuous data
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, col in enumerate(continuous_cols):
    ax = axes.flatten()[i]
    sns.histplot(train.select(col), ax=ax, color=wheel.rak_blue, kde=True)
    ax.set_title(col, fontsize=5)
plt.tight_layout()
plt.show()
