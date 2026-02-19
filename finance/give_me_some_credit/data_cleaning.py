# %%
import data_visualization as dv
import pandas as pd

wheel = dv.ColorWheel()
dv.set_plot_style("cashaback_dark.mplstyle")

# %%
train = pd.read_csv("data/GiveMeSomeCredit/cs-training.csv", index_col=0)
cols = train.columns[1:]

# %% Outliers to remove
# These are based on what feels reasonable when looking at the boxpltos
outliers = {
    "RevolvingUtilizationOfUnsecuredLines": (0, 10000),
    "age": (0, 100),
    "DebtRatio": (0, 50000),
    "MonthlyIncome": (0, 0.25e6),
    "NumberOfOpenCreditLinesAndLoans": (0, 30),
    "NumberRealEstateLoansOrLines": (0, 6),
    "NumberOfTime30-59DaysPastDueNotWorse": (0, 10),
    "NumberOfTime60-89DaysPastDueNotWorse": (0, 10),
    "NumberOfTimes90DaysLate": (0, 10),
    "NumberOfDependents": (0, 5),
}


# %% Deal with missing values
# MonthlyIncome has 29731 missing values, which is about 20% of the data
# NumberOfDependents has 3924 missing values, which is about 2.6% of the data
# For MonthlyIncome, we can fill with the median, since the mean is likely to be skewed by outliers
# For NumberOfDependents, we can fill with the mode, since it's discrete integers
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df.fillna({"MonthlyIncome": df["MonthlyIncome"].median()}, inplace=True)
    df.fillna({"NumberOfDependents": df["NumberOfDependents"].median()}, inplace=True)

    return df


# %% Remove outliers
def detect_outliers(df, col, num_stds=3):
    mean = df[col].mean()
    std = df[col].std()
    outliers = df[(df[col] - mean).abs() > num_stds * std]
    return outliers


def remove_outliers(
    df: pd.DataFrame, outlier_dict: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    new_rows = []
    for col, (lower, upper) in outlier_dict.items():
        new_rows.append(df[(df[col] >= lower) & (df[col] <= upper)])
    final_df = pd.concat(new_rows, axis=0).drop_duplicates()
    return final_df


def clean_df(
    df: pd.DataFrame, outlier_dict: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    df = fill_missing_values(df)
    print(df.isna().sum())
    assert df.isna().sum().sum() == 0, (
        "There are still missing values in the training data"
    )
    df = remove_outliers(df, outlier_dict)
    return df


# %%
clean_train = clean_df(train, outliers)

clean_train.to_csv("data/GiveMeSomeCredit/cs-training-clean.csv")
