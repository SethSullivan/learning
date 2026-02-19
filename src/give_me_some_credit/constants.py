import polars as pl

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
