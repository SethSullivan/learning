import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb
import data_visualization as dv 

dv.set_plot_style("cashaback_dark.mplstyle")

# %% Import data
dtypes = {
    "Loan_ID": pl.String,
    "Gender": pl.String,
    "Married": pl.String,
    "Dependents": pl.String,
    "Education": pl.String,
    "Self_Employed": pl.String,
    "ApplicantIncome": pl.Float64,
    "CoapplicantIncome": pl.Float64,
    "LoanAmount": pl.Float64,
    "Loan_Amount_Term": pl.Float64,
    "Credit_History": pl.Float64,
    "Property_Area": pl.String,
    "Loan_Status": pl.String,
}
# https://www.kaggle.com/datasets/burak3ergun/loan-data-set
df = pl.read_csv("data/loan_portfolio/train.csv", schema_overrides=dtypes)
# Lowercase all columns
df = df.rename({col: col.lower() for col in df.columns})
# %% Data preprocessing
# Binaries to 0 and 1
df = df.with_columns(
    pl.when(pl.col("married") == pl.lit("yes")).then(1).otherwise(0).alias("married"),
    pl.when(pl.col("education") == pl.lit("graduate"))
    .then(1)
    .otherwise(0)
    .alias("education"),
    pl.when(pl.col("loan_status") == pl.lit("Y"))
    .then(1)
    .otherwise(0)
    .alias("loan_status"),
    pl.when(pl.col("self_employed") == pl.lit("Y"))
    .then(1)
    .otherwise(0)
    .alias("self_employed"),
    pl.when(pl.col("gender") == pl.lit("Female")).then(1).otherwise(0).alias("gender"),
)
# %%
# Fill missing values with median and add defaulted as a column
df = df.with_columns(
    pl.col("loanamount").fill_null(pl.col("loanamount").median()),
    pl.col("loan_amount_term").fill_null(pl.col("loan_amount_term").median()),
    pl.col("credit_history").fill_null(pl.col("credit_history").median()),
    pl.when(pl.col("loan_status") == 1).then(0).otherwise(1).alias("defaulted"),
)
print(df)

# %% Decide how much capital
# The bank wants to be 95% confident it can cover a loss of X dollars
# Two things: prob_default and loan_loss to get expected loss
# Going to modify the meaning of the data and say that Loan_Status actually means they defaulted

# 1. Build model to define a probability of default
# 2. Multiply that loan_amount
# 3. Monte carlo simulation using probability of default's to generate a distribution of expected losses
# 4. See where the 95% intervals are and that's the answer

# Finding optimal policy 
# 1. Select a prob_default cutoff and simulate expected loss and profits 

# %% Train/test split
df = df.to_pandas()
X = df.drop(
    columns=["defaulted", "loan_status", "loan_id", "dependents", "property_area"]
)
y = df["defaulted"]
FEATURES = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=123, test_size=0.25
)
# Only doing this so pyright stops yelling
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# XGboost model to get probability of default
majority_instances = np.sum(y_train == 0)
minority_instances = np.sum(y_train == 1)
scale_pos_weight = majority_instances / minority_instances
clf = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    # n_features=2,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
)

cols_to_use = FEATURES
_ = clf.fit(
    X_train.loc[:, cols_to_use],
    y_train,
    eval_set=[(X_train[cols_to_use], y_train), (X_test[cols_to_use], y_test)],
    verbose=100,
)  #

# Print auc for all data, this is cheating bc i'm predicting on what i trained but whatever
preds = clf.predict_proba(X[cols_to_use])[
    :, 1
]  # Second index because that's 1, meaning they did default
auc = roc_auc_score(y, preds)
print(auc)

# %% Get expected loss

loan_amounts = df["loanamount"].to_numpy()
original_expected_loss = preds * loan_amounts
fig, ax = plt.subplots()
ax.hist(original_expected_loss, bins=25)
plt.show()

# %% Monte carlo simulation
n = 10000
mean_losses = []
first_person = []
for i in range(n):
    # Roll a 1 or 0 based on our distribution of people
    defaults = np.random.binomial(n=1, p=preds)

    first_person.append(defaults[0])

    expected_loss = defaults * loan_amounts

    mean_losses.append(np.mean(expected_loss))
conf_interval = [np.percentile(mean_losses, 2.5), np.percentile(mean_losses, 97.5)]
fig, ax = plt.subplots()
ax.hist(mean_losses, bins=100)
ax.axvline(conf_interval[0], color="red")
ax.axvline(conf_interval[1], color="red")
plt.show()

# %%
print(f"Average expected loss: {np.mean(original_expected_loss)}")
print(f"Simulated Average expected loss: {np.mean(mean_losses)}")

print(
    f"Bank can be 95% confident that losses won't exceed {np.percentile(original_expected_loss, 97.5)}"
)
print(f"Bank can be 95% confident that losses won't exceed {conf_interval[1]}")

#%% Now use probabilities to find optimal policy
# Max reward and minimize expected loss

# profits is the interest rate on the loan 
INT_RATE = 0.3
recovery_rate = 0.4 # Hard coded, this should be modeled
df['prob_default'] = preds
# 12 months of accumulating interest, this is assuming that they don't pay off the loan, and just pay the interest along the way
df["expected_profit"] = df['loanamount']*(1 - df["prob_default"])*INT_RATE
# EAD is loan amount (not always true, people might pay back part of the loan, but doing this to simplify)
# PD is prob_default calculated by the model
# LGD is (1 - Recovery Rate)
df['expected_loss'] = df["loanamount"]*df["prob_default"]*(1 - recovery_rate)
df['expected_gain'] = df['expected_profit'] - df['expected_loss']

# %% Histogram of profits losses gains

fig = dv.AutoFigure('a;b;c')
fig.axes['a'].hist(df['expected_profit'], bins=50)
fig.axes['b'].hist(df['expected_loss'], bins=50)
fig.axes['c'].hist(df['expected_gain'], bins=50)

# %% 
fig = dv.AutoFigure('a;b;c', figsize=(10,10))

fig.axes['a'].scatter(df['prob_default'], df['expected_profit'])
fig.axes['b'].scatter(df['prob_default'], df['expected_loss'])
fig.axes['c'].scatter(df['prob_default'], df['expected_gain'])

# %% Simulate with different prob cutoffs
prob_default_cutoffs = np.arange(0.02,1,0.01)
expected_profits = []
expected_losses = []
expected_gains = []
for cutoff in prob_default_cutoffs:
    new_df = df[df['prob_default']<cutoff]
    print(len(new_df))
    expected_profits.append(new_df['expected_profit'].sum())
    expected_losses.append(new_df['expected_loss'].sum())
    expected_gains.append(new_df['expected_gain'].sum())
    
# %% 
fig = dv.AutoFigure('a;b;c', figsize=(7,7))

fig.axes['a'].plot(expected_profits)
fig.axes['b'].plot(expected_losses)
fig.axes['c'].plot(expected_gains)
fig.axes['c'].axvline(np.argmax(expected_gains), ymin=0, ymax = 0.63)
fig.axes['c'].text(np.argmax(expected_gains),np.max(expected_gains), rf"Cutoff Policy"+ "\n" + rf"($\tau = {np.argmax(expected_gains)/100}$)", ha='center', va='bottom', fontweight='bold')

fig.axes['a'].set_ylabel("Expected Profit")
fig.axes['b'].set_ylabel("Expected Loss")
fig.axes['c'].set_ylabel("Expected Gain")

for ax in fig.axes.values():
    ax.set_ylim(0,20000)
    ax.set_xlabel("Probability of Default Cutoff")

       
# %%
