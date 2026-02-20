# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import xgboost as xgb

plt.ion()


# %% Functions
def plot_roc(ax, fpr_train, tpr_train, fpr_test, tpr_test, train_auc, test_auc):
    ax.plot(fpr_train, tpr_train, label="train performance")
    ax.plot(fpr_test, tpr_test, label="test performance")
    ax.text(0.85, 0.1, f"Train AUC: {train_auc:.4f}")
    ax.text(0.85, 0.05, f"Test AUC:  {test_auc:.4f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()


# %%
train = pd.read_csv("data/GiveMeSomeCredit/cs-training-clean.csv", index_col=0)
test = pd.read_csv("data/GiveMeSomeCredit/cs-test.csv", index_col=0)

X = train.drop(columns=["SeriousDlqin2yrs"])
y = train["SeriousDlqin2yrs"]
FEATURES = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=123, test_size=0.25
)
# Only doing this so pyright stops yelling
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# %% Predict serious delinquency using logistic regression

logistic = LogisticRegression(
    max_iter=10000, l1_ratio=0, solver="lbfgs", class_weight="balanced"
).fit(X_train, y_train)

# %%
score = logistic.score(X_train, y_train)
print("Logistic Regression Accuracy on Training data:", score)

score = logistic.score(X_test, y_test)
print("Logistic Regression Accuracy on Testing data:", score)

# %% PLot the coefficients
coef_df = pd.DataFrame(
    {"feature": X.columns, "coefficient": logistic.coef_[0]}
).sort_values(by="coefficient", ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(coef_df["feature"], coef_df["coefficient"])
plt.xticks(rotation=45, fontsize=8)
plt.ylim(-1, 1)
plt.title("Logistic Regression Coefficients")
plt.tight_layout()
plt.show()

# %%
train_predictions = logistic.predict_proba(X_train)[:, 1]
test_predictions = logistic.predict_proba(X_test)[:, 1]

train_preds_sorted = np.sort(train_predictions)
test_preds_sorted = np.sort(test_predictions)
fig, ax = plt.subplots()
ax.hist(train_preds_sorted)
plt.show()
print(train_preds_sorted)

# %% ROC AUC curve (myself)

thresholds = np.linspace(0, 1, 100)
tprs = []
fprs = []
for i in range(len(thresholds)):
    preds = test_predictions >= thresholds[i]
    true_positives = ((preds == 1) & (y_test == 1)).sum()
    false_positives = ((preds == 1) & (y_test == 0)).sum()
    true_negatives = ((preds == 0) & (y_test == 0)).sum()
    false_negatives = ((preds == 0) & (y_test == 1)).sum()

    tprs.append(true_positives / (true_positives + false_negatives))
    fprs.append(false_positives / (false_positives + true_negatives))

fig, ax = plt.subplots()
ax.plot(fprs, tprs)
plt.show()

# %%
fpr_train, tpr_train, threshold = roc_curve(y_train, train_predictions)
fpr_test, tpr_test, threshold = roc_curve(y_test, test_predictions)
train_auc = roc_auc_score(y_train, train_predictions)
test_auc = roc_auc_score(y_test, test_predictions)

fig, ax = plt.subplots()
plot_roc(ax, fpr_train, tpr_train, fpr_test, tpr_test, train_auc, test_auc)
plt.show()

# %% XGBoost
# Create xgb regression model
# n_estimators is the number of trees, more is more complex, but can lead to overfitting
# scale_pos_weight is for unbalanced data sets, negative meaning minority class
majority_instances = np.sum(y_train == 0)
minority_instances = np.sum(y_train == 1)
scale_pos_weight = majority_instances / minority_instances
reg = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    # n_features=2,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
)
cols_to_use = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
]
_ = reg.fit(
    X_train.loc[:, cols_to_use],
    y_train,
    eval_set=[(X_train[cols_to_use], y_train), (X_test[cols_to_use], y_test)],
    verbose=100,
)

# %% Feature Importance, roc and auc
fig, ax = plt.subplots(figsize=(12, 4))
xlocs = np.arange(0, len(cols_to_use))
ax.bar(xlocs, -np.sort(-reg.feature_importances_))
ax.set_xticks(xlocs, labels=cols_to_use, rotation=45)
ax.set_ylabel("Importance")
plt.tight_layout()
plt.show()

# Calculate roc auc
xg_train_preds = reg.predict(X_train[cols_to_use])
xg_test_preds = reg.predict(X_test[cols_to_use])
fpr_train, tpr_train, _ = roc_curve(y_train, xg_train_preds)
fpr_test, tpr_test, _ = roc_curve(y_test, xg_test_preds)
train_auc = roc_auc_score(y_train, xg_train_preds)
test_auc = roc_auc_score(y_test, xg_test_preds)

# Plot roc auc
fig, ax = plt.subplots()
plot_roc(ax, fpr_train, tpr_train, fpr_test, tpr_test, train_auc, test_auc)
plt.show()
