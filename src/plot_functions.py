import numpy as np 

def plot_roc(ax, fpr_train, tpr_train, fpr_test, tpr_test, train_auc, test_auc):
    ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), color="grey")
    ax.plot(fpr_train, tpr_train, label="train performance")
    ax.plot(fpr_test, tpr_test, label="test performance")
    ax.text(0.85, 0.1, f"Train AUC: {train_auc:.4f}")
    ax.text(0.85, 0.05, f"Test AUC:  {test_auc:.4f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()