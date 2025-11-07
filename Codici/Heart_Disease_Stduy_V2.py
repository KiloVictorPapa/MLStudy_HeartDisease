import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception:
    xgb_available = False

# ================= LOAD DATA ===================
df = pd.read_csv("heart.csv")
X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]

# ================== ENCODING ===================
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()

le_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c].astype(str))
    le_dict[c] = le

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# preprocess
preprocess = ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="passthrough")

# =============== PLOT: TARGET DISTRIBUTION ===============
plt.figure(figsize=(6,4))
sns.countplot(x=y, hue=y, palette=["#4C72B0", "#ED8D31"], legend=False)
plt.title("Target Distribution (HeartDisease)")
plt.xlabel("HeartDisease")
plt.ylabel("Count")
legend_elements = [
    mpatches.Patch(color="#4C72B0", label="0 = No"),
    mpatches.Patch(color="#ED8D31", label="1 = Yes")
]
plt.legend(handles=legend_elements, title="Legend")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show(block=False)

# ======================== MODELS ========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "AdaBoost": AdaBoostClassifier(n_estimators=200),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Naive Bayes": GaussianNB()
}
if xgb_available:
    models["XGBoost"] = XGBClassifier(eval_metric="logloss", n_estimators=300, learning_rate=0.05)

# ==================== 5-FOLD ROC (mean ± std) ====================
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
plt.figure(figsize=(10, 8))
mean_fpr = np.linspace(0, 1, 100)
roc_results = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    tprs = []
    aucs = []

    for tr_idx, val_idx in kfold.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        pipe.fit(X_tr, y_tr)
        try:
            y_prob = pipe.predict_proba(X_val)[:, 1]
        except:
            scores = pipe.decision_function(X_val)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min())

        fpr, tpr, _ = roc_curve(y_val, y_prob)
        aucs.append(roc_auc_score(y_val, y_prob))
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_auc = np.mean(aucs)
    roc_results[name] = aucs

    plt.plot(mean_fpr, mean_tpr, lw=2, label=f"{name} (AUC={mean_auc:.3f})")
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1),
                     alpha=0.15)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.title("5-fold ROC w/ band variability (±1 STD)\n")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show(block=False)

# ================ ROC per singolo modello su TEST SET ================
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
    except:
        scores = pipe.decision_function(X_test)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min())

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1], [0,1], "k--")
    plt.title(f"ROC — Test Set ({name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show(block=False)

# ================= METRICHE TEST + MATRICI DI CONDUSIONE =================
test_scores = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
    except:
        scores = pipe.decision_function(X_test)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min())

    test_scores[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, y_prob) if len(np.unique(y_prob))>1 else np.nan
    }

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show(block=False)

final_df = pd.DataFrame(test_scores).T
print("\n=== FINAL TEST SCORES ===")
print(final_df.round(3))

# ================== GRAFICO RIASSUNTIVO DELLE METRICHE ==================
plt.figure(figsize=(16,6))
models_list = final_df.index.tolist()
accuracy = final_df['Accuracy'].fillna(0)
precision = final_df['Precision'].fillna(0)
recall = final_df['Recall'].fillna(0)
f1 = final_df['F1'].fillna(0)
auc = final_df['AUC'].fillna(0)

x = np.arange(len(models_list))
width = 0.15

plt.bar(x - 2*width, accuracy, width=width, label='Accuracy', color='#4C72B0')
plt.bar(x - width, precision, width=width, label='Precision', color='#8172B2')
plt.bar(x, recall, width=width, label='Recall', color='#55A868')
plt.bar(x + width, f1, width=width, label='F1 Score', color='#ED8D31')
plt.bar(x + 2*width, auc, width=width, label='AUC', color='#C44E52')

plt.xticks(x, models_list, rotation=45, ha='right')
plt.ylabel('Score')
plt.title('Performance Metrics of Every Single Model — Test Set')
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show(block=False)

# ======================== FEATURE IMPORTANCE (PERMUTATION) ========================
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    r = permutation_importance(pipe, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1, scoring='accuracy')

    importances = r.importances_mean
    # order features by importance
    indices = np.argsort(importances)[::-1]
    feature_names = X_train.columns.tolist()

    plt.figure(figsize=(10, 6))
    top_n = min(20, len(feature_names))
    top_idx = indices[:top_n]
    plt.barh(np.array(feature_names)[top_idx][::-1], importances[top_idx][::-1], color="#4C72B0")
    plt.title(f"Permutation Feature Importance (accuracy) — {name}")
    plt.xlabel("Mean decrease in accuracy (permutation)")
    plt.tight_layout()
    plt.show(block=False)

plt.show(block=True)
input("\nPremi INVIO per chiudere i grafici e terminare...")
