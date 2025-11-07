import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
# Load data, check also the contents (how many data do I have?)
data = pd.read_csv('heart.csv')
print(f"Shape: {data.shape[0]} lines, {data.shape[1]} columns")
print("Columns:")
print(data.columns.tolist())

# Check for possible duplicates and/or missing data
n_dup = data.shape[0] - data.drop_duplicates().shape[0]
print(f"Duplicate rows found: {n_dup} ({n_dup/data.shape[0]*100:.2f}%)\n")
print("Missing values per column:")
print(data.isnull().sum())

# Splitting of Data -> in our case we decided to opt for an 80/20 division

targetName = 'HeartDisease'
X = data.drop(columns=[targetName])
y = data[targetName]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# DATA PREPROCESSING

# Separate numeric and categorical features
num_cols = X_train.select_dtypes(include=np.number).columns
cat_cols = X_train.select_dtypes(exclude=np.number).columns

# Numeric Data Normalization via z-score principle
scaler = StandardScaler()
X_train_num = pd.DataFrame(
    scaler.fit_transform(X_train[num_cols]),
    columns=num_cols, index=X_train.index
)
X_test_num = pd.DataFrame(
    scaler.transform(X_test[num_cols]),
    columns=num_cols, index=X_test.index
)

# Categorical data conversion:
if len(cat_cols) > 0:
    encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[cat_cols])
    cat_train_encoded = pd.DataFrame(
        encoder.transform(X_train[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_train.index
    )
    cat_test_encoded = pd.DataFrame(
        encoder.transform(X_test[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_test.index
    )
    X_train_final = pd.concat([X_train_num, cat_train_encoded], axis=1)
    X_test_final = pd.concat([X_test_num, cat_test_encoded], axis=1)
else:
    X_train_final = X_train_num
    X_test_final = X_test_num

# Random Forest Model
RFmodel = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight=None
)
RFmodel.fit(X_train_final, y_train)

y_pred = RFmodel.predict(X_test_final)
y_prob = RFmodel.predict_proba(X_test_final)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n####### Model performance on test set #######")
print(f"Accuracy: {accuracy:.3f}")
print(f"AUC: {auc:.3f}\n")

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], '--k')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve - Test Set')
plt.grid(True)
plt.legend(loc='lower right')

plt.show()

# ================================
# CONFUSION MATRIX
# ================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ================================
# METRICS BOX
# ================================
plt.figure(figsize=(5, 3))
plt.axis('off')

metrics_text = (
    f"Accuracy:  {accuracy:.3f}\n"
    f"Precision: {precision:.3f}\n"
    f"Recall:    {recall:.3f}\n"
    f"F1 Score:  {f1:.3f}\n"
    f"AUC:       {auc:.3f}"
)

plt.text(0.5, 0.5, metrics_text,
         ha='center', va='center',
         fontsize=11, fontweight='bold',
         bbox=dict(facecolor='white', edgecolor='black'))
plt.title('Performance Metrics', fontsize=13, fontweight='bold', pad=20)
plt.show()

# ================================
# FEATURE IMPORTANCE
# ================================
importances = RFmodel.feature_importances_
feature_names = X_train_final.columns

# --- Group importance by base feature name ---
grouped_importance = {}
for name, value in zip(feature_names, importances):
    base = name.split('_')[0]
    grouped_importance[base] = grouped_importance.get(base, 0) + value

# Convert to sorted DataFrame
imp_df = pd.DataFrame({
    'Feature': list(grouped_importance.keys()),
    'Importance': list(grouped_importance.values())
}).sort_values('Importance', ascending=False)

# --- Plot horizontal bar ---
plt.figure(figsize=(8, 6))
plt.barh(imp_df['Feature'], imp_df['Importance'], color=(0.2, 0.5, 0.8))
plt.gca().invert_yaxis()
plt.xlabel('Relative Importance')
plt.ylabel('Predictive Variables')
plt.title('Random Forest Feature Importances', fontweight='bold')
plt.grid(axis='x')
plt.show()

# ================================
# FINAL ROC (VALIDATION)
# ================================
fpr_final, tpr_final, _ = roc_curve(y_test, y_prob)
auc_final = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 6))
plt.plot(fpr_final, tpr_final, linewidth=2, label=f'AUC = {auc_final:.3f}')
plt.plot([0, 1], [0, 1], '--k')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve - Validation Set')
plt.grid(True)
plt.legend(loc='lower right')


plt.show()
