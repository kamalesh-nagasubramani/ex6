import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE

# --- STUDENT DETAILS ---
ROLL_NUMBER = "ENTER_YOUR_ROLL_NUMBER"
STUDENT_NAME = "ENTER_YOUR_NAME"
print(f"Scenario 5: SMOTE | {STUDENT_NAME} ({ROLL_NUMBER})")

# 1. Load dataset
path = r"C:\Users\kamal\Downloads\fraud_smote.csv"
df = pd.read_csv(path)
X = df.drop('Fraud', axis=1)
y = df['Fraud']
print(f"Before SMOTE distribution:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Apply SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE distribution:\n{pd.Series(y_train_sm).value_counts()}")

# 3. Train models
rf_before = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
rf_after = RandomForestClassifier(n_estimators=50).fit(X_train_sm, y_train_sm)

# 4. Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(['Before (Fraud)', 'After (Fraud)'], [y_train.sum(), y_train_sm.sum()], color=['red', 'green'])
axes[0].set_title('Class Distribution (Fraud Case)')

p_b, r_b, _ = precision_recall_curve(y_test, rf_before.predict_proba(X_test)[:, 1])
p_a, r_a, _ = precision_recall_curve(y_test, rf_after.predict_proba(X_test)[:, 1])
axes[1].plot(r_b, p_b, label='Before SMOTE', color='red')
axes[1].plot(r_a, p_a, label='After SMOTE', color='green')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()
plt.tight_layout()
plt.savefig('scenario_5_smote.png')
plt.show()
