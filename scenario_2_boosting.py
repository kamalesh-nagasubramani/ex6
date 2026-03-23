import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# --- STUDENT DETAILS ---
ROLL_NUMBER = "ENTER_YOUR_ROLL_NUMBER"
STUDENT_NAME = "ENTER_YOUR_NAME"
print(f"Scenario 2: Boosting | {STUDENT_NAME} ({ROLL_NUMBER})")

# 1. Load dataset
path = r"C:\Users\kamal\Downloads\churn_boosting.csv"
df = pd.read_csv(path)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train AdaBoost
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train, y_train)
ada_acc = accuracy_score(y_test, ada.predict(X_test))

# 3. Train Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))

print(f"AdaBoost Accuracy: {ada_acc:.4f}")
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

# 4. Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fpr_ada, tpr_ada, _ = roc_curve(y_test, ada.predict_proba(X_test)[:, 1])
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb.predict_proba(X_test)[:, 1])
axes[0].plot(fpr_ada, tpr_ada, label=f'AdaBoost (AUC={auc(fpr_ada, tpr_ada):.2f})')
axes[0].plot(fpr_gb, tpr_gb, label=f'GradBoost (AUC={auc(fpr_gb, tpr_gb):.2f})')
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set_title('ROC Curve')
axes[0].legend()

pd.Series(gb.feature_importances_, index=X.columns).nlargest(8).plot(kind='barh', ax=axes[1], color='teal')
axes[1].set_title('Feature Importance (GradBoost)')
plt.tight_layout()
plt.savefig('scenario_2_boosting.png')
plt.show()
