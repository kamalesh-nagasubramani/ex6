import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# --- STUDENT DETAILS ---
ROLL_NUMBER = "ENTER_YOUR_ROLL_NUMBER"
STUDENT_NAME = "ENTER_YOUR_NAME"
print(f"Scenario 3: Random Forest | {STUDENT_NAME} ({ROLL_NUMBER})")

# 1. Load dataset
path = r"C:\Users\kamal\Downloads\income_random_forest.csv"
df = pd.read_csv(path)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('Income', axis=1)
y = df['Income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Tune number of trees
tree_counts = [10, 50, 100, 150]
accuracies = []
for n in tree_counts:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    accuracies.append(acc)
    print(f"Trees: {n} | Accuracy: {acc:.4f}")

# 3. Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(tree_counts, accuracies, marker='o', color='indigo')
axes[0].set_title('Accuracy vs Number of Trees')
rf_final = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
pd.Series(rf_final.feature_importances_, index=X.columns).nlargest(10).plot(kind='barh', ax=axes[1], color='coral')
axes[1].set_title('Feature Importance (n=100)')
plt.tight_layout()
plt.savefig('scenario_3_rf.png')
plt.show()
