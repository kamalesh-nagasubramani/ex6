import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# --- STUDENT DETAILS ---
ROLL_NUMBER = "ENTER_YOUR_ROLL_NUMBER"
STUDENT_NAME = "ENTER_YOUR_NAME"
print(f"Scenario 1: Bagging | {STUDENT_NAME} ({ROLL_NUMBER})")

# 1. Load dataset
path = r"C:\Users\kamal\Downloads\diabetes_bagging.csv"
df = pd.read_csv(path)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

# 3. Apply BaggingClassifier
bag_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
bag_model.fit(X_train, y_train)
bag_pred = bag_model.predict(X_test)
bag_acc = accuracy_score(y_test, bag_pred)

# 4. Results
print(f"Decision Tree Accuracy: {dt_acc:.4f}")
print(f"Bagging Classifier Accuracy: {bag_acc:.4f}")

# 5. Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(['Decision Tree', 'Bagging Classifier'], [dt_acc, bag_acc], color=['skyblue', 'lightgreen'])
axes[0].set_title('Accuracy Comparison')
sns.heatmap(confusion_matrix(y_test, bag_pred), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Confusion Matrix (Bagging)')
plt.tight_layout()
plt.savefig('scenario_1_bagging.png')
plt.show()
