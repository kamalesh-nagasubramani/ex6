import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier

# --- STUDENT DETAILS ---
ROLL_NUMBER = "ENTER_YOUR_ROLL_NUMBER"
STUDENT_NAME = "ENTER_YOUR_NAME"
print(f"Scenario 4: Stacking | {STUDENT_NAME} ({ROLL_NUMBER})")

# 1. Load dataset
path = r"C:\Users\kamal\Downloads\heart_stacking.csv"
df = pd.read_csv(path)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Train base models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier()
}
results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# 3. Stacking
estimators = [('lr', models['Logistic Regression']), ('svm', models['SVM']), ('dt', models['Decision Tree'])]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack.fit(X_train, y_train)
stack_acc = accuracy_score(y_test, stack.predict(X_test))
results['Stacking'] = stack_acc
print(f"Stacking Accuracy: {stack_acc:.4f}")

# 4. Visualization
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange', 'red'])
plt.title('Base Models vs Stacking')
plt.ylabel('Accuracy')
plt.savefig('scenario_4_stacking.png')
plt.show()
