import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data = {
    'income': [50000, 60000, 25000, 80000, 120000, 30000, 70000, 40000, 90000, 100000],
    'debt': [10000, 20000, 15000, 30000, 50000, 10000, 20000, 15000, 40000, 30000],
    'credit_history': [5, 10, 2, 8, 15, 3, 7, 4, 12, 9],
    'num_credit_cards': [2, 4, 1, 3, 5, 1, 3, 2, 4, 5],
    'loan_amount': [20000, 30000, 10000, 40000, 60000, 15000, 25000, 20000, 50000, 45000],
    'employment_status': ['employed', 'employed', 'unemployed', 'employed', 'employed',
                          'unemployed', 'employed', 'unemployed', 'employed', 'employed'],
    'target': [1, 1, 0, 1, 1, 0, 1, 0, 1, 1]  # 1 = Good, 0 = Bad
}

df = pd.DataFrame(data)

print("Dataset:\n", df.head())



# Encode categorical column
le = LabelEncoder()
df['employment_status'] = le.fit_transform(df['employment_status'])

# Features & Target
X = df.drop('target', axis=1)
y = df['target']

# Feature Engineering
X['debt_to_income'] = X['debt'] / X['income']
X['loan_to_income'] = X['loan_amount'] / X['income']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

feature_names = X.columns
importance = model.feature_importances_

print("\nFeature Importance:")
for name, score in zip(feature_names, importance):
    print(f"{name}: {score:.4f}")

print("\nTest New Customer:")

new_customer = pd.DataFrame({
    'income': [75000],
    'debt': [20000],
    'credit_history': [6],
    'num_credit_cards': [3],
    'loan_amount': [30000],
    'employment_status': ['employed']
})

# Encode
new_customer['employment_status'] = le.transform(new_customer['employment_status'])

# Feature Engineering
new_customer['debt_to_income'] = new_customer['debt'] / new_customer['income']
new_customer['loan_to_income'] = new_customer['loan_amount'] / new_customer['income']

# Scale
new_scaled = scaler.transform(new_customer)

# Predict
prediction = model.predict(new_scaled)
probability = model.predict_proba(new_scaled)[0][1]

if prediction[0] == 1:
    print("Prediction: Good Credit Risk ")
else:
    print("Prediction: Bad Credit Risk ")

print("Probability of Good Credit:", probability)