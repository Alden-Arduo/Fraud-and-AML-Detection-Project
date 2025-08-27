import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest

# -----------------------
# 1. Generate synthetic transaction data (hybrid features)
# -----------------------
np.random.seed(42)
N = 1500

# Dates
dates = pd.date_range(start="2024-01-01", periods=N, freq="h")

# Base features
amount = np.random.exponential(scale=100, size=N)
customer_id = np.random.randint(1000, 2000, size=N)
merchant_id = np.random.randint(2000, 3000, size=N)
country = np.random.choice(["NZ", "AU", "US", "KY", "PA"], size=N, p=[0.6, 0.2, 0.15, 0.025, 0.025])

# Anomaly score (unsupervised)
anomaly_score = np.random.uniform(0, 1, size=N)

# Fraud label
fraud = np.random.choice([0, 1], size=N, p=[0.95, 0.05])

# DataFrame
df = pd.DataFrame({
    "date": dates,
    "amount": amount,
    "customer_id": customer_id,
    "merchant_id": merchant_id,
    "country": country,
    "anomaly_score": anomaly_score,
    "fraud": fraud
})

# Feature engineering
# AML-inspired features
df['near_threshold'] = (df['amount'] > 9000).astype(int)
df['is_high_risk_jurisdiction'] = df['country'].isin(['KY', 'PA']).astype(int)
df['log_amount'] = np.log1p(df['amount'])

# -----------------------
# 2. Train-test split
# -----------------------
X = df[['log_amount', 'near_threshold', 'is_high_risk_jurisdiction', 'anomaly_score']]
y = df['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------
# 3. Train supervised model (LightGBM)
# -----------------------
clf = LGBMClassifier(random_state=42)
clf.fit(X_train, y_train)

# -----------------------
# 4. Train unsupervised anomaly detector
# -----------------------
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train)
iso_scores = -iso.score_samples(X_test)

# -----------------------
# 5. Predictions and evaluation
# -----------------------
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# -----------------------
# 6. Feature importance and SHAP
# -----------------------
plt.figure(figsize=(6,4))
sns.barplot(x=clf.feature_importances_, y=X_train.columns)
plt.title('Feature Importance (LightGBM)')
plt.show()

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')
shap.summary_plot(shap_values, X_test)

# -----------------------
# 7. Generate alerts combining supervised and unsupervised scores with transaction details
# -----------------------
alerts = X_test.copy()
alerts['probability'] = y_pred_proba
alerts['anomaly_score'] = iso_scores
alerts['fraud_label'] = y_test.values

# Merge original transaction info
alerts_full = alerts.merge(df[['date', 'amount', 'customer_id', 'merchant_id', 'country']], left_index=True, right_index=True)

# Sort top alerts by probability
top_alerts = alerts_full.sort_values('probability', ascending=False).head(5)
print("\nTop 5 alerts with transaction details:")
print(top_alerts[['date','customer_id','merchant_id','country','amount','probability','anomaly_score','fraud_label']])

# -----------------------
# 8. Visualisations (transaction trends and country risk)
# -----------------------
plt.figure(figsize=(10,5))
sns.histplot(df['amount'], bins=50, kde=False)
plt.axvline(10000, color='red', linestyle='--', label='10k threshold')
plt.title('Transaction Amount Distribution')
plt.legend()
plt.show()

# Suspicious activity over time
df_grouped = df.groupby(df['date'].dt.date)['fraud'].sum().reset_index()
plt.figure(figsize=(10,5))
plt.plot(df_grouped['date'], df_grouped['fraud'], marker='o')
plt.title('Suspicious Transactions Over Time')
plt.xlabel('Date')
plt.ylabel('Count of Suspicious Transactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Country-level suspicious rates
country_summary = df.groupby('country')['fraud'].mean()
plt.figure(figsize=(6,4))
plt.bar(country_summary.index, country_summary.values)
plt.title('Suspicious Rate by Country')
plt.show()
