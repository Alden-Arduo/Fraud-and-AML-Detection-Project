# Fraud-and-AML-Detection-Project

Fraud and AML Detection Project Document
1. Project Title

Hybrid Fraud and Anti-Money Laundering (AML) Detection Using Supervised and Unsupervised Machine Learning

2. Project Objective

The goal of this project is to develop a robust pipeline for proactively detecting suspicious financial transactions and AML red flags. The project demonstrates how a combination of supervised machine learning and unsupervised anomaly detection can surface potential fraud and highlight emerging trends, providing actionable insights for analysts.

3. Tools and Technologies

•	Programming Language: Python

•	Data Processing: pandas, NumPy

•	Machine Learning Models:

  o	LightGBM (supervised classifier)

  o	IsolationForest (unsupervised anomaly detection)

•	Visualization: Matplotlib, Seaborn

•	Explainability: SHAP

4. Data Description

•	Synthetic dataset with 1,500 transactions designed to simulate realistic AML/fraud scenarios.

•	Features include:

  o	Transaction amount (amount)

  o	Customer ID (customer_id)

  o	Merchant ID (merchant_id)

  o	Country (country)

  o	Anomaly score (anomaly_score) from unsupervised detection

  o	Fraud label (fraud) for supervised model training

•	Engineered features:

  o	near_threshold: transactions near reporting thresholds

  o	is_high_risk_jurisdiction: transactions involving high-risk countries

  o	log_amount: log-transformed transaction amounts

5. Methodology

5.1 Data Preparation

•	Synthetic transaction data is generated using random distributions to simulate both normal and suspicious transactions.

•	Feature engineering includes threshold checks, jurisdiction risk indicators, and logarithmic transformation to stabilise variance.

5.2 Model Development

•	Supervised Learning: LightGBM classifier trained on engineered features to predict the probability of fraud.

•	Unsupervised Learning: IsolationForest trained on the same features to assign anomaly scores for unusual patterns.

•	Hybrid Alerts: Top alerts are ranked based on supervised probabilities and anomaly scores for analyst review.

5.3 Model Evaluation

•	Metrics: precision, recall, F1-score, and confusion matrix.

•	Visualizations:

  o	Transaction distribution and threshold markers

  o	Suspicious activity trends over time

  o	Country-level risk analysis

•	Explainability: SHAP provides both global feature importance and local transaction-level explanations.

6. Results

•	Classification Report: High precision and recall on synthetic data due to controlled labels.

•	Top Alerts: Transactions ranked by predicted probability and anomaly score. Includes key details such as customer, merchant, amount, country, and timestamp.

•	Feature Importance: LightGBM and SHAP reveal key drivers of suspicious transactions, including transaction amount and high-risk jurisdiction.

•	Visual Insights:

  o	Histograms and line charts show transaction patterns and trends.

  o	Country-level risk analysis highlights jurisdictions with higher rates of flagged transactions.

7. Key Insights

•	Hybrid detection effectively combines supervised prediction with anomaly detection for broader coverage.

•	SHAP-based explanations improve transparency, allowing analysts to understand why transactions are flagged.

•	Synthetic pipeline demonstrates end-to-end workflow from data preparation to alert generation and visualization.

8. Future Work

•	Apply pipeline to real banking and financial transaction datasets.

•	Include additional features such as transaction velocity, device/IP signals, and network graph relationships.

•	Integrate into an interactive dashboard for real-time monitoring.

•	Implement feedback loops to retrain models based on analyst decisions.

•	Explore ensemble models and threshold tuning for optimal alert prioritization.

9. How to Run

1.	Clone the repository.

2.	Install dependencies:

pip install numpy pandas matplotlib seaborn shap lightgbm scikit-learn

3.	Run the Python script:

python Fraud_AML_Hybrid_Project.py

4.	Review outputs: top alerts, visualizations, SHAP explanations, and evaluation metrics.


10. Conclusion

This project demonstrates a scalable, interpretable, and robust framework for fraud and AML detection. By combining supervised and unsupervised methods, along with explainable AI, financial institutions can detect suspicious activities more effectively while providing transparency to analysts.

11. Visuals/Outputs
<img width="940" height="703" alt="image" src="https://github.com/user-attachments/assets/89c289d9-5158-4462-b2ef-3482da5cc8ce" />
<img width="945" height="670" alt="image" src="https://github.com/user-attachments/assets/c731245a-ec91-4b27-98e4-5716b386e1dd" />
<img width="940" height="452" alt="image" src="https://github.com/user-attachments/assets/61f3d873-fc85-4db1-b2cf-5484b912de13" />
<img width="940" height="430" alt="image" src="https://github.com/user-attachments/assets/dbe07462-b1b2-49de-9973-3d6e6628ba99" />
<img width="940" height="538" alt="image" src="https://github.com/user-attachments/assets/386b5c80-13cd-4db2-9d56-9f828fb8c80d" />
<img width="940" height="616" alt="image" src="https://github.com/user-attachments/assets/51667f0e-56c0-4d2a-98f3-208df8c8d8c5" />
<img width="945" height="730" alt="image" src="https://github.com/user-attachments/assets/60e4c169-057b-4fb1-b2aa-b244ff9d7d80" />







    
       
