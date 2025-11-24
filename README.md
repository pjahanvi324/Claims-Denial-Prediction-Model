## 🏥 Claim Denial Prediction — End-to-End ML Pipeline

This project tackles a core challenge in healthcare analytics: **predicting which medical claims are likely to be denied**.
It uses a fully engineered machine-learning pipeline built on synthetic yet realistic claims data, with a strong focus on domain-driven features and explainability.

---

## 🔍 Project Summary

* Generated **5,000+ synthetic healthcare claims** representing real-world payer scenarios
* Engineered high-signal features such as:

  * allowed/billed ratio
  * paid/allowed ratio
  * prior authorization indicator
  * submission delay
  * underpayment patterns
* Built a clean preprocessing workflow using `ColumnTransformer` + `Pipeline`
* Trained two models: **Logistic Regression** (baseline) and **Random Forest** (nonlinear model)
* Evaluated with comprehensive metrics: accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix
* Extracted **feature importance** to understand key risk drivers

---

## 🧠 Key Insights

Across experiments, denial risk was strongly influenced by:

* Missing prior authorization
* High billed vs allowed discrepancies
* Delays in claim submission
* ER and telehealth claims
* Underpayment indicators

Random Forest achieved higher **recall** and **ROC-AUC**, making it better suited for high-risk claim identification.

---

## 💼 Real-World Relevance

This type of model supports:

* Pre-adjudication denial prevention
* Automated claim routing
* Revenue cycle optimization
* Fraud/waste/abuse detection
* Operational efficiency for insurers and providers

Predicting denials early reduces revenue leakage and improves claim processing workflows.

---

## 🛠 Tech Stack

Python • NumPy • Pandas • scikit-learn • Matplotlib

