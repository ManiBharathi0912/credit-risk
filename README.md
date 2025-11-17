# Interpretable Machine Learning for Credit Risk Modeling using SHAP and LIME

## 📌 Project Overview
This project builds a predictive model for corporate credit default risk using XGBoost.
The focus is not only on predictive accuracy but also on **interpretability** using SHAP (global explanations) and LIME (local explanations).

---

## 🚀 Features
- Data preprocessing (cleaning, encoding, feature engineering)
- Binary classification using **XGBoost**
- Model evaluation with **AUC-ROC**, **Precision-Recall curves**, and **Confusion Matrix**
- Global interpretability with **SHAP summary plots**
- Local interpretability with **LIME explanations** for specific cases
- Concise report summarizing findings and recommendations

---

## 📊 Results
- **AUC-ROC:** 0.9512
- **PR AUC:** 0.9079


Confusion Matrix:
[[5067 28]
 [398 1024]]


**Classification Report:**
```
              precision    recall  f1-score   support

           0     0.9272    0.9945    0.9597      5095
           1     0.9734    0.7201    0.8278      1422

    accuracy                         0.9346      6517
   macro avg     0.9503    0.8573    0.8937      6517
weighted avg     0.9373    0.9346    0.9309      6517

```

**Top 5 SHAP Features:**
- person_income: mean |SHAP| = 0.792357
- loan_percent_income: mean |SHAP| = 0.734730
- loan_int_rate: mean |SHAP| = 0.496188
- person_home_ownership_RENT: mean |SHAP| = 0.358293
- person_home_ownership_OWN: mean |SHAP| = 0.352254

**LIME Explanations:**

### Non-default (lowest p)
- person_home_ownership=2: -0.2644
- loan_intent=5: -0.1392
- person_income > 79000.00: -0.1371
- loan_grade=0: -0.0862
- loan_percent_income <= 0.09: -0.0576
- loan_int_rate <= 8.49: -0.0571
- 5000.00 < loan_amnt <= 8000.00: -0.0503
- person_age > 30.00: -0.0242
- cb_person_cred_hist_length > 8.00: 0.0050
- person_emp_length > 7.00: 0.0037

### Default (highest p)
- loan_percent_income > 0.23: 0.2558
- person_home_ownership=3: 0.2019
- person_income <= 38443.25: 0.1878
- loan_grade=4: 0.1761
- loan_int_rate > 13.11: 0.1602
- loan_intent=2: 0.1547
- person_emp_length <= 2.00: 0.0339
- cb_person_default_on_file=0: -0.0246
- person_age <= 23.00: 0.0204
- loan_amnt <= 5000.00: -0.0164

### Borderline (~0.5)
- loan_percent_income > 0.23: 0.2540
- person_income <= 38443.25: 0.1453
- loan_int_rate > 13.11: 0.1368
- person_home_ownership=0: -0.1184
- loan_intent=3: 0.0695
- loan_grade=2: -0.0658
- loan_amnt > 12000.00: 0.0526
- person_emp_length <= 2.00: 0.0349
- 26.00 < person_age <= 30.00: -0.0102
- cb_person_default_on_file=1: 0.0031


---

## 📦 Deliverables
- Full code implementation (`credit_risk_xai_colab.ipynb`)
- Report (`summary.md`)
- SHAP feature importance values (`shap_top5.csv`)
- LIME explanations (`lime_*.html`)

---

## ✅ Submission
- Use GitIngest to extract code and generate markdown.
- Paste the markdown + report into the submission portal.
- Ensure minimum 100 characters in your submission text.

---

## 📌 Author
- **Sudharsan**
Practical, detail-oriented, and results-driven — focused on interpretable ML for credit risk.
