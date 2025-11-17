# Interpretable Machine Learning for Credit Risk Modeling using SHAP and LIME

## 1. Project Overview
This project builds a predictive model for corporate credit default risk using XGBoost.
The focus is on predictive accuracy **and** interpretability using SHAP (global) and LIME (local).

---

## 2. Model Performance
- **AUC-ROC:** 0.9512
- **Average Precision (PR AUC):** 0.9079


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

---

## 3. Global Interpretability (SHAP)
Top 5 features driving credit default predictions:
- person_income: mean |SHAP| = 0.792357
- loan_percent_income: mean |SHAP| = 0.734730
- loan_int_rate: mean |SHAP| = 0.496188
- person_home_ownership_RENT: mean |SHAP| = 0.358293
- person_home_ownership_OWN: mean |SHAP| = 0.352254

---

## 4. Local Interpretability (LIME)
### Non-default (lowest p)
- person_home_ownership=2: -0.2541
- loan_intent=5: -0.1457
- person_income > 79000.00: -0.1392
- loan_grade=0: -0.1005
- loan_int_rate <= 8.49: -0.0789
- loan_percent_income <= 0.09: -0.0609
- cb_person_default_on_file=0: -0.0301
- person_age > 30.00: -0.0244
- 5000.00 < loan_amnt <= 8000.00: -0.0219
- cb_person_cred_hist_length > 8.00: -0.0148

### Default (highest p)
- loan_percent_income > 0.23: 0.2467
- loan_grade=4: 0.2272
- loan_intent=2: 0.2019
- person_home_ownership=3: 0.1823
- loan_int_rate > 13.11: 0.1684
- person_income <= 38443.25: 0.1522
- person_emp_length <= 2.00: 0.0348
- person_age <= 23.00: 0.0328
- loan_amnt <= 5000.00: -0.0209
- cb_person_default_on_file=0: -0.0176

### Borderline (~0.5)
- loan_percent_income > 0.23: 0.2153
- person_income <= 38443.25: 0.1898
- loan_int_rate > 13.11: 0.1574
- person_home_ownership=0: -0.0976
- loan_intent=3: 0.0702
- loan_amnt > 12000.00: 0.0617
- loan_grade=2: -0.0500
- cb_person_default_on_file=1: 0.0184
- 4.00 < cb_person_cred_hist_length <= 8.00: -0.0140
- person_emp_length <= 2.00: 0.0125

---
## 5. Conclusions
- The model achieves strong AUC and PR AUC, suitable for imbalanced credit default prediction.
- SHAP shows key global drivers such as interest rate, loan amount, and debt-to-income ratio.
- LIME explanations align with SHAP insights, confirming consistency between global and local reasoning.
- Recommendation: Threshold tuning may be required depending on business priorities (precision vs recall).
