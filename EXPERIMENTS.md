# 🧪 Project Experiment & Data Tracking Log

This document tracks all data transformations, quality findings, feature engineering decisions, and model performance metrics across project notebooks.

---

## 📊 Notebook 01: Data Understanding & Exploration

* **Date Executed:** 2026-08-12
* **Primary Objective:** Inspect raw schema, assess class distribution, explore visual trends, and identify quality issues.

### 1. Raw Dataset Overview
* **Dimensions:** 7,043 rows (customers), 21 columns (features)
* **Target Variable:** `Churn`
* **Class Ratio:**
  * Retained (`No`): 5,174 (73.46%)
  * Churned (`Yes`): 1,869 (26.54%)
  * **Class Imbalance Status:** Imbalanced (~3:1 ratio). Imbalance handling required in downstream training.

### 2. Key Exploratory Findings
* **Tenure:** Higher churn rate observed among low-tenure customers (first 1–12 months).
* **Contract Type:** Month-to-month contracts account for the vast majority of customer churn.
* **Internet Service:** Fiber optic customers show significantly higher churn compared to DSL users.
* **Correlation:** `TotalCharges` is strongly correlated with `tenure` (expected, since `TotalCharges ≈ tenure × MonthlyCharges`) — added to the correlation heatmap and outlier boxplots, which originally only covered `tenure` and `MonthlyCharges`.

### 3. Identified Data Quality Issues
* **Missing/Blank Strings:** `TotalCharges` contains 11 blank space strings (`" "`).
* **Identifiers:** `customerID` is a non-predictive unique hash and must be removed.
* **Outliers:** IQR inspection run across all three numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`); detected outliers were **retained rather than removed** — see rationale in Notebook 02.

---

## 🧹 Notebook 02: Data Preprocessing & Cleaning

* **Date Executed:** 2026-08-12
* **Primary Objective:** Clean raw data errors, encode categorical features, map target variable, and export baseline clean dataset.

### 1. Data Cleaning Actions
* **Dropped Feature:** `customerID` removed (reduced dimensions from 21 to 20 columns).
* **Missing Value Imputation:** Converted `TotalCharges` blank spaces to `NaN`, then filled missing values with `0.0` (corresponds to customers with `tenure = 0`).
* **Outlier Decision:** Outliers detected via IQR were **retained, not removed or capped**. Rationale: `tenure`, `MonthlyCharges`, and `TotalCharges` outliers reflect legitimate account attributes (e.g. long-tenured customers, premium plans), not measurement errors. Removing them would discard real customer behavior the model needs to learn from and could bias the dataset toward "typical" customers.

### 2. Feature Encoding & Transformations
* **Target Mapping:** `Churn` mapped directly (`Yes` → `1`, `No` → `0`).
* **Categorical Encoding:** One-Hot Encoding (`pd.get_dummies` with `drop_first=True`) applied to all categorical text columns.
* **Dimension Transformation:**
  * Raw Input Shape: `(7043, 21)`
  * Post-Cleaning & Encoded Shape: `(7043, 31)`

### 3. Output Artifacts
* **File Exported:** `data/telco_cleaned.csv`
* **Null Value Count:** 0 missing values across all 31 features.

---

## ⚖️ Notebook 03: Data Balancing & Feature Selection

* **Date Executed:** 2026-08-12
* **Primary Objective:** Execute a 3-way Stratified Split (Train / Validation / Test), scale continuous features, apply oversampling to training data, and select top predictive features.

### 1. 3-Way Stratified Data Split
* **Train Set:** 70% (`4,930` rows)
* **Validation Set:** 15% (`1,056` rows)
* **Test Set:** 15% (`1,057` rows)
* **Feature Scaling:** `StandardScaler` fitted *only* on `X_train` and transformed onto `X_val` and `X_test` to prevent leakage.

### 2. Imbalance Handling
* **Method:** `SMOTENC` (not vanilla SMOTE) — required because most features are one-hot encoded categorical dummies; vanilla SMOTE would interpolate invalid fractional values between binary columns. SMOTENC samples categorical columns from existing category values while still interpolating the three continuous columns (`tenure`, `MonthlyCharges`, `TotalCharges`).
* **Target Split:** Applied ONLY to training data.
* **Pre-SMOTENC Training Class Ratio:** Retained (`0`): 3,622 | Churned (`1`): 1,308
* **Post-SMOTENC Training Class Ratio:** Retained (`0`): 3,622 | Churned (`1`): 3,622
* **Total Balanced Training Shape:** `(7244, 30)`
* **Validation & Test Sets:** Preserved in original imbalanced ratio (~26.5% positive churn).

### 3. Feature Selection
* **Method:** Mutual Information (`mutual_info_classif`) — chosen over Random Forest importance because it's model-agnostic, captures non-linear feature-target dependency, and handles the mixed binary/continuous feature set correctly (chi-square was ruled out since scaled continuous columns can be negative).
* **Selected Subset:** Top 15 features by MI score.

### 4. Output Artifacts
* **Exported Files:** `X_train.csv`, `X_val.csv`, `X_test.csv`, `y_train.csv`, `y_val.csv`, `y_test.csv`

---

## 🤖 Notebook 04: Model Training, Validation & Testing

* **Date Executed:** 2026-08-12
* **Primary Objective:** Train 4 candidate classifiers, select the final model using validation performance, and evaluate only that model on the held-out test set.

### 1. Model Selection Methodology
* **Candidates:** Logistic Regression, Decision Tree (max_depth=5), Random Forest (n_estimators=100, max_depth=10), Gradient Boosting (n_estimators=100, learning_rate=0.1).
* **Selection Metric:** Highest `Val_F1` (prioritized over accuracy due to class imbalance in the original churn distribution).
* **Test Set Policy:** Evaluated only once, on the single selected model — avoids implicitly overfitting model choice to the test set.

### 2. Results

**Validation metrics (all 4 models):**

| Model | Val_Accuracy | Val_Precision | Val_Recall | Val_F1 | Val_ROC_AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7282 | 0.4921 | 0.7821 | 0.6041 | 0.8279 |
| Decision Tree | 0.7491 | 0.5176 | 0.7857 | 0.6241 | 0.8333 |
| Random Forest | 0.7434 | 0.5110 | 0.7464 | 0.6067 | 0.8255 |
| **Gradient Boosting** | **0.7491** | **0.5173** | **0.8000** | **0.6283** | **0.8354** |

* **Selected Model:** Gradient Boosting (highest Val_F1)

**Test metrics (Gradient Boosting only):**

| Metric | Value |
|---|---|
| Test_Accuracy | 0.7606 |
| Test_Precision | 0.5359 |
| Test_Recall | 0.7438 |
| Test_F1 | 0.6230 |
| Test_ROC_AUC | 0.8286 |

* Test F1 (0.6230) closely tracks validation F1 (0.6283) — no meaningful overfitting to validation.

### 3. Output Artifacts
* `data/model_results.csv`, `models/*.joblib` (all 4 fitted models), `models/best_model_name.joblib`, `data/y_test_pred.npy`, `data/y_test_proba.npy`

---

## 📈 Notebook 05: Model Evaluation & Visualizations

* **Date Executed:** 2026-08-12
* **Primary Objective:** Generate required visualizations for the final model and interpret results for business decision-making.

### 1. Visualizations Produced
* Confusion matrix (Gradient Boosting, test set)
* ROC curves (all 4 models, validation set)
* Precision-Recall curves (all 4 models, validation set)
* Feature importance plot (Gradient Boosting, top 15)
* Validation metric comparison bar chart (all 4 models)

### 2. Top 15 Feature Importances (Gradient Boosting)
1. `tenure`
2. `Contract_Two year`
3. `InternetService_Fiber optic`
4. `PaymentMethod_Electronic check`
5. `MonthlyCharges`
6. `OnlineSecurity_Yes`
7. `TotalCharges`
8. `Dependents_Yes`
9. `InternetService_No`
10. `StreamingMovies_No internet service`
11. `StreamingTV_No internet service`
12. `DeviceProtection_No internet service`
13. `OnlineSecurity_No internet service`
14. `OnlineBackup_No internet service`
15. `TechSupport_No internet service`

### 3. Interpretation
* **Key churn drivers:** `tenure`, `Contract_Two year`, and `InternetService_Fiber optic` dominate by a clear margin, consistent with NB01's exploratory findings — new customers, month-to-month contracts, and fiber optic service are the strongest churn signals.
* **False negative vs. false positive cost:** At 74.38% recall, the model catches roughly 3 in 4 actual churners; missed churners (false negatives) represent lost revenue with no retention intervention, making recall the priority metric over precision here. At 53.59% precision, roughly half of flagged customers are false alarms — an acceptable trade-off for a retention screening tool.
* **Business recommendation:** Prioritize retention offers (contract upgrades, pricing incentives) for new, month-to-month, fiber optic customers; investigate fiber-specific service/pricing issues; review the electronic check payment experience as a secondary friction point.