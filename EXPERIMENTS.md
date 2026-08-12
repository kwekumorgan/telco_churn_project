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

### 3. Identified Data Quality Issues
* **Missing/Blank Strings:** `TotalCharges` contains 11 blank space strings (`" "`).
* **Identifiers:** `customerID` is a non-predictive unique hash and must be removed.
* **Outliers:** IQR inspection shows no extreme or unphysical values in numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`).

---

## 🧹 Notebook 02: Data Preprocessing & Cleaning

* **Date Executed:** 2026-08-12
* **Primary Objective:** Clean raw data errors, encode categorical features, map target variable, and export baseline clean dataset.

### 1. Data Cleaning Actions
* **Dropped Feature:** `customerID` removed (reduced dimensions from 21 to 20 columns).
* **Missing Value Imputation:** Converted `TotalCharges` blank spaces to `NaN`, then filled missing values with `0.0` (corresponds to customers with `tenure = 0`).

### 2. Feature Encoding & Transformations
* **Target Mapping:** `Churn` mapped directly (`Yes` $\rightarrow$ `1`, `No` $\rightarrow$ `0`).
* **Categorical Encoding:** One-Hot Encoding (`pd.get_dummies` with `drop_first=True`) applied to all categorical text columns.
* **Dimension Transformation:**
  * Raw Input Shape: `(7043, 21)`
  * Post-Cleaning & Encoded Shape: `(7043, 31)`

### 3. Output Artifacts
* **File Exported:** `data/telco_cleaned.csv`
* **Null Value Count:** 0 missing values across all 31 features.

# Decision: outliers are retained, not removed or capped.
# Rationale: tenure, MonthlyCharges, and TotalCharges are legitimate account
# attributes (e.g. long-tenured customers, premium plans), not measurement
# errors. Removing them would discard real customer behavior that the model
# needs to learn from, and could bias the dataset toward "typical" customers.

---


## ⚖️ Notebook 03: Data Balancing & Feature Selection

* **Date Executed:** 2026-08-12
* **Primary Objective:** Execute a 3-way Stratified Split (Train / Validation / Test), scale continuous features, apply SMOTE oversampling to training data, and select top predictive features.

### 1. 3-Way Stratified Data Split
* **Train Set:** 70% (`4,930` rows)
* **Validation Set:** 15% (`1,056` rows)
* **Test Set:** 15% (`1,057` rows)
* **Feature Scaling:** `StandardScaler` fitted *only* on `X_train` and transformed onto `X_val` and `X_test` to prevent leakage.

### 2. Imbalance Handling (SMOTE)
* **Target Split:** Applied ONLY to training data.
* **Pre-SMOTE Training Class Ratio:** Retained (`0`): 3,622 | Churned (`1`): 1,308
* **Post-SMOTE Training Class Ratio:** Retained (`0`): 3,622 | Churned (`1`): 3,622
* **Total Balanced Training Shape:** `(7244, 30)`
* **Validation & Test Sets:** Preserved in original imbalanced ratio (~26.5% positive churn).

### 3. Feature Selection
* **Method:** Random Forest Feature Importance.
* **Selected Subset:** Top 15 features (e.g., `tenure`, `TotalCharges`, `MonthlyCharges`, `Contract_Two year`, `InternetService_Fiber optic`).

### 4. Output Artifacts
* **Exported Files:** `X_train.csv`, `X_val.csv`, `X_test.csv`, `y_train.csv`, `y_val.csv`, `y_test.csv`
---

## ⏳ Notebook 04: Training, Validation & Testing (Pending)
*(To be populated after executing Notebook 4)*

---

## ⏳ Notebook 05: Model Evaluation & Visualizations (Pending)
*(To be populated after executing Notebook 5)*
