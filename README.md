# 📊 Exploratory Data Analysis (EDA) Pipeline — Internship Task 02

This repository contains **Task-02** of my internship, focused on building a complete **Exploratory Data Analysis (EDA) pipeline** using Python.

After cleaning and preprocessing the dataset in Task-01, this task analyzes the cleaned data to uncover distributions, correlations, anomalies, and meaningful patterns using both statistics and visualizations.

EDA is a critical step before any modeling, as it provides deep understanding of feature behavior and relationships.

---

## 📌 Objective

The goal of this task is to:

✔ Perform statistical exploration of the dataset  
✔ Generate automated visualizations  
✔ Detect outliers using IQR  
✔ Analyze feature correlations  
✔ Produce structured CSV reports  
✔ Summarize insights for downstream ML readiness  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  

---


---

## 🔍 Analysis Performed

### 1. Statistical Exploration

- Descriptive statistics for numeric columns  
- Frequency counts for categorical columns  
- Missing value inspection  

Output:

- `missing_values_report.csv`

---

### 2. Distribution Analysis

Generated histograms and boxplots for:

- id  
- age  
- income  
- signup_day  
- signup_month  
- signup_year  

These plots help identify skewness, spread, and outliers.

Outputs:

- `histograms_numeric.png`  
- `boxplot_*.png`

---

### 3. Categorical Analysis

Bar charts created for:

- city  
- gender  
- email  
- email_raw  
- email_name_conflict  

Outputs:

- `bar_city.png`  
- `bar_gender.png`  
- `bar_email.png`  
- `bar_email_raw.png`  
- `bar_email_name_conflict.png`

---

### 4. Correlation Analysis

A full correlation matrix was generated along with a heatmap visualization to study relationships between numeric features.

Outputs:

- `correlation_matrix.csv`  
- `correlation_heatmap.png`  

Additionally, strongest correlations were extracted:

- `top_positive_correlations.csv`  
- `top_negative_correlations.csv`

---

### 5. Outlier Detection (IQR Method)

Outliers were detected using Interquartile Range (IQR) for all numeric columns.

The report includes:

- Q1  
- Q3  
- IQR  
- Lower bound  
- Upper bound  
- Outlier count  
- Outlier percentage  

Output:

- `outlier_report_iqr.csv`

---

## ⚙️ Pipeline Workflow

1. Load dataset  
2. Separate numeric and categorical columns  
3. Generate descriptive statistics  
4. Create bar charts for categorical features  
5. Create histograms and boxplots for numeric features  
6. Compute correlation matrix  
7. Plot correlation heatmap  
8. Extract strongest positive & negative correlations  
9. Detect outliers using IQR  
10. Save all plots and reports automatically  

Everything runs from a single script.

---
