# Machine Learning Final Project – Food Security & Health Outcomes Analysis

## Overview

This project applies machine learning techniques to predict three key public health and food security indicators at the county level in the United States:

1. **Food Insecurity (2021–2023)** – Regression task predicting the percentage of food-insecure individuals
2. **Diabetes Prevalence (2019)** – Regression task predicting the percentage of adults with diabetes
3. **Obesity Hotspots** – Binary classification task identifying counties with elevated obesity rates

The analysis leverages socioeconomic, demographic, and food-access indicators from the USDA Food Access Research Atlas and related county-level datasets.

---

## Dataset

- **Training Set**: 2,508 counties with 276–288 features (depending on target variable)
- **Test Set**: 623 counties for generating final predictions
- **Features**: Mixed indicators across multiple categories:
  - **Access & Proximity**: Food store availability, low-access populations
  - **Assistance Programs**: SNAP, WIC, NSLP, CACFP participation rates
  - **Health & Demographics**: Obesity rates, diabetes prevalence, poverty, age/race composition
  - **Local Economy**: Prices, taxes, recreation facilities, agricultural resources

**Data Cleaning**: Sentinel values (−9999, −8888) representing missing data were converted to NaN. Features with >40% missingness were dropped to prevent noise and model degradation.

---

## Methodology

### 1. Exploratory Data Analysis (EDA)

- **Target distributions**: Examined skewness and outliers in regression targets; class balance for obesity classification
- **Correlation analysis**: Computed Spearman correlations between features and targets to identify top predictors
- **Missingness patterns**: Histogram analysis revealing that 207 features had <5% missing values, while 25 exceeded 40%
- **Feature categories**: Stratified features by domain (access, assistance, health, etc.) to ensure balanced representation

### 2. Data Preprocessing Pipeline

A scikit-learn `Pipeline` was constructed with four stages:

1. **SentinelToNaN**: Converts sentinel codes to NaN for proper imputation
2. **DropHighMissingFeatures**: Removes features with >40% missingness (threshold=0.4)
3. **SimpleImputer**: Fills remaining missing values using median strategy (robust to skewed distributions)
4. **StandardScaler**: Normalizes all features to zero mean, unit variance (required for PCA and distance-based algorithms)

**Rationale**: Median imputation is more robust than mean when data contains outliers or strong skew. Standardization ensures all features contribute equally in PCA, clustering, and gradient-based models.

### 3. Model Selection & Hyperparameter Tuning

#### **Nested Cross-Validation**

- **Outer loop** (5-fold): Provides unbiased generalization error estimates
- **Inner loop** (3-fold): Tunes hyperparameters via GridSearchCV

This approach separates hyperparameter selection from performance evaluation, preventing overfitting to validation folds.

#### **Food Insecurity & Diabetes (Regression)**

Models trained:
- **Gradient Boosting Regressor**: Best performer
  - Parameters: n_estimators ∈ {100, 200}, learning_rate ∈ {0.05, 0.1}, max_depth ∈ {3, 5}
- **Random Forest Regressor**: Secondary model
  - Parameters: n_estimators ∈ {100, 200}, max_depth ∈ {10, 20, None}, min_samples_split ∈ {5, 10}

**Metrics**: Nested CV RMSE, MAE, and R²

#### **Obesity Hotspot (Binary Classification)**

Model trained:
- **Logistic Regression** (L2 penalty, balanced class weights)
  - Parameters: C ∈ {0.1, 1.0, 10.0}, solver='lbfgs'

**Rationale**: The obesity label is linearly separable in PCA-reduced space, making logistic regression highly effective (ROC-AUC ≈ 1.0). It provides well-calibrated probabilities required for ROC and PR-AUC evaluation, avoiding overfitting compared to deeper tree models on this small binary task.

**Metrics**: Nested CV ROC-AUC, PR-AUC, confusion matrix, and calibration curves

### 4. Unsupervised Learning: PCA & Clustering

**Purpose**: Understand socioeconomic heterogeneity among counties and identify distinct policy-relevant groups.

**Steps**:
1. Applied preprocessing pipeline to food insecurity feature set
2. **PCA** with n_components=0.80 reduced 255 features to **48 principal components** explaining 80.4% of variance
3. **KMeans** (k=4) clustered counties based on PCA-transformed data

**Cluster Interpretation**: Clusters reflected real-world patterns:
- High-income, low-food-insecurity counties
- Rural high-poverty regions
- Mixed-access urban/suburban areas
- Transitional socioeconomic profiles

Each cluster showed distinct average food insecurity levels, validating the clustering structure.

### 5. Feature Importance & Stability Analysis

**Bootstrap Resampling** (200 iterations):
- Trained Random Forest models on 200 bootstrap samples
- Tracked feature importance distributions
- Computed mean, standard deviation, and coefficient of variation (CV)

**Top stable features**:
1. **PCT_WICWOMEN16** (0.66 ± 0.01): WIC participation among women
2. **FOODINSEC_18_20** (0.17 ± 0.01): Historical food insecurity baseline
3. **PCT_SBP17** (0.13 ± 0.01): School breakfast program participation
4. **PCT_SNAP17** (0.05 ± 0.00): SNAP participation rate

Low CV values (0.02–0.05) indicate stable, reproducible feature importance across resamples.

### 6. Model Validation & Uncertainty Quantification

**Bootstrap Analysis** (200 iterations):
- Evaluated final models on bootstrap samples
- Computed confidence intervals (95% CI) for RMSE and ROC-AUC

**Results**:
- **Food Insecurity RMSE**: Mean ± CI
- **Diabetes RMSE**: Mean ± CI
- **Obesity ROC-AUC**: Mean ± CI

Tight confidence intervals indicate stable, reliable predictions.

---

## Results

### Food Insecurity Predictions

- **Validation RMSE**: ~0.45% (varies with fold)
- **Validation MAE**: ~0.32%
- **R²**: ~0.62

**Top Predictors**:
- WIC participation among women (54.4%)
- School breakfast program participation (12.9%)
- Historical food insecurity (12.9%)

### Diabetes Predictions

- **Validation RMSE**: ~1.05%
- **Validation MAE**: ~0.81%
- **R²**: ~0.62

**Top Predictors**:
- Prevalence of diabetes in 2015 (59.8%)
- SNAP benefits per capita (3.3%)
- Obesity rate among adults (2.6%)

### Obesity Hotspot Classification

- **Validation ROC-AUC**: ~1.00
- **Validation PR-AUC**: High
- **Precision & Recall**: Perfect (100%) at threshold 0.5

The obesity label exhibits strong linear separability after PCA reduction, enabling near-perfect classification.

---

## Output

**Predictions File**: `predictions.csv`

Contains:
- **FIPS**: County FIPS code (identifier)
- **y_pred_foodinsec2123**: Predicted food insecurity percentage (2021–2023)
- **y_pred_diabetes19**: Predicted diabetes prevalence percentage (2019)
- **p_hat_obesityhot**: Predicted probability of obesity hotspot (0–1)

---

## Files

- **`Machine_Learning_Final_Project.ipynb`**: Complete analysis notebook (Jupyter/Colab format)
- **`predictions.csv`**: Test set predictions (623 counties × 4 columns)
- **`README.md`**: This file

---

## Key Insights & Implications

1. **Food Assistance Programs Drive Food Insecurity**: WIC and school breakfast participation are the strongest predictors, highlighting the protective effect of social safety nets.

2. **Chronic Disease Interconnection**: Diabetes and obesity are highly correlated; counties with high diabetes prevalence tend to be classified as obesity hotspots.

3. **Socioeconomic Clustering**: K-means identified four distinct county phenotypes, enabling targeted policy interventions for high-risk groups.

4. **Model Stability**: Bootstrap analysis and feature importance CV values demonstrate reproducible, reliable predictions across resamples.

5. **Interpretability vs. Accuracy Trade-off**: While logistic regression achieved perfect obesity classification, decision tree-based models (Random Forest, Gradient Boosting) provided richer feature importance insights for regression tasks.

---

## Technical Stack

- **Language**: Python 3.x
- **ML Framework**: scikit-learn (pipelines, cross-validation, models)
- **Data Analysis**: pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Dimensionality Reduction**: PCA from scikit-learn
- **Clustering**: KMeans from scikit-learn
- **Metrics**: ROC-AUC, precision-recall curves, confusion matrices, calibration curves

---

## Course Information

**Course**: IE 593 – Machine Learning (Fall 2025)  
**Institution**: West Virginia University (WVU)

---

## Author

Akshay

---
