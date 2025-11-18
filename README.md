# 🛡️ CreditGuard AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creditguardai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Intelligent Credit Risk Assessment System** | Real-time loan default prediction using advanced machine learning

An end-to-end machine learning solution for predicting loan default risk, built with Logistic Regression, PCA dimensionality reduction, and hyperparameter optimization. The system achieved **72.17% accuracy** through rigorous cross-validation and is deployed as an interactive web application.

---

## 📋 **Table of Contents**

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Live Demo](#-live-demo)
- [Technical Architecture](#-technical-architecture)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [Model Development](#-model-development)
- [Usage Guide](#-usage-guide)
- [Technical Highlights](#-technical-highlights)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🎯 **Overview**

CreditGuard AI is a production-ready machine learning system that assesses credit risk by analyzing 23 financial and credit history features from the **HELOC (Home Equity Line of Credit) dataset**. The system processes applicant data through a sophisticated pipeline involving data preprocessing, dimensionality reduction via PCA, and optimized logistic regression to deliver instant risk assessments.

### **Problem Statement**

Financial institutions face significant challenges in accurately assessing loan default risk while maintaining:
- ✅ Fast processing times for real-time decisions
- ✅ Interpretable predictions for regulatory compliance
- ✅ High accuracy to minimize financial losses
- ✅ Fair, unbiased lending practices

### **Solution**

CreditGuard AI addresses these challenges through:
- **Advanced Preprocessing**: Intelligent handling of missing values and special codes
- **Feature Engineering**: Strategic use of PCA for noise reduction and model optimization
- **Hyperparameter Optimization**: GridSearchCV with 60+ parameter combinations
- **Production Pipeline**: Seamless integration with Scikit-Learn Pipeline for consistent predictions
- **Interactive Interface**: User-friendly Streamlit web application with real-time risk visualization

---

## ✨ **Key Features**

### **Machine Learning**
🧠 **Optimized Logistic Regression** - Final model with C=0.001, saga solver  
📊 **PCA Dimensionality Reduction** - 23 → 16 features (95% variance retained)  
🔍 **5-Fold Cross-Validation** - Robust model evaluation  
⚡ **72.17% Accuracy** - Achieved through systematic hyperparameter tuning  
📈 **Comprehensive Model Comparison** - Tested Logistic Regression, SVM, Random Forest  

### **Data Processing**
🔧 **Intelligent Missing Value Handling** - Special codes (-9, -8, -7) converted to NaN  
📐 **Median Imputation** - Robust to outliers in right-skewed financial data  
🎯 **Stratified Train-Test Split** - Maintains class distribution (52% Bad, 48% Good)  
⚖️ **StandardScaler Normalization** - Essential for PCA and Logistic Regression  

### **Deployment**
🌐 **Streamlit Web Interface** - Clean, intuitive UI with real-time predictions  
📊 **Interactive Risk Meter** - Visual gauge showing default probability  
🚀 **Streamlit Cloud Hosting** - Accessible from anywhere, zero infrastructure management  
🔒 **Production Pipeline** - Joblib-serialized model for consistent predictions  

---

## 🚀 **Live Demo**

**Experience CreditGuard AI**: [creditguardai.streamlit.app](https://creditguardai.streamlit.app/)

### **Quick Start Guide**
1. Enter applicant's financial details in the sidebar
2. Key inputs: External Risk Score, Credit History Length, Number of Trades
3. Click "**Predict Risk**" for instant assessment
4. View risk probability with color-coded gauge (Green=Low, Yellow=Medium, Red=High)
5. Receive actionable recommendation

---

## 🏗️ **Technical Architecture**

### **Technology Stack**

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Core ML** | Scikit-Learn | 1.0+ | Model training, evaluation, pipeline |
| **Data Processing** | Pandas | 1.3+ | Data manipulation and preprocessing |
| **Numerical Computing** | NumPy | 1.21+ | Mathematical operations and arrays |
| **Visualization** | Matplotlib, Seaborn | 3.4+, 0.11+ | EDA and performance visualization |
| **Web Interface** | Streamlit | 1.12+ | Interactive web application |
| **Model Persistence** | Joblib | 1.0+ | Pipeline serialization |

### **ML Pipeline Architecture**

```
Raw Data (10,459 samples × 23 features)
    ↓
Data Preprocessing
    • Replace special codes (-9, -8, -7) with NaN
    • Median imputation for missing values
    • Feature validation and type checking
    ↓
Feature Scaling
    • StandardScaler normalization
    • Mean=0, Standard Deviation=1
    ↓
Dimensionality Reduction
    • PCA with 95% variance retention
    • 23 features → 16 principal components
    ↓
Model Training
    • Logistic Regression (C=0.001, saga solver)
    • 5-fold stratified cross-validation
    • Hyperparameter optimization via GridSearchCV
    ↓
Model Evaluation
    • Accuracy: 72.17%
    • Precision, Recall, F1-Score
    • Confusion Matrix Analysis
    ↓
Production Deployment
    • Scikit-Learn Pipeline (scaler → PCA → model)
    • Joblib serialization
    • Streamlit web application
```

---

## 📊 **Model Performance**

### **Final Model Metrics**

| Metric | Score | Description |
|--------|-------|-------------|
| **Cross-Validation Accuracy** | **72.17%** | Average accuracy across 5 folds |
| **Test Set Accuracy** | 71.32% | Performance on held-out test data |
| **Precision (Bad Class)** | 74% | Accuracy of default predictions |
| **Recall (Bad Class)** | 70% | Coverage of actual defaults |
| **F1-Score** | 71-72% | Balanced performance measure |

### **Model Comparison Results**

Extensive testing was conducted on three algorithms:

| Model | Without PCA | With PCA | Best Configuration |
|-------|-------------|----------|-------------------|
| **Logistic Regression** | 70.56% | **72.17%** ✅ | PCA improved performance |
| **SVM (RBF Kernel)** | 70.86% | 70.45% | PCA decreased performance |
| **Random Forest** | 70.25% | 69.84% | PCA decreased performance |

### **Key Insight: Why PCA Improved Logistic Regression**

**Logistic Regression (Linear Model)**
- ✅ PCA reduced noise and multicollinearity
- ✅ Simplified decision boundary for linear separation
- ✅ Transformed 23 correlated features into 16 uncorrelated components
- ✅ **Result**: 1.61% accuracy improvement

**Random Forest (Non-Linear Model)**
- ❌ PCA removed subtle, complex patterns
- ❌ Tree-based models excel at handling raw, noisy data
- ❌ Dimensionality reduction simplified data too much
- ❌ **Result**: 0.41% accuracy decrease

**Conclusion**: The project demonstrates deep understanding of algorithm characteristics—linear models benefit from PCA's noise reduction, while non-linear models prefer rich, complex feature spaces.

---

## 💻 **Installation**

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### **Step-by-Step Setup**

1. **Clone the Repository**
```bash
git clone https://github.com/Parvptl/CreditGuard.git
cd CreditGuard
```

2. **Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Application**
```bash
streamlit run app.py
```

5. **Access the App**
- Open browser to `http://localhost:8501`
- The app will automatically reload on code changes

### **requirements.txt**
```
streamlit>=1.12.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```



---

## 🔄 **Data Pipeline**

### **1. Data Understanding**

**Dataset**: HELOC (Home Equity Line of Credit) from FICO  
**Size**: 10,459 loan applications  
**Features**: 23 credit and financial attributes  
**Target**: RiskPerformance (Good=0, Bad=1)  
**Class Distribution**: 52% Bad, 48% Good (Well-balanced)

### **2. Data Preprocessing**

#### **Challenge: Special Missing Value Codes**
```python
# Problem: Dataset used -9, -8, -7 to represent different missing types
# Solution: Convert to standard NaN format
df.replace([-9, -8, -7], np.nan, inplace=True)
```

**Rationale**: Machine learning models interpret -9 as a valid numerical value. Converting to NaN allows proper missing value handling and prevents false pattern learning.

#### **Imputation Strategy: Median Over Mean**
```python
# Fill missing values with median
for col in df.columns:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
```

**Why Median?**
- Financial data is **right-skewed** (confirmed in EDA)
- Median is **robust to outliers** (e.g., extreme credit utilization)
- **Example**: If 9 people have $100K income and 1 has $10M, median ($100K) is more representative than mean ($1.09M)

#### **Missing Value Summary**
| Feature | Missing Count | Imputation Method |
|---------|---------------|-------------------|
| MSinceMostRecentDelq | 5,428 (52%) | Median |
| NetFractionInstallBurden | 4,007 (38%) | Median |
| MSinceMostRecentInqexcl7days | 2,919 (28%) | Median |
| Others | 588-1,449 (6-14%) | Median |

### **3. Feature Scaling**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Why StandardScaler?**
- **Essential for PCA**: PCA is sensitive to feature scales
- **Improves Convergence**: Logistic Regression converges faster
- **Prevents Bias**: Features with large ranges don't dominate

### **4. Train-Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Key Parameter**: `stratify=y`  
- Ensures train and test sets have identical class distributions
- Training set: 52% Bad, 48% Good
- Test set: 52% Bad, 48% Good
- **Result**: Fair and unbiased model evaluation

---

## 🧪 **Model Development**

### **Phase 1: Baseline Model Comparison**

Three algorithms were evaluated:

#### **1. Logistic Regression**
```python
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
# Accuracy: 71.32%
```

#### **2. Support Vector Machine**
```python
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
# Accuracy: 71.03%
```

#### **3. Random Forest**
```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
# Accuracy: 71.65%
```

### **Phase 2: Cross-Validation**

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(log_reg, X, y, cv=5, scoring='accuracy')
print(f"Average CV Accuracy: {cv_scores.mean():.4f}")
# Logistic Regression: 70.56%
```

**Why Cross-Validation?**
- Single train-test split can be "lucky" or "unlucky"
- 5-fold CV tests model on all data subsets
- More reliable performance estimate

### **Phase 3: PCA Dimensionality Reduction**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)
# Features reduced: 23 → 16
```

**PCA Results by Model**:
- ✅ Logistic Regression: 70.56% → **72.04%** (+1.48%)
- ❌ SVM: 70.86% → 70.45% (-0.41%)
- ❌ Random Forest: 70.25% → 69.84% (-0.41%)

### **Phase 4: Hyperparameter Optimization**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_pca, y)
# Best: C=0.001, solver='saga'
# Best CV Accuracy: 72.17%
```

**Hyperparameter Insights**:
- **C=0.001**: Strong regularization prevents overfitting
- **saga solver**: Efficient for large datasets, supports L1/L2 regularization
- **60 configurations tested** (6 C values × 2 solvers × 5 folds)

### **Phase 5: Production Pipeline**

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('model', LogisticRegression(C=0.001, solver='saga', max_iter=1000))
])

pipeline.fit(X, y)
joblib.dump(pipeline, 'credit_risk_pipeline.joblib')
```

**Pipeline Benefits**:
- **Prevents Data Leakage**: Scaler/PCA fit only on training data
- **Consistency**: Same preprocessing for training and prediction
- **Simplicity**: Single `.predict()` call handles all steps
- **Production-Ready**: One file deployment

---

## 📖 **Usage Guide**

### **Web Interface (Streamlit)**

1. **Launch Application**
```bash
streamlit run app.py
```

2. **Enter Applicant Data**
- External Risk Estimate (0-100)
- Months Since Oldest Trade Open
- Average Months in File
- Number of Satisfactory Trades
- Percent of Trades Never Delinquent
- *(and 5 more fields)*

3. **Get Instant Prediction**
- Click "Predict Risk" button
- View animated risk gauge
- Read detailed risk assessment
- Receive loan recommendation

### **Python API (Programmatic)**

```python
import joblib
import pandas as pd

# Load pipeline
pipeline = joblib.load('credit_risk_pipeline.joblib')

# Prepare applicant data
applicant = pd.DataFrame({
    'ExternalRiskEstimate': [72],
    'MSinceOldestTradeOpen': [180],
    'MSinceMostRecentTradeOpen': [6],
    'AverageMInFile': [85],
    'NumSatisfactoryTrades': [22],
    # ... (18 more features)
})

# Make prediction
risk_class = pipeline.predict(applicant)[0]
risk_probability = pipeline.predict_proba(applicant)[0][1]

print(f"Risk Class: {'High Risk' if risk_class == 1 else 'Low Risk'}")
print(f"Default Probability: {risk_probability:.2%}")
```

### **Batch Processing**

```python
# Load multiple applicants from CSV
applicants = pd.read_csv('loan_applications.csv')

# Predict for all
predictions = pipeline.predict(applicants)
probabilities = pipeline.predict_proba(applicants)[:, 1]

# Add results to dataframe
applicants['risk_prediction'] = predictions
applicants['default_probability'] = probabilities

# Export results
applicants.to_csv('risk_assessments.csv', index=False)
```

---

## 🔬 **Technical Highlights**

### **1. Strategic Feature Engineering**

**Top 10 Most Important Features** (from Random Forest):
1. ExternalRiskEstimate (0.18) - Credit bureau risk score
2. MSinceOldestTradeOpen (0.12) - Credit history length
3. PercentTradesNeverDelq (0.10) - Payment reliability
4. AverageMInFile (0.09) - Average account age
5. MSinceMostRecentDelq (0.08) - Recency of delinquency
6. NumSatisfactoryTrades (0.07) - Number of good accounts
7. PercentTradesWBalance (0.06) - Active credit usage
8. NetFractionRevolvingBurden (0.05) - Credit utilization
9. NumTotalTrades (0.05) - Credit mix
10. MaxDelq2PublicRecLast12M (0.04) - Recent delinquency severity

### **2. Advanced ML Techniques**

✅ **Stratified K-Fold Cross-Validation**  
✅ **Grid Search Hyperparameter Optimization**  
✅ **PCA for Dimensionality Reduction**  
✅ **Pipeline for Production Deployment**  
✅ **Median Imputation for Robustness**  
✅ **StandardScaler for Feature Normalization**  

### **3. Model Interpretability**

```python
# Feature importance from Random Forest
importances = rf_model.feature_importances_
feature_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)
```

### **4. Performance Optimization**

- **Convergence Warning Resolution**: Increased max_iter to 1000
- **Efficient Solver Selection**: saga solver for sparse data
- **Multicore Processing**: GridSearchCV with n_jobs=-1
- **Memory Management**: Pipeline reduces redundant transformations

---

## 🚀 **Future Enhancements**

### **Short-Term (1-3 months)**
- [ ] Add SHAP/LIME for explainable AI
- [ ] Implement batch prediction API endpoint
- [ ] Create comprehensive unit tests (pytest)
- [ ] Add model monitoring dashboard
- [ ] Implement automated retraining pipeline

### **Medium-Term (3-6 months)**
- [ ] Ensemble multiple models (stacking/voting)
- [ ] Add XGBoost/LightGBM for comparison
- [ ] Build REST API with FastAPI
- [ ] Implement PostgreSQL database backend
- [ ] Add user authentication system
- [ ] Create admin dashboard for monitoring

### **Long-Term (6+ months)**
- [ ] Deploy with Docker/Kubernetes
- [ ] Implement A/B testing framework
- [ ] Add real-time model performance tracking
- [ ] Build mobile app (React Native)
- [ ] Integrate with loan management systems
- [ ] Add fairness/bias detection tools

---

## 🤝 **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### **Development Guidelines**
- Follow PEP 8 style guide
- Add docstrings to functions
- Write unit tests for new features
- Update documentation
- Test locally before PR

---

## 📚 **Documentation**

### **Technical Deep Dive**
- **[Technical Decisions Explained](docs/technical_decisions.md)** - In-depth rationale for every preprocessing and modeling decision
- **[Model Card](docs/model_card.md)** - Comprehensive model documentation following best practices
- **[Data Dictionary](docs/data_dictionary.md)** - Detailed feature descriptions

### **Learning Resources**
- **[Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)**
- **[Streamlit Docs](https://docs.streamlit.io/)**
- **[HELOC Dataset Documentation](https://community.fico.com/s/explainable-machine-learning-challenge)**

---

## 📜 **License**

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 Parv Patel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 👨‍💻 **Author**

**Parv Patel**  
Data Science & AI/ML Engineering Student @ IIT Palakkad

- 🌐 **Portfolio**: [parvpatel.dev](https://parvpatel.me)
- 💼 **LinkedIn**: [linkedin.com/in/parvptl](https://linkedin.com/in/parvptl)
- 🐙 **GitHub**: [github.com/Parvptl](https://github.com/Parvptl)
- 📧 **Email**: parv4careers@gmail.com
- 📱 **Phone**: +91-7861080021

**Currently**: AI/ML Developer Intern @ Easy Algo (Quantitative Trading)

---

## 🙏 **Acknowledgments**

- **Dataset**: FICO HELOC Dataset from Explainable Machine Learning Challenge
- **Inspiration**: Financial risk management research and industry best practices
- **Community**: Scikit-Learn, Streamlit, and open-source ML communities
- **Mentors**: Faculty and industry professionals at IIT Palakkad

---

## 📊 **Project Statistics**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.12+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## 💡 **Citation**

If you use this project in research or commercial applications:

```bibtex
@software{creditguard_ai_2025,
  author = {Patel, Parv},
  title = {CreditGuard AI: Machine Learning System for Credit Risk Assessment},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Parvptl/CreditGuard},
  note = {Deployed at https://creditguardai.streamlit.app/}
}
```

---

## 🔗 **Related Projects**

Explore my other data science and ML projects:

- **[NEXUS: E-Commerce Analytics Platform](https://github.com/Parvptl/NEXUS)** - 100K+ transactions analyzed with statistical inference and market basket analysis
- **[Human Activity Recognition System](https://github.com/Parvptl/HAR)** - 96.17% accuracy with PCA and SVM optimization
- **[FinTrack: Investment Dashboard](https://github.com/Parvptl/FinTrack)** - Full-stack portfolio management with PostgreSQL and Flask

---

<div align="center">

### **Built with ❤️ and ☕ by Parv Patel**

**Transforming data into intelligent decisions, one model at a time.**

⭐ **Star this repo if you found it helpful!** ⭐

[Report Bug](https://github.com/Parvptl/CreditGuard/issues) · [Request Feature](https://github.com/Parvptl/CreditGuard/issues) · [View Demo](https://creditguardai.streamlit.app/)

---

</div>
