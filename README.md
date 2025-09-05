# Heart Disease Project

# Heart Disease Prediction - Complete Machine Learning Pipeline

A comprehensive machine learning project for predicting heart disease using the UCI Heart Disease dataset. This project demonstrates the full ML pipeline from data preprocessing to model deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Steps](#pipeline-steps)
- [Results](#results)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)

## Project Overview

This project implements a complete machine learning pipeline for heart disease prediction, including:

- Data preprocessing and cleaning
- Feature selection and dimensionality reduction
- Multiple supervised learning algorithms
- Unsupervised learning for pattern discovery
- Hyperparameter optimization
- Model deployment readiness

### Objectives

- Build robust binary classification models to predict heart disease presence
- Compare multiple ML algorithms and optimization techniques
- Implement both supervised and unsupervised learning approaches
- Create a reproducible and well-documented ML pipeline

## Dataset

**Source**: UCI Heart Disease Dataset https://archive.ics.uci.edu/dataset/45/heart+disease
**Size**: 303 instances, 14 attributes  
**Target**: Binary classification (0 = No heart disease, 1 = Heart disease present)

### Features

- `age`: Age in years
- `sex`: Gender (1 = male, 0 = female)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure
- `chol`: Serum cholesterol in mg/dl
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting electrocardiographic results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of peak exercise ST segment
- `ca`: Number of major vessels colored by fluoroscopy
- `thal`: Thalium stress test result

## Project Structure

```
Heart_Disease_Project/
├── data/
│   ├── raw/
│   │   └── processed.cleveland.data
│   └── processed/
│       ├── cleaned_data.csv
│       ├── pca_data.csv
│       └── selected_features.csv
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── models/
│   ├── best_model.pkl
│   ├── best_optimized_model.pkl
│   └── model_results.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Pipeline Steps

### 1. Data Preprocessing & Cleaning

- Handle missing values (dropna approach)
- Convert multi-class target to binary classification
- Feature standardization for numerical variables
- Exploratory Data Analysis with visualizations

### 2. Dimensionality Reduction (PCA)

- Principal Component Analysis to reduce feature space
- Retain 95% of variance
- Visualization of explained variance ratios
- Create PCA-transformed dataset

### 3. Feature Selection

- **Random Forest Feature Importance**: Identify most predictive features
- **Recursive Feature Elimination (RFE)**: Iterative feature elimination
- **Chi-Square Test**: Statistical significance testing
- **Consensus Approach**: Select features chosen by multiple methods

### 4. Supervised Learning

Binary classification using:

- **Logistic Regression**: Linear probabilistic model
- **Decision Tree**: Interpretable tree-based model
- **Random Forest**: Ensemble of decision trees
- **Support Vector Machine**: Margin-based classification

**Evaluation Metrics**:

- Accuracy, Precision, Recall, F1-Score
- ROC Curve and AUC Score
- Cross-validation performance

### 5. Unsupervised Learning

- **K-Means Clustering**: Partition-based clustering
- **Hierarchical Clustering**: Agglomerative clustering with dendrograms
- **Cluster Evaluation**: Silhouette Score, Adjusted Rand Index
- **Pattern Discovery**: Compare clusters with actual disease labels

### 6. Hyperparameter Tuning

- **GridSearchCV**: Exhaustive parameter search
- **5-Fold Cross Validation**: Robust performance estimation
- **F1-Score Optimization**: Balanced precision-recall optimization
- **Baseline Comparison**: Performance improvement quantification

## Results

### Best Model Performance

- **Algorithm**: [Will be determined by your results]
- **F1-Score**: [Your best F1-score]
- **Accuracy**: [Your best accuracy]
- **AUC**: [Your best AUC score]

### Feature Importance

The most important features for heart disease prediction:

1. [Feature 1] - [Importance score]
2. [Feature 2] - [Importance score]
3. [Feature 3] - [Importance score]

### Key Insights

- [Add your key findings from the analysis]
- [Model performance comparisons]
- [Feature importance insights]

## Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score | AUC   |
| ------------------- | -------- | --------- | ------ | -------- | ----- |
| Logistic Regression | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    | 0.XXX |
| Decision Tree       | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    | 0.XXX |
| Random Forest       | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    | 0.XXX |
| SVM                 | 0.XXX    | 0.XXX     | 0.XXX  | 0.XXX    | 0.XXX |

## Technologies Used

- **Python 3.8+**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib
- **Development**: Jupyter Notebook

### Key Libraries

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
joblib>=1.0.0
```

## Acknowledgments

- UCI Machine Learning Repository for providing the Heart Disease dataset
- Scikit-learn community for excellent machine learning tools
- Contributors to the open-source libraries used in this project
