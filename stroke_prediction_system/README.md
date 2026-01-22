🧠 Stroke Prediction Using Machine Learning
📌 Project Overview

Brain stroke is one of the leading causes of death and long-term disability worldwide. Early identification of individuals at high risk can significantly improve medical outcomes.

This project presents a machine learning–based stroke prediction system that analyzes clinical and lifestyle data to predict the likelihood of a brain stroke. The system follows a research-grade, modular ML workflow including data preprocessing, feature engineering, model comparison, optimization, and explainability.

🎯 Objectives

Predict whether a person is at risk of brain stroke (binary classification)

Compare multiple machine learning algorithms

Handle severe class imbalance in medical data

Optimize models with medical priority on recall

Provide model explainability using feature importance

Build a deployable and extensible ML core (UI-ready)

🧠 Problem Type

Type: Binary Classification

Target Variable: stroke

0 → No Stroke

1 → Stroke

📊 Dataset

Source: Public Stroke Prediction Dataset (Kaggle)

Records: ~5,100

Features Include:

Age

Gender

Hypertension

Heart Disease

BMI

Average Glucose Level

Smoking Status

Work Type

Residence Type

⚠️ The dataset is highly imbalanced, with very few stroke cases — a key challenge addressed in this project.

🏗️ Project Structure
stroke_prediction_system/
│
├── data/
│   ├── stroke_data.csv
│   └── processed_stroke_data.csv
│
├── notebooks/
│   ├── part1_data_analysis.ipynb
│   ├── part2_feature_engineering.ipynb
│   ├── part3_model_training.ipynb
│   └── part4_model_optimization.ipynb
│
├── pipelines/
│   ├── scaler.pkl
│   └── smote.pkl
│
├── experiments/
│   └── model_comparison.csv
│
├── models/
│   └── final_stroke_model.pkl
│
├── requirements.txt
└── README.md

⚙️ Tech Stack

Language: Python

Libraries:

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

Imbalanced-learn (SMOTE)

Tools: Jupyter Notebook, VS Code

Version Control: GitHub

🧪 Methodology (Step-by-Step)
🔹 Part 1: Data Engineering & Analysis

Dataset inspection & validation

Missing value handling (BMI imputation)

Class imbalance analysis

Exploratory Data Analysis (EDA)

Correlation analysis

Clean dataset generation

🔹 Part 2: Feature Engineering & Pipeline

One-hot encoding of categorical variables

Feature scaling using StandardScaler

Class imbalance handling using SMOTE

Saving preprocessing pipeline for reuse

🔹 Part 3: Model Training & Comparison

Trained and evaluated the following models:

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

Support Vector Machine (SVM)

Evaluation Metrics:

Accuracy

Precision

Recall (priority metric)

F1-Score

ROC-AUC

🔹 Part 4: Optimization & Explainability

Hyperparameter tuning using GridSearchCV

Recall-focused optimization (medical priority)

Feature importance analysis (model explainability)

Final model serialization for deployment

📈 Key Results

Best Model: Random Forest Classifier

Why: Achieved the best balance between high recall and ROC-AUC

Medical Insight: Reduced false negatives (missed stroke cases)

Top Predictive Features:

Age

Average Glucose Level

Hypertension

Heart Disease

BMI

🩺 Why Recall Matters in This Project

In healthcare applications, false negatives are dangerous.
Predicting “No Stroke” for a high-risk patient can lead to delayed treatment.

➡️ Therefore, recall was prioritized over accuracy during model optimization.

▶️ How to Run the Project
1️⃣ Clone or Download the Repository
git clone <repository-link>


OR download ZIP from GitHub and extract.

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Notebooks (In Order)

Open Jupyter Notebook or VS Code and run:

part1_data_analysis.ipynb

part2_feature_engineering.ipynb

part3_model_training.ipynb

part4_model_optimization.ipynb

⚠️ Run cells sequentially (top to bottom).

🚀 Future Enhancements

Web-based UI using Streamlit or Flask

Real-time clinical data integration

Deep learning models (ANN)

Model explainability using SHAP

Deployment as a REST API

👥 Team Contribution

Ayush Bassi:

Machine Learning implementation

Data preprocessing

Model training & optimization

Evaluation & explainability

📜 Disclaimer

This project is for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.

⭐ Final Note

This project follows industry-standard ML practices, making it suitable for:

Academic evaluation

Research demonstration

Resume & portfolio

Future deployment