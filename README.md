# RandomForest-Oral-Cancer-Risk-Prediction

This project builds a machine learning pipeline to estimate the risk of
oral cancer using demographic, lifestyle, and clinical indicators.

The focus is on **risk assessment, feature analysis, and model evaluation**,
not clinical diagnosis.

---

## Why This Project?

Oral cancer risk is influenced by a combination of:
- Lifestyle factors (tobacco, alcohol, diet)
- Infections (HPV)
- Clinical symptoms (lesions, bleeding)
- Demographics and geography

This project explores how these factors interact and contribute to
cancer risk using interpretable machine learning models.

---

## Dataset

- Tabular dataset with demographic, behavioral, and clinical features
- Includes categorical and numerical variables
- Target: **Oral Cancer (Diagnosis)**

> ⚠️ Dataset is used for educational and research purposes only.

---

## Pipeline Overview

### Stage 1: Data Preprocessing
- Categorical encoding (label + one-hot)
- Leakage feature removal
- Correlation analysis
- Multicollinearity reduction using VIF

### Stage 2: Feature Selection
- Mutual Information scoring
- Domain-informed feature filtering

### Stage 3: Model Training
- Logistic Regression (interpretable baseline)
- Random Forest (nonlinear model)
- Cross-validation and class imbalance handling

### Stage 4: Evaluation
- Accuracy, ROC-AUC, PR-AUC
- Confusion matrix
- Feature importance analysis

### Stage 5: Risk Prediction Tool
- Interactive CLI for individual risk estimation
- Probability-based output (not binary diagnosis)

---

## Models Used

- Logistic Regression
- Random Forest Classifier

---

## Important Note

This project **does not provide medical diagnosis**.
Predictions are probabilistic risk estimates intended
for academic exploration only.

---

## Project structure 
```
Oral-Cancer-Risk-Prediction/
├── src/
│   ├── data_preprocessing.py        # Encoding, leakage removal, EDA
│   ├── model_training.py            # Logistic Regression & Random Forest
│   └── risk_prediction_cli.py       # Interactive risk prediction
│
├── data/
│   └── README.md
├── README.md
├── requirements.txt
└── .gitignore
```
<img width="823" height="567" alt="image" src="https://github.com/user-attachments/assets/82697b4a-0b4e-42dc-bcbe-7eee9f06122e" />

## Author
Gowtham Vuppaladhadiam
