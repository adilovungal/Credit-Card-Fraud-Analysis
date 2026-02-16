# Credit Card Fraud Detection (Imbalanced Classification)

## Problem Statement
This project builds machine learning models to detect fraudulent credit card transactions in a highly imbalanced dataset. Because fraud cases are rare, the focus is on recall, precision, and ROC-AUC rather than accuracy.

## Dataset
- Source: Credit Card Fraud Detection dataset
- Total transactions: 284,807
- Fraud cases: 492 (~0.17%)
- Features: PCA-transformed numerical variables + transaction amount

## Approach

### Data Preparation
- Removed missing values
- Performed stratified train-test split
- Applied StandardScaler for Logistic Regression
- Addressed class imbalance using class-weighted models

### Models Implemented
- Logistic Regression (class_weight='balanced')
- Random Forest (class_weight='balanced')

### Evaluation Metrics
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision–Recall analysis
- Stratified cross-validation

## Results

| Model | CV ROC-AUC (mean) | CV Std |
|------|-------------------|--------|
| Logistic Regression | ~0.979 | ~0.008 |
| Random Forest | ~0.951 | ~0.008 |

## Key Insights
- Logistic Regression achieved the highest ROC-AUC, indicating strong linear separability in the PCA-transformed feature space.
- Random Forest provided competitive performance with more conservative fraud flagging.
- Model selection should depend on business tolerance for false positives vs missed fraud.

## Business Interpretation
In fraud detection systems, recall is often prioritized because missed fraudulent transactions directly translate to financial loss. However, excessive false positives increase operational costs. The final model choice should align with risk tolerance.

## How to Run

```bash
pip install -r requirements.txt
python fraud_detection.py
