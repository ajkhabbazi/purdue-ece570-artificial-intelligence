# NASA NEOs Classification

This project implements and compares multiple machine learning classifiers on NASA’s Near-Earth Objects (NEOs) dataset. It was developed as part of an assignment focused on model training, feature pruning, and performance evaluation.

## Overview
The notebook covers:
1. **Data preprocessing** with scaling and feature cleaning  
2. **Model training** using K-Nearest Neighbors (KNN) and Logistic Regression  
3. **Feature pruning** using mutual information to reduce dimensionality  
4. **Advanced modeling** with XGBoost to improve balanced accuracy and F1-score on imbalanced data  

## Models and Evaluation
- All models were tuned with `GridSearchCV` using cross-validation  
- Metrics include accuracy, balanced accuracy, precision, recall, and F1-score  
- The XGBoost model achieved the best performance on minority-class metrics  

## Files
- `a4_nasa_neos_classification_ajk.ipynb` — Main notebook  
- `classifiers.pdf` — Assignment and grading rubric  

## Requirements
- Python 3.9+  
- scikit-learn  
- xgboost  
- pandas  
- numpy  
- matplotlib
