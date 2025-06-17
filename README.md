# 🏠 Milan House Price Prediction – XGBoost Regression

This project predicts house prices in Milan using structured real estate listing data. By combining careful feature engineering with a tuned XGBoost model, it aims to forecast log-transformed housing prices with strong accuracy and interpretability.

## 📁 Files

- `Cleaning.ipynb` – Cleans and preprocesses the housing dataset with categorical grouping, encoding, and missing value handling  
- `Model.ipynb` – Trains and tunes the XGBoost regression model using cross-validation  
- `ML_report.pdf` – Summarizes the project’s methodology, data decisions, and model performance  

## 📊 Summary

### 1. Data Preparation  
- Combined and cleaned listing data (`train.csv`, `test.csv`)  
- Applied groupings to low-frequency categories (e.g., zones, construction year)  
- Created dummies for missing values where informative  
- Encoded ordinal variables like energy class and number of rooms  
- Log-transformed price target to reduce skew and stabilize variance  

### 2. Feature Engineering & Modeling  
- Created over 200 features including categorical encodings and ordinal bins  
- Used **XGBoost** due to its robustness with tabular data  
- Tuned model with a mix of **RandomizedSearchCV** and **GridSearchCV**  
- Applied 10-fold cross-validation to ensure generalization

### 3. Results  
- Achieved **MSE of 0.181** on log-price during cross-validation  
- Mean Absolute Error on real prices was ~**€67,500**  
- Model favored a large number of shallow trees (depth 3) with high regularization

## 🔧 Tools & Libraries

- **Python** (pandas, numpy, matplotlib, scikit-learn, xgboost)
- Grid/Randomized search for hyperparameter tuning

## 📌 Notes

- Zone grouping was done manually based on Milan metro proximity  
- The modeling target was log(price), predictions can be exponentiated if needed  
- Continuous square meter variable showed strong positive correlation with price

---

📉 Completed as part of the course Machine Learning I.
