# 🏦 Credit Card Fraud Detection Using Machine Learning

## 📌 Project Overview

Credit card fraud detection is an important application of machine learning in the banking and financial sector.

This project develops a machine learning system that predicts whether a credit card transaction is **legitimate or fraudulent** based on transaction-related information.

The project covers data analysis, preprocessing, machine learning model training, model comparison, hyperparameter tuning, cross-validation, threshold tuning, and deployment using Streamlit.

---

## 🎯 Objectives

* Detect fraudulent credit card transactions.
* Analyze transaction data and identify useful patterns.
* Preprocess numerical and categorical features.
* Handle the imbalanced fraud dataset.
* Compare multiple machine learning models.
* Tune the Random Forest model.
* Evaluate the model using multiple performance metrics.
* Build a web application for fraud prediction.
* Deploy the application using Streamlit.
* Maintain the project using Git and GitHub.

---

## 📊 Dataset

The project uses the `credit_card_fraud_10k.csv` dataset.

### Dataset Information

* **Total transactions:** 10,000
* **Legitimate transactions:** 9,849
* **Fraudulent transactions:** 151
* **Target variable:** `is_fraud`

### Target Values

| Value | Meaning                |
| ----- | ---------------------- |
| `0`   | Legitimate Transaction |
| `1`   | Fraudulent Transaction |

---

## 🧾 Features Used

The following features were used for prediction:

| Feature               | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `amount`              | Transaction amount                                             |
| `transaction_hour`    | Hour at which the transaction occurred                         |
| `merchant_category`   | Category of the merchant                                       |
| `foreign_transaction` | Indicates whether the transaction is foreign                   |
| `location_mismatch`   | Indicates a mismatch between transaction and expected location |
| `device_trust_score`  | Trust score of the device                                      |
| `velocity_last_24h`   | Number of transactions in the previous 24 hours                |
| `cardholder_age`      | Age of the cardholder                                          |

### Removed Feature

`transaction_id` was removed before final model training because it is an identifier and does not provide meaningful predictive information.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
One-Hot Encoding
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Comparison
   ↓
Class Imbalance Handling
   ↓
Hyperparameter Tuning
   ↓
5-Fold Cross-Validation
   ↓
Threshold Tuning
   ↓
Final Model
   ↓
Model Saving
   ↓
Streamlit Deployment
```

---

## 🧹 Data Preprocessing

### 1. Train-Test Split

The dataset was divided into:

* **80% training data**
* **20% testing data**

Stratified splitting was used to maintain the fraud/non-fraud distribution.

### 2. Categorical Encoding

`OneHotEncoder` was used for the `merchant_category` feature.

```python
OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)
```

### 3. Feature Scaling

`StandardScaler` was used to scale the processed features.

### 4. Feature Selection

The `transaction_id` column was removed before final model training.

---

## ⚖️ Handling Class Imbalance

The dataset contains significantly more legitimate transactions than fraudulent transactions.

```text
Legitimate: 9849
Fraud:       151
```

To address the class imbalance, the project experimented with:

* `class_weight='balanced'`
* SMOTE
* Precision, Recall and F1-score evaluation
* Probability threshold tuning

---

## 🤖 Machine Learning Models

Three main machine learning approaches were evaluated:

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Random Forest

Used as the main tree-based classification model.

### 3. XGBoost

Used as an additional gradient-boosting model for comparison.

---

## 📈 Model Comparison

The consistent evaluation results were:

| Model               | Accuracy | Precision |  Recall | F1 Score |
| ------------------- | -------: | --------: | ------: | -------: |
| Logistic Regression |   95.85% |    26.55% |    100% |   41.96% |
| Random Forest       |  100.00% |   100.00% | 100.00% |  100.00% |
| XGBoost             |   99.90% |   100.00% |  93.33% |   96.55% |

The models were compared using accuracy, precision, recall, F1-score and ROC-AUC where available.

---

## 🔧 Hyperparameter Tuning

`RandomizedSearchCV` was used to tune the Random Forest model.

### Configuration

* **20 parameter combinations**
* **5-fold cross-validation**
* **F1-score** used as the optimization metric

### Best Parameters

```text
n_estimators = 200
max_depth = 15
min_samples_split = 10
min_samples_leaf = 4
max_features = log2
```

---

## 🔁 5-Fold Cross-Validation

The tuned Random Forest was evaluated using 5-fold cross-validation.

### F1 Scores

```text
[0.8571, 0.7368, 0.8000, 0.8293, 0.8095]
```

### Mean F1 Score

```text
0.8066
```

### Standard Deviation

```text
0.0400
```

Cross-validation was used to check the consistency of model performance across different data splits.

---

## 🎚️ Threshold Tuning

The model's fraud probability was evaluated using different classification thresholds.

Tested thresholds included:

```text
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.45
0.50
```

The final threshold used in the Streamlit application is:

```text
0.40
```

At this threshold:

* **Precision:** 100%
* **Recall:** 93.33%
* **F1 Score:** 96.55%

The threshold was selected to provide a strong balance between identifying fraudulent transactions and avoiding false fraud alerts.

---

## 🏆 Final Model

The final selected model is a:

### **Tuned Random Forest Classifier**

The trained model is saved as:

```text
fraud_model.pkl
```

The preprocessing objects are saved as:

```text
encoder.pkl
scaler.pkl
```

---

## 📊 Final Model Performance

The tuned Random Forest achieved:

| Metric    |   Score |
| --------- | ------: |
| Accuracy  |  99.90% |
| Precision | 100.00% |
| Recall    |  93.33% |
| F1 Score  |  96.55% |
| ROC-AUC   |  99.98% |

### Confusion Matrix

```text
[[1970    0]
 [   2   28]]
```

---

## 🌐 Streamlit Web Application

A Streamlit web application was developed for interactive fraud prediction.

The user can enter:

* Transaction Amount
* Transaction Hour
* Merchant Category
* Foreign Transaction
* Location Mismatch
* Device Trust Score
* Velocity Last 24 Hours
* Cardholder Age

The application calculates the fraud probability and displays the prediction.

### Possible Outputs

```text
Fraud Probability: 0.00%

✅ Legitimate Transaction
```

or

```text
🚨 Fraudulent Transaction Detected
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Preprocessing

* OneHotEncoder
* StandardScaler

### Machine Learning Models

* Logistic Regression
* Random Forest
* XGBoost

### Model Optimization

* RandomizedSearchCV
* 5-Fold Cross-Validation
* Threshold Tuning

### Class Imbalance

* Class Weight
* SMOTE

### Deployment

* Streamlit

### Development Tools

* Google Colab
* Visual Studio Code

### Version Control

* Git
* GitHub

### Model Persistence

* Pickle
* Joblib

---

## 📁 Project Structure

```text
credit_card_fraud_app/
│
├── app.py
├── fraud_model.pkl
├── encoder.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

### File Description

| File               | Purpose                                  |
| ------------------ | ---------------------------------------- |
| `app.py`           | Streamlit web application                |
| `fraud_model.pkl`  | Trained final Random Forest model        |
| `encoder.pkl`      | OneHotEncoder for merchant category      |
| `scaler.pkl`       | StandardScaler used during preprocessing |
| `requirements.txt` | Required Python packages                 |
| `README.md`        | Project documentation                    |

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Snehayadawad/credit-card-fraud-detection.git
```

### Step 2: Open the Project

```bash
cd credit-card-fraud-detection
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

### Step 5: Use the Application

Enter the transaction details and click:

**🔍 Predict Fraud**

The application will display the fraud probability and classification result.

---

## 🚀 Future Improvements

* Use a larger real-world credit card fraud dataset.
* Improve handling of highly imbalanced data.
* Perform threshold selection using a dedicated validation set.
* Add explainable AI techniques such as SHAP.
* Add real-time transaction monitoring.
* Improve the Streamlit user interface.
* Deploy the application to a cloud platform.
* Continuously retrain the model using new transaction data.

---

## 👩‍💻 Author

**Sneha Yadawad**

GitHub:
https://github.com/Snehayadawad/credit-card-fraud-detection
