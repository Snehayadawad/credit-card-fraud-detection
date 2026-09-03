# credit-card-fraud-detection
Machine learning project to detect fraudulent credit card transactions using Python and Streamlit.
# 💳 Credit Card Fraud Detection

## 📌 Project Overview

Credit card fraud is an important problem in the banking and financial sector. This project uses **Machine Learning** to predict whether a credit card transaction is **fraudulent or legitimate**.

I developed the Machine Learning part using **Google Colab** and then created a **Streamlit web application** using **VS Code**. The trained model and preprocessing files are saved and loaded by the Streamlit application to make predictions.

The application allows the user to enter transaction details and predicts whether the transaction is fraudulent or legitimate.

---

## 🎯 Project Objectives

* Analyze credit card transaction data.
* Understand fraudulent and legitimate transactions.
* Preprocess numerical and categorical data.
* Train a Machine Learning classification model.
* Evaluate the trained model.
* Save the trained model and preprocessing objects.
* Build a Streamlit web application.
* Make fraud predictions from user-entered transaction details.

---

## 📊 Dataset

The project uses a credit card fraud dataset containing:

* **10,000 transactions**
* **10 columns**

### Dataset Columns

| Column                | Description                                                                   |
| --------------------- | ----------------------------------------------------------------------------- |
| `transaction_id`      | Unique transaction identifier                                                 |
| `amount`              | Transaction amount                                                            |
| `transaction_hour`    | Hour at which the transaction occurred                                        |
| `merchant_category`   | Category of the merchant                                                      |
| `foreign_transaction` | Indicates whether the transaction is foreign                                  |
| `location_mismatch`   | Indicates whether the transaction location differs from the expected location |
| `device_trust_score`  | Trust score of the device                                                     |
| `velocity_last_24h`   | Number of transactions in the last 24 hours                                   |
| `cardholder_age`      | Age of the cardholder                                                         |
| `is_fraud`            | Target variable                                                               |

### Target Variable

The target variable is `is_fraud`.

```text
0 → Legitimate Transaction
1 → Fraudulent Transaction
```

The dataset contains:

```text
Legitimate transactions: 9849
Fraudulent transactions: 151
```

This shows that fraudulent transactions are much fewer than legitimate transactions.

---

# 🧪 Work Done in Google Colab

The Machine Learning part of the project was developed in **Google Colab**.

## 1. Importing Libraries

The following Python libraries were used:

* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib

Pandas was used for data handling and analysis, NumPy for numerical operations, Matplotlib for visualization, Scikit-learn for preprocessing, splitting and Machine Learning, and Joblib for saving the trained model and preprocessing objects.

---

## 2. Loading the Dataset

The dataset was loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv("/content/credit_card_fraud_10k.csv")
```

---

## 3. Exploring the Dataset

The dataset was explored using commands such as:

```python
df.shape
```

Output:

```text
(10000, 10)
```

The structure and data types were checked using:

```python
df.info()
```

The target variable was also analyzed using:

```python
df["is_fraud"].value_counts()
```

Output:

```text
0    9849
1     151
```

---

## 4. Preparing Features and Target

The target variable `is_fraud` was separated from the input features.

```python
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]
```

---

## 5. Removing Transaction ID

`transaction_id` was not required for prediction, so it was removed from the features.

```python
X = X.drop("transaction_id", axis=1)
```

---

## 6. Splitting the Dataset

The dataset was divided into training and testing data using `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The data was divided into:

* 80% training data
* 20% testing data

`stratify=y` was used to maintain the distribution of fraudulent and legitimate transactions.

---

## 7. Encoding Categorical Data

The `merchant_category` column contains categorical values, so **OneHotEncoder** was used.

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

This converts categorical values into numerical form so that the Machine Learning model can use them.

---

## 8. Scaling Numerical Features

Numerical features were scaled using **StandardScaler**.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Scaling helps put numerical features on a comparable scale.

---

## 9. Combining Features

The encoded categorical features and scaled numerical features were combined using NumPy.

```python
import numpy as np

X_train_final = np.hstack([
    X_train_scaled,
    X_train_encoded
])

X_test_final = np.hstack([
    X_test_scaled,
    X_test_encoded
])
```

---

# 🤖 Machine Learning Model

A classification Machine Learning model was trained using the processed transaction data.

The model learns patterns from the training data and predicts whether a new transaction is fraudulent or legitimate.

The trained model was then tested using the test dataset.

---

# 📈 Model Evaluation

The trained model was evaluated using the test data.

The prediction results were compared with the actual `is_fraud` values to understand how well the model detects fraudulent transactions.

The project focuses particularly on fraud detection because identifying fraudulent transactions correctly is important for this application.

---

# 💾 Saving the Model

After training, the Machine Learning model and preprocessing objects were saved using Joblib.

The following files were created:

```text
fraud_model.pkl
scaler.pkl
encoder.pkl
```

### File Purpose

| File              | Purpose                              |
| ----------------- | ------------------------------------ |
| `fraud_model.pkl` | Saved trained Machine Learning model |
| `scaler.pkl`      | Saved StandardScaler                 |
| `encoder.pkl`     | Saved OneHotEncoder                  |

These files allow the Streamlit application to use the already-trained model without training it again every time.

---

# 🖥️ Streamlit Web Application

After completing the Machine Learning work in Google Colab, a web application was created using **Streamlit**.

The application was developed using **Visual Studio Code**.

The application takes the following inputs:

* Transaction Amount
* Transaction Hour
* Merchant Category
* Foreign Transaction
* Location Mismatch
* Device Trust Score
* Velocity Last 24 Hours
* Cardholder Age

The user enters the transaction details and clicks:

**Predict Fraud**

The application then displays the prediction.

---

# 🔄 Project Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Feature & Target Separation
   ↓
Remove Transaction ID
   ↓
Train-Test Split
   ↓
Categorical Encoding
   ↓
Numerical Feature Scaling
   ↓
Feature Combination
   ↓
Machine Learning Model
   ↓
Model Evaluation
   ↓
Save Model & Preprocessors
   ↓
Streamlit Application
   ↓
User Enters Transaction Details
   ↓
Fraud Prediction
```

---

# 📁 Project Structure

```text
credit-card-fraud-detection/
│
├── app.py
├── fraud_model.pkl
├── scaler.pkl
├── encoder.pkl
├── requirements.txt
└── README.md
```

### Files

**`app.py`**
Contains the Streamlit application code.

**`fraud_model.pkl`**
Contains the trained Machine Learning model.

**`scaler.pkl`**
Contains the fitted StandardScaler.

**`encoder.pkl`**
Contains the fitted OneHotEncoder.

**`requirements.txt`**
Contains the Python libraries required to run the project.

**`README.md`**
Contains the project documentation.

---

# 🛠️ Technologies and Tools Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Streamlit

### Development Tools

* Google Colab
* Visual Studio Code
* GitHub

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/Snehayadawad/credit-card-fraud-detection.git
```

## 2. Open the Project Folder

```bash
cd credit-card-fraud-detection
```

## 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The Streamlit application will open in the browser.

---

# 🔍 Example Prediction

The application can classify a transaction as:

### 🚨 Fraudulent Transaction

or

### ✅ Legitimate Transaction

The prediction is generated by the trained Machine Learning model using the transaction details entered by the user.

---

# 🔮 Future Improvements

The project can be improved in the future by:

* Using a larger real-world dataset.
* Testing additional Machine Learning algorithms.
* Improving fraud detection performance.
* Handling class imbalance with appropriate techniques.
* Adding more visualizations to the Streamlit application.
* Deploying the Streamlit application online.
* Adding transaction history and monitoring features.

---

# 👩‍💻 Author

**Sneha Yadawad**

Computer Science & Engineering Student

---

## 📜 Disclaimer

This project is developed for **educational and internship purposes**. It is a Machine Learning demonstration and should not be used as a real banking fraud detection system without additional testing, security measures, and validation.

