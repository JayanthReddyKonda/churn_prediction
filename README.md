# Customer Churn Prediction with ANN

A deep learning project that predicts customer churn probability using an Artificial Neural Network (ANN) built with TensorFlow/Keras, served through an interactive Streamlit web application.

## 📋 Overview

Customer churn prediction is a critical business problem that helps companies identify customers who are likely to leave their service. This project trains an ANN model on customer data to predict churn probability and deploys it as a user-friendly web application.

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow / Keras
- **Data Processing**: Pandas, NumPy
- **Preprocessing**: Scikit-learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`)
- **Model Serialization**: Joblib, Keras (`.keras` format)
- **Web Framework**: Streamlit

## 📂 Project Structure

```
OWN_ML/DL/churn_prediction/
├── app.py                          # Streamlit web application
├── churn_model.keras               # Trained ANN model
├── column_transformer.joblib       # Saved preprocessing pipeline
├── churn_prediction.ipynb          # Notebook for data preprocessing & model training
├── prediction.ipynb                # Notebook for testing model predictions
├── logs/
│   └── fit/                        # TensorBoard logs (training history)
├── .gitignore
└── README.md
```

## 📊 Dataset

The project uses the **Churn_Modelling.csv** dataset (located at `OWN_ML/DataSets/`) which contains customer information for a bank:

| Feature | Description |
|---------|-------------|
| RowNumber | Row identifier |
| CustomerId | Unique customer ID |
| Surname | Customer surname |
| CreditScore | Customer credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Years as a customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products |
| HasCrCard | Has credit card (0/1) |
| IsActiveMember | Is active member (0/1) |
| EstimatedSalary | Estimated annual salary |
| **Exited** | **Target: Churned (1) or retained (0)** |

## 🔧 Data Preprocessing

The preprocessing pipeline (`ColumnTransformer`) performs the following steps:

1. **Drop Unnecessary Columns**: Removes `Surname`, `CustomerId`, and `RowNumber` (non-predictive identifiers)
2. **One-Hot Encoding**: Converts `Geography` into dummy variables
3. **Ordinal Encoding**: Encodes `Gender` as `Female=0, Male=1`
4. **Standard Scaling**: Normalizes numerical features (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`)

The `ColumnTransformer` is saved using Joblib for consistent preprocessing during inference.

## 🧠 Model Architecture

The ANN is built using Keras `Sequential` API:

```
Input Layer: (N, 11)  ──>  11 input features after preprocessing
Dense(64, relu)
Dense(32, relu)
Dense(1, sigmoid)    ──>  Output probability (0-1)
```

**Training Configuration**:
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: EarlyStopping (patience=10, restore best weights) + TensorBoard
- **Epochs**: Up to 100 (early stopping applied)
- **Validation Split**: 20%

## 🚀 Installation & Setup

### Prerequisites

Ensure you have Python 3.8+ installed along with the required packages:

```bash
pip install pandas numpy scikit-learn tensorflow streamlit joblib
```

### Required Files

Make sure the following files are present in the same directory as `app.py`:
- `churn_model.keras`
- `column_transformer.joblib`

### Run the Application

1. Navigate to the project directory:
   ```bash
   cd OWN_ML/DL/churn_prediction
   ```

2. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to interact with the web app at `http://localhost:8501`

## 💡 How It Works

### Web Application Flow

1. **User Input**: Enter customer details via the form (Credit Score, Geography, Gender, Age, Tenure, Balance, etc.)
2. **Preprocessing**: The input data is transformed using the saved `ColumnTransformer`
3. **Prediction**: The ANN model outputs a churn probability
4. **Result**: Displays whether the customer is likely to churn along with the probability percentage

### Example Usage

The `prediction.ipynb` notebook demonstrates how to load and use the trained model:

```python
import tensorflow as tf
import pandas as pd
import joblib

# Load artifacts
model = tf.keras.models.load_model("churn_model.keras")
preprocessor = joblib.load("column_transformer.joblib")

# Create input data
df_test = pd.DataFrame({
    'RowNumber': [0],
    'CustomerId': [15634602],
    'Surname': ['Hargrave'],
    'CreditScore': [619],
    'Geography': ['France'],
    'Gender': ['Female'],
    'Age': [42],
    'Tenure': [2],
    'Balance': [0.00],
    'NumOfProducts': [1],
    'HasCrCard': [1],
    'IsActiveMember': [1],
    'EstimatedSalary': [101348.88]
})

# Predict
X = preprocessor.transform(df_test)
probability = model.predict(X, verbose=0)
```

## 📈 Model Training

To retrain the model, open `churn_prediction.ipynb` and execute the cells in order. Key steps include:

1. Load and explore the dataset
2. Build and fit the `ColumnTransformer`
3. Split data into train/test sets
4. Transform data using the preprocessing pipeline
5. Define and compile the ANN architecture
6. Train with EarlyStopping and TensorBoard callbacks
7. Save the trained model as `churn_model.keras`

## 🖼️ TensorBoard Visualization

Training logs are automatically saved to `logs/fit/`. To visualize training history:

1. Open the notebook and run:
   ```python
   %load_ext tensorboard
   %tensorboard --logdir logs/fit/
   ```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2026 Jayanth Reddy Konda**
