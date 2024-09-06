import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('/kaggle/input/ibm-churn-rate/dataset.csv')

def preprocess_data(df):
    churn_column = [col for col in df.columns if 'churn' in col.lower()][0]
    # Remove unnecessary columns
    df = df.drop(['customerID'], axis=1)
    
    # Convert 'TotalCharges' to numeric, replacing empty strings with NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    
    # Impute numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    
    # Impute categorical columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    
    # Convert target variable to numeric
    df[churn_column] = df[churn_column].map({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
    df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 1, 'No': 0})
    df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 1, 'No': 0})
    df['TechSupport'] = df['TechSupport'].map({'Yes': 1, 'No': 0})
    df['StreamingTV'] = df['StreamingTV'].map({'Yes': 1, 'No': 0})
    df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 1, 'No': 0})
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    df['OnlineBackup'] = df['OnlineBackup'].map({'Yes': 1, 'No': 0})
    
    features_to_drop = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod', 'MonthlyCharges']
    df = df.drop(features_to_drop, axis=1)
    df = df.fillna(0)
    
    # Separate features and target
    X = df.drop(churn_column, axis=1)
    y = df[churn_column]
        
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)

print("\nShape of training data:", X_train.shape)
print("Shape of testing data:", X_test.shape)
print("\nFeatures:", features)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Print feature importances
for feature, importance in zip(features, model.coef_[0]):
    print(f"{feature}: {importance}")
    
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.4f}")
