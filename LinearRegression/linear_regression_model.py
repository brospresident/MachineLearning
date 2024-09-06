import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# dataset url: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

train = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv')
stores = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/stores.csv')
oil = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/oil.csv')
holidays = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv')

 # Merge datasets
df = train.merge(stores, on='store_nbr', how='left')
df = df.merge(oil, on='date', how='left')

print(df.columns)

# Ensure all features are present and drop any rows with missing values
# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Create time-based features
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Select the features from dataset
features = ['store_nbr', 'onpromotion', 'day_of_week', 'month', 'year', 'dcoilwtico', 'cluster']

X = df[features]
y = df['sales']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Add polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(X.columns)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

print("Shape of X after cleaning:", X_scaled.shape)
print("Shape of y after cleaning:", y.shape)

# Now try splitting and fitting the model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2 * 100}")
