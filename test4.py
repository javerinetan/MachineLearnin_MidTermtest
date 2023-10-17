import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load your dataset
data = pd.read_csv('C:/Users/javer/OneDrive/Desktop/OSEP Gachon/Machine Learning/Homework 1/dataset.csv')

# Convert the 'i_day' column to a datetime format
data['i_day'] = pd.to_datetime(data['i_day'])

# Filter the DataFrame to include only records from 2017
data_2017 = data[data['i_day'].dt.year == 2017]

# Select the features (X) and the target variable (y)
X = data_2017[['m_food_wst_cnt']]
y = data_2017['m_food_wst_amt']

# Generate some random data for testing
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Create an instance of LinearRegression
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict using the trained model
X_new = np.array([[2.0], [3.0]])
y_pred = model.predict(X_new)

# Create a DataFrame to display the results
results = pd.DataFrame({'Food waste count (X_new)': X_new[:, 0], 'Predicted Food waste count': y_pred[:, 0]})
print("Predictions for Food waste count:")
print(results)
