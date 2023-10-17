import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your dataset
data = pd.read_csv('C:/Users/javer/OneDrive/Desktop/OSEP Gachon/Machine Learning/Homework 1/dataset.csv')

# Convert the 'i_day' column to a datetime format
data['i_day'] = pd.to_datetime(data['i_day'])

# Filter the DataFrame to include only records from 2017
data_2017 = data[data['i_day'].dt.year == 2017]

# Select the features (X) and the target variable (y)
X = data_2017[['m_food_wst_cnt']]
y = data_2017['m_food_wst_amt']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Create a linear regression model and fit it to the training data
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions on the test data
predictions = reg.predict(X_test)

# Calculate the Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

new_waste_amount = 2500 #change this value on what you wanna predict
predicted_waste_count = (new_waste_amount - reg.intercept_) / reg.coef_

print(f"Predicted Waste Count for {new_waste_amount} Waste Amount: {predicted_waste_count[0]:.2f}")
print(f"Prediction: {predictions}")