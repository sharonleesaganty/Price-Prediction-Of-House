# Price-Prediction-Of-House

**Project Description**

A House Price Prediction model using machine learning (regression model) to predict house 
prices based on various features such as area, number of rooms, location, and age of the house.

**OUTPUTS:** 
• Input: A dataset containing features like area, number of rooms, location, and age of the house.

• Output: The predicted price of a house based on the input features.

**PSEUDOCODE:** 
# Step 1: Import necessary libraries 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 
 
# Step 2: Load dataset 
dataset = pd.read_csv('house_price_data.csv') 
 
# Step 3: Data Preprocessing 
# Drop any missing values or handle them appropriately 
dataset = dataset.dropna() 
 
# Step 4: Define the feature columns and the target variable 
X = dataset[['area', 'num_rooms', 'location', 'age']] 
y = dataset['price'] 
 
# Step 5: Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# Step 6: Initialize the regression model 
model = LinearRegression() 
 
 
 
17  
# Step 7: Train the model on the training data 
model.fit(X_train, y_train) 
# Step 8: Predict house prices using the test data 
y_pred = model.predict(X_test) 
 
# Step 9: Evaluate the model performance 
error = mean_absolute_error(y_test, y_pred) 
print(f'Mean Absolute Error: {error}') 
# Step 10: Predict the price for a new house 
new_house = np.array([[1200, 3, 'Downtown', 5]])  # Example input 
predicted_price = model.predict(new_house) 
print(f'Predicted House Price: {predicted_price[0]}')


** Programming Environment: **  
   - Jupyter Notebook: Used for writing and testing Python code in an interactive environment, 
enabling easy visualization and debugging.   
** Programming Language:  ** 
   - Python: The primary language used for data preprocessing, analysis, and implementing machine  learning models which include libraries as well.


**Dataset**

Used a dataset from Kaggle.
