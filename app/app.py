#Importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels
import matplotlib.pyplot as plt
import math
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

## Reading the Data
airline_df = pd.read_csv(r"C:\Users\magak\Desktop\Projects\Airline Analysis\Data\Airline Dataset.csv")
airline_df.head()

#Features and target variable

X = airline_df.drop(columns=['coach_price', 'firstclass_price'])
y = airline_df['coach_price']

# Define the preprocessing steps for numerical and categorical features
numerical_features = ['miles', 'passengers', 'delay', 'hours']
categorical_features = ['inflight_meal', 'inflight_entertainment', 'inflight_wifi', 'day_of_week', 'redeye', 'weekend']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'model__n_estimators': [100],
    'model__max_depth': [None],
    'model__min_samples_split': [2]
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", -grid_search.best_score_)

# Evaluate the model on the test set
y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error on test set: ", mse)
print("R^2 Score on test set: ", r2)



st.title('Airline Price Prediction')
st.write('This app predicts the price of a coach ticket for an airline based on the features provided.')

# Input form for the user
miles = st.number_input('Miles', min_value=0)
passengers = st.number_input('Passengers', min_value=0)
delay = st.number_input('Delay', min_value=0)
hours = st.number_input('Hours', min_value=0)
inflight_meal = st.selectbox('Inflight Meal', ['Yes', 'No'])
inflight_entertainment = st.selectbox('Inflight Entertainment', ['Yes', 'No'])
inflight_wifi = st.selectbox('Inflight WiFi', ['Yes', 'No'])
day_of_week = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
redeye = st.selectbox('Redeye', ['Yes', 'No'])
weekend = st.selectbox('Weekend', ['Yes', 'No'])

# Create a DataFrame with the user input
user_input = pd.DataFrame({
    'miles': [miles],
    'passengers': [passengers],
    'delay': [delay],
    'hours': [hours],
    'inflight_meal': [inflight_meal],
    'inflight_entertainment': [inflight_entertainment],
    'inflight_wifi': [inflight_wifi],
    'day_of_week': [day_of_week],
    'redeye': [redeye],
    'weekend': [weekend]
})

if st.button('Make Prediction'):
    # Process the user input with the preprocessor first
    processed_input = preprocessor.transform(user_input)
    
    # Make a prediction using the trained model
    prediction = grid_search.predict(processed_input)
    
    # Display the prediction
    st.subheader('Predicted Coach Price')
    st.write('The predicted price of a coach ticket is $', round(prediction[0], 2))

