# Airline Price Prediction
In this project, I perform exploratory data analysis (EDA) to uncover insights on factors influencing coach ticket prices for airline travel. I then use machine learning to predict these prices based on inflight services, travel miles, delays, and other variables. The project demonstrates the entire process, from data exploration to deploying a predictive model via a Streamlit application.

# Project Overview
With an ever-growing demand for air travel, understanding the factors influencing ticket prices is crucial for both passengers and airline companies. This project analyzes various features impacting coach ticket prices and provides a predictive model to estimate these prices based on specific parameters.

# Objectives
1. Analyze Factors Influencing Ticket Prices
2. Examine the Effect of Inflight Services on Pricing
3. Identify Patterns in Ticket Prices Based on Travel Schedules
4. Build a Predictive Model for Price Estimation
5. Provide Data-Driven Insights for Dynamic Pricing Strategies
6. Evaluate the Impact of Delays on Ticket Costs

# Data
The dataset includes variables like:

Travel distance (miles)<br>
Number of passengers (passengers) <br>
Delays (delay)<br>
Inflight services (inflight_meal, inflight_entertainment, inflight_wifi)<br>
Flight specifics (day_of_week, weekend, redeye)<br>

You can access the dataset from the data/ folder or provide a link if it's public.<br>

# Exploratory Data Analysis
Conducted an t test and ANOVA analysis to understand the impact of inflight services on coach and first-class prices.
Visualizations were created to examine relationships between ticket prices and factors such as travel distance and inflight amenities.

# Machine Learning Model
The project utilizes a RandomForestRegressor model:
- Data preprocessing was handled through pipelines for numerical and categorical features.<br>
- A GridSearchCV was employed to optimize hyperparameters.<br>
- Model evaluation metrics included Mean Squared Error (MSE) and R² Score.<br>

# Results
The best model achieved:

Mean Squared Error (MSE): 1038 <br>
R² Score: 0.775 <br>

These results demonstrate the model’s accuracy in predicting coach ticket prices based on the input features.

# Deployment
The model is deployed as a Streamlit web app, enabling users to input parameters and receive price predictions in real time. The app file (app.py) can be found in the app/ directory.
