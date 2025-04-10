import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
data = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw4/Q3/data/covid-data-us-national.txt')
mortality_data = data['death_jhu_incidence']  # Replace with the exact column name for mortality data

# Select training and test data
train_data = mortality_data[9:35]  # Adjust index to match epiweek 202010 to 202035
test_data = mortality_data[35:39]   # Adjust index to match epiweek 202036 to 202039

predictions = []
training_series = train_data.copy()

for i in range(len(test_data)):
    # Fit ARIMA model on current training data
    model = ARIMA(training_series, order=(2, 0, 2))
    model_fit = model.fit()
    
    # Forecast next week
    forecast = model_fit.forecast(steps=1)
    predictions.append(forecast.iloc[0])

    # Update training data with forecast
    forecast_series = pd.Series(forecast, index=[training_series.index[-1] + 1])
    training_series = pd.concat([training_series, forecast_series])

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data, predictions))
print(predictions)
print(f'Average RMSE for epiweeks 202036-202039: {rmse}')
