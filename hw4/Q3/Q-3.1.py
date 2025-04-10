import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

'''CHANGE FILE PATH HERE'''
df = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw4/Q3/data/covid-data-us-national.txt')

df_train = df.loc[((df['epiweek'] >= 202010) & (df['epiweek'] <= 202035))]
data = df_train['death_jhu_incidence'].to_list()
# order = (2, 0, 2)
# first 2 = p -> go 2 observations back to forecast the value
# 0 = d -> number of times the raw observations are differenced
# second 2 -> size of the moving avg window

# make predictions
weeks_to_predict = 4
data_forecast = []
for i in range(weeks_to_predict):
    model = ARIMA(data, order=(2, 0, 2))
    model_fit = model.fit()
    
    forecast = model_fit.forecast(step=1)
    data.append(forecast[0])
    data_forecast.append(forecast[0])

# get ground truth values for epiweeks specified
df_ground_truth = df.loc[((df['epiweek'] >= 202036) & (df['epiweek'] <= 202039))]
data_ground_truth = df_ground_truth['death_jhu_incidence'].to_list()

# calculate the RMSE for forecasted and ground_truth values
RMSE = []
for i in range(weeks_to_predict):
    rmse = np.sqrt((data_forecast[i] - data_ground_truth[i]) ** 2)
    RMSE.append(rmse)

# calculate the avg of RMSE values
avg_RMSE = np.mean(RMSE)
print(RMSE)
print(avg_RMSE)