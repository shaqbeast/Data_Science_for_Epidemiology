import pandas as pd
from statsmodels.regression.linear_model import OLS
import numpy as np
import statsmodels.api as sm

'''CHANGE FILE PATH HERE'''
df = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw4/Q3/data/covid-data-us-national.txt')

df_train = df.loc[((df['epiweek'] >= 202010) & (df['epiweek'] <= 202035))]

# define variables
weeks_to_predict = 4
data_forecast = []
predictor_columns = ['covid_cases', 'mobility', 'totalTests']
X = df_train[predictor_columns]
X = sm.add_constant(X)
y = df_train['death_jhu_incidence']
start_epiweek = 202036
for i in range(weeks_to_predict):
    # deaths = m1 * covid_cases + m2 * mobility + m3 * totalTests + C
    df_forecast = df.loc[((df['epiweek'] == start_epiweek))]
    model = sm.OLS(y, X).fit()
    
    X_next = df_forecast[predictor_columns]
    X_next = sm.add_constant(X_next, has_constant='add')
    prediction = model.predict(X_next)
    data_forecast.append(prediction)
    X = pd.concat([X, X_next], ignore_index=True)# adds the next predictor values to the last row of X data
    y = pd.concat([y, prediction], ignore_index=True)
    start_epiweek += 1

# get ground truth values for epiweeks specified
df_ground_truth = df.loc[((df['epiweek'] >= 202036) & (df['epiweek'] <= 202039))]

data_ground_truth = df_ground_truth['death_jhu_incidence'].to_list()

# calculate the RMSE for forecasted and ground_truth values
RMSE = []
for i in range(weeks_to_predict):
    rmse = np.sqrt((data_forecast[i] - data_ground_truth[i]) ** 2)
    RMSE.append(rmse)

# calculate the avg of RMSE values
print(data_forecast)
print(RMSE)
avg_RMSE = np.mean(RMSE)
print(avg_RMSE)
