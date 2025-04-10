import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import numpy as np

def model_ARIMA(data, weeks_to_predict):
    weeks_to_predict = 4
    data_forecast_ARIMA = []
    for i in range(weeks_to_predict):
        model = ARIMA(data, order=(2, 0, 2))
        model_fit = model.fit()
    
        forecast = model_fit.forecast(step=1)
        data.append(forecast[0])
        data_forecast_ARIMA.append(forecast[0])
        
    return data_forecast_ARIMA

def model_OLS(data, weeks_to_predict):
    data_forecast_OLS = []
    predictor_columns = ['covid_cases', 'mobility', 'totalTests']
    X = df_train[predictor_columns]
    X = sm.add_constant(X)
    y = df_train['death_jhu_incidence']
    start_epiweek = 202036
    for i in range(weeks_to_predict):
        df_forecast = df.loc[((df['epiweek'] == start_epiweek))]
        model = sm.OLS(y, X).fit()
    
        X_next = df_forecast[predictor_columns]
        X_next = sm.add_constant(X_next, has_constant='add')
        prediction = model.predict(X_next)
        data_forecast_OLS.append(prediction)
        
        X = pd.concat([X, X_next], ignore_index=True) # adds the next predictor values to the last row of X data
        y = pd.concat([y, prediction], ignore_index=True)
        start_epiweek += 1
    
    return data_forecast_OLS
    

'''CHANGE FILE PATH HERE'''
df = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw4/Q3/data/covid-data-us-national.txt')

df_train = df.loc[((df['epiweek'] >= 202010) & (df['epiweek'] <= 202035))]
data = df_train['death_jhu_incidence'].to_list()

df_ground_truth = df.loc[((df['epiweek'] >= 202036) & (df['epiweek'] <= 202039))]
data_ground_truth = df_ground_truth['death_jhu_incidence'].to_list()
weeks_to_forecast = 4

# data = training data
# ARIMA
data_forecast_ARIMA = model_ARIMA(data, weeks_to_forecast)

# OLS
data_forecast_OLS = model_OLS(data, weeks_to_forecast)

''' EM1 '''
em1 = []
for week in range(weeks_to_forecast):
    arima = data_forecast_ARIMA[week]
    ols = data_forecast_OLS[week]
    
    solution = (arima + ols) / 2
    em1.append(solution)

rmse_em1 = np.sqrt(mean_squared_error(em1, data_ground_truth))
print(rmse_em1)

''' EM2 '''
rmse_ARIMA = np.sqrt(mean_squared_error(data_forecast_ARIMA, data_ground_truth))
rmse_OLS = np.sqrt(mean_squared_error(data_forecast_OLS, data_ground_truth))

w1 = 1 / rmse_ARIMA
w2 = 1 / rmse_OLS

em2 = []
for week in range(weeks_to_forecast):
    arima = data_forecast_ARIMA[week]
    ols = data_forecast_OLS[week]
    
    solution = ((w1 * arima) + (w2 * ols)) / (w1 + w2)
    em2.append(solution)

rmse_em2 = np.sqrt(mean_squared_error(em2, data_ground_truth))
print(rmse_em2)
