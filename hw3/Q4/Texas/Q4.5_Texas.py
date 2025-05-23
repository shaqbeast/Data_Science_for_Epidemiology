import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q4/data/ILINet_states.csv')
df = df.loc[(df['REGION'] == 'Texas') & (((df['YEAR'] == 2018) & (df['WEEK'] >= 40)) | ((df['YEAR'] == 2019) & (df['WEEK'] <= 20)))]

df_2018 = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q4/data/2018_symptoms_dataset.csv')
df_2018 = df_2018.loc[(df_2018['sub_region_1'] == 'Texas')]
df_2018 = df_2018.tail(13)

df_2019 = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q4/data/2019_weekly_symptoms_dataset.csv')
df_2019 = df_2019.loc[(df_2019['sub_region_1'] == 'Texas')]
df_2019 = df_2019.head(20)

df_2018_2019 = pd.concat([df_2018, df_2019], axis=0)

ILI = []
for index, row in df.iterrows():
    unweighted_ILI = float(row['%UNWEIGHTED ILI'])
    ILI.append(unweighted_ILI)

s_values = []
for index, row in df_2018_2019.iterrows():
    value = row['symptom:Low-grade fever']
    s_values.append(value)

df['EPIWEEK'] = 0
epiweek = 1
for index, row in df.iterrows():
    df.loc[index, 'EPIWEEK'] = epiweek
    epiweek += 1

pearson_values = []
for t_prime in range(0, 32):
    ILI_min = (t_prime)
    s_max = 33 - t_prime
    
    ILI_time_series = ILI[ILI_min:33]
    s_time_series = s_values[0:s_max]
    
    pcc_lg_fever, p_value = stats.pearsonr(ILI_time_series, s_time_series)
    pearson_values.append(pcc_lg_fever)

max = np.argmax(pearson_values)
print("Lead Time: " + str(max))