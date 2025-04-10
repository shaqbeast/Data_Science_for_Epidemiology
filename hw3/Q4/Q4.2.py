import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# Read 2018 data (only GA and the last 13 weeks: 40 - 52)
df_2018 = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q4/data/2018_symptoms_dataset.csv')
df_2018 = df_2018.loc[(df_2018['sub_region_1'] == 'Georgia')]
df_2018 = df_2018.tail(13)

# Read 2019 data (only GA and the first 20 weeks: 1 - 20)
df_2019 = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q4/data/2019_weekly_symptoms_dataset.csv')
df_2019 = df_2019.loc[(df_2019['sub_region_1'] == 'Georgia')]
df_2019 = df_2019.head(20)

# concat the two 
df = pd.concat([df_2018, df_2019], axis=0)
epiweeks = []
for i in range(1, 34):
    epiweeks.append(i)

plt.plot(epiweeks, df['symptom:Fever'], label="Fever")
plt.plot(epiweeks, df['symptom:Low-grade fever'], label="Low-grade fever")
plt.plot(epiweeks, df['symptom:Cough'], label="Cough")
plt.plot(epiweeks, df['symptom:Sore throat'], label="Sore throat")
plt.plot(epiweeks, df['symptom:Headache'], label="Headache")
plt.plot(epiweeks, df['symptom:Fatigue'], label="Fatigue")
plt.plot(epiweeks, df['symptom:Muscle weakness'], label="Muscle weakness")
plt.xlabel("Epiweeks")
plt.ylabel("Values")
plt.title("Flu Symptom Trends for 2018-2019 Season")
plt.legend()
plt.show()

