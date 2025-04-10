import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q4/data/ILINet_states.csv')
df = df.loc[(df['REGION'] == 'New York') & (((df['YEAR'] == 2018) & (df['WEEK'] >= 40)) | ((df['YEAR'] == 2019) & (df['WEEK'] <= 20)))]

df_2018 = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q4/data/2018_symptoms_dataset.csv')
df_2018 = df_2018.loc[(df_2018['sub_region_1'] == 'New York')]
df_2018 = df_2018.tail(13)

df_2019 = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q4/data/2019_weekly_symptoms_dataset.csv')
df_2019 = df_2019.loc[(df_2019['sub_region_1'] == 'New York')]
df_2019 = df_2019.head(20)

df_2018_2019 = pd.concat([df_2018, df_2019], axis=0)

ILI = []
for index, row in df.iterrows():
    unweighted_ILI = float(row['%UNWEIGHTED ILI'])
    ILI.append(unweighted_ILI)

pcc_fever, p_value = stats.pearsonr(ILI, df_2018_2019['symptom:Fever'])
pcc_low_grade_fever, p_value = stats.pearsonr(ILI, df_2018_2019['symptom:Low-grade fever'])
pcc_cough, p_value = stats.pearsonr(ILI, df_2018_2019['symptom:Cough'])
pcc_sore_throat, p_value = stats.pearsonr(ILI, df_2018_2019['symptom:Sore throat'])
pcc_headache, p_value = stats.pearsonr(ILI, df_2018_2019['symptom:Headache'])
pcc_fatigue, p_value = stats.pearsonr(ILI, df_2018_2019['symptom:Fatigue'])
pcc_muscle, p_value = stats.pearsonr(ILI, df_2018_2019['symptom:Muscle weakness'])

print("PCC Fever: " + str(pcc_fever))
print("PCC Low-grade fever: " + str(pcc_low_grade_fever))
print("PCC Cough: " + str(pcc_cough))
print("PCC Sore Throat: " + str(pcc_sore_throat))
print("PCC Headache: " + str(pcc_headache))
print("PCC Fatigue: " + str(pcc_fatigue))
print("PCC Muscle: " + str(pcc_muscle))