import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q4/data/ILINet_states.csv')

df = df.loc[(df['REGION'] == 'Alaska') & (((df['YEAR'] == 2018) & (df['WEEK'] >= 40)) | ((df['YEAR'] == 2019) & (df['WEEK'] <= 20)))]

df['EPIWEEK'] = 0
epiweek = 1
for index, row in df.iterrows():
    df.loc[index, 'EPIWEEK'] = epiweek
    epiweek += 1

print(df['EPIWEEK'])
print(df['%UNWEIGHTED ILI'])


X = []
y = []
for index, row in df.iterrows():
    epiweek = row['EPIWEEK']
    unweighted_ILI = float(row['%UNWEIGHTED ILI'])
    X.append(epiweek)
    y.append(unweighted_ILI)

plt.plot(X, y)
plt.xlabel('Epiweek')
plt.ylabel('% Unweighted ILI')
plt.title('Alaska Flu Season: 2018-2019 (Unweighted ILI)')
plt.show()