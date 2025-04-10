import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats 

def map_epiweek(previous_epiweek, week):
    # return nothing if the value of the week is between 21-39 
    # if the epiweek is 0, make sure to not graph that point on your plot 
    epiweek = 0
    if week >= 21 and week <= 39:
        return epiweek  
    else:
        return previous_epiweek + 1
    
# read file 
df = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q2/data/ILINet.csv')

# filters out region and year to be national and above year 2000
df = df.loc[(df['REGION'] == 'Region 10') & (df['YEAR'] >= 2000) & (df['YEAR'] <= 2020)]

# populates the epiweeks for df 
df['EPIWEEK'] = 0 # populates all rows with a new column name "EPIWEEK" with 0
df['SEASON'] = ""
previous_epiweek = 13 
for index, row in df.iterrows():
    week = row['WEEK']
    epiweek = map_epiweek(previous_epiweek, week)
    previous_epiweek = epiweek
    df.loc[index, 'EPIWEEK'] = epiweek
    
# filter out any columns with an epiweek that's not part of flu season
df = df.loc[df['EPIWEEK'] > 0]

# add the season for each respective epiweek
for index, row in df.iterrows():
    year = row['YEAR']  
    week = row['WEEK'] 
    epiweek = row['EPIWEEK']
    if week <= 20:
        previous_year = year - 1
        df.loc[index, 'SEASON'] = str(previous_year) + "-" + str(year)
    else: 
        next_year = year + 1
        df.loc[index, 'SEASON'] = str(year) + "-" + str(next_year)

# filter out the seasons that are 2019-2020
df_2019_2020 = df.loc[df['SEASON'] == "2019-2020"]
df = df.loc[df['SEASON'] != "2019-2020"]


# create a dict with values from 1-34 
epiweek_dict = dict()
for key in range(1, 35):
    epiweek_dict[key] = []

# populate dictionary with ILI values for each epiweek from the seasons WITHOUT 2019-2020 season
for index, row in df.iterrows():
    ILI = row['% WEIGHTED ILI']
    epiweek = row['EPIWEEK']
    epiweek_dict[epiweek].append(ILI)

# get the mean and standard deviation for each epiweek 
    
# loop through the df for 2019-2020 season
print("Anomalous Weeks:")
for index, row in df_2019_2020.iterrows():
    test_week = row['WEEK']
    test_epiweek = row['EPIWEEK']
    test_ILI = row['% WEIGHTED ILI']
    test_year = row['YEAR']
    
    # find the mean of the previous seasons for the epiweek we're looking at for the 2019-2020 season
    mean = np.mean(epiweek_dict[test_epiweek])
    std = np.std(epiweek_dict[test_epiweek])
    
    sigma3_above = mean + (3 * std)
    sigma3_below = mean - (3 * std)
    
    if test_ILI < sigma3_below or test_ILI > sigma3_above:
        print(test_week)