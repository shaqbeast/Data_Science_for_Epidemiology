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
df = df.loc[(df['REGION'] == 'Region 7') & (df['YEAR'] >= 2000) & (df['YEAR'] <= 2019)]

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
df = df.loc[df['SEASON'] != "2019-2020"]

# create a dict with values from 1-34 
# each value for the key in these dicts will be a list of ILI values for the corresponding epiweek for each season
# run mean and standard deviation on each key 
epiweek_dict = dict()
for key in range(1, 35):
    epiweek_dict[key] = []

# populate dictionary with ILI values
for index, row in df.iterrows():
    ILI = row['% WEIGHTED ILI']
    epiweek = row['EPIWEEK']
    epiweek_dict[epiweek].append(ILI)

results_list = []

# remember key is the epiweek
# finding mean, std, and saving it into an array
for key in range(1, 35):
    mean = np.mean(epiweek_dict[key]) # finds mean of a specific epiweek over the seasons
    std = np.std(epiweek_dict[key]) # finds std of a specific epiweek over the seasons
    
    results_list.append({
        'Epiweek': key,
        'Mean': mean,
        'Standard Deviation': std
    })
    
    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    gaussian_distribution = stats.norm.pdf(x, mean, std)
    plot_label = "Gaussian Distribution - Epiweek: " + str(key)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, gaussian_distribution, color='blue')
    plt.title(plot_label)
    plt.xlabel('ILI')
    plt.ylabel('Probability Distribution')
    plt.grid()
    plt.show()

results = pd.DataFrame(results_list)

results.to_csv('Q2.5-Region7_Epiweek_mean_std.csv', index=False)  