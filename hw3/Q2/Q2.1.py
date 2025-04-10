import pandas as pd 
import matplotlib.pyplot as plt

def map_epiweek(previous_epiweek, week):
    # return nothing if the value of the week is between 21-39 
    # if the epiweek is 0, make sure to not graph that point on your plot 
    epiweek = 0
    if week >= 21 and week <= 39:
        return epiweek  
    else:
        return previous_epiweek + 1

'''CHANGE FILE NAME HERE'''
df = pd.read_csv('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q2/data/ILINet.csv')

# get the values of year, week, and %ILI
# filter out all the rows that are below year 2000 and National region
# create a dictionary that will map the value of the year of the week with the epiweek 
# week 40 of the year = Epiweek 1 ... 

# filter out the region and year
df = df.loc[(df['REGION'] == 'X') & (df['YEAR'] >= 2000)]

# itterrows gives us 2 return values within a tuple
# since the first week we start with is 1 and the previous year wasn't a leap year, this will be week 13
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

'''
array = []
for i in range(len(df)):
    if i % 75 == 0:
        array.append(i)
plt.xticks(array)
'''

# get a dataframe that's different for each season 
df1 = df.loc[df['SEASON'] == "2000-2001"]
df2 = df.loc[df['SEASON'] == "2001-2002"]
df3 = df.loc[df['SEASON'] == "2002-2003"]
df4 = df.loc[df['SEASON'] == "2003-2004"]
df5 = df.loc[df['SEASON'] == "2004-2005"]
df6 = df.loc[df['SEASON'] == "2005-2006"]
df7 = df.loc[df['SEASON'] == "2006-2007"]
df8 = df.loc[df['SEASON'] == "2007-2008"]
df9 = df.loc[df['SEASON'] == "2008-2009"]
df10 = df.loc[df['SEASON'] == "2009-2010"]
df11 = df.loc[df['SEASON'] == "2010-2011"]
df12 = df.loc[df['SEASON'] == "2011-2012"]
df13 = df.loc[df['SEASON'] == "2012-2013"]
df14 = df.loc[df['SEASON'] == "2013-2014"]
df15 = df.loc[df['SEASON'] == "2014-2015"]
df16 = df.loc[df['SEASON'] == "2015-2016"]
df17 = df.loc[df['SEASON'] == "2016-2017"]
df18 = df.loc[df['SEASON'] == "2017-2018"]
df19 = df.loc[df['SEASON'] == "2018-2019"]
df20 = df.loc[df['SEASON'] == "2019-2020"]

# plotting 
plt.plot(df1['EPIWEEK'], df1['% WEIGHTED ILI'], label="2000-2001")
plt.plot(df2['EPIWEEK'], df2['% WEIGHTED ILI'], label="2001-2002")
plt.plot(df3['EPIWEEK'], df3['% WEIGHTED ILI'], label="2002-2003")
plt.plot(df4['EPIWEEK'], df4['% WEIGHTED ILI'], label="2003-2004")
plt.plot(df5['EPIWEEK'], df5['% WEIGHTED ILI'], label="2004-2005")
plt.plot(df6['EPIWEEK'], df6['% WEIGHTED ILI'], label="2005-2006")
plt.plot(df7['EPIWEEK'], df7['% WEIGHTED ILI'], label="2006-2007")
plt.plot(df8['EPIWEEK'], df8['% WEIGHTED ILI'], label="2007-2008")
plt.plot(df9['EPIWEEK'], df9['% WEIGHTED ILI'], label="2008-2009")
plt.plot(df10['EPIWEEK'], df10['% WEIGHTED ILI'], label="2009-2010")
plt.plot(df11['EPIWEEK'], df11['% WEIGHTED ILI'], label="2010-2011")
plt.plot(df12['EPIWEEK'], df12['% WEIGHTED ILI'], label="2011-2012")
plt.plot(df13['EPIWEEK'], df13['% WEIGHTED ILI'], label="2012-2013")
plt.plot(df14['EPIWEEK'], df14['% WEIGHTED ILI'], label="2013-2014")
plt.plot(df15['EPIWEEK'], df15['% WEIGHTED ILI'], label="2014-2015")
plt.plot(df16['EPIWEEK'], df16['% WEIGHTED ILI'], label="2015-2016")
plt.plot(df17['EPIWEEK'], df17['% WEIGHTED ILI'], label="2016-2017")
plt.plot(df18['EPIWEEK'], df18['% WEIGHTED ILI'], label="2017-2018")
plt.plot(df19['EPIWEEK'], df19['% WEIGHTED ILI'], label="2018-2019")
plt.plot(df20['EPIWEEK'], df20['% WEIGHTED ILI'], label="2019-2020")
plt.xlabel('Epiweek')
plt.ylabel('ILI')
plt.title("ILI vs Epiweek - US National Region")
plt.legend()
plt.show()


