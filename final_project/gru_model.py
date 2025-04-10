import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from modules.gru_network import GRUNetwork
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

'''Data Preprocessing'''
def map_to_epiweek(year, week):
    """
    Maps a regular week of a year to its corresponding Epiweek in a flu season.

    Args:
        year (int): The calendar year. This year needs to be the later year.
        (If the year value is 2014-2015, 2015 needs to be passed into the method)
        week (int): The calendar week (1-52/53)

    Returns:
        tuple: (season_start_year, epiweek), or None if not part of a flu season.
    """
    if week >= 40:
        season_start_year = year
        epiweek = week - 39
    elif week <= 20:
        season_start_year = year - 1
        epiweek = week + 13
    else:
        return 0 # weeks 21-39 are not epiweeks

    return season_start_year, epiweek

data = pd.read_csv("final_project/data/NY_influenza.csv")

# Time Information
data['Week Ending Date'] = pd.to_datetime(data['Week Ending Date'])

data['Year'] = data['Week Ending Date'].dt.year
data['Month'] = data['Week Ending Date'].dt.month
data['Week'] = data['Week Ending Date'].dt.isocalendar().week

data[['season_start', 'season_end']] = data['Season'].str.split("-", expand=True).astype(int)

disease_encoder = LabelEncoder()
data['Disease_encoded'] = disease_encoder.fit_transform(data['Disease'])

# Counts
scaler = MinMaxScaler()
data['Count_normalized'] = scaler.fit_transform(data[['Count']])

data['Count_log_scaled'] = np.log1p(data['Count'])


# Missing Values
# data['Count_filled'] = data['Count'].fillna(method='ffill')
data['Count_filled'] = data.groupby(['County', 'Week'])['Count'].transform(lambda x: x.fillna(x.mean()))

data['Epiweek'] = 0
for index, row in data.iterrows():
  year = row['Year']
  week = row['Week']
  tup = map_to_epiweek(year, week)

  if tup != 0:
    season_start, epiweek = tup
    data.loc[index, 'Epiweek'] = epiweek

# print(data.columns)

# Aggregate weekly counts
# Group by the Year/Epiweek
# 2009, Epiweek 1 CountX ... 
# 2009, Epiweek 2 CountX ... Sum all counts grouped by year and epiweek 
weekly_counts = data.groupby(['Year', 'Epiweek'])['Count'].sum().reset_index()

# getting lag counts 
# Lagged count 1 shifts the counts 1 down (you lose that last value at epiweek 33)
# Lagged count 2 shifts the counts 2 down (you lose count values at epiweek 32 and 33)
weekly_counts['Lagged Count 1'] = weekly_counts['Count'].shift(1)
weekly_counts['Lagged Count 2'] = weekly_counts['Count'].shift(2)
weekly_counts.fillna(0, inplace=True)

# Normalize features
# Scale the data
# Import so that we don't have outliers cause a ton of change in the model
scaler = MinMaxScaler()
weekly_counts[['Normalized Count', 'Lagged Count 1', 'Lagged Count 2']] = scaler.fit_transform(
    weekly_counts[['Count', 'Lagged Count 1', 'Lagged Count 2']]
)

# Epiweeks
# Choose season to train on
train_data = weekly_counts[(weekly_counts['Year'] < 2022) & (weekly_counts['Epiweek'] != 0)]
test_data = weekly_counts[(weekly_counts['Year'] == 2022) & (weekly_counts['Epiweek'] != 0)]


'''GRU Analysis'''

# Training data: Sequences and labels
# .values gives numpy array
# tensors are multidimensional arrays that only the model accepts
# train_sequences contains the values that will be used to create predictions
# train_labels performs the supervised learning
train_sequences, train_labels = train_data[['Lagged Count 1', 'Lagged Count 2']].values, train_data['Normalized Count'].values
train_sequences = torch.tensor(train_sequences, dtype=torch.float32).unsqueeze(-1)
train_labels = torch.tensor(train_labels, dtype=torch.float32)

# Create DataLoader for training (optional, but useful for batching)
train_dataset = torch.utils.data.TensorDataset(train_sequences, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = GRUNetwork(input_size=1, hidden_size=64, output_size=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs): # run through the datasets 10 times
    model.train() # put the model in training mode
    for seqs, labels in train_loader:
        optimizer.zero_grad() # clears any previous gradients so that the model is training fresh
        outputs = model(seqs) # model's predictions for the next week's normalized flu counts
        loss = loss_function(outputs.squeeze(), labels) # computes loss between predicted and targeted values
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    

# Store predictions for 2022
test_sequences, test_labels = test_data[['Lagged Count 1', 'Lagged Count 2']].values, test_data['Normalized Count'].values
test_sequences = torch.tensor(test_sequences, dtype=torch.float32).unsqueeze(-1)

# Predict for each week in the test data (2022)
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    # Predict the next week's count
    predictions = model(test_sequences).numpy()

rescaled_test = scaler.inverse_transform(
    np.column_stack([test_labels, np.zeros((len(test_labels), 2))])
)[:, 0]
rescaled_predictions = scaler.inverse_transform(
    np.column_stack([predictions, np.zeros((len(predictions), 2))])
)[:, 0]

epiweeks = []
for i in range(1, 34):
    epiweeks.append(i)

# `predictions` now contains the predicted influenza case counts for 2022

# Pearson Correlation Coefficient
pearson_corr, _ = pearsonr(rescaled_test, rescaled_predictions)
print(f"Pearson Correlation Coefficient: {pearson_corr}")

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(rescaled_test, rescaled_predictions))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Mean Absolute Error (MAE)
mae = mean_absolute_error(rescaled_test, rescaled_predictions)
print(f"Mean Absolute Error (MAE): {mae}")

# Plot actual vs. predicted counts
plt.plot(epiweeks, rescaled_test, label='Actual', marker='o')
plt.plot(epiweeks, rescaled_predictions, color='salmon', marker='x', label='Predicted', linestyle='--')
plt.xlabel('Epiweek')
plt.xticks(ticks=range(1, 35, 2))
plt.ylabel('Influenza Case Count')
plt.title('Predicted vs Actual Influenza Cases for 2022')
plt.grid()
plt.legend()
plt.show()

# Residuals
residuals = rescaled_predictions - rescaled_test

plt.scatter(epiweeks, residuals, label='Residuals')
plt.xlabel('Epiweek')
plt.ylabel('Influenza Case Count')
plt.title('Residuals between Predicted and Actual Influenza Cases for 2022')
plt.legend()
plt.show()

