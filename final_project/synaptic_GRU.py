import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks import dataset_benchmark
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from avalanche.training import SynapticIntelligence
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from modules.gru_network import GRUNetwork

'''Data Preprocessing'''
def map_to_epiweek(year, week):
    if week >= 40:
        season_start_year = year
        epiweek = week - 39
    elif week <= 20:
        season_start_year = year - 1
        epiweek = week + 13
    else:
        return 0

    return season_start_year, epiweek

seed = 100
np.random.seed(seed)
torch.manual_seed(seed)

data = pd.read_csv("/Users/shaqbeast/CSE 8803 EPI/final_project/data/NY_influenza.csv")

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
data['Count_filled'] = data.groupby(['County', 'Week'])['Count'].transform(lambda x: x.fillna(x.mean()))

data['Epiweek'] = 0
for index, row in data.iterrows():
  year = row['Year']
  week = row['Week']
  tup = map_to_epiweek(year, week)

  if tup != 0:
    season_start, epiweek = tup
    data.loc[index, 'Epiweek'] = epiweek

# Aggregate weekly counts
weekly_counts = data.groupby(['Year', 'Epiweek'])['Count'].sum().reset_index()

# Lagged counts
weekly_counts['Lagged Count 1'] = weekly_counts['Count'].shift(1)
weekly_counts['Lagged Count 2'] = weekly_counts['Count'].shift(2)
weekly_counts.fillna(0, inplace=True)

# Normalize features
scaler = MinMaxScaler()
weekly_counts[['Normalized Count', 'Lagged Count 1', 'Lagged Count 2']] = scaler.fit_transform(
    weekly_counts[['Count', 'Lagged Count 1', 'Lagged Count 2']]
)

# Train and test data
year = 2021
task1 = weekly_counts[(weekly_counts['Year'] <= year) & (weekly_counts['Epiweek'] != 0)]
'''task2 = weekly_counts[(weekly_counts['Year'] > 2020) & (weekly_counts['Year'] <= 2021) & (weekly_counts['Epiweek'] != 0)]'''
forecast = weekly_counts[(weekly_counts['Year'] == 2022) & (weekly_counts['Epiweek'] != 0)]

'''Synaptic Intelligence Model'''
# Task 1 Tensor
task1_sequences, task1_labels = task1[['Lagged Count 1', 'Lagged Count 2']].values, task1['Normalized Count'].values
task1_sequences = torch.tensor(task1_sequences, dtype=torch.float32).unsqueeze(-1)
task1_labels = torch.tensor(task1_labels, dtype=torch.float32)

# Task 2 Tensor
'''task2_sequences, task2_labels = task2[['Lagged Count 1', 'Lagged Count 2']].values, task2['Normalized Count'].values
task2_sequences = torch.tensor(task2_sequences, dtype=torch.float32).unsqueeze(-1)
task2_labels = torch.tensor(task2_labels, dtype=torch.float32)'''

# Datasets
dataset1 = TensorDataset(task1_sequences, task1_labels)
'''dataset2 = TensorDataset(task2_sequences, task2_labels)'''

# ----- 2. Define a Continual Learning Benchmark -----
# Split the datasets into tasks
benchmark = dataset_benchmark(train_datasets=[dataset1], test_datasets=[dataset1])

# Create model
model = GRUNetwork(input_size=1, hidden_size=64, output_size=1)

# Logs only loss for regression (accuracy is not meaningful for regression tasks)
evaluation_plugin = EvaluationPlugin(
    loss_metrics(minibatch=True, epoch=True, experience=True),      # Measure loss for each task
    loggers=[InteractiveLogger()]  # Print metrics during training
)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# Synaptic Intelligence regularization for continual learning
cl_strategy = SynapticIntelligence(
    model=model,
    optimizer=optimizer,
    criterion=nn.MSELoss(),  # Mean Squared Error loss for regression
    si_lambda=0.1,  # Strength of the regularization
    train_mb_size=32,  # Mini-batch size
    train_epochs=50,    # Number of epochs per task
    eval_mb_size=32,   # Mini-batch size for evaluation
    evaluator=evaluation_plugin,
    device='cpu'  # Use 'cuda' if a GPU is available
)

# Train model
print("Starting training...")
for task_id, experience in enumerate(benchmark.train_stream):
    print(f"Training on Task {task_id + 1}")
    cl_strategy.train(experience)  # Train on current task
    print("Evaluating...")
    cl_strategy.eval(benchmark.test_stream)  # Evaluate on all tasks so far

# Forecasting
forecast_sequences, forecast_labels = forecast[['Lagged Count 1', 'Lagged Count 2']].values, forecast['Normalized Count'].values
forecast_sequences = torch.tensor(forecast_sequences, dtype=torch.float32).unsqueeze(-1)
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    # Predict the next week's count
    predictions = model(forecast_sequences).numpy()

rescaled_test = scaler.inverse_transform(
    np.column_stack([forecast_labels, np.zeros((len(forecast_labels), 2))])
)[:, 0]
rescaled_predictions = scaler.inverse_transform(
    np.column_stack([predictions, np.zeros((len(predictions), 2))])
)[:, 0]

scaling_factor = np.mean(rescaled_test) / np.mean(rescaled_predictions)
rescaled_predictions = rescaled_predictions * scaling_factor

epiweeks = []
for i in range(1, 34):
    epiweeks.append(i)

# Pearson Correlation Coefficient
pearson_corr, _ = pearsonr(rescaled_test, rescaled_predictions)
print(f"Pearson Correlation Coefficient: {pearson_corr}")

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(rescaled_test, rescaled_predictions))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Mean Absolute Error (MAE)
mae = mean_absolute_error(rescaled_test, rescaled_predictions)
print(f"Mean Absolute Error (MAE): {mae}")

plt.plot(epiweeks, rescaled_test, label='Actual', marker='o')
plt.plot(epiweeks, rescaled_predictions, label='Predicted', linestyle='--', marker='x')
plt.xlabel('Epiweek')
plt.xticks(ticks=range(1, 35, 2))
plt.ylabel('Influenza Case Count')
plt.title(f'Predicted vs Actual Influenza Cases 2022 - Continual Learning Model Trained up until {year}')
plt.legend()
plt.grid()
plt.show()



