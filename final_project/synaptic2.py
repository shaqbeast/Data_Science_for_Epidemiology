import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
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

def set_seed(seed):
    """
    Set the seed for reproducibility in PyTorch, NumPy, and Python's random module.
    """
    # Python random seed
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch random seeds
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For single GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU (if applicable)

    # Ensures deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Can slow down training slightly

# Example Usage:
seed_value = 42
set_seed(seed_value)


def create_tensors(task):
    task_sequences, task_labels = task[['Lagged Count 1', 'Lagged Count 2']].values, task['Normalized Count'].values
    task_sequences = torch.tensor(task_sequences, dtype=torch.float32).unsqueeze(1)
    task_labels = torch.tensor(task_labels, dtype=torch.float32)
    
    return task_sequences, task_labels

class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bias=True)  # GRU layer
        self.fc = nn.Linear(hidden_size, output_size, bias=True)  # Dense layer
    
    def forward(self, x):
        out, _ = self.gru(x)  # Pass through GRU
        # Check if the sequence length is 1, and use the appropriate indexing
        if len(out.shape) == 3:  # [batch_size, sequence_length, hidden_size]
            out = out[:, -1, :]  # Use the last output of the GRU
        else:  # [batch_size, hidden_size] (if sequence_length == 1)
            out = out.squeeze(1)  # Remove the sequence dimension

        out = self.fc(out)  # Dense layer to predict the output
        return out


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

print(weekly_counts.head(20))

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
task1 = weekly_counts[(weekly_counts['Year'] >= 2009) & (weekly_counts['Year'] <= 2015) & (weekly_counts['Epiweek'] != 0)]
task2 = weekly_counts[(weekly_counts['Year'] >= 2016) & (weekly_counts['Year'] <= 2021) & (weekly_counts['Epiweek'] != 0)]
'''task3 = weekly_counts[(weekly_counts['Year'] >= 2013) & (weekly_counts['Year'] <= 2014) & (weekly_counts['Epiweek'] != 0)]
task4 = weekly_counts[(weekly_counts['Year'] >= 2015) & (weekly_counts['Year'] <= 2016) & (weekly_counts['Epiweek'] != 0)]
task5 = weekly_counts[(weekly_counts['Year'] >= 2017) & (weekly_counts['Year'] <= 2018) & (weekly_counts['Epiweek'] != 0)]
task6 = weekly_counts[(weekly_counts['Year'] >= 2019) & (weekly_counts['Year'] <= 2020) & (weekly_counts['Epiweek'] != 0)]
task7 = weekly_counts[(weekly_counts['Year'] == 2021) & (weekly_counts['Epiweek'] != 0)]'''
forecast = weekly_counts[(weekly_counts['Year'] == 2022) & (weekly_counts['Epiweek'] != 0)]

'''Synaptic Intelligence Model'''
# Task 1 Tensor
task1_sequences, task1_labels = create_tensors(task1)

# Task 2 Tensor
task2_sequences, task2_labels = create_tensors(task2)

'''# Task 3 Tensor
task3_sequences, task3_labels = create_tensors(task3)

# Task 4 Tensor
task4_sequences, task4_labels = create_tensors(task4)

# Task 5 Tensor
task5_sequences, task5_labels = create_tensors(task5)

# Task 6 Tensor
task6_sequences, task6_labels = create_tensors(task6)

# Task 7 Tensor
task7_sequences, task7_labels = create_tensors(task7)'''

# Datasets
dataset1 = TensorDataset(task1_sequences, task1_labels)
dataset2 = TensorDataset(task2_sequences, task2_labels)
'''dataset3 = TensorDataset(task3_sequences, task3_labels)
dataset4 = TensorDataset(task4_sequences, task4_labels)
dataset5 = TensorDataset(task5_sequences, task5_labels)
dataset6 = TensorDataset(task6_sequences, task6_labels)
dataset7 = TensorDataset(task7_sequences, task7_labels)'''

# ----- 2. Define a Continual Learning Benchmark -----
# Split the datasets into tasks
benchmark = dataset_benchmark(train_datasets=[dataset1, dataset2], 
                              test_datasets=[dataset1, dataset2])

# Create model
# model = CustomMLP(input_size=2, hidden_size=64, output_size=1)
model = GRUNetwork(input_size=2, hidden_size=64, output_size=1) 

# Logs only loss for regression (accuracy is not meaningful for regression tasks)
evaluation_plugin = EvaluationPlugin(
    loss_metrics(),      # Measure loss for each task
    loggers=[InteractiveLogger()]  # Print metrics during training
)

# Synaptic Intelligence regularization for continual learning
cl_strategy = SynapticIntelligence(
    model=model,
    optimizer=optim.SGD(model.parameters(), lr=0.1),
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

forecast_sequences, forecast_labels = forecast[['Lagged Count 1', 'Lagged Count 2']].values, forecast['Normalized Count'].values
forecast_sequences = torch.tensor(forecast_sequences, dtype=torch.float32)
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    # Predict the next week's count
    predictions = model(forecast_sequences).numpy()

# Transform normalized counts back to normal counts
rescaled_test = scaler.inverse_transform(
    np.column_stack([forecast_labels, np.zeros((len(forecast_labels), 2))])
)[:, 0]
rescaled_predictions = scaler.inverse_transform(
    np.column_stack([predictions, np.zeros((len(predictions), 2))])
)[:, 0]

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

plt.plot(epiweeks, rescaled_test, label='Actual')
plt.plot(epiweeks, rescaled_predictions, label='Predicted', linestyle='--')
plt.xlabel('Epiweek')
plt.ylabel('Influenza Case Count')
plt.title('Predicted vs Actual Influenza Cases for 2022 - Continual Learning Model')
plt.legend()
plt.show()

