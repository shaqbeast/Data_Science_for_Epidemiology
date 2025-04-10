import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
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

print(data.head())

# Epiweeks
# Choose season to train on
train_data = data[(data['Year'] >= 2009) & (data['Year'] <= 2016) & (data['Epiweek'] != 0)]
test_data = data[(data['Year'] >= 2017) & (data['Epiweek'] != 0)]

# Get average values for all years in training and testing data sets
epiweek_counts_train = train_data.groupby('Epiweek')['Count_normalized'].sum().reset_index()
epiweek_counts_test = test_data.groupby('Epiweek')['Count_normalized'].sum().reset_index()

# Define the dataset
class FluDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.features = torch.tensor(
            data[['Epiweek']].values, dtype=torch.float32
        )
        self.labels = torch.tensor(
            data['Count_normalized'].values, dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Neural network model
class FluModel(nn.Module):
    def __init__(self, input_dim):
        super(FluModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Synaptic Intelligence loss
class SynapticIntelligence:
    def __init__(self, model, importance_coefficient=0.1):
        self.model = model
        self.importance_coefficient = importance_coefficient
        self.previous_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.importance = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    def update_importance(self, gradients):
        for n, g in gradients.items():
            if n in self.importance:
                self.importance[n] += g.abs()

    def compute_penalty(self):
        penalty = 0.0
        for n, p in self.model.named_parameters():
            if n in self.importance:
                penalty += (
                    self.importance_coefficient
                    * self.importance[n]
                    * (p - self.previous_params[n]).pow(2)
                ).sum()
        return penalty

# Training loop with SI
def train_model_with_si(model, train_loader, optimizer, criterion, si, epochs=50):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)
            base_loss = criterion(outputs, labels)

            # SI penalty
            gradients = {n: p.grad for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
            si.update_importance(gradients)
            penalty = si.compute_penalty()
            total_loss = base_loss + penalty

            # Backward pass
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

# Prepare datasets
train_dataset = FluDataset(epiweek_counts_train)
test_dataset = FluDataset(epiweek_counts_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model, optimizer, criterion
input_dim = train_dataset.features.shape[1]
model = FluModel(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
si = SynapticIntelligence(model)

# Train model
train_model_with_si(model, train_loader, optimizer, criterion, si)

# Evaluate model
def evaluate_and_plot(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:  # Iterate over the DataLoader
            predictions = model(features)
            all_predictions.extend(predictions.numpy())  # Collect predictions
            all_labels.extend(labels.numpy())  # Collect actual labels

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Plot predictions vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_predictions, alpha=0.7, color='blue', label="Data Points")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label="Perfect Fit")
    plt.title("Predicted vs Actual Normalized Flu Cases")
    plt.xlabel("Actual Normalized Flu Cases")
    plt.ylabel("Predicted Normalized Flu Cases")
    plt.legend()
    plt.grid(True)
    plt.show()


test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
evaluate_and_plot(model, test_loader)
