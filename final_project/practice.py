import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from avalanche.benchmarks import nc_benchmark
from avalanche.models import SimpleMLP
from avalanche.training import SynapticIntelligence
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

# ----- 1. Create Synthetic Datasets (Regression) -----
# Task 1 data (floating point)
X1 = torch.rand(1000, 10)  # 1000 samples, 10 features
y1 = torch.rand(1000, 1)  # Continuous labels for regression
dataset1 = TensorDataset(X1, y1)

# Task 2 data (floating point)
X2 = torch.rand(1000, 10)
y2 = torch.rand(1000, 1)
dataset2 = TensorDataset(X2, y2)

# ----- 2. Define a Continual Learning Benchmark -----
# Split the datasets into tasks
benchmark = nc_benchmark(
    train_dataset=[dataset1, dataset2],  # Train datasets for Task 1 and Task 2
    test_dataset=[dataset1, dataset2],  # Test datasets for Task 1 and Task 2
    task_labels=True,  # Use task labels to distinguish tasks
    n_experiences=2
)

# ----- 3. Create the Model -----
# Simple Multi-Layer Perceptron (MLP) for regression
model = SimpleMLP(input_size=10, hidden_size=50, output_size=1)  # 10 features -> 1 continuous output

# ----- 4. Define Evaluation Plugin -----
# Logs only loss for regression (accuracy is not meaningful for regression tasks)
evaluation_plugin = EvaluationPlugin(
    loss_metrics(),      # Measure loss for each task
    loggers=[InteractiveLogger()]  # Print metrics during training
)

# ----- 5. Define the Synaptic Intelligence Strategy -----
# Synaptic Intelligence regularization for continual learning
cl_strategy = SynapticIntelligence(
    model=model,
    optimizer=optim.SGD(model.parameters(), lr=0.01),
    criterion=nn.MSELoss(),  # Mean Squared Error loss for regression
    si_lambda=0.1,  # Strength of the regularization
    train_mb_size=32,  # Mini-batch size
    train_epochs=5,    # Number of epochs per task
    eval_mb_size=32,   # Mini-batch size for evaluation
    evaluator=evaluation_plugin,
    device='cpu'  # Use 'cuda' if a GPU is available
)

# ----- 6. Train the Model on Sequential Tasks -----
print("Starting training...")
for task_id, experience in enumerate(benchmark.train_stream):
    print(f"Training on Task {task_id + 1}")
    cl_strategy.train(experience)  # Train on current task
    print("Evaluating...")
    cl_strategy.eval(benchmark.test_stream)  # Evaluate on all tasks so far
