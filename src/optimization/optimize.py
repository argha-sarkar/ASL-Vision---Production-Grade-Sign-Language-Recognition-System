# 1. Import the DataLoader class
from src.data.loader import DataLoader
from src.optimization.optimizer import HyperparameterOptimizer

# 2. Use the static methods to load the data into the expected variables
train_dataset = DataLoader.load_train()
validation_dataset = DataLoader.load_test()

# 3. Now the optimizer will find the variables
optimizer = HyperparameterOptimizer(
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    input_shape=(28, 28, 1),
    num_classes=24,
    epochs=15,
    n_trials=20,
)

study = optimizer.optimize()
