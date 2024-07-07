import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import ray.tune as tune
from ray.tune.schedulers import ASHAScheduler

# Initialize Ray
ray.init()

# Define a simple neural network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_cnn(config, checkpoint_dir=None):
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN()
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(checkpoint_dir)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Save checkpoint
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = checkpoint_dir + "/checkpoint"
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=loss.item())

# Define hyperparameter search space
config = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([32, 64, 128]),
    "epochs": 10
}

# Define scheduler and reporter
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=10,
    grace_period=1,
    reduction_factor=2
)

reporter = tune.CLIReporter(
    metric_columns=["loss", "training_iteration"]
)

# Run the hyperparameter tuning
tune.run(
    train_cnn,
    resources_per_trial={"cpu": 2, "gpu": 1},
    config=config,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=reporter
)
