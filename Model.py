import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


test_set = 'C:\\path\\to\\testing_data.csv'
train_set = 'C:\\path\\to\\training_data.csv'
batch = 64
lr = 0.1
epoch = 3

def Transform_Data(path:str):
    """
    Transform data from CSV to PyTorch readable tensor.
    Returns: Y = first row (labels) | X = rest of the data
    """
    data = pd.read_csv(path, dtype=np.float32)
    data = np.array(data).T
    row, column = data.shape
    data = np.ndarray(shape=(row,column), dtype=np.float32, buffer=data)
    data = torch.as_tensor(data, dtype=torch.float32)
    Y = data[0].float()
    X = data[1:].float()
    X = X / 255. # Generalize data
    return X, Y

# Turn CSV into PyTorch readable tensor.
trainX, trainY = Transform_Data(train_set)
testX, testY = Transform_Data(test_set)

# Create PyTorch custom TensorDataset -> Feed it to the PyTorch DataLoader
trainDataset = TensorDataset(trainX.T, trainY)
trainDataLoader = DataLoader(trainDataset, batch_size=batch)

testDataset = TensorDataset(testX.T, testY)
testDataLoader = DataLoader(testDataset, batch_size=batch)

# Select device for training process. (CPU or CUDA). 
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# The model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Define Model, loss function and optimizer
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Training
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y.long())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Testing
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y.long()).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test results: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

# Actual training and testing loop with ability to save the most accurate model.
for i in range(epoch):
    print(f'Training {i+1} epoch out of {epoch}.\n ------------------')
    train(trainDataLoader, model, loss_fn, optimizer)
    last = test(testDataLoader, model, loss_fn)
    best = 0

    if last > best:
        torch.save(model.state_dict(), 'C:\\path\\to\\save_data.csv')
        best = last




