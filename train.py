import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from net import NeuralNetwork

# Initializations
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f'Using {device}')
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Data Preparation
train_set = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())     # Download from open datasets.
test_set = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)
for X, y in train_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Model Preparation
model = NeuralNetwork().to(device)

# Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train and Test
def train(dataloader, model, loss_fn, optimizer, losses_train):
    size = len(dataloader.dataset)
    loss_train_avg = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward propagation
        pred = model(X)
        loss = loss_fn(pred, y)

        # Back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_train_avg += loss.item()

        if batch % 100 == 99:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    losses_train.append(loss_train_avg / len(dataloader))

def test(dataloader, model, loss_fn, accuracies, losses_test):
    size = len(dataloader.dataset)
    loss_test_avg = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss_test_avg += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size      
    losses_test.append(loss_test_avg / len(dataloader))
    accuracies.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss_test_avg:>8f} \n")

epochs = 30
losses_train = []
losses_test = []
accuracies = []
best_epoch = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer, losses_train)
    test(test_loader, model, loss_fn, accuracies, losses_test)
    if accuracies[t] > accuracies[best_epoch]:                  # Save the best model
        best_epoch = t
        torch.save(model.state_dict(), "model\\best_model.pth")
print(f"Best model is saved at epoch {best_epoch+1}")

# Plot the loss and accuracy
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(10, 5))
ax0.plot(losses_train, label='train')
ax0.plot(losses_test, label='test')
ax0.set_title('Loss')
ax0.set_xlabel('epoch')
ax0.set_ylabel('loss')
ax0.legend(loc='upper right')
ax0.grid(True)

ax1.plot(accuracies)
ax1.set_ylim(0.5, 1)
ax1.set_title('Accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.grid(True)

plt.show()
