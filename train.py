import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Initializations
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print('Using ' + device)
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Data Preparation
train_set = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())     # Download from open datasets.
test_set = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)
for X, y in train_loader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Define Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Fisrt Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=1, padding=1)      # Convolution Layer: input channel is 1, output channel is 8, kernel size is 3*3, stride is 1, padding is 1
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)                                             # Maxpooling Layer: kernel size is 2*2, stride is 2
        self.relu1 = nn.ReLU()                                                                              # Activation Function: ReLU

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.relu2 = nn.ReLU()

        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.relu3 = nn.ReLU()

        # Fully Connected Layer
        self.flatten = nn.Flatten()                                                                         # Flatten the tensor from 3 dimensions to 1 dimension, while keeping the batch size
        self.linear1 = nn.Linear(in_features=3*3*32, out_features=128)
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=128, out_features=10)                                          # Output: 10 categories

                                                        # batch_size, channel, height, width
    def forward(self, x):                               # torch.Size([1, 1, 28, 28])
        x = self.conv1(x)                               # torch.Size([1, 8, 28, 28])
        x = self.pool1(x)                               # torch.Size([1, 8, 14, 14])
        x = self.relu1(x)                               # torch.Size([1, 8, 14, 14])

        x = self.conv2(x)                               # torch.Size([1, 16, 14, 14])
        x = self.pool2(x)                               # torch.Size([1, 16, 7, 7])
        x = self.relu2(x)                               # torch.Size([1, 16, 7, 7])

        x = self.conv3(x)                               # torch.Size([1, 32, 7, 7])
        x = self.pool3(x)                               # torch.Size([1, 32, 3, 3])
        x = self.relu3(x)                               # torch.Size([1, 32, 3, 3])

                                                        # batch_size, features
        x = self.flatten(x)                             # torch.Size([1, 288])
        x = self.linear1(x)                             # torch.Size([1, 128])
        x = self.relu4(x)                               # torch.Size([1, 128])
        x = self.linear2(x)                             # torch.Size([1, 10])
        return x

model = NeuralNetwork().to(device)
print(model)

# Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Train and Test
def train(dataloader, model, loss_fn, optimizer, losses):
    size = len(dataloader.dataset)
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

        if batch % 100 == 99:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, accuracies):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracies.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 30
losses = []
accuracies = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer, losses)
    test(test_loader, model, loss_fn, accuracies)
torch.save(model.state_dict(), "model.pth")                         # Save the trained model

# Plot the loss and accuracy
fig, ax1 = plt.subplots()
ax = fig.add_subplot(1, 2, 1)               # 1 row, 2 columns, 1st subplot
ax.plot(losses)
ax.set_title('Training Loss')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

ax = fig.add_subplot(1, 2, 2)
ax.plot(accuracies)
ax.set_title('Test Accuracy')
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

plt.show()
