import torch
import torch.nn as nn


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
    
if __name__ == '__main__':
    model = NeuralNetwork()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)
    print(model)
