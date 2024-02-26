# Imports necessary pytorch elements
import torch

print(torch.__version__)

# Imports necessary neural network builds
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Sets the default run device to cuda if it is installed, if not then runs on pc's cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Class inherits from neural network module class, which virtually turns the class into a neural network
class MNIST(nn.Module):

    # Upon instantiation, this function runs
    def __init__(self):
        super().__init__()  # ASK ABOUT THIS, NEURAL NETWORK TEMPLATE INSTANCE???
        self.flatten = nn.Flatten()  # Flattens a multidimensional image into one dimension

        # The following creates a container of seperate layers where the data flows through
        self.linear_relu_stack = nn.Sequential(nn.Linear(28 * 28, 512), # Takes 28x28 image and transforms it into linear output, "input layer"
                                               nn.ReLU(),  # Activation function akin to sigmoid
                                               nn.Linear(512, 512),  # Rehashes, likely "hidden layer"
                                               nn.ReLU(),
                                               nn.Linear(512, 10)) # Outlines final output values (10 nodes for 10 digits)

    # Outlines the flow of data in the network
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == '__main__':
    model = MNIST().to(device)  # Instantiates the model in the main method
    print(model)
    data = datasets.MNIST('./data', train=True, download=True)