# Imports necessary pytorch stuff
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Defines the neural network model
class NeuralNet(nn.Module):

    # Defines what to do upon initialization
    def __init__(self):
        super(NeuralNet, self).__init__()  # Initializes the parent class of the current neural network
        self.fc1 = nn.Linear(28 * 28, 128)  # Creates the fully connected layer, flattening the 28*28 image
        self.relu = nn.ReLU()  # Sets the activation function for the first layer
        self.fc2 = nn.Linear(128, 10)  # Generates an output layer based on the layer that is feeding it

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flattens the 28*28 image into a usable 1D format
        x = self.fc1(x)  # Creates a fully connected layer from the inputted image
        x = self.relu(x)  # Applies the activation function to the layer that was set up in line 22
        x = self.fc2(x)  # Creates the output layer based off of the values taken from layer 1 after ReLU transformation
        return x  # Returns the given output


# Load MNIST dataset

# Concatenates the transforms taken from the torchvision package
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Gets the MNIST dataset from the torchvision package, sets as training set
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Root specifies where data is stored, train is whether it is a training set, transform applies a transform to the data,
# and download checks if the dataset exists in the given root directory, if not then it downloads it
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Loads the data onto a DataLoader object and sets the batch size for each epoch, also shuffles the batch
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Similarly loads the test data set, but does not shuffle
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Initialize the network, loss function, and optimizer
model = NeuralNet()  # Instantiates a neural network object
criterion = nn.CrossEntropyLoss()  # Instantiates a loss function that utilizes cross entropy
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Performs gradient descent for each node using the learning rate specified


# Outlines a function for training the network
def train(epoch):
    model.train()  # Sets the model in training mode so things that act differently during training process properly
    for batch_idx, (data, target) in enumerate(train_loader):  # Loops through the training data
        # (batch_idx = batch index, data = data values, target = data labels)
        optimizer.zero_grad()  # Wipes previously calculated gradients to prevent accidental accumulation
        output = model(data)  # Passes the data into the model and saves its output onto a variable
        loss = criterion(output, target)  # Uses the loss function to compare the output values to the target values
        loss.backward()  # Does backpropagation (updates weights based on loss results) through the neural network
        optimizer.step()  # Updates the parameters of the model based on the new computed gradients
        if batch_idx % 100 == 0:  # Prints the data calculated for every 100 datapoints
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,  # Prints the current epoch being calculated
                batch_idx * len(data),  # Prints the current batch
                len(train_loader.dataset),  # Prints the total amount of data to be processed
                100. * batch_idx / len(train_loader),  # Prints the percentage of data processed
                loss.item()))  # Prints the value calculated by the loss function


# Testing the network
def test():
    model.eval()  # Sets the model to evaluation mode, as things may act differently than in training mode
    test_loss = 0  # Sets/Resets a variable for the total test loss across all batches
    correct = 0  # Sets/Resets a variable for the amount of correctly predicted test values
    with torch.no_grad():  # Temporarily disables gradient descent calculations for the inner block (mainly for speed)
        for data, target in test_loader:  # Iterates through batches of the test data
            output = model(data)  # Passes the data into the model and saves its output onto a variable
            test_loss += criterion(output, target).item()  # Computes the loss of the current batch and adds it to test_loss
            pred = output.argmax(dim=1, keepdim=True)  # Pulls the maximum value of the 10 output nodes, or the "prediction"
            correct += pred.eq(target.view_as(pred)).sum().item()  # Pulls the correct labels and compares to the predicted,
                                                                   # Then adds the correctly estimated values to the variable

    test_loss /= len(test_loader.dataset)  # Averages the test loss across the number of values
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,  # Prints the average test loss
        correct,  # Prints the number of correctly predicted images
        len(test_loader.dataset),  # Prints the length of the complete dataset
        100. * correct / len(test_loader.dataset)))  # Prints the percentage of values that were predicted correctly


# Train and test the network
for epoch in range(1, 6):
    train(epoch)
    test()
