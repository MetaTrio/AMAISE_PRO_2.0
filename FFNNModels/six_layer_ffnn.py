import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        
        self.fc1 = nn.Linear(512, 2048)   # Input layer with 512 features -- 5mer
        self.fc2 = nn.Linear(2048, 1024)  # Second hidden layer with 1024 neurons
        self.fc3 = nn.Linear(1024, 512)   # Third hidden layer with 512 neurons
        self.fc4 = nn.Linear(512, 256)    # Fourth hidden layer with 256 neurons
        self.fc5 = nn.Linear(256, 128)    # Fifth hidden layer with 128 neurons
        self.fc6 = nn.Linear(128, 6)      # Output layer with 6 classes
        
        # Activation function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Forward pass through the deeper network
        x = self.relu(self.fc1(x))  # 1st hidden layer
        x = self.relu(self.fc2(x))  # 2nd hidden layer
        x = self.relu(self.fc3(x))  # 3rd hidden layer
        x = self.relu(self.fc4(x))  # 4th hidden layer
        x = self.relu(self.fc5(x))  # 5th hidden layer
        x = self.softmax(self.fc6(x))  # Output layer (with softmax for classification)
        return x
