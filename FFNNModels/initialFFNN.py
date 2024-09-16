import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(32, 1024)   # Input layer with 32 features, 1024 neurons
        self.fc2 = nn.Linear(1024, 256)  # Hidden layer with 256 neurons
        self.fc3 = nn.Linear(256, 128)   # Hidden layer with 128 neurons
        self.fc4 = nn.Linear(128, 6)     # Output layer with 6 classes
        
        # Activation function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Forward pass
        x = self.relu(self.fc1(x))  # Input to 1st hidden layer
        x = self.relu(self.fc2(x))  # 1st hidden to 2nd hidden layer
        x = self.relu(self.fc3(x))  # 2nd hidden to 3rd hidden layer
        # x = self.softmax(self.fc4(x))  # Final output (with softmax for classification)
        x = self.softmax(self.fc4(x))
        return x
