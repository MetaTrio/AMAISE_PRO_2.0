import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()

        # Input layer matches 512 input features
        self.fc1 = nn.Linear(512, 1024)  # Hidden layer with 1024 neurons
        self.fc2 = nn.Linear(1024, 512)  # Hidden layer with 512 neurons
        self.fc3 = nn.Linear(512, 256)   # Hidden layer with 256 neurons
        self.fc4 = nn.Linear(256, 128)   # Hidden layer with 128 neurons
        self.fc5 = nn.Linear(128, 6)     # Output layer with 6 classes

        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(0.3)

        # Activation function
        self.relu = nn.ReLU()

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

        # Softmax for output classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass through the layers with batch normalization, dropout, and ReLU
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)

        x = self.softmax(self.fc5(x))
        return x
