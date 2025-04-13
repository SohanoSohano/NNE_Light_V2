# model_definition.py
import torch
import torch.nn as nn
import torch.nn.functional as F # Often used for activation functions

# --- Simple CNN Example ---
# Replace this with YOUR actual model definition class.
# The architecture defined here MUST EXACTLY MATCH the architecture
# whose weights are saved in your .pth/.pt file.

class SimpleCNN(nn.Module):
    """
    A very basic CNN example for demonstration.
    Assumes input images are grayscale and roughly 28x28 (like MNIST).
    """
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        print(f"--- Initializing SimpleCNN (input_channels={input_channels}, num_classes={num_classes}) ---")

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5, padding=2) # Output: [batch, 6, 28, 28]
        # -> padding = 2 keeps size 28x28 with kernel 5
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: [batch, 6, 14, 14]

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # Output: [batch, 16, 10, 10]
        # -> kernel 5 on 14x14 input -> 10x10 output
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: [batch, 16, 5, 5]

        # Fully Connected Layers
        # Calculate the flattened size: 16 filters * 5x5 feature map size
        flattened_size = 16 * 5 * 5
        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        print("--- SimpleCNN Initialization Complete ---")


    def forward(self, x):
        # Apply layers with activation functions
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1) # Flatten all dimensions except batch

        # Apply fully connected layers with activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No activation on the final output layer (logits)
        return x

# --- Simple RNN Example (Alternative - Uncomment and adapt if needed) ---
# class SimpleRNN(nn.Module):
#     def __init__(self, input_size=10, hidden_size=32, num_layers=1, num_classes=5):
#         super().__init__()
#         print(f"--- Initializing SimpleRNN ---")
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
#         print(f"--- SimpleRNN Initialization Complete ---")
#
#     def forward(self, x):
#         # Set initial hidden and cell states
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#
#         # Forward propagate LSTM
#         out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
#
#         # Decode the hidden state of the last time step
#         out = self.fc(out[:, -1, :])
#         return out
