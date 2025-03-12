import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate random data
X = torch.randn(1000, 10)  # Input with 1000 samples
y = torch.randint(0, 3, (100,))  

# Define a simple MLP classifier
class MLP(nn.Module):
    def __int__(self):  # Typo in __init__
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(10, 32)
        self.layer2 = nn.Linear(10, 3)  
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x) 
        return x

# Initialize model, loss function, and optimizer
model = MLP()
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    model.train()
    
    # Move data to device
    X = X.to(device)
    y = y.to(device)
    
    # Forward pass
    outputs = model(X)
    loss = loss_fn(outputs, y)  
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Test model
test_input = torch.randn(1, 10).to(device)
model.eval()
with torch.no_grad():
    test_output = model(test_input)
    
print("Predicted class:", torch.argmax(test_output, dim=1).item())
