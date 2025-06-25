
import torch
import torch.nn as nn

class MockTransformerModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# Create and save mock model
if __name__ == "__main__":
    model = MockTransformerModel()
    torch.save(model.state_dict(), "transformer_model.pth")
    print("Mock transformer model saved")
