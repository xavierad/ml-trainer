import torch

class SimpleNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(SimpleNN, self).__init__()
        self.fc1: torch.nn.Linear = torch.nn.Linear(input_size, hidden_size)
        self.relu: torch.nn.ReLU = torch.nn.ReLU()
        self.fc2: torch.nn.Linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out