from math import prod
from typing import Tuple

import torch


class SimpleNN(torch.nn.Module):
    def __init__(self, input_size: int | Tuple, hidden_size: int, output_size: int) -> None:
        super(SimpleNN, self).__init__()
                
        flat_input_size: int = prod((input_size,) if isinstance(input_size, int) else input_size)
        self.flatten: torch.nn.Flatten = torch.nn.Flatten()
        self.fc1: torch.nn.Linear = torch.nn.Linear(flat_input_size, hidden_size)
        self.relu: torch.nn.ReLU = torch.nn.ReLU()
        self.fc2: torch.nn.Linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out