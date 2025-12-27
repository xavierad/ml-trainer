from typing import Tuple
from pydantic import BaseModel

import torch
from torch.utils.data import Dataset

class CustomDataDataset(BaseModel, Dataset):
    X: torch.Tensor
    Y: torch.Tensor

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]