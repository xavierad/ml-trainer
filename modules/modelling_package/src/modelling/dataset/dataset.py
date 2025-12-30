from typing import Callable, Tuple
from pydantic import BaseModel

import torch
from torch.utils.data import Dataset

class CustomDataDataset(BaseModel, Dataset):
    features: torch.Tensor
    labels: torch.Tensor
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None

    def _transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.transform:
            return self.transform(data)
        return data

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._transform(self.features[idx]), 
            self.labels[idx]
        )