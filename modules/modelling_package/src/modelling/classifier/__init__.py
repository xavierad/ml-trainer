import os
from copy import deepcopy
from time import time
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm


class Classifier:
    def __init__(
        self,
        model_name: str,
        version: str,
        accuracy: float,
        model: torch.nn.Module,
        labels: List[str]
    ) -> None:
        
        self.model_name = model_name
        self.version = version
        self.accuracy = accuracy
        self.model = model
        self.labels = labels    
        
    def _run_epoch_phase(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        phase: str,
        device: torch.device
    ) -> Tuple[float, float]:
        """Run a single phase (train or validation) of an epoch.
        
        Returns:
            tuple: (epoch_loss, epoch_accuracy)
        """
        is_training: bool = phase == 'train'
        model.train(is_training)
        
        running_loss: float = 0.0
        running_corrects: torch.Tensor = torch.tensor(0, dtype=torch.int64)
        
        loader: tqdm = tqdm(data_loader, leave=False, desc=phase.capitalize())
        
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(is_training):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if is_training:
                    loss.backward()
                    optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            loader.set_postfix(loss=loss.item())
        
        dataset_size: int = len(data_loader) if data_loader.batch_size is None else len(data_loader) * data_loader.batch_size
        epoch_loss: float = running_loss / dataset_size
        epoch_acc: float = running_corrects.double().item() / dataset_size
        
        return epoch_loss, epoch_acc


    def _check_early_stopping(
        self,
        val_loss: float,
        min_val_loss: float,
        epochs_no_improve: int,
        patience: int
    ) -> Tuple[bool, float, int]:
        """Check if early stopping criteria is met.
        
        Returns:
            tuple: (should_stop, updated_min_loss, updated_epochs_no_improve)
        """
        if val_loss < min_val_loss:
            return False, val_loss, 0
        
        epochs_no_improve += 1
        should_stop = epochs_no_improve >= patience
        return should_stop, min_val_loss, epochs_no_improve


    def _update_best_model(
        self,
        model: torch.nn.Module,
        current_acc: float,
        best_acc: float,
        best_weights: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Update best model weights if current accuracy is better.
        
        Returns:
            tuple: (updated_best_acc, updated_best_weights)
        """
        if current_acc > best_acc:
            return current_acc, deepcopy(model.state_dict())
        return best_acc, best_weights


    def _finalize_training(
        self,
        model: torch.nn.Module,
        best_weights: Dict[str, Any],
        best_acc: float,
        start_time: float,
        save_path: str = "/models"
    ) -> None:
        """Finalize training by loading best weights and saving model."""
        time_elapsed = time() - start_time
        
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        
        model.load_state_dict(best_weights)
        self.accuracy = best_acc
        torch.save(model.state_dict(), os.path.join(save_path, f"{self.model_name}_{self.version}.pth"))
        print('Finished Training')


    def fit(
        self,
        model: torch.nn.Module,
        training_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        early_stopping: int | None = None,
        device: torch.device = torch.device('cpu')
    ) -> None:
        """Train the model with optional early stopping."""
        
        start_time = time()
        data_loaders = {'train': training_loader, 'val': validation_loader}
        
        # Initialize tracking variables
        best_model_wts = deepcopy(model.state_dict())
        best_acc = 0.0
        epochs_no_improve = 0
        min_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)
            
            for phase in ['train', 'val']:
                # Run single phase
                epoch_loss, epoch_acc = self._run_epoch_phase(
                    model, data_loaders[phase], criterion, optimizer, phase, device
                )
                
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Validation phase: update best model and check early stopping
                if phase == 'val':
                    best_acc, best_model_wts = self._update_best_model(
                        model, epoch_acc, best_acc, best_model_wts
                    )
                    
                    if early_stopping is not None:
                        should_stop, min_val_loss, epochs_no_improve = self._check_early_stopping(
                            epoch_loss, min_val_loss, epochs_no_improve, early_stopping
                        )
                        
                        if should_stop:
                            print(f'\nEarly stopping triggered after {epoch + 1} epochs!')
                            self._finalize_training(model, best_model_wts, best_acc, start_time)
                            return
        
        # Training completed without early stopping
        self._finalize_training(model, best_model_wts, best_acc, start_time)

    def load(self, folder: str) -> None:
        """Loads the model to memory. To be implemented by the data scientist.
        """
        self.model.load_state_dict(
            torch.load(
                os.path.join(folder, f"{self.model_name}_{self.version}.pth"), 
                weights_only=True
            )
        )

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            return outputs

