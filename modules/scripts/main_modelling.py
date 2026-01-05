import os

import torchvision
import torch
import torch.nn as nn
from torchvision import transforms

from modelling.classifier import Classifier
from modelling.architecture.base import SimpleNN


ON_GPU: bool = os.getenv('ON_GPU', 'false').lower() == 'true'
EPOCHS: int = int(os.getenv('EPOCHS', '2'))
BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '4'))

seed = 42
torch.manual_seed(seed)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), 
            (0.5, 0.5, 0.5)
        )
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root='/data', 
    train=True,
    download=True, 
    transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=BATCH_SIZE,
    shuffle=True, 
    num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='/data', 
    train=False,
    download=True, 
    transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=2
)

device: torch.device = torch.device("cpu")
if torch.cuda.is_available() and ON_GPU:
    device = torch.device("cuda")

dummy_model: SimpleNN = SimpleNN(input_size=(3, 32, 32), hidden_size=128, output_size=10).to()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.001, momentum=0.9)

dummy_classifier: Classifier = Classifier(
    model_name="SimpleNN",
    version="1.0",
    accuracy=0.0,
    model=dummy_model,
    labels=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
)
dummy_classifier.fit(
    model=dummy_model, 
    training_loader=trainloader, 
    validation_loader=testloader, 
    epochs=EPOCHS,
    criterion=criterion, 
    optimizer=optimizer,
    device=torch.device("cpu"),
    early_stopping=2
)