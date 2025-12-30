import torchvision
import torch
import torch.nn as nn
from torchvision import transforms

from src.modelling.classifier import Classifier
from src.modelling.architecture.base import SimpleNN

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

batch_size = 4

trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=True, 
    num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)
testloader = torch.utils.data.DataLoader(

    testset, 
    batch_size=batch_size,
    shuffle=False, 
    num_workers=2
)

dummy_model: SimpleNN = SimpleNN(input_size=(3, 32, 32), hidden_size=128, output_size=10)
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
    epochs=5,
    criterion=criterion, 
    optimizer=optimizer,
    device=torch.device("cpu"),
    early_stopping=2
)