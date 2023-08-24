import ivy
import torch
from torchvision import datasets, transforms

from layers import BaseDense
from model import BaseModel
import losses
import optimizers

def main():
    
    ivy.set_backend("torch")
    
    BS = 1
    LR = 0.0001
    EPOCHS = 10
    
    model_layers = [
        BaseDense(784,64,BS),
        BaseDense(64,32,BS),
        BaseDense(32,32,BS),
        BaseDense(32,10,BS)
    ]
    model = BaseModel(*model_layers)
    
    criterion = losses.CCE
    criterion_back = losses.CCE_back
    
    optimizers.SGD(LR, model)
    
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BS)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BS)
    
    for e in range(EPOCHS):
        print(f"Starting Epoch: {e+1}")
        for batch_idx, (data, target) in enumerate(train_loader):
            data = ivy.flatten(data, start_dim=1)
            target = ivy.one_hot(target,10)
            logits = model(data)
            loss = criterion(logits, target)
            if batch_idx % 1000 == 0 and batch_idx != 0:
                print(f"Loss at Batch {batch_idx}: {loss}")
            grads = criterion_back(logits, target)
            model.backward(grads)
            model.update()
        
        mean_loss = 0
        correct = 0
        
        for data, target in test_loader:
            data = ivy.flatten(data, start_dim=1)
            logits = model(data)
            loss = criterion(logits, target)
            mean_loss += loss
            correct += 1 if logits.argmax() == target.argmax() else 0
        
        mean_loss /= len(test_loader)
        print(f"Epoch: {e} | Loss: {mean_loss} | Accuracy = {correct/len(test_loader): .2f}%")
        
    
if __name__ == "__main__":
    main()
    