import torch
from torchvision import datasets, transforms
import time

from layers.base_layers import Conv2D, Dense, MaxPool2D, Flatten
from models.test import BaseModel
import losses
import optimizers

BS = 32
LR = 0.0001
EPOCHS = 10
device = "cpu"

def trainloop1(train_dataloader, test_dataloader):
    torch.manual_seed(0)
    model_layers = [
        Conv2D(1, 6, BS, (5,5)),
        Conv2D(6, 16, BS, (5,5)),
        MaxPool2D(BS),
        Conv2D(16,16,BS, (3,3)),
        MaxPool2D(BS),
        Flatten(),
        Dense(16*4*4, 120, BS),
        Dense(120,84,BS),
        Dense(84,10,BS)
    ]

    model = BaseModel(*model_layers)
    
    model.to(device)
    
    criterion = losses.softmax_CCE
    criterion_back = losses.softmax_CCE_back
    
    optimizers.SGD(LR, model)
    
    for e in range(EPOCHS):
        start = time.time_ns()
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data = data.to(torch.float32).to(device)
            print(data.shape)
            target = torch.nn.functional.one_hot(target, num_classes=10).to(torch.float32).to(device)
            logits = model(data)
            #print(torch.exp(logits)/torch.sum(torch.exp(logits)))
            loss = criterion(logits, target)
            grads = criterion_back(logits, target)
            #print(grads)
            model.backward(grads)
            model.update()
        end = time.time_ns()
        
        mean_loss = 0
        correct = 0
        
        for data, target in test_dataloader:
            data = data.to(torch.float32).to(device)
            target = torch.nn.functional.one_hot(target, num_classes=10).to(torch.float32).to(device)
            logits = model(data)
            loss = criterion(logits, target)
            #print(loss)
            #print(mean_loss)
            mean_loss += torch.mean(loss)
            correct += torch.sum(logits.argmax(dim=1) == target.argmax(dim=1))

        mean_loss /= len(test_dataloader)
        #print(correct, len(test_dataloader)*BS)
        print(f"\n Epoch: {e+1} | Loss: {mean_loss: .6f} | Accuracy: {(correct/(len(test_dataloader)*BS))*100}% | Time: {(end-start)*1e-9: .2f}\n")




def main():
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BS, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BS, drop_last=True)
    
    trainloop1(train_loader, test_loader)
    
if __name__ == "__main__":
    main()