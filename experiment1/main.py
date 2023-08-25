import torch
from torchvision import datasets, transforms

from layers import BaseDense
from model import BaseModel
import losses
import optimizers


def main():
    
    BS = 16
    LR = 0.0001
    EPOCHS = 10
    
    model_layers = [
        BaseDense(784,784,BS),
        BaseDense(784,784,BS),
        BaseDense(784,784,BS),
        BaseDense(784,10,BS)
    ]
    model = BaseModel(*model_layers)
    
    criterion = losses.softmax_CCE
    criterion_back = losses.softmax_CCE_back
    
    optimizers.SGD(LR, model)
    
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BS)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BS)
    for e in range(EPOCHS):
        print(f"Starting Epoch: {e+1}")
        for batch_idx, (data, target) in enumerate(train_loader):
            data = torch.flatten(data, start_dim=1).to(torch.float32)
            target = torch.nn.functional.one_hot(target, num_classes=10).to(torch.float32)
            logits = model(data)
            #print(torch.exp(logits)/torch.sum(torch.exp(logits)))
            loss = criterion(logits, target)
            if batch_idx*16 % 10000 == 0 and batch_idx != 0:
                print(f"Loss at Batch {batch_idx*16}: {torch.mean(loss): .4f}")
            grads = criterion_back(logits, target)
            #print(grads)
            model.backward(grads)
            model.update()
        
        mean_loss = 0
        correct = 0
        
        for data, target in test_loader:
            data = torch.flatten(data, start_dim=1)
            target = torch.nn.functional.one_hot(target, num_classes=10).to(float)
            logits = model(data)
            loss = criterion(logits, target)
            print(loss)
            print(mean_loss)
            mean_loss += torch.mean(loss)
            correct += torch.sum(logits.argmax(dim=1) == target.argmax(dim=1))

        mean_loss /= len(test_loader)
        print(correct, len(test_loader))
        print(f"\n\n Epoch: {e+1} | Loss: {mean_loss: .6f} | Accuracy = {(correct/(len(test_loader)*BS))*100}% \n\n")
        
    
if __name__ == "__main__":
    main()