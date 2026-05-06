import torch
torch.backends.cudnn.benchmark = True
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

from torchsummary import summary
from torchvision.models import resnet152

from data.dummy import get_train_dataloader
from utils import benchmark

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 5
gbs = 32
learning_rate = 0.0001

train_loader = get_train_dataloader(256, (3,224,224), gbs, vocab_size=1000, dtype="image")

# Initialize the model
model = resnet152()
model = model.train()
model.fc = nn.Linear(model.fc.in_features, 1000)  # CIFAR-10 has 10 classes
model = model.to(device)
summary(model, (3, 224, 224), depth=5, col_names=("output_size", "num_params", "kernel_size"))


# Loss and optimizer
@torch.jit.script
def criterion(logits, y):
    y_hat = torch.softmax(logits, dim=-1)
    return torch.sum(-y*torch.log(y_hat))

optimizer = optim.SGD(model.parameters(), lr=learning_rate)
param_mem = torch.cuda.memory_allocated()
total_params = count_parameters(model)

print(f"No. Params: {total_params:,}")
print(f"GPU Mem: {param_mem/1e9:.4f} GB")

# Train the model
for epoch in range(num_epochs):
    with benchmark("Epoch %d" % epoch, 256):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                
param_mem = (torch.cuda.max_memory_reserved())
print(f"GPU Mem: {param_mem/1e9:.4f} GB")

print("Training complete!")
