import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define a simple model split across two GPUs
class ModelParallelNN(nn.Module):
    def __init__(self):
        super(ModelParallelNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512).to('cuda:0')
        self.fc2 = nn.Linear(512, 512).to('cuda:0')
        self.fc3 = nn.Linear(512, 512).to('cuda:1')
        self.fc4 = nn.Linear(512, 10).to('cuda:1')

    def forward(self, x):
        x = x.view(-1, 28 * 28).to('cuda:0')
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = x.to('cuda:1')
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the model
model = ModelParallelNN()

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Load training data
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Training loop
# During training, the data is initially moved to cuda:0 for the first part of the network and then to cuda:1 for the second part.
# The loss is computed on cuda:1 since the final output is on cuda:1.
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        # The loss is computed using the target moved to cuda:1.
        # loss.backward() computes gradients, and PyTorch handles the communication between GPUs to propagate gradients correctly.
        loss = criterion(output, target.to('cuda:1'))
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')

# Train the model
for epoch in range(1, 3):  # 2 epochs for demonstration
    train(model, train_loader, criterion, optimizer, epoch)
