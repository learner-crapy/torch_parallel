import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)
# Initializes the distributed process group
# Rank:
# Each process in the distributed setup is assigned a unique identifier known as the rank.
# Ranks are numbered from 0 to world_size - 1. For instance, in a setup with 4 GPUs (world size = 4), the ranks will be 0, 1, 2, and 3.
# The rank is used to distinguish between the different processes and to manage data distribution and aggregation.
def setup(rank, world_size):
    #  the environment variables MASTER_ADDR and MASTER_PORT
    #  are used to set up the communication between different processes

    os.environ['MASTER_ADDR'] = 'localhost'
    # MASTER_ADDR specifies the address of the master node (also known as the rank 0 process).
    # This address is used by all processes to know where the master node is located.
    # Typically, in a multi-node setup, this would be the IP address or hostname of the machine designated as the master node.
    os.environ['MASTER_PORT'] = '12355'
    # MASTER_PORT specifies the port on the master node that processes use to communicate.
    # This port should be free and open for communication on the master node.
    # It ensures that all processes connect to the same communication endpoint on the master node.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
#  Destroys the process group after training.
def cleanup():
    dist.destroy_process_group()
# total number of processes involved in the training, Each process typically corresponds to one GPU.
# Therefore, the world size is usually equal to the number of GPUs being used for the distributed training.
def train(rank, world_size):
    setup(rank, world_size)

    # Create model and move it to GPU with id rank
    model = SimpleModel().cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Create a loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Load training data
    train_dataset = datasets.MNIST('.', train=True, download=True,
                                   transform=transforms.ToTensor())
    #  ensures that each process gets a unique subset of the dataset. This helps in parallelizing the data loading
    #  and ensures that each GPU works on different parts of the data, preventing overlap.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               sampler=train_sampler)

    # Training loop
    for epoch in range(2):  # 2 epochs for demonstration
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda(rank)
            target = target.cuda(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
