import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_utils import TrainHandler
from torch.utils.data.dataset import random_split

dataset = TensorDataset(torch.randn(1000, 10), torch.empty(1000).random_(2))
train_length = int(len(dataset) * 0.95)
train, test = random_split(dataset, [train_length, len(dataset) - train_length])

train_loader = DataLoader(train, batch_size=64)
valid_loader = DataLoader(test, batch_size=64)
model = None
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
model_path = 'model.p'
handler = TrainHandler(train_loader,
                       valid_loader,
                       model,
                       criterion,
                       optimizer,
                       model_path,
                       batch_size=32,
                       epochs=5,
                       scheduler=None,
                       gpu_num=0)
handler.train()
