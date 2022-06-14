import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tqdm import tqdm

CUDA_LAUNCH_BLOCKING = 1
log = 'basic'

data = pd.read_csv('data.csv')
target = data['tm'].values
data = data.drop(['tm', 'index'], axis=1)

d_train, d_test, t_train, t_test = train_test_split(data, target, test_size=0.2, random_state=35)

d_train = normalize(d_train)
d_test = normalize(d_test)

BATCH_SIZE = 128
EPOCH = 50
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
train_tensor = torch.tensor(d_train)
train_target_tensor = torch.tensor(t_train)
train_dataset = torch.utils.data.TensorDataset(train_tensor, train_target_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_tensor = torch.tensor(d_test)
test_target_tensor = torch.tensor(t_test)
test_dataset = torch.utils.data.TensorDataset(test_tensor, test_target_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(12, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs):
        # preparing features attention tensor
        x = self.dropout(inputs)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train(model, train_loader, optimizer):
    model.train()

    for epoch in tqdm(range(EPOCH)):
        avg_loss = 0
        for batch, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            data = data.float()
            targets = torch.reshape(targets, (BATCH_SIZE, 1)).float()
            optimizer.zero_grad()
            outputs = model(data)
            loss_func = nn.L1Loss(reduction='sum')
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader.dataset)
        print("train loss : ", avg_loss)

    return model

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    output = np.array([])
    target = np.array([])

    with torch.no_grad():
        for data, targets in test_loader:
            target = np.concatenate((target, targets.detach().cpu().numpy())) if target.size else targets.detach().cpu().numpy()
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            data = data.float()
            targets = torch.reshape(targets, (BATCH_SIZE, 1)).float()
            outputs = model(data)
            output = np.concatenate((output, outputs.detach().cpu().numpy())) if output.size else outputs.detach().cpu().numpy()
            loss_func = nn.L1Loss(reduction='sum')
            loss = loss_func(outputs, targets)
            test_loss += loss.item() / len(test_loader.dataset)
    print('test_loss : ', test_loss)

    return test_loss, target, output

model = Model().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

model = train(model, train_loader, optimizer)
save_name = '{0}_model.pt'.format(log)
torch.save(model.state_dict(), save_name)
# model.load_state_dict(torch.load("main_model.pt"))
test_loss, target, output = evaluate(model, test_loader)

output = output.squeeze(1)
result = {'output':output, 'target':target}
result = pd.DataFrame(result)
result.to_csv('result_vanilla_dnn.csv', index=False)