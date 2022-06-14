import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
import os
import hdbscan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from tqdm import tqdm

random.seed(0)
os.environ["PYTHONHASHSEED"] = str(0)

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


CUDA_LAUNCH_BLOCKING = 1
log = 'hdbscan'
# export OMP_NUM_THREADS=1

data = pd.read_csv('total_data.csv')
target = data['tm'].values
data = data.drop(['tm', 'r2extent', 'total_energy', 'hlg', 'smd'], axis=1)

d_train, d_test, t_train, t_test = train_test_split(data, target, test_size=0.2, random_state=35)

d_train = normalize(d_train)
d_test = normalize(d_test)

BATCH_SIZE = 64
EPOCH = 55
FEATURE_NUM = 10
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')


kmeans = KMeans(n_clusters=11, random_state=35).fit(d_train)
train_labels = np.expand_dims(kmeans.labels_, axis=1)
d_train = np.concatenate((d_train, train_labels), axis=1)
test_labels = np.expand_dims(kmeans.predict(d_test), axis=1)
d_test = np.concatenate((d_test, test_labels), axis=1)

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

        # for feature embedding
        self.embedding = nn.Linear(1, 300)
        self.cluster_embedding = nn.Embedding(11, 300)
        self.w1 = nn.Linear(300, 300)
        self.w2 = nn.Linear(300, 300)
        self.w3 = nn.Linear(300, 300)
        self.softmax = nn.Softmax()

        # for encoder output
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(300, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 300)
        self.batchnorm = nn.BatchNorm1d(FEATURE_NUM)
        self.layernorm1 = nn.LayerNorm([BATCH_SIZE, FEATURE_NUM, 300])
        self.layernorm2 = nn.LayerNorm([BATCH_SIZE, FEATURE_NUM, 1])

        # for tm
        self.fc4 = nn.Linear(300, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 1)
        self.fc7 = nn.Linear(FEATURE_NUM, 1)
        self.v = nn.Linear(300, 1)

    def forward(self, inputs):
        # preparing features attention tensor
        need_emb = inputs[:, :FEATURE_NUM-2]
        emb_mol_features = inputs[:, FEATURE_NUM-2:-1]
        emb_mol_features = torch.transpose(emb_mol_features, 1, 2)
        emb_cluster = self.cluster_embedding(inputs[:, -1].int())
        embedded_inputs = torch.cat((self.embedding(need_emb), emb_cluster, emb_mol_features), dim=1)
        q = self.w1(embedded_inputs)
        k = self.w2(embedded_inputs)
        v = self.w3(embedded_inputs)
        attention = torch.matmul(self.softmax(torch.matmul(q, torch.transpose(k, 1, 2))), v)

        # encoder output
        attention = self.layernorm1(embedded_inputs + attention)
        encoder_output = self.relu(self.fc1(attention))
        encoder_output = self.relu(self.fc2(encoder_output))
        encoder_output = self.fc3(encoder_output)
        encoder_output = self.layernorm1(attention + encoder_output)

        tm = self.dropout(encoder_output) # (128, 10, 300)
        tm = self.relu(self.fc4(tm))
        tm = self.relu(self.fc5(tm))
        tm = self.relu(self.fc6(tm))
        tm = self.layernorm2(tm + self.v(encoder_output)).squeeze(2)
        tm = self.fc7(tm)

        return tm


def train(model, train_loader, optimizer):
    model.train()

    for epoch in tqdm(range(EPOCH)):
        avg_loss = 0
        for batch, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            data = torch.unsqueeze(data, dim=2).float()
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
            data = torch.unsqueeze(data, dim=2).float()
            targets = torch.reshape(targets, (BATCH_SIZE, 1)).float()
            outputs = model(data)
            output = np.concatenate((output, outputs.detach().cpu().numpy())) if output.size else outputs.detach().cpu().numpy()
            loss_func = nn.L1Loss(reduction='sum')
            loss = loss_func(outputs, targets)
            test_loss += loss.item() / len(test_loader.dataset)
    print('test_loss : ', test_loss)

    return test_loss, target, output

model = Model().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.001)

model = train(model, train_loader, optimizer)
save_name = '{0}_model.pt'.format(log)
torch.save(model.state_dict(), save_name)
# model.load_state_dict(torch.load("main_model.pt"))
test_loss, target, output = evaluate(model, test_loader)

output = output.squeeze(1)
result = {'output':output, 'target':target}
result = pd.DataFrame(result)
result.to_csv('results/result_with_embedded_molvec.csv', index=False)