import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tqdm import tqdm
import sys
print(sys.path)
exit()

CUDA_LAUNCH_BLOCKING = 1

data = pd.read_csv('data.csv')
target = data['tm'].values
data = data.drop(['tm', 'index'], axis=1)

d_train, d_test, t_train, t_test = train_test_split(data, target, test_size=0.2, random_state=35)

d_train = normalize(d_train)
d_test = normalize(d_test)

BATCH_SIZE = 128
EPOCH = 10
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(DEVICE)
train_tensor = torch.tensor(d_train)
train_target_tensor = torch.tensor(t_train)
train_dataset = torch.utils.data.TensorDataset(train_tensor, train_target_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_tensor = torch.tensor(d_test)
test_target_tensor = torch.tensor(t_test)
test_dataset = torch.utils.data.TensorDataset(test_tensor, test_target_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


class FeaturesAttention(nn.Module):
    def __init__(self):
        super(FeaturesAttention, self).__init__()
        # for self-attention
        self.embedding = nn.Linear(1, 10)
        self.w1 = nn.Linear(10, 10)
        self.w2 = nn.Linear(10, 10)
        self.w3 = nn.Linear(10, 10)
        self.softmax = nn.Softmax()

        # for encoder output
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(10, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 10)
        self.batchnorm = nn.BatchNorm1d(12)

    def forward(self, inputs):
        # preparing features attention tensor
        embedded = self.embedding(inputs)
        q = self.w1(embedded)
        k = self.w2(embedded)
        v = self.w3(embedded)
        att_features = torch.matmul(self.softmax(torch.matmul(q, torch.transpose(k, 1, 2))), v)

        # encoder output
        encoder_output = self.dropout(att_features)
        encoder_output = self.batchnorm(embedded + encoder_output)
        encoder_output = self.relu(self.fc1(encoder_output))
        encoder_output = self.relu(self.fc2(encoder_output))
        encoder_output = self.fc3(encoder_output)

        return encoder_output, embedded


class EmbeddingTarget(nn.Module):
    def __init__(self):
        super(EmbeddingTarget, self).__init__()

        self.embedding = nn.Linear(1, 10)
        self.w = nn.Linear(10, 10)

    def forward(self, inputs):
        emb_target = self.embedding(inputs)
        emb_target = self.w(emb_target)

        return emb_target


class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()

        self.w1 = nn.Linear(20, 10)
        self.w2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs, attention, embedded_tm):
        context_vector = []
        for i in range(BATCH_SIZE):
            context_value = []
            for j in range(12):
                context_value = inputs[i][j] * attention[i][j] if j == 0 else context_value + inputs[i][j] * attention[i][j]
            context_vector.append(context_value)
        context_vector = torch.stack(context_vector).to(DEVICE)
        x = torch.cat((embedded_tm, context_vector), dim=1)
        tm = self.relu(self.w1(x))
        tm = self.w2(tm)

        return tm

softmax = nn.Softmax(dim=0)
def train(attention_model, emb_tm_model, main_model, train_loader, optimizer):
    attention_model.train()
    emb_tm_model.train()
    main_model.train()

    for epoch in tqdm(range(EPOCH)):
        avg_loss = 0
        for batch, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            data = torch.reshape(data, (BATCH_SIZE, 12, 1)).float()
            targets = torch.reshape(targets, (BATCH_SIZE, 1)).float()
            optimizer.zero_grad()
            features_attention, data = attention_model(data)
            embedded_tm = emb_tm_model(targets)
            for i in range(embedded_tm.shape[0]):
                features_attention_ith = features_attention[i]
                embedded_tm_ith = embedded_tm[i].unsqueeze(1)
                if i == 0:
                    attention = softmax(torch.mm(features_attention_ith, embedded_tm_ith))
                else:
                    attention = torch.cat((attention, softmax(torch.mm(features_attention_ith, embedded_tm_ith))), dim=1)
            attention = torch.transpose(attention, 0, 1)
            outputs = main_model(data, attention, embedded_tm)
            loss_func = nn.L1Loss()
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader.dataset)
        print("train loss : ", avg_loss)

    return attention_model, emb_tm_model, main_model

def evaluate(attention_model, emb_tm_model, main_model, test_loader):
    attention_model.eval()
    emb_tm_model.eval()
    main_model.eval()
    test_loss = 0
    output = np.array([])
    target = np.array([])

    with torch.no_grad():
        for data, targets in test_loader:
            target = np.concatenate((target, targets.detach().cpu().numpy())) if target.size else targets.detach().cpu().numpy()
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            data = torch.reshape(data, (BATCH_SIZE, 12, 1)).float()
            targets = torch.reshape(targets, (BATCH_SIZE, 1)).float()
            features_attention, data = attention_model(data)
            embedded_tm = emb_tm_model(targets)
            for i in range(embedded_tm.shape[0]):
                features_attention_ith = features_attention[i]
                embedded_tm_ith = embedded_tm[i].unsqueeze(1)
                if i == 0:
                    attention = softmax(torch.mm(features_attention_ith, embedded_tm_ith))
                else:
                    attention = torch.cat((attention, softmax(torch.mm(features_attention_ith, embedded_tm_ith))), dim=1)
            attention = torch.transpose(attention, 0, 1)
            outputs = main_model(data, attention, embedded_tm)
            output = np.concatenate((output, outputs.detach().cpu().numpy())) if output.size else outputs.detach().cpu().numpy()
            loss_func = nn.L1Loss()
            loss = loss_func(outputs, targets)
            test_loss += loss.item() / len(test_loader.dataset)
    print('test_loss : ', test_loss)

    return test_loss, target, output

attention_model = FeaturesAttention().to(DEVICE)
emb_tm_model = EmbeddingTarget().to(DEVICE)
main_model = MainModel().to(DEVICE)
optimizer = torch.optim.SGD(main_model.parameters(), lr=0.001)

attention_model, emb_tm_model, main_model = train(attention_model, emb_tm_model, main_model, train_loader, optimizer)
save_name_1 = 'attention_model.pt'
save_name_2 = 'emb_tm_model.pt'
save_name_3 = 'main_model.pt'
torch.save(attention_model.state_dict(), save_name_1)
torch.save(emb_tm_model.state_dict(), save_name_2)
torch.save(main_model.state_dict(), save_name_3)
# attention_model.load_state_dict(torch.load("attention_model.pt"))
# emb_tm_model.load_state_dict(torch.load("emb_tm_model.pt"))
# main_model.load_state_dict(torch.load("main_model.pt"))
test_loss, target, output = evaluate(attention_model, emb_tm_model, main_model, test_loader)

output = output.squeeze(1)
result = {'output':output, 'target':target}
result = pd.DataFrame(result)
result.to_csv('result.csv', index=False)