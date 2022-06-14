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

BATCH_SIZE = 256
EPOCH = 50
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
THRESHOLD = 0.1

train_tensor = torch.tensor(d_train)
train_target_tensor = torch.tensor(t_train)
train_dataset = torch.utils.data.TensorDataset(train_tensor, train_target_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_tensor = torch.tensor(d_test)
test_target_tensor = torch.tensor(t_test)
test_dataset = torch.utils.data.TensorDataset(test_tensor, test_target_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


class MaskModel(nn.Module):
    def __init__(self):
        super(MaskModel, self).__init__()

        # softmax
        self.softmax_0 = nn.Softmax(dim=0)
        self.softmax_1 = nn.Softmax(dim=1)

        # for feature embedding
        self.embedding = nn.Linear(1, 10)

        # for target embedding
        self.embedding_t = nn.Linear(1, 10)

        # for tm output
        self.w1 = nn.Linear(20, 50)
        self.w2 = nn.Linear(50, 10)
        self.w3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

        # mask
        self.mask = nn.Linear(12, 12)
        torch.nn.init.xavier_normal(self.mask.weight)

    def forward(self, inputs, targets):
        # preparing features attention tensor
        emb_feature = self.embedding(inputs) # (128, 12, 10)

        # embedding target
        emb_target = self.embedding_t(targets)

        for i in range(emb_target.shape[0]):
            emb_feature_ith = emb_feature[i]
            emb_target_ith = emb_target[i].unsqueeze(1)
            mat = torch.mm(emb_feature_ith, emb_target_ith)
            mat = mat / torch.max(mat)
            if i == 0:
                attention = self.softmax_0(mat)
            else:
                attention = torch.cat((attention, self.softmax_0(mat)), dim=1)
        attention = torch.transpose(attention, 0, 1)

        context_vector = []
        for i in range(BATCH_SIZE):
            context_value = []
            for j in range(12):
                context_value = emb_feature[i][j] * attention[i][j] if j == 0 else context_value + emb_feature[i][j] * \
                                                                                attention[i][j]
            context_vector.append(context_value)
        context_vector = torch.stack(context_vector).to(DEVICE)

        x = torch.cat((emb_target, context_vector), dim=1)
        tm = self.relu(self.w1(x))
        tm = self.relu(self.w2(tm))
        tm = self.w3(tm)

        attention_mean = torch.mean(attention)
        mask = torch.where(attention > attention_mean, 1, 0)

        return tm, mask


class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()

        self.fc1 = nn.Linear(11, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs, mask):
        mask = torch.mean(mask, dim=0)
        keep_indices = torch.argsort(mask, descending=True)[:-1]
        keep_indices = torch.sort(keep_indices).values
        inputs = inputs[:, keep_indices]
        output = self.relu(self.fc1(inputs))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)

        return output


def mask_train(mask_model, train_loader, mask_optimizer):
    mask_model.train()
    mask = None

    for epoch in tqdm(range(EPOCH)):
        avg_loss = 0
        for batch, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            data = torch.reshape(data, (BATCH_SIZE, 12, 1)).float()
            targets = torch.reshape(targets, (BATCH_SIZE, 1)).float()
            mask_optimizer.zero_grad()
            outputs, mask = mask_model(data, targets)
            outputs = outputs.float()
            mask = mask.float()
            loss_func = nn.L1Loss(reduction='sum')
            loss = loss_func(outputs, targets)
            loss.backward()
            mask_optimizer.step()
            avg_loss += loss.item() / len(train_loader.dataset)
        print("train step 1 loss : ", avg_loss)

    return mask_model, mask

def main_train(main_model, train_loader, main_optimizer, mask):
    main_model.train()

    for epoch in tqdm(range(EPOCH)):
        avg_loss = 0
        for batch, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            data = data.float()
            targets = torch.reshape(targets, (BATCH_SIZE, 1)).float()
            mask_optimizer.zero_grad()
            outputs = main_model(data, mask).float()
            loss_func = nn.L1Loss(reduction='sum')
            loss = loss_func(outputs, targets)
            loss.backward()
            main_optimizer.step()
            avg_loss += loss.item() / len(train_loader.dataset)
        print("train step 2 loss : ", avg_loss)

    return main_model

def evaluate(main_model, test_loader, mask):
    main_model.eval()
    test_loss = 0
    output = np.array([])
    target = np.array([])

    with torch.no_grad():
        for data, targets in test_loader:
            target = np.concatenate((target, targets.detach().cpu().numpy())) if target.size else targets.detach().cpu().numpy()
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            data = data.float()
            targets = torch.reshape(targets, (BATCH_SIZE, 1)).float()
            outputs = main_model(data, mask)
            output = np.concatenate((output, outputs.detach().cpu().numpy())) if output.size else outputs.detach().cpu().numpy()
            loss_func = nn.L1Loss(reduction='sum')
            loss = loss_func(outputs, targets)
            test_loss += loss.item() / len(test_loader.dataset)
    print('test_loss : ', test_loss)

    return test_loss, target, output

mask_model = MaskModel().to(DEVICE)
main_model = MainModel().to(DEVICE)
mask_optimizer = torch.optim.SGD(mask_model.parameters(), lr=5e-4, weight_decay=0.01)
main_optimizer = torch.optim.SGD(main_model.parameters(), lr=1e-5, weight_decay=0.01)

mask_model, mask = mask_train(mask_model, train_loader, mask_optimizer)
save_name_1 = 'mask_model.pt'
torch.save(mask_model.state_dict(), save_name_1)

main_model = main_train(main_model, train_loader, main_optimizer, mask)
save_name_2 = 'main_model.pt'
torch.save(main_model.state_dict(), save_name_2)
# main_model.load_state_dict(torch.load("basic_main_model.pt"))
# main_model.load_state_dict(torch.load("basic_main_model.pt"))
test_loss, target, output = evaluate(main_model, test_loader, mask)

output = output.squeeze(1)
result = {'output':output, 'target':target}
result = pd.DataFrame(result)
result.to_csv('result.csv', index=False)