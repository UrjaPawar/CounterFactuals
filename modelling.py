import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import itertools
from collections import defaultdict
import pickle
import numpy as np

class HeartDiseaseNN(nn.Module):
    def __init__(self):
        super(HeartDiseaseNN, self).__init__()
        self.fc1 = nn.Linear(25, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def run_model_heart(data):
    net = HeartDiseaseNN()
    optimizer = optim.AdamW(net.parameters())
    criterion = nn.CrossEntropyLoss()
    losses = []
    max_test = 0
    best_params = net.state_dict()
    x_train = torch.tensor((data.train_df.drop(data.target, axis=1)).values).float()
    x_test = torch.tensor((data.test_df.drop(data.target, axis=1)).values).float()
    y_train = torch.tensor((data.train_df[data.target]).values).long()
    y_test = torch.tensor((data.test_df[data.target]).values).long()
    for epoch in range(1, 200):
        optimizer.zero_grad()
        outputs = net(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        _, preds_y = torch.max(outputs, 1)
        train_acc = accuracy_score(y_train, preds_y)

        pred_test = net(x_test)
        _, preds_test_y = torch.max(pred_test, 1)
        test_acc = accuracy_score(y_test, preds_test_y)
        if test_acc > max_test:
            max_test = test_acc
            best_params = net.state_dict()
            torch.save(net, "models/heart.pkl")
    print(max_test)

# 44 99 0.75
# 96 120 0.75