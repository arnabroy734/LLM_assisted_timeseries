from data.dataset import get_dataset
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from architecture.lstm import LSTMModel
from architecture.mixedmodel import BERTConcat, BERTAttn, BERTAttn1
from torch import nn
from pathlib import Path
import json
import pickle

def train_time_model(
    model,
    optimiser,
    loss_fn,
    epochs,
    batch,
    lag,
    name,
    type,
    modelpath
):
    trainds, testds, scaler = get_dataset(name, type, lag)
    trainloader = DataLoader(trainds, batch_size=batch)
    testloader = DataLoader(testds, batch_size=batch)

    device = torch.device('cuda' if torch.cuda.is_available() else  'cpu')
    model.to(device)
    tr_losses = list()
    val_losses = list()

    for i in range(epochs):
        train_loss = list()
        val_loss = list()
        for j, (X, y) in enumerate(trainloader):
            n = X.shape[0]
            tr_size = int(n*0.9)
            X_train, y_train = X[:tr_size, :], y[:tr_size]
            X_val, y_val = X[tr_size:, :], y[tr_size:]
            
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_train)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss.append(loss.item())
            # if (j+1)%3 == 0:
            # print(f"Epoch - {i+1}, step - {j+1}, train_loss = {loss.item()}")
            with torch.no_grad():
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred_val = model(X_val)
                val_loss.append(loss_fn(y_pred_val, y_val).item())

        tr_losses.append(np.mean(train_loss))
        val_losses.append(np.mean(val_loss))
        print(f'Epoch {i+1} | train loss - {tr_losses[-1]} | val loss - {val_losses[-1]}')

    plt.plot(tr_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title(f"Window Size = {lag}")
    plt.savefig(modelpath/'training.png')
    plt.show()

    torch.save(model, modelpath/'model.pt')

    pred = list()
    actual = list()
    for X, y in testloader:
        X = X.to(device)
        y_pred = model(X)
        y = y.detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        y = scaler.inverse_transform(y.reshape((-1,1))).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape((-1,1))).flatten()
        pred.extend(list(y_pred))
        actual.extend(list(y))

    plt.figure(figsize=(20,5))
    plt.plot(pred, 'r--', label='predicted' )
    plt.plot(actual, 'g--', label='actual')
    plt.legend()
    test_error = mean_squared_error(pred, actual)
    plt.grid()
    plt.title(f"Test error - {test_error:0.3f}, forecasting window = 1")
    plt.savefig(modelpath/'prediction.jpg')
    plt.show()

def train_mixed_model(
    lr,
    loss_fn,
    epochs,
    batch,
    lag,
    name, 
    type,
    modelpath,
    modelClass
):
    trainds, _, _ = get_dataset(name, type, lag)
    trainloader = DataLoader(trainds, batch_size=batch)
    try:
        model = torch.load(modelpath/'model_latest.pt')
        print('Model loaded succesfully')
    except Exception as e:
        # model = BERTConcat(in_size=1, hid_size=512)
        model = modelClass(in_size=1, hid_size=512)

    try:
        optimiser = torch.load(modelpath/'optimiser.pt')
        print('Optimiser loaded succesfully')

    except Exception as e:
        optimiser = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=10**(-3))
    optimiser.lr = lr

    try:
        with open(modelpath/'loss.json', 'r') as f:
            losses = json.load(f)
            f.close()
    except Exception as e:
        losses = {'train': [], 'validation':[]}

    tr_losses = list()
    val_losses = list()
    device = torch.device('cuda' if torch.cuda.is_available() else  'cpu')
    # device = 'cpu'
    model.to(device)

    for i in range(epochs):
        train_loss = list()
        val_loss = list()
        for j, (X, tokens, masks, y) in enumerate(trainloader):
            n = X.shape[0]
            tr_size = int(n*0.9)
            y = y.reshape((-1,1))
            X, tokens, masks, y = X.to(device), tokens.to(device), masks.to(device), y.to(device)
            X_train, tokens_train = X[:tr_size, :],tokens[:tr_size, :] 
            mask_train, y_train = masks[:tr_size, :], y[:tr_size, :]
            X_val, tokens_val,  = X[tr_size:, :], tokens[tr_size:, :] 
            mask_val, y_val = masks[tr_size:, :], y[tr_size:, :]

            y_pred = model(X_train, tokens_train, mask_train)
            loss = loss_fn(y_pred, y_train)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss.append(loss.item())
            if (j+1)%10 == 0:
                print(f"Epoch - {i+1}, step - {j+1}, train_loss = {loss.item()}")
            with torch.no_grad():
                y_pred_val = model(X_val, tokens_val, mask_val)
                val_loss.append(loss_fn(y_pred_val, y_val).item())

        tr_losses.append(np.mean(train_loss))
        val_losses.append(np.mean(val_loss))
        print(f'Epoch {i+1} | train loss - {tr_losses[-1]} | val loss - {val_losses[-1]}')
    
    losses['train'].extend(tr_losses)
    losses['validation'].extend(val_losses)
    ep = len(losses['train'])
    torch.save(model, modelpath/f'model_{ep}.pt')
    torch.save(model, modelpath/f'model_latest.pt')
    torch.save(optimiser, modelpath/f'optimiser.pt')
    with open(modelpath/'loss.json', 'w') as f:
        json.dump(losses, f)
        f.close()

def plot_training_curve(modelpath, lag):
    with open(modelpath/'loss.json', 'r') as f:
        losses = json.load(f)
        f.close()

    plt.plot(losses['train'], label="Train Loss")
    plt.plot(losses['validation'], label="Validation Loss")
    plt.legend()
    plt.title(f"Window Size = {lag}")
    plt.savefig(modelpath/'training.png')
    plt.show()

def plot_result(modelpath):
    with open(modelpath/'result.json', 'r') as f:
        result = json.load(f)
        f.close()
    predictions = result['prediction']
    actuals = result['actual']
    test_error = result['mse']
    plt.figure(figsize=(20,5))
    plt.plot(predictions, 'r--', label='predicted' )
    plt.plot(actuals, 'g--', label='actual')
    plt.legend()
    plt.grid()
    plt.title(f"Test error - {test_error:0.3f}, forecasting window = 1")
    plt.savefig(modelpath/'prediction.png')
    plt.show()

def test_mixed_model(
        name,
        type,
        lag,
        modelpath
):
    _, testds, scaler = get_dataset(name, type, lag)
    testloader = DataLoader(testds, batch_size=32)
    model = torch.load(modelpath/'model_latest.pt')
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    actuals = list()
    predictions = list()

    for X, tokens, masks, y in testloader:
        X, tokens, masks = X.to(device), tokens.to(device), masks.to(device)
        y_pred = model(X, tokens, masks)
        y = y.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y = y.reshape((-1,1))
        y = scaler.inverse_transform(y)
        y_pred = scaler.inverse_transform(y_pred)
        actuals.extend(y.flatten().tolist())
        predictions.extend(y_pred.flatten().tolist())

    test_error = mean_squared_error(predictions, actuals)
    result = {'prediction': predictions, 'actual': actuals, 'mse': test_error}
    with open(modelpath/'result.json', 'w') as f:
        json.dump(result, f)
        f.close()
    



if __name__ == "__main__":
    # Experiment 1
    # batch = 8
    # epochs = 100
    # lag = 7
    # model = LSTMModel(in_size=1, hid_size=512)
    # optimiser = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=10**(-4))
    # loss_fn = nn.MSELoss()
    # train_time_model(model, optimiser, loss_fn,
    #                   epochs, batch, lag, 'wheat', 'time', Path.cwd()/'models/lstm_wheat')

    # Experiment 2
    # batch = 32
    # epochs = 5
    # lag = 7
    # loss_fn = nn.MSELoss()
    # train_mixed_model(0.005,loss_fn,
    #                   epochs, batch, lag, 'wheat', 'text', Path.cwd()/'models/bert1_wheat')
    # Testing
    # modelpath = Path.cwd()/'models/bert1_wheat'
    # plot_training_curve(modelpath, 7)
    # test_mixed_model('wheat', 'text', 7, modelpath)
    # plot_result(modelpath)

    # Experiment 3
    # batch = 32
    # epochs = 10
    # lag = 7
    # loss_fn = nn.MSELoss()
    # train_mixed_model(0.001,loss_fn,
    #                   epochs, batch, lag, 'wheat', 'text', Path.cwd()/'models/bert_attn_wheat', BERTAttn1)
    # Testing
    modelpath = Path.cwd()/'models/bert_attn_wheat'
    # plot_training_curve(modelpath, 7)
    # test_mixed_model('wheat', 'text', 7, modelpath)
    plot_result(modelpath)
    pass
   