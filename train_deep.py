import torch
import torch.nn as nn

import sota_models.metrics as metrics

import time
import wandb
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

def run_epoch(model, optimizer, criterion, dataloader, device, epoch, metric='accuracy', mode='train'):
    """
    The function which deals with epoch running
    """
    is_train = (mode == 'train')
    if is_train:
        model.train()
    else:
        model.eval()
    
    if metric == 'accuracy':
        calculate_metric = metrics.accuracy_score
    elif metric == 'f1 score':
        calculate_metric = metrics.calculate_f1
    elif metric == 'precision':
        calculate_metric = metrics.calculate_precision
    elif metric == 'recall':
        calculate_metric = metrics.calculate_recall
    else:
        print('Unknown metric.')
        return 0
    
    epoch_loss, epoch_metric = 0.0, 0.0
    with torch.set_grad_enabled(is_train):
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred_raw = model(X)
            loss = criterion(y_pred_raw, y)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            y_pred = y_pred_raw.softmax(dim=1).argmax(dim=1)
            epoch_metric += calculate_metric(y, y_pred, average='weighted')
            
    return epoch_loss / len(dataloader), epoch_metric / len(dataloader)

def train(model, optimizer, criterion, train_loader, val_loader, device, n_epochs, metric='accuracy', project_name='crypto_results',\
          scheduler=None, checkpoint=True, ch_name='./model_best.pth', early_stopping=False, es_patience=10, freq=None, verbose=True):
    if verbose and freq is None:
        freq = max(1, n_epochs // 10)
    
    wandb.init(project=project_name)
    
    best_val_m = float('-inf')
    bad_epochs = 0
    for epoch in range(n_epochs):
        epoch_start = time.time()
        train_loss, train_m = run_epoch(model, optimizer, criterion, train_loader, device, epoch, metric, 'train')
        val_loss, val_m     = run_epoch(model, None, criterion, val_loader, device, epoch, metric, 'val')
        train_val_time = time.time() - epoch_start

        wandb.log({
            "Train loss" : train_loss,
            "Train {}".format(metric) : train_m,
            "Validation loss" : val_loss,
            "Validation {}".format(metric) : val_m
            }
        )
        
        if val_m > best_val_m:
            bad_epochs = 0
            best_val_m = val_m
            
            if checkpoint:
                torch.save(model.state_dict(), ch_name)
        
        if scheduler is not None:
            scheduler.step()
        
        if verbose and epoch % freq == 0:
            print("Epoch {}: train loss - {:.4f} | validation loss - {:.4f}".format(epoch, train_loss, val_loss))
            print("train {} - {:.2f} | validation {} - {:.2f}".format(metric, train_m, metric, val_m))
            print("Elapsed time: {:.2f} s".format(train_val_time))
        
        if early_stopping:
            bad_epochs += 1
            if bad_epochs > es_patience:
                print("Stopped at", epoch, "because patience threshold for epochs",\
                      "without validation loss improvement was reached.")
                break

def infer_model(model, test_loader, device):
    model.eval()

    f1, acc, prec, rec = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for (X, y) in test_loader:
            X, y = X.to(device), y.to(device)

            y_pred_raw = model(X)
            y_pred = y_pred_raw.softmax(dim=1).argmax(dim=1)

            f1  += metrics.calculate_f1(y, y_pred, average='weighted')
            acc += metrics.accuracy_score(y, y_pred)
            prec += metrics.calculate_precision(y, y_pred, average='weighted')
            rec += metrics.calculate_recall(y, y_pred, average='weighted')
    return acc / len(test_loader), f1 / len(test_loader), prec / len(test_loader), rec / len(test_loader)