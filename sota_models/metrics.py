import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def accuracy_score(y, y_pred, average=None):
    """
    Calculate accuracy as a percentage.
    Input:
        y - tensor of true labels,
        y_pred - tensor of predicted labels,
        average - dummy variable for consistency
    Return:
        accuracy score in percents
    """
    return (y_pred == y).float().mean().item() * 100

def calculate_f1(y, y_pred, average='weighted'):
    """
    Calculate f1 score as a percentage.
    Input:
        y - tensor of true labels,
        y_pred - tensor of predicted labels,
        average - averaging method
    Return:
        f1 score in percents
    """
    return f1_score(y.detach().cpu(), y_pred.detach().cpu(), average=average) * 100

def calculate_precision(y, y_pred, average='weighted'):
    """
    Calculate precision as a percentage.
    Input:
        y - tensor of true labels,
        y_pred - tensor of predicted labels,
        average - averaging method
    Return:
        precision score in percents
    """
    return precision_score(y.detach().cpu(), y_pred.detach().cpu(), average=average) * 100

def calculate_recall(y, y_pred, average='weighted'):
    """
    Calculate recall as a percentage.
    Input:
        y - tensor of true labels,
        y_pred - tensor of predicted labels,
        average - averaging method
    Return:
        recall score in percents
    """
    return recall_score(y.detach().cpu(), y_pred.detach().cpu(), average=average) * 100