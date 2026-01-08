"""
Evaluation metrics
"""

import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

def compute_confusion_matrix(predictions, labels, num_classes, ignore_label=-1):
    """
    Compute confusion matrix
    
    Args:
        predictions: (N,) predicted labels
        labels: (N,) ground truth labels
        num_classes: Number of classes
        ignore_label: Label to ignore
    
    Returns:
        confusion_matrix: (num_classes, num_classes) array
    """
    # Filter out ignore label
    mask = labels != ignore_label
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Compute confusion matrix
    cm = sklearn_confusion_matrix(
        labels,
        predictions,
        labels=list(range(num_classes))
    )
    
    return cm

def compute_miou(confusion_matrix):
    """
    Compute mean IoU from confusion matrix
    
    Args:
        confusion_matrix: (C, C) array
    
    Returns:
        miou: Mean IoU
        iou_per_class: IoU for each class
    """
    # IoU = TP / (TP + FP + FN)
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    
    iou_per_class = tp / (tp + fp + fn + 1e-10)
    miou = np.mean(iou_per_class)
    
    return miou, iou_per_class

def compute_accuracy(confusion_matrix):
    """Compute overall accuracy"""
    return np.diag(confusion_matrix).sum() / confusion_matrix.sum()

def compute_precision_recall_f1(confusion_matrix):
    """Compute precision, recall, F1 per class"""
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return precision, recall, f1
