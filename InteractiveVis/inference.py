#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import pandas as pd

class InferenceEngine:
    def __init__(self, model_path):
        """Initialize the inference engine with a trained model.
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model = load_model(model_path)
        
    def predict(self, image, batch_size=1):
        """Make predictions on input images.
        
        Args:
            image (np.ndarray): Input image or batch of images
            batch_size (int): Batch size for prediction
            
        Returns:
            np.ndarray: Predicted probabilities for each class
        """
        if len(image.shape) == 4:  # Single image
            image = np.expand_dims(image, axis=0)
        return self.model.predict(image, batch_size=batch_size)
    
    def evaluate_performance(self, images, labels, class_names=['CN', 'MCI', 'AD']):
        """Evaluate model performance and generate metrics.
        
        Args:
            images (np.ndarray): Input images
            labels (np.ndarray): True labels
            class_names (list): Names of the classes
            
        Returns:
            dict: Dictionary containing performance metrics
        """
        predictions = self.predict(images)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_classes, pred_classes)
        
        # Calculate metrics for each class
        metrics = {}
        n_classes = len(class_names)
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - (tp + fp + fn)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
            
            metrics[class_names[i]] = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'accuracy': accuracy,
                'ppv': ppv
            }
            
            # Calculate ROC curve and AUC
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
                roc_auc = auc(fpr, tpr)
                metrics[class_names[i]]['auc'] = roc_auc
                metrics[class_names[i]]['fpr'] = fpr
                metrics[class_names[i]]['tpr'] = tpr
        
        metrics['confusion_matrix'] = cm
        return metrics
    
    def plot_confusion_matrix(self, confusion_matrix, class_names=['CN', 'MCI', 'AD']):
        """Plot confusion matrix using seaborn.
        
        Args:
            confusion_matrix (np.ndarray): Confusion matrix to plot
            class_names (list): Names of the classes
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, class_name):
        """Plot ROC curve for binary classification.
        
        Args:
            fpr (np.ndarray): False positive rates
            tpr (np.ndarray): True positive rates
            roc_auc (float): Area under the ROC curve
            class_name (str): Name of the positive class
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {class_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        return plt.gcf() 