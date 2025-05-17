#!/usr/bin/env python
# coding: utf-8

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from InteractiveVis.datamodel import Model
from InteractiveVis.config import selected_model
import matplotlib.pyplot as plt

def load_test_data(data_path, labels_path):
    """Load test data and labels from HDF5 files."""
    with h5py.File(data_path, 'r') as f:
        images = np.array(f['test_images'])
    with h5py.File(labels_path, 'r') as f:
        labels = np.array(f['test_labels'])
    return images, labels

def main():
    # Initialize model
    model = Model()
    model.set_model(selected_model)
    
    # Load test data (replace with your actual data paths)
    images, labels = load_test_data('data/test_images.h5', 'data/test_labels.h5')
    
    # Get performance metrics
    metrics = model.get_performance_metrics(images, labels)
    
    # Print metrics for each class
    class_names = ['CN', 'MCI', 'AD']
    for class_name in class_names:
        if class_name in metrics:
            print(f"\nMetrics for {class_name}:")
            print(f"Sensitivity: {metrics[class_name]['sensitivity']:.3f}")
            print(f"Specificity: {metrics[class_name]['specificity']:.3f}")
            print(f"Accuracy: {metrics[class_name]['accuracy']:.3f}")
            print(f"PPV: {metrics[class_name]['ppv']:.3f}")
            if 'auc' in metrics[class_name]:
                print(f"AUC: {metrics[class_name]['auc']:.3f}")
    
    # Plot confusion matrix
    cm_plot = model.inference_engine.plot_confusion_matrix(
        metrics['confusion_matrix'], 
        class_names=class_names
    )
    cm_plot.savefig('confusion_matrix.png')
    
    # Plot ROC curves if binary classification
    if len(class_names) == 2:
        for class_name in class_names:
            if 'auc' in metrics[class_name]:
                roc_plot = model.inference_engine.plot_roc_curve(
                    metrics[class_name]['fpr'],
                    metrics[class_name]['tpr'],
                    metrics[class_name]['auc'],
                    class_name
                )
                roc_plot.savefig(f'roc_curve_{class_name}.png')
    
    plt.close('all')

if __name__ == '__main__':
    main() 