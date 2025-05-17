import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import streamlit as st

def plot_confusion_matrix(y_true, y_pred, classes=['CN', 'MCI', 'AD']):
    """
    Plot confusion matrix using seaborn's heatmap.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        classes (list): List of class names
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and axes
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Return the figure for Streamlit
    return plt.gcf()

def display_confusion_matrix_metrics(y_true, y_pred, classes=['CN', 'MCI', 'AD']):
    """
    Display confusion matrix and related metrics in Streamlit.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        classes (list): List of class names
    """
    st.subheader("Confusion Matrix Analysis")
    
    # Plot confusion matrix
    fig = plot_confusion_matrix(y_true, y_pred, classes)
    st.pyplot(fig)
    plt.close()
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics for each class
    metrics = {}
    n_classes = len(classes)
    
    for i in range(n_classes):
        # True Positives, False Positives, False Negatives
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        # Calculate metrics
        accuracy = (tp + tn) / np.sum(cm)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        metrics[classes[i]] = {
            'Accuracy': f'{accuracy:.3f}',
            'Sensitivity': f'{sensitivity:.3f}',
            'Specificity': f'{specificity:.3f}',
            'Precision': f'{precision:.3f}'
        }
    
    # Display metrics in a nice format
    st.subheader("Classification Metrics")
    
    # Create columns for each class
    cols = st.columns(len(classes))
    
    # Display metrics for each class in separate columns
    for idx, (class_name, class_metrics) in enumerate(metrics.items()):
        with cols[idx]:
            st.write(f"**{class_name}**")
            for metric_name, value in class_metrics.items():
                st.write(f"{metric_name}: {value}") 