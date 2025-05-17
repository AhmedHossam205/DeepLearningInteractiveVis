import streamlit as st
import numpy as np
from components.ConfusionMatrix import plot_confusion_matrix, display_confusion_matrix_metrics

def app():
    st.title("Confusion Matrix Visualization")
    
    # Get the model instance from session state
    if 'model' not in st.session_state:
        st.error("Please load a model first!")
        return
        
    model = st.session_state.model
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Interactive Confusion Matrix")
        
        # Add controls for test data
        test_data_option = st.selectbox(
            "Choose Test Data Source",
            ["Sample Data", "Real Predictions", "Custom Input"]
        )
        
        if test_data_option == "Sample Data":
            if st.button("Load Sample Data"):
                # Example data with different scenarios
                test_data = [
                    ("CN", "CN"), ("CN", "CN"), ("CN", "MCI"),  # Some CN cases
                    ("MCI", "MCI"), ("MCI", "CN"), ("MCI", "AD"),  # Some MCI cases
                    ("AD", "AD"), ("AD", "AD"), ("AD", "MCI")  # Some AD cases
                ]
                model.true_labels = []
                model.predicted_labels = []
                for true_label, pred_label in test_data:
                    model.add_prediction_result(true_label, pred_label)
                st.success("Sample data loaded!")
                
        elif test_data_option == "Custom Input":
            st.write("Add custom prediction pairs:")
            true_label = st.selectbox("True Label", ["CN", "MCI", "AD"], key="true")
            pred_label = st.selectbox("Predicted Label", ["CN", "MCI", "AD"], key="pred")
            if st.button("Add Prediction"):
                model.add_prediction_result(true_label, pred_label)
                st.success(f"Added prediction: True={true_label}, Predicted={pred_label}")
    
    with col2:
        # Controls for visualization
        st.subheader("Visualization Controls")
        if st.button("Clear All Data"):
            model.true_labels = []
            model.predicted_labels = []
            st.success("All data cleared!")
            
        st.write("---")
        st.write("Current Statistics:")
        n_predictions = len(model.true_labels)
        st.write(f"Total predictions: {n_predictions}")
        if n_predictions > 0:
            correct = sum(t == p for t, p in zip(model.true_labels, model.predicted_labels))
            accuracy = (correct / n_predictions) * 100
            st.write(f"Overall accuracy: {accuracy:.1f}%")
    
    # Display the confusion matrix and metrics
    if n_predictions > 0:
        st.write("---")
        display_confusion_matrix_metrics(
            model.true_labels,
            model.predicted_labels,
            classes=['CN', 'MCI', 'AD']
        )
        
        # Display prediction history
        st.write("---")
        st.subheader("Prediction History")
        history_data = {
            "True Label": model.true_labels,
            "Predicted Label": model.predicted_labels,
            "Correct": [t == p for t, p in zip(model.true_labels, model.predicted_labels)]
        }
        st.dataframe(history_data)
    else:
        st.info("No prediction data available. Please add some predictions using the controls above.") 