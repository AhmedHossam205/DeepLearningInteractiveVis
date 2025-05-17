import streamlit as st
from datamodel import Model
import numpy as np

def app():
    st.title("Model Performance Analysis")
    
    # Get the model instance from session state
    if 'model' not in st.session_state:
        st.error("Please load a model first!")
        return
        
    model = st.session_state.model
    
    # Add some example predictions if needed (for testing)
    if st.button("Add Sample Test Data"):
        # Example data - replace with your actual test data
        test_data = [
            ("CN", "CN"),
            ("CN", "MCI"),
            ("MCI", "MCI"),
            ("MCI", "AD"),
            ("AD", "AD"),
            ("AD", "MCI"),
        ]
        for true_label, pred_label in test_data:
            model.add_prediction_result(true_label, pred_label)
        st.success("Added sample test data!")
    
    # Clear predictions button
    if st.button("Clear All Predictions"):
        model.true_labels = []
        model.predicted_labels = []
        st.success("Cleared all prediction data!")
    
    # Display metrics
    st.write("### Model Performance Metrics")
    st.write("This page shows the confusion matrix and related metrics for the model's predictions.")
    
    # Show number of predictions
    n_predictions = len(model.true_labels)
    st.write(f"Number of predictions: {n_predictions}")
    
    if n_predictions > 0:
        # Display confusion matrix and metrics
        model.analyze_model_performance()
        
        # Display raw prediction data in a table
        st.write("### Raw Prediction Data")
        data = {
            "True Label": model.true_labels,
            "Predicted Label": model.predicted_labels,
            "Correct": [t == p for t, p in zip(model.true_labels, model.predicted_labels)]
        }
        st.dataframe(data)
    else:
        st.info("No prediction data available. Run some predictions or add sample test data to see the analysis.")