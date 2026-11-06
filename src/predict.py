import joblib
import pandas as pd

def load_model(model_path="models/model.joblib"):
    # Load the saved Scikit-learn Pipeline
    return joblib.load(model_path)

def predict(model, input_data):
    # Make predictions using the loaded pipeline
    return model.predict(input_data)

if __name__ == "__main__":
    print("Loading model...")
    pipeline = load_model()
    
    print("Loading sample data...")
    X_test_sample = pd.read_csv("data/processed/X_test.csv").head(5)
    
    # NEW: Load the actual labels to compare against
    y_test_sample = pd.read_csv("data/processed/y_test.csv").head(5)
    
    # Convert the label dataframe to a flat list/array for easy indexing
    actual_y = y_test_sample.values.flatten()
    
    print("Generating predictions...")
    predictions = predict(pipeline, X_test_sample)
    
    print("\n--- Prediction Results ---")
    for i, pred in enumerate(predictions):
        # FIXED: Changed {prediction} to {pred}
        print(f"Sample {i+1}: Actual Class {actual_y[i]} | Predicted Class {pred}")