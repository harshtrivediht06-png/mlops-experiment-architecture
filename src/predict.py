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
    
    # Load a few samples from our processed test set to simulate "new" records
    print("Loading sample data...")
    X_test_sample = pd.read_csv("data/processed/X_test.csv").head(5)
    
    print("Generating predictions...")
    predictions = predict(pipeline, X_test_sample)
    
    print("\n--- Prediction Results ---")
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: Actual Class {actual_y[i]} | Predicted Class {prediction}")