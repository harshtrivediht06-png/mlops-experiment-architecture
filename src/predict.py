# import pandas as pd
# import joblib
# from sklearn.metrics import accuracy_score

# def run_inference():
#     # 1. Define paths to the root models folder and processed data folder
#     model_path = "models/random_forest_model.pkl"
#     x_test_path = "data/processed/X_test.csv"
#     y_test_path = "data/processed/y_test.csv"

#     print(f"Loading model from {model_path}...")
#     model = joblib.load(model_path)

#     print(f"Loading processed test data from {x_test_path}...")
#     X_test = pd.read_csv(x_test_path)
#     y_test = pd.read_csv(y_test_path)

#     # 2. Execute predictions
#     print("Generating predictions...")
#     predictions = model.predict(X_test)

#     # 3. Calculate accuracy
#     accuracy = accuracy_score(y_test, predictions)
    
#     # 4. Display the results
#     print("\n==============================")
#     print(f"Prediction Accuracy: {accuracy:.4f}")
#     print("==============================")
    
#     results_df = pd.DataFrame({
#         "Actual Label": y_test.values.flatten(),
#         "Predicted Label": predictions
#     })
    
#     print("\nSample of First 10 Predictions:")
#     print(results_df.head(10))

# if __name__ == "__main__":
#     run_inference()



import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def run_inference():
    # 1. Define paths
    model_path = "models/random_forest_model.pkl"
    x_test_path = "data/processed/X_test.csv"
    y_test_path = "data/processed/y_test.csv"
    predictions_dir = "data/predictions"
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading processed test data from {x_test_path}...")
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    # 2. Execute predictions
    print("Generating predictions...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # 3. Create DataFrame of results
    results_df = pd.DataFrame({
        "Actual Label": y_test.values.flatten(),
        "Predicted Label": predictions
    })
    
    # 4. Save to physical file
    os.makedirs(predictions_dir, exist_ok=True)
    save_path = f"{predictions_dir}/predictions.csv"
    results_df.to_csv(save_path, index=False)
    
    print(f"\nSuccess! Predictions saved physically to: {save_path}")
    print(f"Prediction Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    run_inference()