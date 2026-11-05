# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split

# def load_and_split_data(test_size, random_state):
#     # Load Data
#     data = load_breast_cancer()
#     X = data.data
#     y = data.target
    
#     # Preprocess & Split Dataset
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, 
#         test_size=test_size, 
#         random_state=random_state
#     )
    
#     return X_train, X_test, y_train, y_test, data.target_names

import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def process_and_save_data(test_size, random_state):
    # 1. Ensure physical folders exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 2. Load dataset and save as RAW physical file
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    df.to_csv("data/raw/breast_cancer.csv", index=False)
    
    # 3. Split the data
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # 4. Save splits as PROCESSED physical files
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    
    return X_train, X_test, y_train, y_test, data.target_names