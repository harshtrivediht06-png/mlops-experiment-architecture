import yaml
import argparse
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature  
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# Import modules from your src folder
from data_preprocessing import process_and_save_data
from evaluate import evaluate_and_report

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main(config_path):
    # Load Configuration
    config = load_config(config_path)
    mlflow.set_experiment(config['experiment']['name'])
    
    # 2. ENABLE AUTOLOGGING (But disable auto-model saving to prevent conflicts)
    mlflow.sklearn.autolog(log_models=False)
    
    with mlflow.start_run():
        # Load, Preprocess, and Split Dataset
        X_train, X_test, y_train, y_test, target_names = process_and_save_data(
            test_size=config['training']['test_size'],
            random_state=config['training']['random_state']
        )
        
        # Train Model
        rf = RandomForestClassifier(
            n_estimators=config['model']['n_estimators'],
            max_depth=config['model']['max_depth'],
            min_samples_split=config['model']['min_samples_split'],
            random_state=config['training']['random_state']
        )
        rf.fit(X_train, y_train)
        
        # Evaluate Model & Generate Report
        y_pred = rf.predict(X_test)
        metrics = evaluate_and_report(y_test, y_pred, target_names)
        
        # Save Model locally in the models/ folder
        os.makedirs("models", exist_ok=True)
        joblib.dump(rf, "models/random_forest_model.pkl")
        
        # 3. INFER THE MODEL SIGNATURE
        signature = infer_signature(X_train, rf.predict(X_train))
        
        # Log Custom Artifacts
        mlflow.log_artifact("reports/figures/confusion_matrix.png")
        mlflow.log_artifact("reports/classification_report.txt")
        
        # 4. MANUALLY LOG THE MODEL WITH THE SIGNATURE
        mlflow.sklearn.log_model(rf, "random_forest_model", signature=signature)
        
        print(f"Pipeline executed successfully. Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)