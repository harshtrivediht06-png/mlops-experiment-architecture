import yaml
import subprocess
import time

# 1. Define the matrix of hyperparameters from Part H
experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 200, "max_depth": None} # None translates to 'null' in YAML
]

config_path = "configs/config.yaml"

for i, exp in enumerate(experiments, 1):
    print(f"\n======================================")
    print(f"Running Experiment {i}/6")
    print(f"n_estimators: {exp['n_estimators']} | max_depth: {exp['max_depth']}")
    print(f"======================================")
    
    # 2. Read the existing config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # 3. Update the hyperparameters
    config['model']['n_estimators'] = exp['n_estimators']
    config['model']['max_depth'] = exp['max_depth']
    
    # 4. Save the updated config back to the YAML file
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
        
    # 5. Execute the train.py script via the terminal
    try:
        # This is equivalent to running `python src/train.py --config configs/config.yaml`
        subprocess.run(["python", "src/train.py", "--config", config_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing run {i}: {e}")
        
    # Brief pause to ensure MLflow registers timestamps properly
    time.sleep(1)

print("\nAll 6 experiments completed successfully!")
print("Run 'mlflow ui' in your terminal to compare the results.")