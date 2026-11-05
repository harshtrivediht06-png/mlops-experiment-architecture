import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def evaluate_and_report(y_true, y_pred, target_names):
    # Evaluate Model
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
    
    # Generate Report / Artifacts
    os.makedirs("reports/figures", exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("reports/figures/confusion_matrix.png")
    plt.close()
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    with open("reports/classification_report.txt", "w") as f:
        f.write(report)
        
    return metrics