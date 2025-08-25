import mlflow
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main(run_id):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    print("Metrics:", run.data.metrics)

    # Example confusion matrix plot (assuming you stored predictions separately)
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig("confusion.png")

    mlflow.log_artifact("confusion.png")

if __name__ == "__main__":
    import sys
    run_id = sys.argv[1]
    main(run_id)