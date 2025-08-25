import mlflow
from mlflow.tracking import MlflowClient

def main():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("titanic")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.test_auc DESC"], max_results=1)

    best_run = runs[0]
    run_id = best_run.info.run_id

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, "titanic_model")

    client.transition_model_version_stage(
        name="titanic_model",
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"Registered model v{mv.version} as Production")

if __name__ == "__main__":
    main()