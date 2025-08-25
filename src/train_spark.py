from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark

def main():
    spark = SparkSession.builder.appName("TitanicTrain").getOrCreate()

    df = spark.read.parquet("data/processed/train")
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    mlflow.set_experiment("titanic")

    with mlflow.start_run():
        lr = LogisticRegression(labelCol="Survived", featuresCol="features", maxIter=20)
        model = lr.fit(train)

        preds = model.transform(test)
        evaluator = BinaryClassificationEvaluator(labelCol="Survived")
        auc = evaluator.evaluate(preds)

        print("Test AUC:", auc)

        mlflow.log_param("maxIter", 20)
        mlflow.log_metric("test_auc", auc)

        mlflow.spark.log_model(model, "model")

if __name__ == "__main__":
    main()