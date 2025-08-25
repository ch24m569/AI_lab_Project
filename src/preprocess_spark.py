from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

def main():
    spark = SparkSession.builder.appName("TitanicPreprocess").getOrCreate()

    df = spark.read.csv("data/raw/train.csv", header=True, inferSchema=True)

    # Basic preprocessing
    df = df.drop("PassengerId", "Name", "Ticket", "Cabin")

    # Handle missing values
    df = df.na.fill({"Age": df.agg({"Age": "mean"}).first()[0],
                     "Embarked": "S",
                     "Fare": df.agg({"Fare": "mean"}).first()[0]})

    categorical = ["Sex", "Embarked", "Pclass"]
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx", handleInvalid="keep") for col in categorical]
    encoders = [OneHotEncoder(inputCol=col+"_idx", outputCol=col+"_vec") for col in categorical]

    features = ["Age", "SibSp", "Parch", "Fare"] + [col+"_vec" for col in categorical]
    assembler = VectorAssembler(inputCols=features, outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    model = pipeline.fit(df)
    processed = model.transform(df)

    processed.select("Survived", "features").write.mode("overwrite").parquet("data/processed/train")

    print("Processed data saved in data/processed/train")

if __name__ == "__main__":
    main()