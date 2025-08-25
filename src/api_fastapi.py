from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

model = mlflow.pyfunc.load_model("models:/titanic_model/Production")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    preds = model.predict(df)
    return {"prediction": int(preds[0])}