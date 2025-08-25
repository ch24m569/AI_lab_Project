import requests

sample = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

r = requests.post("http://localhost:8000/predict", json=sample)
print(r.json())