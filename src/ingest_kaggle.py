import os
import pandas as pd
from pathlib import Path

def main():
    os.makedirs("data/raw", exist_ok=True)
    # Titanic dataset from Kaggle (download manually or via API).
    # For now, assume 'train.csv' and 'test.csv' are placed in same folder.
    BASE_DIR = Path(__file__).resolve().parent.parent  # goes from src/ → project
    DATA_PATH = BASE_DIR / "data" / "raw"


    train = pd.read_csv(DATA_PATH / "train.csv")
    test = pd.read_csv(DATA_PATH / "test.csv")

    #train.to_csv("data/raw/train.csv", index=False)
    #test.to_csv("data/raw/test.csv", index=False)

    print(train.head())

if __name__ == "__main__":
    main()