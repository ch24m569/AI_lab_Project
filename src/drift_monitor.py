from evidently.test_suite import TestSuite
from evidently.tests import TestDataDrift
import pandas as pd

def main():
    ref = pd.read_csv("data/raw/train.csv")
    new = pd.read_csv("data/raw/new_data.csv")  # simulate new arrivals

    suite = TestSuite(tests=[TestDataDrift()])
    suite.run(reference_data=ref, current_data=new)

    suite.save_html("drift_report.html")
    print("Drift report generated")

if __name__ == "__main__":
    main()