import subprocess

def main():
    # Assume new data dropped into data/raw/
    subprocess.run(["dvc", "repro"], check=True)
    print("Pipeline retrained successfully!")

if __name__ == "__main__":
    main()