import pandas as pd

def profile_csv(path: str):
    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))

    print("\nNull counts (top):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\nSample rows:")
    print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    profile_csv("data/input/amazon_dataset.csv")