import pandas as pd

def feature_engineer(input_path, output_path):
    df = pd.read_csv(input_path)

    # Example feature: binning cost
    df["cost_bin"] = pd.cut(
        df["cost"], 
        bins=[0, 50, 100, 200, 500, 10000],
        labels=["very_low", "low", "medium", "high", "very_high"]
    )

    # Example feature: cuisine diversity (count commas in cuisine list)
    df["cuisine_diversity"] = df["cuisine"].astype(str).apply(lambda x: len(x.split(",")))

    df.to_csv(output_path, index=False)
    print(f"Feature engineering complete → {output_path}")

if __name__ == "__main__":
    feature_engineer("data/processed.csv", "data/features.csv")
