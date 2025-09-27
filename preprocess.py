import pandas as pd
import sys
import os

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)

    # Example cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=['cuisine', 'cost', 'rating_number'])
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    df = df.dropna(subset=['cost'])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    input_path = "zomato_df_final_data.csv"
    output_path = "data/processed.csv"
    preprocess(input_path, output_path)
