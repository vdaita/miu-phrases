from tqdm import tqdm
import re
import pandas as pd
from rapidfuzz import fuzz, process
from fire import Fire
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

def main(input_csv: str = "company_website_second_round_with_additional_firms.csv", output_csv: str = "company_website_second_round_with_additional_firms_without_redundant.csv"):
    df = pd.read_csv(input_csv, low_memory=True, index_col=[0])
    df = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed:')]) # If th

    non_temporal_df = df.iloc[:, 0:13] # Will need to change indexing if more non-temporal columns are added
    df = df.drop(df.columns[0:13], axis=1)
    assert not any(re.match(r"\d{4}-\d{2}", col) for col in non_temporal_df.columns), "Columns with the pattern 'YYYY-MM' are present in the non-temporal dataframe"
    assert all(re.match(r"\d{4}-\d{2}", col) for col in df.columns), "Not all columns follow the 'YYYY-MM' pattern"
    df = df[df.columns[::-1]] # Reverse the columns so that the latest dates are on the right

    saved = 0
    considering = 0
    for idx in tqdm(df.index, desc="Processing rows"):
        previous_str = ""
        row = df.loc[idx]
        
        for col in df.columns:
            val = row[col]
            
            if not isinstance(val, str):
                continue
                
            if fuzz.ratio(val, previous_str) >= 95:
                df.at[idx, col] = -1
                saved += 1
            else:
                previous_str = val
                considering += 1
                

    print(f"Saved {saved} cells as there are similar matches, considering {considering} cells")
    
    df.to_csv(output_csv)

if __name__ == "__main__":
    Fire(main)