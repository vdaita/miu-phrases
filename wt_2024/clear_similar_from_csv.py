from tqdm import tqdm
import re
import pandas as pd
from rapidfuzz import fuzz
from fire import Fire
from joblib import Parallel, delayed

def main(input_csv: str = "company_website_second_round_with_additional_firms.csv", output_csv: str = "company_website_second_round_with_additional_firms_without_redundant.csv"):
    df = pd.read_csv(input_csv, low_memory=True, index_col=[0])
    df = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed:')]) # If th

    non_temporal_df = df.iloc[:, 0:13] # Will need to change indexing if more non-temporal columns are added
    df = df.drop(df.columns[0:13], axis=1)
    assert not any(re.match(r"\d{4}-\d{2}", col) for col in non_temporal_df.columns), "Columns with the pattern 'YYYY-MM' are present in the non-temporal dataframe"
    assert all(re.match(r"\d{4}-\d{2}", col) for col in df.columns), "Not all columns follow the 'YYYY-MM' pattern"

    def compute_hash(string):
        ans = 0
        for i in range(0, min(len(string), 8)):
            ans += ord(string[i]) * (7**i)
        return ans

    table = {}
    saved = 0
    considering = 0
    def process_row(index, row):
        nonlocal saved, considering
        local_table = {}
        for column_index, value in enumerate(row):
            if pd.isna(value) or type(value) == int or type(value) == float:
                continue
            value_hash = compute_hash(value)
            
            if not(value_hash in local_table):
                local_table[value_hash] = []

            max_similarity = 0
            for existing_value in local_table[value_hash]:
                max_similarity = max(max_similarity, fuzz.ratio(existing_value, value))

            if max_similarity < 0.95:
                local_table[value_hash].append(value)
                considering += 1
            else:
                saved += 1
                df.iloc[index, column_index - 1] = 0 # subtracting 1 from the column_index because the first one is index or something, setting 0 leads to default fallback

    Parallel(n_jobs=4, timeout=120)(delayed(process_row)(index, row) for index, row in tqdm(enumerate(df.itertuples())))
                
    print(saved, considering)
    df.to_csv(output_csv)

if __name__ == "__main__":
    Fire(main)