import pandas as pd
import fire

def append_strings_from_csv(input_csv="company_website_second_round_with_additional_firms_without_redundant_cleaned.csv", output_file="corpus_cleaned.txt"):
    df = pd.read_csv(input_csv, header=None)
    with open(output_file, mode='a', encoding='utf-8') as outfile:
        for row in df.itertuples(index=False, name=None):
            valid_strings = [item for item in row if isinstance(item, str)]
            if valid_strings:
                outfile.write(' '.join(valid_strings) + '\n')

if __name__ == "__main__":
    fire.Fire(append_strings_from_csv)
