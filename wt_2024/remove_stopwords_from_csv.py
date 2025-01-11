import pandas as pd
from fire import Fire
import spacy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def remove_stopwords_from_csv(file_path="company_website_second_round_with_additional_firms_without_redundant.csv", output_file_path="company_website_second_round_with_additional_firms_without_redundant_cleaned.csv"):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Load the spaCy model for English
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000

    # Function to remove stopwords from a string
    def remove_stopwords(text):
        if isinstance(text, str):
            doc = nlp(text)
            return " ".join([token.text for token in doc if not token.is_stop])
        return text

    # Iterate through each row in the DataFrame
    def process_row(row):
        for col in df.columns[14:]:
            if pd.notna(row[col]):
                row[col] = remove_stopwords(row[col])
        return row

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_row, [row for _, row in df.iterrows()]), total=df.shape[0]))

    df = pd.DataFrame(results)

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    # Report the maximum length of a string in the given CSV file
    Fire(remove_stopwords_from_csv)