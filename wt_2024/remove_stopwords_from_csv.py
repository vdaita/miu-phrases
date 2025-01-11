import pandas as pd
from fire import Fire
import spacy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor



def remove_stopwords_from_csv(file_path="company_website_second_round_with_additional_firms.csv", output_file_path="company_website_second_round_with_additional_firms_cleaned.csv"):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    max_length = df.applymap(lambda x: len(str(x))).max().max()
    print(f"The maximum length of a string in the CSV file is: {max_length}")
    # Load Spacy model
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = max_length
    
    # Function to remove stopwords from a text using Spacy
    def remove_stopwords(text):
        doc = nlp(text)
        filtered_words = [token.text for token in doc if not token.is_stop]
        return ' '.join(filtered_words)
    
    # Apply the function to each cell after the first 14 columns with progress tracking
    def process_column(col):
        df[col] = df[col].astype(str).apply(remove_stopwords)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_column, df.columns[14:]), total=len(df.columns[14:]), desc="Processing columns"))
    # Save the modified DataFrame back to a CSV file
    df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    # Report the maximum length of a string in the given CSV file
    Fire(remove_stopwords_from_csv)