import theme_counts

from theme_counts import compute_hash, remove_redundant_computations, get_phrases, process_keywords

import pandas as pd
from rapidfuzz import fuzz
import re
import json
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

def main():

    cosine_similarity_threshold = .4 # Used in comparing phrases found in phrase_extraction.py and text phrases 
                                     # to determine whether a phrase can be classified into a theme
    phrases_for_extraction = {
        "jobs": ["job growth", "hiring from local communities", "boosting local jobs", "hire american"],
        "antiforeign": ["non-imported", "low-quality foreign manufacturing", "low-quality outsourcing"],
        "quality": ["american-made quality"],
        "labor": ["labor condition and benefits", "sweat shops", "worker benefits", "partnerships with unions"],
        "national_pride": ["pride", "proudly", "patriot", "patriotism"],
        "military": ["veteran", "military", "first responder", "law enforcement", "supporting our troops"],
        "revival": ["make america great again", "MAGA", "american manufacturing", "reshoring manufacturing", "boosting america", "america first"]
    }
    
    phrases_for_extraction = {"Made in america": [
        "made in america", "made in usa", "american made",
        "buy american", "support america", "patriot",
        "choose american", "national pride",
        "usa based", "usa produced", "usa manufactured",
        "american worker", "american job",
        "veteran owned", "handcrafted in america",
        "america heritage", "america tradition", "america value",
        "icon of america", "america manufacturer"
    ]}

    
    df = pd.read_csv(r"C:\Users\theal\PycharmProjects\miu-phrases\sp_2024\chatgpt-model\v2\company_website_second_round_with_additional_firms.csv", low_memory=True, index_col=[0], nrows=10)
    df = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed:')])
    print("Loaded data")
    
    non_temporal_df = df.iloc[:, 0:13] # Will need to change indexing if more non-temporal columns are added /// IMPORTANT ///
    df = df.drop(df.columns[0:13], axis=1) # Drop all non-temporal columns
    assert not any(re.match(r"\d{4}-\d{2}", col) for col in non_temporal_df.columns), "Columns with the pattern 'YYYY-MM' are present in dataframe"
    assert all(re.match(r"\d{4}-\d{2}", col) for col in df.columns), "Not all columns follow the 'YYYY-MM' pattern"
    print("Isolated non-temporal data")

    df_computations = remove_redundant_computations(df) # Mark similar pages as redundant to save on computation time
    print("Removed redundant computations")

    embeddings_model = SentenceTransformer("all-MiniLM-L6-v2") # Using this model for embeddings
    print("Loaded embeddings model")
    
    process_keywords(phrases_for_extraction, embeddings_model, df_computations, cosine_similarity_threshold) # Calculate and record keyword similarity for website data
    print("Processed keywords")



if __name__ == "__main__":
    print("Starting main function.")
    main()
    print("Script completed.")