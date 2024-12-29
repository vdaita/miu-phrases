import pandas as pd
from rapidfuzz import fuzz
import re
import json
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import os
import nltk
import heapq
from openai import OpenAI
from rapidfuzz import fuzz
from nltk.corpus import stopwords

nltk.download("stopwords")

# os.environ["OPENAI_API_KEY"] = "YOUR-API-KEY-HERE"
# model = OpenAI(
#     api_key=os.environ["OPENAI_API_KEY"]
# )



def compute_hash(string):
    """
    Computes a hash value for the given string by summing 
    up to the first 8 ASCII values each multiplied by powers of 7.

    Same hash function in phrase_extraction.py
    """

    ans = 0                                   # Initalize our hash to 0
    for i in range(0, min(len(string), 8)):   # Loop over the first 8 characters if the string is > 8, otherwise use all characters
        ans += ord(string[i]) * (7**i)        # ord converts value to ASCII, and we multiply by 7 to the power of i here
    return ans                                # Return the calculated hash


def remove_redundant_computations(df, verbose=True):
    """
    Using the hash we help improve comutational efficiency by avoiding processesing websites that have very similar.
    We use the normalized indel distance to compute phrase similarity and use a cutoff of .95 to determine similarity.
    Processing rows here is also done in reverse order to speed up the process, and does not affect results.

    Potential issues: Using only first 8 words for hash increases amount of Type I errors.
    """
                                    
    df_copy = df.copy()                 # Make a copy of the original DataFrame to return
    table = {}                          # Initialize hash table
    computations_to_skip = 0            # Used to keep track of amount of computations(LLM prompts) saved
    unique_values_to_process = 0        # Used to keep track amount of computations(LLM  prompts) needed

    for index, row in enumerate(df_copy.itertuples()):                              # Iterate over each row
                                                                                    #
        if verbose and index % 100 == 0:                                            # 
            print(f"Finished processing {index} out of {len(df_copy.index)} rows.") # 
                                                                                    #
        for column_index, value in enumerate(row):                                  # Iterate over each column 
            if pd.isna(value) or isinstance(value, (int, float)):                   # Skip NA values
                continue                                                            # 
                                                                                    #
            value_hash = compute_hash(value)                                        # Compute hash value for the current value
            if value_hash not in table:                                             # Check if the hash value already exists in the hash table
                table[value_hash] = []                                              # If not, initialize the table for the given hash
                                                                                    #
            max_similarity = 0                                                      # Initialize max similarity
            for existing_value in table[value_hash]:                                # Iterate through the existing hash values
                max_similarity=max(max_similarity,fuzz.ratio(existing_value,value)) # Check max similarity using fuzz for normalized indel distance 
                                                                                    #
            if max_similarity < 0.95:                                               # Determine if the value is similar enough
                table[value_hash].append(value)                                     # If it isn't similar add to hash table
                unique_values_to_process += 1                                       # We will be prompting this in LLM so add to total
            else:                                                                   #
                computations_to_skip += 1                                           # If the similarity is high, mark it as a saved computation
                df_copy.iloc[index, column_index - 1] = -1                          # Set the position in df as -1 to signify no need to compute
                                                                                    # 
    if verbose:                                                                     #
        print(computations_to_skip, unique_values_to_process)                       # Print stats
    return df_copy                                                                  # Return


def get_phrases(corpus, length=20, split=15):
    """
    Extracts overlapping phrases from a text corpus. It also makes all characters lowercase.

    Parameters:
    - corpus (str): Text corpus for phrase extraction.
    - length (int, optional): Length of each phrase in words (default is 20).
    - split (int, optional): Word interval to start a new phrase (default is 15).

    Returns:
    - List of extracted phrases (str).

    This is different than get_phrases in phrase_extraction as they datatype of corpus is different
    """
    
    total_words = [word.lower() for word in corpus.split()]       # Get a list of lowercase words from the corpus
    phrases = []                                                  # initialize an empty list of phrases
                                                                  #
    for i in range(0, len(total_words), split):                   # Iterate through the words and increment 15 at a time
        right_idx = min(len(total_words), i + length)             #         
        phrases.append(" ".join(total_words[i:right_idx]))        # Append to phrases
                                                                  #
    return phrases                                                # Return


def process_keywords(phrases_for_extraction, embeddings_model, df, cosine_similarity_threshold):
    """
    Processes a DataFrame using a dictionary of keywords and their associated list of phrases.
    Each text entry in the DataFrame is compared against predefined phrases using cosine similarity
    to identify and count similar phrases, but only if similarity > threshold do we pass to LLM.
    If LLM classification is 'Yes', we increment the count by 1.

    Args:
        phrases_for_extraction (dict): Dictionary where keys are keywords (e.g. 'jobs', 'labor', etc.)
                                       and values are lists of phrases to compare.
        embeddings_model (SentenceTransformer): The model used to embed phrases for similarity.
        df (pd.DataFrame): The DataFrame whose text columns will be processed.
        cosine_similarity_threshold (float): Threshold above which we consider a text-phrase
                                             sufficiently similar to one of the predefined phrases.
    """

    # 1) Define which returned LLM classifications should count as "Yes" for each keyword
    valid_classifications_dict = {
        "jobs": {"OFFER", "GENERAL"},
        "antiforeign": {"FOREIGN"},
        "labor": {"FAIRNESS"},
        "military": {"MILITARY"},
        "national_pride": {"PRIDE"},  # maps to ask_llm_pride
        "quality": {"AMERICAN"},
        "revival": {"REVIVAL"}
    }

    llm_prompt_count = 0
        
    # 2) Define a helper function that calls the appropriate LLM function for each keyword
    def get_llm_response(keyword, phrase):
        if keyword == "jobs":
            classification, _ = ask_llm_jobs(phrase)
        elif keyword == "antiforeign":
            classification, _ = ask_llm_anti_foreign(phrase)
        elif keyword == "labor":
            classification, _ = ask_llm_labor_fairness(phrase)
        elif keyword == "military":
            classification, _ = ask_llm_military(phrase)
        elif keyword == "national_pride":
            classification, _ = ask_llm_pride(phrase)
        elif keyword == "quality":
            classification, _ = ask_llm_quality(phrase)
        elif keyword == "revival":
            classification, _ = ask_llm_revival(phrase)
        else:
            classification = "NOT"

        return classification

    # 3) Loop over each keyword and its associated list of phrases
    for keyword, existing_phrases in phrases_for_extraction.items():
        print(f"Current keyword being processed: {keyword}")
        collected_phrases = []

        # Embed the loaded (existing) phrases for the current keyword
        existing_embedded = embeddings_model.encode(existing_phrases, convert_to_tensor=True)

        cell_counter = 0
        time_last = time.time()

        def count_phrases(text):
            """
            Count how many phrases in the text are classified as "Yes" by LLM after passing
            the cosine similarity > threshold check.
            """
            nonlocal time_last, cell_counter, existing_embedded, cosine_similarity_threshold, llm_prompt_count

            if pd.isna(text) or isinstance(text, (float, int)):
                return text

            # Extract phrases from text
            phrases = get_phrases(text)
            if not phrases:
                phrases = [text]

            # Embed the newly extracted phrases
            phrases_embedded = embeddings_model.encode(phrases, convert_to_tensor=True)

            # Compute cosine similarities: 
            # shape => (#existing_phrases_for_keyword, #phrases_in_text)
            cosine_scores = util.cos_sim(existing_embedded, phrases_embedded)

            phrase_count = 0

            # For each phrase in the text
            for j in range(cosine_scores.shape[1]):
                # Check if any existing keyword-phrase is above threshold
                for i in range(cosine_scores.shape[0]):
                    if cosine_scores[i][j] > cosine_similarity_threshold:
                        # Call the LLM classification function
                        
                        
                        if False: 
                            classification = get_llm_response(keyword, phrases[j])
                        else:
                            classification = "NOT RUNNING FOR NOW"
                            llm_prompt_count += 1
                            
                        # If classification is in the valid set, count it
                        if classification in valid_classifications_dict.get(keyword, set()):
                            phrase_count += 1
                            collected_phrases.append(phrases[j])
                        # Break so we do not double count the same text-phrase
                        break

            cell_counter += 1
            if cell_counter % 500 == 0:
                print(f"Processed {cell_counter} cells in {time.time() - time_last} seconds.")
                time_last = time.time()
            # print(f"LLM prompt count: {llm_prompt_count}")
            return phrase_count

        # Apply the count function across the entire DataFrame
        df2 = df.applymap(count_phrases)
        print(f"Final LLM prompt count: {llm_prompt_count}")
        df2.to_csv(f"theme_panel_data/prelim_{keyword}_nofix_panel.csv")

        # Drop any columns with 'Unnamed:' in their name
        df2 = df2.drop(columns=[col for col in df2.columns if col.startswith('Unnamed:')])

        # "Forward fill" logic from right-to-left columns if the row's current column is NaN or 0 or -1
        for index, row in df2.iterrows():
            prev_value = None
            for col in reversed(df2.columns):
                value = row[col]
                if pd.isna(value) or value in [-1, 0]:
                    if prev_value is not None:
                        df2.at[index, col] = prev_value
                else:
                    prev_value = value

        # Print aggregate sum and save final
        print(df2.sum())
        df2.to_csv(f"theme_panel_data/prelim_{keyword}_panel.csv")
        
        with open(f"phrase_data/{keyword}_phrases.txt", "w", encoding="utf-8") as f:
            for p in collected_phrases:
                f.write(p + "\n")
        
        
def ask_llm_made_in_america(phrase):
    """
    Sends a phrase to the OpenAI API to classify it as MADE-IN-AMERICA or NOT based on the context of promoting American manufacturing, products, or values.

    Args:
        phrase (str): The phrase to be classified.

    Returns:
        tuple: A tuple containing the classification result and the string "MADE-IN-AMERICA".
    """
    messages = [
        {"role": "system", "content": """Classify the phrase as the following: MADE-IN-AMERICA, NOT.
        The phrase should be classified as MADE-IN-AMERICA if it explicitly or implicitly refers to supporting or prioritizing American-made products, encouraging the purchase of American goods, highlighting American manufacturing, workers, or businesses, or emphasizing national pride, patriotism, or values tied to domestic production.
        The phrase should be classified as NOT if it does not align with the themes of American manufacturing, products, or patriotic values tied to domestic production.
        """},
        
        {"role": "user", "content": "Phrase: " + phrase},
    ]
        
    response = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
        temperature=0.1
    )

    return response.choices[0].message.content, "MADE-IN-AMERICA"




def ask_llm_jobs(phrase):
    """
    Sends a phrase to the OpenAI API to classify it as GENERAL, OFFER, or NEITHER based on the context of job offers.

    Args:
        phrase (str): The phrase to be classified.

    Returns:
        tuple: A tuple containing the classification result and the string "GENERAL".
    """
    messages = [
        {"role": "system", "content": """Classify the following phrase as either: GENERAL, OFFER, or NEITHER. 
        The phrase should be classified as OFFER if it refers to a specific job offer the company is providing. 
        The phrase should be classified as GENERAL if it refers to a general attitude of supporting job growth in the United States.
        The phrase should be classified as NEITHER if it refers to neither or is miscellaneous."""},
        {"role": "user", "content": "Phrase: " + phrase},
    ]
        
    response = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
        temperature=0.1
    )

    return response.choices[0].message.content, "GENERAL"


def ask_llm_anti_foreign(phrase):
    """
    Sends a phrase to the OpenAI API to classify it as ANTI-FOREIGN or NOT based on the context.

    Args:
        phrase (str): The phrase to be classified.

    Returns:
        tuple: A tuple containing the classification result and the string "ANTI-FOREIGN".
    """
    messages = [
        {"role": "system", "content": """Classify the following phrase as either: ANTI-FOREIGN, NOT. 
        The phrase should be classified as ANTI-FOREIGN if it reflects opposition to foreign workers, imports, goods, 
        or cultural influences. Examples include rejecting foreign products, claiming foreign goods are inferior, 
        or expressing concerns about foreign cultural impact.
        
        The phrase should be classified as NOT if it does not contain these ideas.
        """},
        
        {"role": "user", "content": "Phrase: " + phrase},
    ]
        
    response = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
        temperature=0.1
    )

    return response.choices[0].message.content, "FOREIGN"



def ask_llm_labor_fairness(phrase):
    """
    Sends a phrase to the OpenAI API to classify it as FAIRNESS or NOT based on the context of labor conditions.

    Args:
        phrase (str): The phrase to be classified.

    Returns:
        tuple: A tuple containing the classification result and the string "FAIRNESS".
    """
    messages = [
        {"role": "system", "content": """Classify the following phrase as either: FAIRNESS, NOT. 
        The phrase should be classified as FAIRNESS if it describes ideas such as paying a fair wage, offering decent wages to American workers, or providing good working conditions, among other related concepts.
        The phrase should be classified as NOT if it does not.
        """},
        
        {"role": "user", "content": "Phrase: " + phrase},
    ]
        
    response = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
        temperature=0.1
    )

    return response.choices[0].message.content, "FAIRNESS"


def ask_llm_military(phrase):
    """
    Sends a phrase to the OpenAI API to classify it as MILITARY or NOT based on the context of support for military and law enforcement.

    Args:
        phrase (str): The phrase to be classified.

    Returns:
        tuple: A tuple containing the classification result and the string "MILITARY".
    """
    messages = [
        {"role": "system", "content": """Classify the phrase as the following: MILITARY, NOT.
        The phrase should be classified as MILITARY if it explicitly states support for The United State's military, law enforcement, and other public enforcement workers.
        The phrase should be classified as NOT if it does not.
        """},
        
        {"role": "user", "content": "Phrase: " + phrase},
    ]
        
    response = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
        temperature=0.1
    )

    return response.choices[0].message.content, "MILITARY"


def ask_llm_pride(phrase):
    """
    Sends a phrase to the OpenAI API to classify it as PRIDE or NOT based on the context of American patriotism.

    Args:
        phrase (str): The phrase to be classified.

    Returns:
        tuple: A tuple containing the classification result and the string "PRIDE".
    """
    messages = [
        {"role": "system", "content": """Classify the phrase as the following: PRIDE, NOT.
        The phrase should be classified as PRIDE if it explicitly states pride and patriotism for the United States.
        The phrase should be classified as NOT if it does not.
        """},
        
        {"role": "user", "content": "Phrase: " + phrase},
    ]
        
    response = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
        temperature=0.1
    )

    return response.choices[0].message.content, "PRIDE"


def ask_llm_quality(phrase):
    """
    Sends a phrase to the OpenAI API to classify it as AMERICAN or NOT based on the context of product quality associated with American manufacturing.

    Args:
        phrase (str): The phrase to be classified.

    Returns:
        tuple: A tuple containing the classification result and the string "AMERICAN".
    """
    messages = [
        {"role": "system", "content": """Classify the phrase as the following: AMERICAN, NOT.
        The phrase should be classified as AMERICAN if it explicitly states the quality increase from goods being manufactured in the United States.
        The phrase should be classified as NOT if it does not.
        """},
        
        {"role": "user", "content": "Phrase: " + phrase},
    ]
        
    response = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
        temperature=0.1
    )

    return response.choices[0].message.content, "AMERICAN"



def ask_llm_revival(phrase):
    """
    Sends a phrase to the OpenAI API to classify it as REVIVIAL or NOT based on the context of product quality associated with American manufacturing.

    Args:
        phrase (str): The phrase to be classified.

    Returns:
        tuple: A tuple containing the classification result and the string "REVIVAL".
    """
    messages = [
            {"role": "system", "content": """Classify the phrase as the following: REVIVAL, NOT.
            The phrase should be classified as REVIVAL if it explicitly states restoring the American economy and American greatness.
            The phrase should be classified as NOT if it does not.
            """},
            
            {"role": "user", "content": "Phrase: " + phrase},
        ]
        
    response = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
        temperature=0.1
    )

    return response.choices[0].message.content, "REVIVAL"