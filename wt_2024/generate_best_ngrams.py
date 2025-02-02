from gensim.utils import tokenize
import faiss
import numpy as np
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from rapidfuzz import process
from fire import Fire

load_dotenv()

def remove_substrings(input_list):
    result_list = []

    for string in input_list:
        has_substring = False
        for other_string in input_list:
           if other_string in string and other_string != string:
                has_substring = True
                break
        if not has_substring:
            result_list.append(string)

    return result_list

def get_phrase_vector(phrase, model):
    """Get vector representation of a phrase using available n-grams."""
    wv = model.wv
    # Normalize and clean the phrase
    words = phrase.lower().split()
    word_vectors = []
    used_ngrams = []
    i = 0
    
    while i < len(words):
        # Try trigrams first
        if i < len(words) - 2:
            trigram = f"{words[i]}_{words[i+1]}_{words[i+2]}"
            if trigram in wv:
                word_vectors.append(wv[trigram] * 1.5)  # Weight trigrams higher
                used_ngrams.append(trigram)
                i += 3
                continue
                
        # Try bigrams
        if i < len(words) - 1:
            bigram = f"{words[i]}_{words[i+1]}"
            if bigram in wv:
                word_vectors.append(wv[bigram] * 1.2)  # Weight bigrams slightly higher
                used_ngrams.append(bigram)
                i += 2
                continue
        
        # Fall back to unigrams
        if words[i] in wv:
            word_vectors.append(wv[words[i]])
            used_ngrams.append(words[i])
            i += 1
        else:
            # Handle unknown words by using subwords or skipping
            subwords = words[i].split('-')  # Handle hyphenated words
            subword_vectors = []
            for subword in subwords:
                if subword in wv:
                    subword_vectors.append(wv[subword])
            if subword_vectors:
                word_vectors.append(np.mean(subword_vectors, axis=0))
                used_ngrams.append(words[i])
            i += 1

    # Print debug info
    print(f"Phrase: {phrase}")
    print(f"Used n-grams: {used_ngrams}")

    if not word_vectors:
        return None

    # Normalize the final vector
    final_vector = np.mean(word_vectors, axis=0)
    return final_vector / np.linalg.norm(final_vector)

def get_most_similar_words(seed_words, model):
    wv = model.wv
    dim = model.wv.vector_size
    index = faiss.IndexFlatL2(dim)  # Create index with correct dimensions

    # Collect ngrams and their embeddings
    ngrams = []
    ngram_embeddings = []
    for ngram in wv.key_to_index:
        ngrams.append(ngram)
        ngram_embeddings.append(wv[ngram])
    
    # Convert to numpy array and normalize
    ngram_embeddings = np.array(ngram_embeddings).astype('float32')
    faiss.normalize_L2(ngram_embeddings)  # Normalize vectors
    
    # Add vectors to the index
    index.add(ngram_embeddings)
    print(f"Added {len(ngram_embeddings)} vectors to index")

    # Get average vector for seed words
    vectors = [get_phrase_vector(phrase, model) for phrase in seed_words]
    vectors = [v for v in vectors if v is not None]
    query_vector = np.mean(vectors, axis=0).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # Search similar vectors
    print("Ngrams: ", len(ngrams))
    k = min(400, len(ngrams))  # Number of results to retrieve
    D, I = index.search(query_vector, k)  # D: distances, I: indices
    
    # Get corresponding ngrams
    for dist, idx in zip(D[0], I[0]):
        if idx >= 0:
            print(f"Ngram: {ngrams[idx]}, Similarity Score: {1 - dist}")
    similar_ngrams = [ngrams[i] for i in I[0] if i >= 0]  # Filter out -1 indices
    similar_ngrams = remove_substrings(similar_ngrams)
    similar_ngrams = [ngram.replace("_", " ") for ngram in similar_ngrams]

    return similar_ngrams
    
if __name__ == "__main__":
    seed_words = json.load(open("chat_extracted_keywords.json", "r"))["direct_extracted"]["antiforeign"]
    model = Word2Vec.load("ngram_model.model")

    with open("chat_extracted_keywords_extended.json", "w+") as f:
        sim_words = get_most_similar_words(seed_words, model)
        print("Similar words: ", len(sim_words))
        json.dump(
         {
            "extracted_expanded": {
                "antiforeign": sim_words + seed_words
            }
         },
         f
        )
    # Fire(get_best_ngrams)