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
    index = faiss.IndexFlatL2(384)

    ngrams = []
    ngram_embeddings = []
    for ngram in wv.key_to_index:
        ngrams.append(ngram)
        ngram_embeddings.append(wv[ngram])

    vectors = [get_phrase_vector(phrase, model) for phrase in seed_words]
    vectors = [vector for vector in vectors if vector is not None]
    average_vector = np.mean(vectors, axis=0)

    similar_ngrams = index.search(average_vector.reshape(1, -1), 400)
    similar_ngrams = [ngrams[i] for i in similar_ngrams[1][0]]
    similar_ngrams = remove_substrings(similar_ngrams)
    for i in range(len(similar_ngrams)):
        similar_ngrams[i] = similar_ngrams[i].replace("_", " ")

    return similar_ngrams

def get_best_ngrams(model_path: str = "ngram_model.model", seed_words_path: str = "seed_words.json", description_path_json: str = "descriptions.json", output_path_json: str = "generated_words.json",  output_path_txt: str = "generated_words.txt"):
    model = Word2Vec.load(model_path)
    model = model.wv
    print("Word2Vec model loaded")

    index = faiss.IndexFlatL2(384)
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )


    ngrams = []
    ngram_embeddings = []

    # Load in the word_vectors and find the most similar ones
    for ngram in model.key_to_index:
        ngrams.append(ngram)
        ngram_embeddings.append(model[ngram])

    # print(ngram_embeddings)
    
    index.add(np.array(ngram_embeddings))


    seed_words = json.load(open(seed_words_path, 'r'))
    all_words = {
        "refined": {},
        "unrefined": {}
    } # create 2 options, without refinement, and with refinement

    all_words_txt = "# Generated n-grams\n\n"
    descriptions = json.load(open(description_path_json, 'r'))

    for category in seed_words:
        vectors = [get_phrase_vector[word] for word in seed_words[category]]
        average_vector = np.mean(vectors, axis=0)
        
        similar_ngrams = index.search(average_vector.reshape(1, -1), 400)
        similar_ngrams = [ngrams[i] for i in similar_ngrams[1][0]]
        similar_ngrams = remove_substrings(similar_ngrams)
        for i in range(len(similar_ngrams)):
            similar_ngrams[i] = similar_ngrams[i].replace("_", " ")

        all_words["unrefined"][category] = similar_ngrams

        all_words_txt += f"## Unrefined {category}\n\n"
        all_words_txt += "\n".join(similar_ngrams) + "\n\n"
        
        # Refine the words using ChatGPT
        response = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": """Select the 150 most relevant keywords that should be included in this. Response JSON format: {"sorted_keywords": ["ngram1", "ngram2", ...]}"""
            }, {
                "role": "user",
                "content": f"# Category: {category} sentiment from an American company \n Category description: {descriptions[category]} \n\n# Bigrams:\n{json.dumps(similar_ngrams)}"
            }],
            model="gpt-4o",
            # temperature=0.0,
            response_format={"type": "json_object"}
        )
        response = response.choices[0].message.content
        response = json.loads(response)
        chatgpt_refined_ngrams = response["sorted_keywords"]
        verified_refined_ngrams = []
        for bigram in chatgpt_refined_ngrams:
            if bigram in similar_ngrams:
                verified_refined_ngrams.append(bigram)
        all_words["refined"][category] = verified_refined_ngrams

        all_words_txt += f"## Refined {category}\n\n"
        all_words_txt += "\n".join(verified_refined_ngrams) + "\n\n"

    with open(output_path_json, 'w') as f:
        json.dump(all_words, f, indent=4)
    
    with open(output_path_txt, 'w') as f:
        f.write(all_words_txt)

if __name__ == "__main__":
    seed_words = json.load(open("chat_extracted_keywords.json", "r"))["direct_extracted"]["antiforeign"]
    model = Word2Vec.load("ngram_model.model")

    with open("chat_extracted_keywords_extended.json", "w+") as f:
        json.dump(
         {
            "extracted_expanded": {
                "antiforeign": get_most_similar_words(seed_words, model)
            }
         },
         f
        )
    # Fire(get_best_ngrams)