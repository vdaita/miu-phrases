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

def remove_substrings(input_set):
    sorted_set = sorted(input_set, key=len)
    result_set = set()

    for string in sorted_set:
        if not any(string in other for other in result_set):
            result_set.add(string)

    return list(result_set)

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
        filtered_words = [word for word in seed_words[category] if word in model.key_to_index]
        vectors = [model[word] for word in filtered_words]
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
    Fire(get_best_ngrams)