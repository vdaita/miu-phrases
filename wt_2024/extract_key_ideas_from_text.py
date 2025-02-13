from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

load_dotenv()

# Global client and embedding for reference phrase
client = OpenAI()
model = "text-embedding-3-small"

def get_similarity_for_phrase(phrase):
    """Worker function to process a single phrase"""
    try:
        # Get reference embedding
        id_phrase = client.embeddings.create(
            input="a sentiment from an american company against foreign companies",
            model="text-embedding-3-small"
        ).data[0].embedding
        
        # Get phrase embedding
        response = client.embeddings.create(
            input=phrase,
            model="text-embedding-3-small"
        ).data[0].embedding
        
        similarity = cosine_similarity([id_phrase], [response])[0][0]
        return (phrase, similarity)
    except Exception as e:
        print(f"Error processing phrase: {phrase}, {str(e)}")
        return (phrase, 0.0)

def extract_most_relevant_phrases(words):
    phrase_size = 7
    split_length = 5
    phrases = [" ".join(words[i:min(i + phrase_size, len(words))]) 
              for i in range(0, len(words), split_length)]

    similarities = []
    max_workers = os.cpu_count()

    # Process phrases in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all phrases for processing
        future_to_phrase = {executor.submit(get_similarity_for_phrase, phrase): phrase 
                          for phrase in phrases}
        
        # Collect results as they complete
        for future in tqdm(as_completed(future_to_phrase), total=len(phrases)):
            result = future.result()
            if result:
                similarities.append(result)

    return sorted(similarities, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    if not os.path.exists("corpus_cleaned_sample.txt"):
        corpus_text = open("corpus_cleaned.txt", "r").read().split(" ")[:25000]
        with open("corpus_cleaned_sample.txt", "w+") as f:
            f.write(" ".join(corpus_text))
    
    corpus_text = open("corpus_cleaned_sample.txt", "r").read().split(" ")
    phrases = extract_most_relevant_phrases(corpus_text)
    
    with open("relevant_phrases.json", "w+") as f:
        json.dump(phrases, f)