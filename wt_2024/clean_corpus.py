import spacy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from typing import List

# Load spaCy with only needed components
nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'lemmatizer', 'attribute_ruler'])

def replace_patterns_with_spacy_batch(texts: List[str]):
    # Process multiple texts in batches
    results = []
    
    # Use spaCy's pipe for batch processing
    for doc in nlp.pipe(texts):
        chunk_tokens = [
            "[number]" if token.like_num
            else "[date]" if token.ent_type_ == "DATE"
            else "[time]" if token.ent_type_ == "TIME"
            else token.text
            for token in doc
            if not token.is_stop and len(token.text) <= 12
        ]
        results.append(' '.join(chunk_tokens))
    
    return results

# For single text processing
def replace_patterns_with_spacy(text: str) -> str:
    chunk_size = 100000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    processed_chunks = replace_patterns_with_spacy_batch(chunks)
    return ''.join(processed_chunks)

def chunk_text(text, chunk_size=500000):  # Increased chunk size
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_file_with_spacy(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()
    chunks = chunk_text(content)
    
    with ProcessPoolExecutor(max_workers=12) as executor:  # Increased workers
        processed_chunks = list(tqdm(executor.map(replace_patterns_with_spacy, chunks), total=len(chunks)))
    
    # More efficient joining
    with open(output_file, 'w') as file:
        file.write(' '.join(processed_chunks))

if __name__ == "__main__":
    input_file = 'corpus.txt'
    output_file = 'corpus_cleaned.txt'
    process_file_with_spacy(input_file, output_file)