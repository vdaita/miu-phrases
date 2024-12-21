import spacy
from tqdm import tqdm

def replace_patterns_with_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    new_text = []
    for token in doc:
        if token.like_num:
            new_text.append("[number]")
        elif token.ent_type_ == "DATE":
            new_text.append("[date]")
        elif token.ent_type_ == "TIME":
            new_text.append("[time]")
        else:
            new_text.append(token.text)
    
    return " ".join(new_text)

def chunk_text(text, chunk_size=200000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_file_with_spacy(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()
    
    chunks = chunk_text(content)
    processed_chunks = []
    for chunk in tqdm(chunks):
        processed_chunks.append(replace_patterns_with_spacy(chunk))
    processed_content = " ".join(processed_chunks)
    
    with open(output_file, 'w') as file:
        file.write(processed_content)

if __name__ == "__main__":
    input_file = 'corpus.txt'  
    output_file = 'corpus_replaced.txt' 
    process_file_with_spacy(input_file, output_file)
