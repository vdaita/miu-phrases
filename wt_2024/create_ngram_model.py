from gensim.models import Phrases, Word2Vec
from gensim.models.phrases import Phraser
from fire import Fire
from tqdm import tqdm
import gensim
import multiprocessing
from itertools import islice
import re

class TqdmCallback(gensim.models.callbacks.CallbackAny2Vec):
    def __init__(self):
        self.pbar = None
        self.epoch = 0
        
    def on_epoch_begin(self, model):
        print(f"\nEpoch {self.epoch} beginning...")
        if self.pbar is None:
            self.pbar = tqdm(total=model.epochs, desc="Training Word2Vec model")
            
    def on_epoch_end(self, model):
        self.epoch += 1
        self.pbar.update(1)
        
def preprocess_text(text):
    """Clean and normalize text"""
    # Convert to lowercase and remove special chars
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def main(input_file: str = "corpus_cleaned.txt", 
         output_file: str = "ngram_model.model", 
         concept: str = "manufacturing",
        ):
    
    print("Loading and preprocessing corpus...")
    with open(input_file, "r") as f:
        text = f.read()
        
    # Preprocess entire text
    text = preprocess_text(text)
    
    # Create sentences (30 words each)
    words = text.split()
    print(f"Total words before crop: {len(words)}")
    print(f"'manufacturing' count before crop: {words.count('manufacturing')}")
    
    # Crop to 15M words if needed
    if len(words) > 15000000:
        words = words[:15000000]
    
    print(f"Total words after crop: {len(words)}")
    print(f"'manufacturing' count after crop: {words.count('manufacturing')}")
    
    # Create sentences
    corpus = [words[i:i+30] for i in range(0, len(words), 30)]
    
    print("Training phrase models...")
    # Train bigram model
    bigram = Phrases(corpus, min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram)
    
    # Train trigram model on top of bigrams
    trigram = Phrases(bigram_phraser[corpus], min_count=5, threshold=10)
    trigram_phraser = Phraser(trigram)
    
    # Apply both models
    processed_corpus = [trigram_phraser[bigram_phraser[sentence]] for sentence in tqdm(corpus)]
    
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=processed_corpus,
        vector_size=384,
        window=10,  # Increased context window
        min_count=10,  # Reduced to catch more words
        workers=multiprocessing.cpu_count(),
        batch_words=10000,
        callbacks=[TqdmCallback()],
        sg=1,  # Skip-gram
        epochs=20,  # Increased epochs
        alpha=0.025,  # Initial learning rate
        min_alpha=0.0001  # Final learning rate
    )
    
    print(f"\nVocabulary size: {len(model.wv.key_to_index)}")
    
    # Save model
    model.save(output_file)

    print("Index to key: ", model.wv.index_to_key[:50])
    
    # Test the model
    if concept in model.wv:
        print(f"\nTesting model with concept: '{concept}'")
        similar_words = model.wv.most_similar(concept, topn=20)
        print(f"Most similar words to '{concept}':")
        for word, similarity in similar_words:
            word = word.replace("_", " ")
            print(f"{word}: {similarity:.4f}")
    else:
        print(f"\nWARNING: '{concept}' not in vocabulary!")
        print("Available similar words:", [w for w in model.wv.key_to_index if concept in w])

if __name__ == "__main__":
    Fire(main)