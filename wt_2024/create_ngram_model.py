from gensim.models import Phrases, Word2Vec
from gensim.models.phrases import Phraser
from fire import Fire
from tqdm import tqdm
import gensim
import multiprocessing
from itertools import islice

class TqdmCallback(gensim.models.callbacks.CallbackAny2Vec):
    def __init__(self):
        self.pbar = None

    def on_epoch_begin(self, model):
        if self.pbar is None:
            self.pbar = tqdm(total=model.epochs, desc="Training Word2Vec model")

    def on_epoch_end(self, model):
        self.pbar.update(1)

    def on_train_end(self, model):
        self.pbar.close()

def process_chunk(chunk):
    return [line.lower().split() for line in chunk]

def batch_reader(file_path, batch_size=10000):
    with open(file_path, "r") as f:
        while True:
            batch = list(islice(f, batch_size))
            if not batch:
                break
            yield batch

def main(input_file: str = "corpus_cleaned.txt", 
         output_file: str = "ngram_model.model", 
         concept: str = "american",
        ):
    
    with open(input_file, "r") as f:
        corpus = f.read().split(" ")
        corpus = [" ".join([word for word in corpus[i:i+30] if len(word) < 10]) for i in tqdm(range(0, len(corpus), 30))]
        corpus = [line.lower().split() for line in corpus]

    
    print(f"Corpus length: {len(corpus)}")

    # Optimize bigram detection
    bigram = Phrases(corpus, min_count=25, threshold=10)
    bigram_phraser = Phraser(bigram)
    bigram_corpus = [bigram_phraser[sentence] for sentence in corpus]
    
    # Optimize Word2Vec parameters
    model = Word2Vec(
        sentences=bigram_corpus,
        vector_size=384,
        window=5,
        min_count=25,  # Increased to reduce vocabulary size
        workers=10,  # Use all CPU cores
        batch_words=10000,  # Larger batch size
        callbacks=[TqdmCallback()],
        sg=1  # Use skip-gram (usually faster)
    )
    
    model.save(output_file)

    if concept:
        similar_words = model.wv.most_similar(concept, topn=20)
        print(f"Most similar words to '{concept}':")
        for word, similarity in similar_words:
            word = word.replace("_", " ")
            print(f"{word}: {similarity}")

if __name__ == "__main__":
    Fire(main)