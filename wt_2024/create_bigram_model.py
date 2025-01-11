from gensim.models import Phrases, Word2Vec
from gensim.models.phrases import Phraser
from fire import Fire

def main(input_file: str = "corpus_cleaned.txt", output_file: str = "ngram_model.model"):
    with open(input_file, "r") as f:
        corpus = f.read()
    corpus = [corpus.split(" ")]
    bigram = Phrases(corpus, min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram)
    bigram_corpus = [bigram_phraser[sentence] for sentence in corpus]
    model = Word2Vec(sentences=bigram_corpus, vector_size=100, window=5, min_count=1, workers=4)
    model.save(output_file)

if __name__ == "__main__":
    Fire(main)