from gensim.models import Phrases, Word2Vec
from gensim.models.phrases import Phraser
from fire import Fire

def main(input_file: str = "corpus_cleaned.txt", output_file: str = "ngram_model.model", concept: str = "american"):
    with open(input_file, "r") as f:
        corpus = f.read().split(" ")
    corpus = [" ".join(corpus[i:i+30]) for i in range(0, len(corpus), 30)]
    corpus = [line.lower().split() for line in corpus]

    bigram = Phrases(corpus, min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram)
    bigram_corpus = [bigram_phraser[sentence] for sentence in corpus]
    model = Word2Vec(sentences=bigram_corpus, vector_size=384, window=5, min_count=1, workers=4)
    model.save(output_file)

    if concept:
        similar_words = model.wv.most_similar(concept, topn=20)
        print(f"Most similar words to '{concept}':")
        for word, similarity in similar_words:
            word = word.replace("_", " ")
            print(f"{word}: {similarity}")

if __name__ == "__main__":
    Fire(main)