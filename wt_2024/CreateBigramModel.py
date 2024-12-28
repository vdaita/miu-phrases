from gensim.models import Word2Vec
from gensim.utils import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

corpus = open("corpus_replaced.txt", "r").read()
corpus = list(tokenize(corpus, lowercase=True, deacc=True))

print(corpus[:10])

bigrams = []
for i in range(len(corpus) - 1):
    bigram = (corpus[i], corpus[i + 1])
    bigrams.append(bigram)

model = Word2Vec(bigrams, vector_size=384, window=5, min_count=1, workers=4)
model.save("bigram_model.model")