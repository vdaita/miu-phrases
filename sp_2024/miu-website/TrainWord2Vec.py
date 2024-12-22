from gensim.models import Word2Vec

data = open("corpus_replaced.txt", "r").readlines()

# join 20 lines together to make one chunk
data = [" ".join(data[i:i+50]) for i in range(0, len(data), 50)]
data = [line.lower().split() for line in data]

model = Word2Vec(sentences=data, vector_size=384, window=5, min_count=1, workers=4)
model.save("word2vec.model")