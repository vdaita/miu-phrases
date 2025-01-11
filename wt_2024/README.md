# Calculating related keyword using single words and bigrams

## Clean the compiled corpus text
```python clean_corpus.py```

## Generate a word2vec model based on the cleaned text. 
```python create_ngram_model.py```

## Then, generate the keywords (after stopwords removal and everything)
```python generate_ngram_vectors.py```

## Remove redundancies in the CSV file
```python clear_similar_from_csv.py```

## Remove stopwords from the text in those CSV files
```python remove_stopwords_from_csv.py```

## Produce the counts and graphs
```python generate_graphs.py```