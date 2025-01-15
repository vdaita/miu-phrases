# Calculating related keyword using single words and bigrams

## Remove redundancies in the CSV file
```python clear_similar_from_csv.py```

## Generate the corpus from the cleaned CSV file
```python generate_corpus_from_cleaned_file.py```

## Clean the compiled corpus text
```python clean_corpus.py```

## Generate a word2vec model based on the cleaned text. 
```python create_ngram_model.py```

## Then, generate the keywords (after stopwords removal and everything)
```python generate_ngram_vectors.py```


## Remove stopwords from the text in those CSV files
```python remove_stopwords_from_csv.py```

## Produce the counts and graphs
This will automatically save in ./graph_data and ./keyword_charts, so those folders should be created beforehand.
```python generate_graphs.py```