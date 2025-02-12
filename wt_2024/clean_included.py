from generate_best_ngrams import remove_substrings
import json

file = "manual_keywords_fuzzy.json"
with open(file, "r") as f:
    keywords_original = json.load(f)
    for key1 in keywords_original:
        for key2 in keywords_original[key1]:
            keywords_original[key1][key2] = remove_substrings(keywords_original[key1][key2])
    json.dump(keywords_original, open(file, "w+"), indent=4)