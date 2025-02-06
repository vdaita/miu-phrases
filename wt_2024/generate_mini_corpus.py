"""
Select one element from each row of the dataframe, arbitrarily to make sure it isn't biased towards one time period or another
"""

import pandas as pd
import random
from tqdm import tqdm
import re

df = pd.read_csv("company_website_second_round_with_additional_firms.csv")
corpus = ""

for index, row in tqdm(df.iterrows()):
    consideration = [element for element in row if isinstance(element, str) and len(element) > 10]
    if len(consideration) > 0:
        corpus += random.choice(consideration) + "\n"
    else:
        print("No elements found in row", index)

corpus = corpus.splitlines()
new_corpus = []

for line in corpus:
    if len(line.strip().split(" ")) > 2:
        new_corpus.append(line)

def split_capitalization(text):
    while re.search(r'([a-z])([A-Z])', text):
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text

print("Split capitalization demo: ", "HelloWorld how is it going? HiAm fine. -->", split_capitalization("HelloWorld how is it going? HiAm fine."))

corpus = "\n".join(new_corpus)
corpus = "\n".join([split_capitalization(line) for line in new_corpus])

words = corpus.split()
words = [word for word in words if len(word) < 20]
for i in tqdm(range(len(words))):
    if any(char.isdigit() for char in words[i]):
        words[i] = '[num]'

corpus = " ".join(words)

with open("mini_corpus.txt", "w") as file:
    file.write(corpus)