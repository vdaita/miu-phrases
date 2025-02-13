import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

def extract_phrases_from_texts(texts):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """Extract the antiforeign keywords from the following text, and return in a JSON response: {"keywords": []}. Make these phrases as clear as possible. Make the keywords as general as possible. """
            },
            {
                "role": "user",
                "content": "\n\n".join(texts)
            }
        ],
        model="gpt-4o",
        response_format={"type": "json_object"}
    )

    keywords = json.loads(response.choices[0].message.content)
    for i in range(len(keywords["keywords"])):
        keywords["keywords"][i] = ''.join(char if char.isalpha() else ' ' for char in keywords["keywords"][i]).lower()
    return keywords

if __name__ == "__main__":
    with open("antiforeign_candidate_phrases_v2-2_comparison.csv", "r") as f:
        texts = f.read()

    print("Texts: ", texts)

    chunk_size = 1000
    lines = texts.splitlines()
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    all_keywords = {"direct_extracted": {"antiforeign": []}}
    for chunk in tqdm(chunks):
        chunk_text = "\n".join(chunk)
        keywords = extract_phrases_from_texts([chunk_text])
        all_keywords["direct_extracted"]["antiforeign"].extend(keywords["keywords"])

    with open("chat_extracted_keywords.json", "w+") as f:
        json.dump(all_keywords, f)