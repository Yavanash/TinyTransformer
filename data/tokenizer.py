import unicodedata
import pickle
from transformers import AutoTokenizer

fra_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
eng_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

filepath = "eng-fra.txt"

tokens = {"eng": [], "fra": []}

with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
        pairs = line.strip().split('\t')
        
        eng_sent, fra_sent = pairs[0], pairs[1]

        eng_sent = unicodedata.normalize("NFKC", eng_sent)
        fra_sent = unicodedata.normalize("NFKC", fra_sent)

        eng_ids = eng_tokenizer.encode(eng_sent, add_special_tokens=True)
        fra_ids = fra_tokenizer.encode(fra_sent, add_special_tokens=True)

        tokens["eng"].append(eng_ids)
        tokens["fra"].append(fra_ids)
        
with open("tokens.pkl", "wb") as f:
    pickle.dump(tokens, f)