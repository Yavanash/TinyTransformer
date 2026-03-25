import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from main import Transformer, Vocab, custom_padding, load_checkpoint

import pandas as pd
import string
import unicodedata

eng_vocab = "../data/eng.json"
fra_vocab = "../data/fra.json"
filepath = "../data/test.csv"
checkpt_path = "../checkpoints/checkpt9.pth"

class TestDataset(Dataset):
    def __init__(self, filepath, eng_vocab_path=None, fra_vocab_path=None):
        super().__init__()

        test = pd.read_csv(filepath, encoding="utf-8")
        self.df = test.rename(columns={"English words/sentences": "eng", "French words/sentences": "fra"})
        self.df["eng"] = self.df["eng"].apply(self.preprocess)
        self.df["fra"] = self.df["fra"].apply(self.preprocess)
        self.fra_vocab = Vocab(lang="fra", min_freq=5)
        self.eng_vocab = Vocab(lang="eng", min_freq=5)

        with open(eng_vocab_path, 'r') as f:
            data = self.eng_vocab.load_vocab(eng_vocab_path)
            self.eng_vocab.stoi = data["stoi"]
            self.eng_vocab.itos = data["itos"]
            self.eng_vocab.nxt_idx = data["nxt_idx"]

        with open(fra_vocab_path, 'r') as f:
            data = self.fra_vocab.load_vocab(fra_vocab_path)
            self.fra_vocab.stoi = data["stoi"]
            self.fra_vocab.itos = data["itos"]
            self.fra_vocab.nxt_idx = data["nxt_idx"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        eng_tokens = self.eng_vocab.encode(self.df["eng"][idx])
        fra_tokens = self.fra_vocab.encode(self.df["fra"][idx])

        eng_tensor = torch.tensor(eng_tokens, dtype=torch.long)
        fra_tensor = torch.tensor(fra_tokens[1:], dtype=torch.long)

        return fra_tensor, eng_tensor 
    
    def preprocess(self, sent):
        sent = unicodedata.normalize("NFKC", sent)
        TABLE = str.maketrans("","",string.punctuation.replace("'",""))
        return sent.lower().translate(TABLE)
    
testdata = TestDataset(filepath, eng_vocab, fra_vocab)
test_loader = DataLoader(testdata, collate_fn=custom_padding, drop_last=True, batch_size=32, num_workers=2)

D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 512
SRC_VOCAB_SIZE = testdata.fra_vocab.nxt_idx
TGT_VOCAB_SIZE = testdata.eng_vocab.nxt_idx
MAX_SEQ_LEN = 25

model = Transformer(D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LEN, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model, optimizer, epoch, losses = load_checkpoint(model, optimizer, checkpt_path)

def main():
    print(next(iter(test_loader)))

if __name__ == "__main__":
    main()