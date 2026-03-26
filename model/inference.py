import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from main import Transformer, Vocab, custom_padding, load_checkpoint

import pandas as pd
from tqdm import tqdm
import string
import unicodedata

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eng_vocab = "../data/eng.json"
fra_vocab = "../data/fra.json"
filepath = "../data/test.csv"
checkpt_path = "../checkpoints/checkpt75.pth"


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
            self.eng_vocab.load_vocab(eng_vocab_path)

        with open(fra_vocab_path, 'r') as f:
            self.fra_vocab.load_vocab(fra_vocab_path)

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


class InferenceTransformer(Transformer):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_len, src_vocab_size, tgt_vocab_size):
        super().__init__(d_model, num_heads, num_layers, d_ff, max_seq_len, src_vocab_size, tgt_vocab_size)
    
    def forward(self, src, k=3):
        # src = [bs, sl]
        src = src[:, :MAX_SEQ_LEN]
        sos = torch.full((src.size(0), 1), 1, dtype=torch.long, device=device) # 1 = SOS
        src_mask, out_mask = self.generate_mask(src, sos)

        input_src = self.positional_encoding(self.src_embed(src)) # no dropout in inference
        out_sos = self.positional_encoding(self.tgt_embed(sos))

        enc_out = input_src
        for enc_layer in self.encoder_block:
            context = enc_layer(enc_out, src_mask)

        beams = [(out_sos, 1)]

        for _ in range(3):
            new_beam = []
            for out, prob in beams:
                pass

    def forward_step(self, out, prob, new_beam, enc_out, src_mask, out_mask, k):
        # input = tgt = SOS
        dec_out = out
        for dec_layer in self.decoder_block:
            dec_out = dec_layer(dec_out, enc_out, src_mask, out_mask)
        # after all layers, dec_out = [bs, 1, d] => next token predicted

        x = F.log_softmax(self.fc(dec_out))
        vals, idxs = torch.topk(x, k, dim=-1, sorted=True) 



    def beam_search(self, beams, new_token, k):
        pass
        


testdata = TestDataset(filepath, eng_vocab, fra_vocab)
test_loader = DataLoader(testdata, collate_fn=custom_padding, drop_last=True, batch_size=32, num_workers=2)

D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 512
SRC_VOCAB_SIZE = testdata.fra_vocab.nxt_idx
TGT_VOCAB_SIZE = testdata.eng_vocab.nxt_idx
MAX_SEQ_LEN = 54
MAX_LEN = 25

model = InferenceTransformer(D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LEN, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model, _, _, _ = load_checkpoint(model, optimizer, checkpt_path)

def validate(model, criterion, test_loader):
    model.eval()
    test_losses = []

    progress = tqdm(test_loader, total=len(test_loader))

    for src, tgt in progress:
        pred = model(src) # pred = [bs, sl, vocab]
        pred_loss = pred.contiguous().view(-1, TGT_VOCAB_SIZE)
        tgt_loss = tgt.contiguous().view(-1)

        loss = criterion(pred_loss, tgt_loss)
        test_losses.append(loss.item())
    
    return test_losses

def translate():
    pass

def main():
    print(next(iter(test_loader))[0].shape)
    print(model(next(iter(test_loader))[0]))

if __name__ == "__main__":
    main()