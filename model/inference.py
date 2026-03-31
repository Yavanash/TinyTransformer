import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from main import Transformer, Vocab, load_checkpoint
from nltk.translate.bleu_score import corpus_bleu

import pandas as pd
from tqdm import tqdm
import string
from pickle import dump
import unicodedata

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eng_vocab = "../data/eng.json"
fra_vocab = "../data/fra.json"
filepath = "../data/test.csv"
checkpt_path = "../checkpoints/checkpt125.pth"
# filepath = "test.csv"
# eng_vocab = "eng.json"
# fra_vocab = "fra.json"
# checkpt_path = "checkpt100.pth"

class TestDataset(Dataset):
    def __init__(self, filepath, eng_vocab_path, fra_vocab_path):
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
        eng_tokens = self.eng_vocab.encode(self.df["eng"][idx].split())
        fra_tokens = self.fra_vocab.encode(self.df["fra"][idx].split())
        
        fra_tensor = torch.tensor(fra_tokens, dtype=torch.long)

        return fra_tensor, eng_tokens
    
    def preprocess(self, sent):
        sent = unicodedata.normalize("NFKC", sent)
        TABLE = str.maketrans("","",string.punctuation.replace("'",""))
        return sent.lower().translate(TABLE)


def custom_padding(batch, pad_idx = 0):
    fra, eng = zip(*batch)
    batch_size = len(batch)

    max_fra = max(x.size(0) for x in fra)
    max_eng = max(len(x) for x in eng)

    fra_batch = torch.full((batch_size, max_fra), pad_idx, dtype=torch.long)
    eng_batch = [[pad_idx for _ in range(max_eng)] for _ in range(batch_size)]

    for i in range(batch_size):
        fra_batch[i, :fra[i].size(0)] = fra[i]
        eng_batch[i][:len(eng[i])] = eng[i]

    return fra_batch, eng_batch

testdata = TestDataset(filepath, eng_vocab, fra_vocab)
test_loader = DataLoader(testdata, collate_fn=custom_padding, drop_last=True, batch_size=32, num_workers=2)


class InferenceTransformer(Transformer):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_len, src_vocab_size, tgt_vocab_size):
        super().__init__(d_model, num_heads, num_layers, d_ff, max_seq_len, src_vocab_size, tgt_vocab_size)
    
    def forward(self, src, k=3):
        src = src[:, :MAX_SEQ_LEN]
        dec_in = torch.full((src.size(0), 1), 1, dtype=torch.long, device=device)
        src_mask, _ = self.generate_mask(src, dec_in)

        input_src = self.positional_encoding(self.src_embed(src))
        enc_out = input_src
        for enc_layer in self.encoder_block:
            enc_out = enc_layer(enc_out, src_mask)

        initial_probs = torch.zeros(src.size(0), dtype=torch.float, device=device)
        beams = [(dec_in, initial_probs)]
        completed = []

        for _ in range(MAX_LEN):
            active = []
            new_beam = []

            for inpt, prob in beams:
                if inpt[0, -1].item() == 2:  # 2 = EOS token index
                    completed.append((inpt, prob))
                else:
                    active.append((inpt, prob))

            if not active:  # all beams have hit EOS
                break

            for inpt, prob in active:
                self.forward_step(src, inpt, prob, new_beam, enc_out, src_mask, k)

            beams = sorted(new_beam, key=lambda x: x[1].sum().item() / x[0].size(1), reverse=True)[:k]

        # prefer completed beams, fall back to active ones if EOS never appeared
        final = completed if completed else beams
        return sorted(final, key=lambda x: x[1].sum().item() / x[0].size(1), reverse=True)[:k]

    def forward_step(self, src, inpt, prob, new_beam, enc_out, src_mask, k):
        # inpt = [bs, sl]
        out = self.positional_encoding(self.tgt_embed(inpt))
        _, out_mask = self.generate_mask(src, inpt)
        dec_out = out

        for dec_layer in self.decoder_block:
            dec_out = dec_layer(dec_out, enc_out, src_mask, out_mask)
        
        dec_out = self.fc(dec_out)
        final_token = dec_out[:, -1, :] # = [bs, vocab]

        vals, idxs = torch.topk(F.log_softmax(final_token, dim=-1), k, dim=-1, largest=True) 
        # idxs = [bs, k], vals = [bs, k], prob = [bs]
        for i in range(k):
            new_inpt = torch.cat((inpt, idxs[:,i].unsqueeze(-1)), dim=-1)
            new_prob = prob + vals[:,i]
            new_beam.append((new_inpt, new_prob))     


D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 512
SRC_VOCAB_SIZE = testdata.fra_vocab.nxt_idx
TGT_VOCAB_SIZE = testdata.eng_vocab.nxt_idx
MAX_SEQ_LEN = 55
MAX_LEN = 10

tgt_vocab = testdata.eng_vocab
src_vocab = testdata.fra_vocab
preprocess = testdata.preprocess

model = InferenceTransformer(D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, MAX_SEQ_LEN, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)

model, _, _ = load_checkpoint(model, checkpt_path)

model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def calc_bleu_score(model, test_loader, tgt_vocab = tgt_vocab):
    generated = []
    references = []

    progress = tqdm(test_loader, total=len(test_loader))
    for src, tgt in progress:
        src = src.to(device)
        beams = model(src)

        # print(beams)
        # break ; there are multiple outputs so select only top
        outputs = tgt_vocab.decode(beams[0][0]) # beams = list of ( [bs,sl], [bs] )
        generated.extend(outputs)
        references.extend([seq for seq in tgt]) # tgt = [bs, sl]
    
    bleu_score = corpus_bleu(references, generated)
    return bleu_score

def translate(model, src, src_vocab = src_vocab, tgt_vocab = tgt_vocab, preprocess = preprocess):
    pre = preprocess(src).split()
    tokens = src_vocab.encode(pre)
    tensor = torch.tensor(tokens, dtype=torch.long, device=device)

    beams = model(tensor.unsqueeze(0))
    outputs = [tgt_vocab.decode(out) for out, _ in beams]
    return outputs

def main():
    model.eval()
    # src = input("Enter sentence to translate: ")
    # # src = ["Une table pour deux, s'il vous plaît", "Comment vous appelez-vous ?", "Où puis-je trouver un restaurant ?"]
    # with torch.no_grad():
    #     outputs = translate(model, src)
    # print(outputs)
    bleu_score = calc_bleu_score(model, test_loader)
    print(bleu_score)

if __name__ == "__main__":
    main()