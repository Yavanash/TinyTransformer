import torch
import torch.nn as nn
from collections import Counter
import math
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = '/content/drive/MyDrive/transformer'

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"
MIN_FREQ = 5


class Vocab():
    def __init__(self, lang, min_freq):
        self.stoi = {PAD:0,SOS:1,EOS:2,UNK:3}
        self.itos = {0:PAD,1:SOS,2:EOS,3:UNK}
        self.freqs = {}
        self.nxt_idx = 4
        self.lang = lang
        self.min_freq = min_freq

    def build_vocab(self, corpus): #we should get the corpus as an iterable of all the words
        freqs = Counter(corpus)
        # VERY IMP: Counter can store in random order for diff runs so sort before vocab.
        self.freqs = dict(sorted(freqs.items(), key=lambda x:x[1], reverse=True)) #higher freq -> lower index
        for word, freq in self.freqs.items():
            if freq >= self.min_freq and word not in self.stoi:
                self.stoi[word] = self.nxt_idx
                self.itos[self.nxt_idx] = word
                self.nxt_idx += 1

    def encode(self, sent):
        tokens = [self.stoi[SOS]]
        tokens += [self.stoi.get(word, self.stoi[UNK]) for word in sent]
        tokens += [self.stoi[EOS]]

        return tokens

    def decode(self, batch):
        # batch = [bs, sl]
        out = []
        for seq in batch:
            words = []
            for token in seq:
                if token.item() == self.stoi[SOS] or token.item() == self.stoi[PAD]:
                    continue
                if token.item() == self.stoi[EOS]:
                    break
                words.append(self.itos.get(token.item(), UNK))
            out.append(" ".join(words))

        return out
    
    def save_vocab(self, filepath):
        with open(filepath, "w") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos, "nxt_idx": self.nxt_idx}, f)

    def load_vocab(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.stoi = data["stoi"]
            self.itos = {int(k):v for k,v in data["itos"].items()}
            self.nxt_idx = int(data["nxt_idx"])


def custom_padding(batch, pad_idx = 0):
    fra, eng = zip(*batch)

    max_fra = max(x.size(0) for x in fra)
    max_eng = max(x.size(0) for x in eng)

    fra_batch = torch.full((len(batch), max_fra), pad_idx, dtype=torch.long)
    eng_batch = torch.full((len(batch), max_eng), pad_idx, dtype=torch.long)

    for i in range(len(batch)):
        fra_batch[i, :fra[i].size(0)] = fra[i]
        eng_batch[i, :eng[i].size(0)] = eng[i]

    return fra_batch, eng_batch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divible by num heads."

        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)

    def split_heads(self, x):
        # x = [bs, sl, d] -> [bs, sl, nh, dk] -> [bs, nh, sl, dk]
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)

    def combine_heads(self, x):
        # x = [bs, nh, sl, dk] -> [bs, sl, nh, dk] -> [bs, sl, d]
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)

    def scaled_dot_product_attn(self, Q, K, V, mask=None):
        # Q = [bs, nh, sl, dk]
        alignment_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # alignment_scores = [bs, nh, sl, sl] = [bs, nh, querylen, keylen]

        if mask is not None: # apply mask
            alignment_scores = alignment_scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = torch.softmax(alignment_scores, dim=-1)
        contextual_embed = attn_weights @ V
        # contextual_embed = [bs, nh, sl, dk]

        return contextual_embed

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        embed = self.scaled_dot_product_attn(Q, K, V, mask)
        output = self.W_o(self.combine_heads(embed))

        return output
    

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()

        # PE = sin or cos(pos / 10000^(2i/d))
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(max_seq_len, d_model)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000) / d_model)) #this is the denominator in angle
        pe[:, 0::2] = torch.sin(position * div_term) # position = [msl, 1], div = [d/2]
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = embedding = [bs, sl, d]
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, ):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):
        # input x = embed + positional encoding = [bs, sl, d]
        z = self.mha(x, x, x, mask)
        z_norm1 = self.norm1(self.dropout(z) + x) # residual connections
        z = self.ffn(z_norm1)
        z_norm2 = self.norm2(self.dropout(z) + z_norm1)

        return z_norm2


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # src_mask = fra mask = padding only
        # tgt_mask = eng mask = causal + padding mask
        # non auto regressive in training

        z = self.masked_mha(x, x, x, tgt_mask)
        z1 = self.norm1(self.dropout(z) + x)
        z2 = self.cross_mha(z1, enc_output, enc_output, src_mask)
        z3 = self.norm2(self.dropout(z2) + z1)
        y = self.ffn(z3)
        out = self.norm3(self.dropout(y) + z3)

        return out
    
    
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_len, src_vocab_size, tgt_vocab_size):
        super().__init__()

        # input
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # blocks
        self.encoder_block = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_block = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

        #output
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(0.1)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        # print(type(src_mask))

        seqlen = tgt.size(1)
        no_peak_mask = (1 - torch.triu(torch.ones(1, seqlen, seqlen, device=device), diagonal=1)).bool()

        tgt_mask = tgt_mask & no_peak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):

        src_mask, tgt_mask = self.generate_mask(src, tgt)
        input_src = self.dropout(self.positional_encoding(self.src_embed(src)))
        input_tgt = self.dropout(self.positional_encoding(self.tgt_embed(tgt)))

        enc_out = input_src
        for enc_layer in self.encoder_block:
            # print(type(enc_out))
            enc_out = enc_layer(enc_out, src_mask)

        dec_out = input_tgt
        for dec_layer in self.decoder_block:
            dec_out = dec_layer(dec_out, enc_out, src_mask, tgt_mask)

        output = self.fc(dec_out)
        return output
    

def load_checkpoint(model, optimizer, checkpt_path):
    checkpt = torch.load(checkpt_path, map_location=device)
    model.load_state_dict(checkpt['model_state_dict'])
    optimizer.load_state_dict(checkpt['optimizer_state_dict'])
    epoch = checkpt['epoch']
    losses = checkpt['loss']

    return model, optimizer, epoch, losses