import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim 

class PositionalEncoding(nn.Module):
    def __init__(self, dropout, emb_size, maxlen=5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2)*math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(den*pos)
        pos_embedding[:, 1::2] = torch.cos(den*pos)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
    def forward(self, token_embedding):
        return self.dropout(self.pos_embedding[:, :token_embedding.size(1)] + token_embedding)
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 enc_vocab_size, dec_vocab_size,
                 dim_feedforward, dropout):
        super().__init__()
        self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=num_encoder_layers, 
                                         num_decoder_layers=num_decoder_layers, 
                                         nhead=nhead,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout, 
                                        )
        self.enc_emb = nn.Embedding(enc_vocab_size, d_model)
        self.dec_emb = nn.Embedding(dec_vocab_size, d_model)
        self.predict = nn.Linear(d_model, dec_vocab_size)
        self.pos_encoding = PositionalEncoding(0.2, d_model)

    def forward(self, enc_inp, dec_inp, dec_mask, enc_pad_mask, dec_pad_mask):
        enc_emb = self.enc_emb(enc_inp)
        dec_emb = self.dec_emb(dec_inp)

        enc_emb = enc_emb.transpose(0, 1)
        dec_emb = dec_emb.transpose(0, 1)

        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=dec_mask, src_key_padding_mask=enc_pad_mask, tgt_key_padding_mask=dec_pad_mask)
        return self.predict(outs)
    
    def encoder(self, enc_inp):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        out = self.transformer.encoder(enc_emb)
        return self.predict(out)
    
    def decoder(self, dec_inp, memory, dec_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        out = self.transformer.decoder(dec_emb, memory, dec_mask)
        return self.predict(out)

def build_vocab(tokens):
    special_token = ['<pad>', '<s>', '</s>', '<unk>']
    vocab = {token: idx for idx, token in enumerate(special_token)}
    for token in set(tokens):
        vocab[token] = len(vocab)
    return vocab

def get_proc(batch):
    enc_ids, dec_ids = [],[]
    for enc, dec in batch:
        enc_idx = torch.tensor([vocab[tk] for tk in enc], dtype=torch.int64)
        dec_idx = torch.tensor([vocab[tk] for tk in dec], dtype=torch.int64)
        enc_ids.append(enc_idx)
        dec_ids.append(dec_idx)
    enc_ids = pad_sequence(enc_ids, batch_first=True, padding_value=0)
    dec_ids = pad_sequence(dec_ids, batch_first=True, padding_value=0)
    return enc_ids, dec_ids

def generate_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 1, float(0.0)).masked_fill(mask == 0, float('-inf'))
    return mask

def create_mask(enc_inp, dec_inp):
    dec_seq_size = dec_inp.size(1)
    dec_mask = generate_mask(dec_seq_size)

    enc_pad_mask = enc_inp == 0
    dec_pad_mask = dec_inp == 0
    return dec_mask, enc_pad_mask, dec_pad_mask


if __name__ == '__main__':
    chs = '天生我才必有用，莫使金樽空对月'

    enc_list, dec_list = [],[]
    chs = list(chs)
    for i in range(len(chs) - 1):
        # print(chs[:i+1], chs[i+1:])
        enc_list.append(chs[:i+1])
        dec_list.append(['<s>']+ chs[i+1:] + ['</s>'])

    vocab = build_vocab(chs)
    
    dataset = list(zip(enc_list, dec_list))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=get_proc)  

    model = Seq2SeqTransformer(216, 3, 2, 2, len(vocab), len(vocab), 216, 0.3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        for enc, dec in dataloader:
            dec_input = dec[:, :-1]
            dec_target = dec[:, 1:]
            dec_mask, enc_pad_mask, dec_pad_mask = create_mask(enc, dec_input)
            out = model(enc, dec_input, dec_mask, enc_pad_mask, dec_pad_mask)
            loss = criterion(out.view(-1, out.size(-1)), dec_target.contiguous().view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'epoch: {epoch +1}, loss: {loss}')

    import json
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f)

    torch.save(model.state_dict(), 'model_state.bin')