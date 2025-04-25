import torch 
import torch.nn as nn

class Attention(nn.Module):
    def __init(self):
        super().__init__()
        
    def forward(self, enc_out, dec_out):
        a_c = torch.bmm(dec_out, enc_out.permute(0, 2, 1))
        a_c = torch.softmax(a_c, dim=-1)
        a_h = torch.bmm(a_c, enc_out)
        return a_h

class Encoder(nn.Module):
    def __init__(self, vocab_dim, emb_dim, hid_dim, dropout, mode='concat'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True, batch_first=True, dropout=dropout)
    
    def forward(self, token_ids):
        emb = self.embedding(token_ids)
        out, hid = self.rnn(emb)
        hid = torch.cat([hid[0], hid[1]], dim=1) if mode == 'concat' else torch.add(hid[0], hid[1])
        return out, hid

class Decoder(nn.Module):
    def __init__(self, vocab_dim, emb_dim, hid_dim, dropout, mode='concat'):
        super().__init__()
        self.attention = Attention()
        self.attfc = nn.Linear(hid_dim*4, hid_dim*2)
        self.embedding = nn.Embedding(vocab_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim*2, batch_first=True, dropout=dropout) if mode == 'concat' else \
                    nn.GRU(emb_dim, hid_dim, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hid_dim*2, vocab_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, hid, enc_out):
        emb = self.embedding(token_ids)
        emb = self.dropout(emb)
        dec_out, hid = self.rnn(emb, hid.unsqueeze(0))
        a_h = self.attention(enc_out, dec_out)
        out = nn.Tanh(torch.cat([a_h, dec_out], dim=2))
        out = self.attfc(out)
        out = self.fc(out)
        return out, hid

class Seq2seq(nn.Module):
    def __init__(self, enc_vocab_dim, dec_vocab_dim, emb_dim, hid_dim, dropout, mode='concat'):
        super().__init__()
        self.encoder = Encoder(enc_vocab_dim, emb_dim, hid_dim, dropout, mode)
        self.decoder = Decoder(dec_vocab_dim, emb_dim, hid_dim, dropout, mode)

    def forward(self, enc_token_ids, dec_token_ids):
        enc_out, hid = self.encoder(enc_token_ids)
        out, hid = self.decoder(dec_token_ids, hid, enc_out)
        return out, hid

if __name__ == '__main__':
    model = Seq2seq(100, 100, 128, 64, 0.5)
    enc_token_ids = torch.randint(0, 100, (32, 10))
    dec_token_ids = torch.randint(0, 100, (32, 10))
    model = Seq2seq(100, 100, 128, 64, 0.5)
    out, hid = model(enc_token_ids, dec_token_ids)
    print(out.shape)
    print(hid.shape)

