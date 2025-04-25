import torch
import torch.optim as optim 
import torch.nn as nn
import json
import pickle
from EncoderDecoder import Seq2seq
from process import get_process
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.tensorboard as tqbar

if __name__ == '__main__':
    with open('data.pkl', 'rb') as f:
        in_data, out_data = pickle.load(f)
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
    
    writer = SummaryWriter()

    emb_dim = 128
    hid_dim = 64
    dropout = 0.5
    batch_size = 32
    seq2seq = Seq2seq(len(vocab), len(vocab), emb_dim, hid_dim, dropout)
    optimizer = optim.Adam(seq2seq.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = list(zip(in_data, out_data))
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=get_process(vocab))

    modes = ['concat', 'add']
    for mode in modes:
        for epoch in range(10):
            seq2seq.train()
            epoch_loss = []
            tqbar = tqdm(dl)
            for enc_ids, dec_ids, labels in tqbar:
                optimizer.zero_grad()
                out, _ = seq2seq(enc_ids, dec_ids)
                loss = criterion(out.view(-1, out.size(-1)), labels.view(-1))
                epoch_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            avg_loss = np.mean(epoch_loss)
            writer.add_scalar(f'loss_{mode}', avg_loss, epoch)
            print(f"epoch {epoch+1}, loss: {avg_loss:.4f}")
        torch.save(seq2seq.state_dict(), f'{mode}_seq2seq_state.bin')
        writer.close()



        


