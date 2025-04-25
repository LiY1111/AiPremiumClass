import os 
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def read_data(dir, filename, is_out=False):
    filepath = os.path.join(dir, filename)
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().split('\n')
        for line in content:
            line = line.replace('， ', '')
            if is_out:
                line = 'BOS ' + line + 'EOS'
            data.append(line)
        data = [line for line in data if len(line)>=5]
        return data

def build_vocab(vocab_name):
    with open(vocab_name, 'r', encoding='utf-8') as f:
        words = f.read().split('\n')
        words  =['PAD', 'UNK'] + words + ['BOS', 'EOS']
        token_id = {word:i for i , word in enumerate(words)}
        return token_id

def get_process(vocab):
    def batch_process(batch):
        enc_ids, dec_ids, labels = [], [], []
        for enc, dec in batch:
            enc = enc.split(' ')
            dec = dec.split(' ')
            enc_idx = torch.tensor([vocab.get(word, 'UNK') for word in enc])
            dec_idx = torch.tensor([vocab.get(word, 'UNK') for word in dec])
            enc_ids.append(enc_idx)
            dec_ids.append(dec_idx[:-1])
            labels.append(dec_idx[1:])
        enc_ids = pad_sequence(enc_ids, batch_first=True, padding_value=0)
        dec_ids = pad_sequence(dec_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)
        return enc_ids, dec_ids, labels
    return batch_process

        


if __name__ == '__main__':
    data1 = read_data('test', 'in.txt')
    data2 = read_data('test', 'out.txt', True)
    # print(data1[:10])
    # print(data2[:10])

    vocab = build_vocab('vocabs')
    print(len(vocab))

    import json
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    
    import pickle
    with open('data.pkl', 'wb') as f:
        pickle.dump((data1, data2), f)

    dataset = list(zip(data1, data2))
    print(dataset[:2])

    dl = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=get_process(vocab))
    for enc_ids, dec_ids, labels in dl:
        print(enc_ids.shape)
        print(dec_ids.shape)
        print(labels.shape)
        break
