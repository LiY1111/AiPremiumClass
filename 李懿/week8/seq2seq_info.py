import torch
import pickle
import json
from EncoderDecoder import Seq2seq, Encoder, Decoder
from process import read_data, get_process, build_vocab
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def eval_seq2seq(seq2seq, dl):
    seq2seq.eval()
    with torch.no_grad():
        acc = []
        for enc_ids, dec_ids, labels in dl:
            out, _ = seq2seq(enc_ids, dec_ids)
            out = torch.argmax(out, dim=-1)
            acc.append((out == labels).sum().item() / labels.numel())
        
        plt.plot(acc)
        plt.xlabel('batch')
        plt.ylabel('acc')
        plt.title('acc of seq2seq on test data')
        plt.show()


if __name__ == '__main__':
   
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
    state_dict = torch.load('seq2seq_state.bin')

    in_data = read_data('test', 'in.txt')
    out_data = read_data('test', 'out.txt', True)

    dataset = list(zip(in_data, out_data))
    dl = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=get_process(vocab))
    print(dataset[:2])

    seq2seq = Seq2seq(len(vocab), len(vocab), 128, 64, 0.5)
    seq2seq.load_state_dict(state_dict)

    # eval_seq2seq(seq2seq, dl)

    # 随机生成一个句子，输入到模型中，输出翻译结果
    max_len = 20
    comment = '少壮不努力'
    comment = list(comment)
    enc_input = torch.tensor([[vocab[tk] for tk in comment]])

    dec_input = torch.tensor([[vocab['BOS']]])
    voc_inv = {v:k for k, v in vocab.items()}
    with torch.no_grad():
        tokens = []
        seq2seq.eval()
        enc_out, hid = seq2seq.encoder(enc_input)
        while True:
            if len(tokens) >= max_len:
                break
            out, hid = seq2seq.decoder(dec_input, hid, enc_out)
            out = torch.argmax(out, dim=-1)
            token = voc_inv[out.squeeze(0).item()]
            if token == 'EOS':
                break
            tokens.append(token)
            hid = hid.view(-1, hid.size(-1))
        print(''.join(tokens))
    


