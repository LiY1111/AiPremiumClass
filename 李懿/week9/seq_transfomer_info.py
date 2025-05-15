from seq_transformer import Seq2SeqTransformer, generate_mask
import json
import torch


def process_input(text_input):
    tokens = list(text_input)
    tokens_ids = [vocab[tk] for tk in tokens]
    return torch.tensor(tokens_ids, dtype=torch.int64).unsqueeze(0)


def generate(model, text_input, maxlen=50):
    model.eval()
    token_input = process_input(text_input)
    memory = model.encoder(token_input)

    id2token = {id: tk for tk, id in vocab.items()}

    dec_input = torch.tensor([[vocab['<s>']]], dtype=torch.int64)
    generate_ids = []
    for _ in range(maxlen):
        with torch.no_grad():
            mask = generate_mask(dec_input.size(1))

            out = model.decoder(dec_input, memory, mask)
            next_token_logits = torch.argmax(out, dim=-1)
            next_token_id = next_token_logits[0]

            if id2token[next_token_id.item()] == '</s>': break
            generate_ids.append(next_token_id.itme())

            dec_input = torch.cat([dec_input, next_token_id], dim=1)
    generate_text = ''.join([id2token[id] for id in generate_ids])
    return generate_text



if __name__ == '__main__':
    with open('vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    #216, 3, 2, 2, len(vocab), len(vocab), 216, 0.3
    d_model = 216
    nhead = 3
    num_encoder_layers = 2
    num_decoder_layers = 2
    vocab_size = len(vocab)
    dim_feedforward = 216
    dropout = 0.3

    model = Seq2SeqTransformer(d_model, nhead, num_encoder_layers, num_decoder_layers, vocab_size, dim_feedforward, dim_feedforward, dropout)
    model.load_state_dict(torch.load('model_state.bin'))

    text_input = '天生我才'

    gen_text = generate(model, text_input)
    print(f'预测输出：{gen_text}')