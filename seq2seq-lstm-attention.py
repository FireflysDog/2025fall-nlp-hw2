from sacrebleu.metrics import BLEU
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import argparse, time
import numpy as np

# 加载数据
def load_data(num_train):
    zh_sents, en_sents = {}, {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for split in ['train', 'val', 'test']:
        zh_sents[split], en_sents[split] = [], []
        with open(os.path.join(base_dir, "data", f"zh_en_{split}.txt"), encoding='utf-8') as f:
            for line in f:
                zh, en = line.strip().split("\t")
                zh_sents[split].append(zh.split())
                en_sents[split].append(en.split())
    if num_train != -1:
        zh_sents['train'] = zh_sents['train'][:num_train]
        en_sents['train'] = en_sents['train'][:num_train]
    print("训练集 验证集 测试集大小:",
          len(zh_sents['train']), len(zh_sents['val']), len(zh_sents['test']))
    return zh_sents, en_sents


# 词表
class Vocab:
    def __init__(self):
        self.word2idx, self.idx2word = {}, []
        for w in ["[BOS]", "[EOS]", "[UNK]", "[PAD]"]:
            self.add_word(w)

    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)

    def add_sent(self, sent):
        for w in sent:
            self.add_word(w)

    def index(self, w):
        return self.word2idx.get(w, self.word2idx["[UNK]"])

    def encode(self, sent, max_len):
        return [self.word2idx["[BOS]"]] + \
               [self.index(w) for w in sent][:max_len] + \
               [self.word2idx["[EOS]"]]

    def decode(self, ids, strip=True):
        res = []
        for i in ids:
            w = self.idx2word[i]
            if strip and w in ["[BOS]", "[EOS]", "[PAD]"]:
                continue
            res.append(w)
        return res

    def __len__(self):
        return len(self.idx2word)


# 合并门控的 LSTMCell
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 输入和隐状态的线性映射合并
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, state):
        h_prev, c_prev = state
        gates = self.W_x(x) + self.W_h(h_prev)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


# Encoder
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = LSTMCell(emb_dim, hidden_size)

    def forward(self, src, state):
        emb = self.dropout(self.embed(src))
        h, c = self.rnn(emb, state)
        return h, c


# Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        # Using General Attention: score(h_t, h_s) = h_t^T W h_s
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch_size, hidden_size] (current decoder hidden state)
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        # mask: [batch_size, seq_len] (True for padding)
        
        # Project encoder outputs
        # energy: [batch_size, seq_len, hidden_size]
        # 计算注意力分数: [batch, seq_len]
        # hidden.unsqueeze(2): [batch, hidden, 1]
        # 使用 mask 屏蔽 PAD
        energy = self.attn(encoder_outputs) 
        
        # Calculate scores: [batch_size, seq_len]
        # hidden.unsqueeze(2): [batch_size, hidden_size, 1]
        scores = torch.bmm(energy, hidden.unsqueeze(2)).squeeze(2)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        return F.softmax(scores, dim=1)


# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = Attention(hidden_size)
        self.rnn = LSTMCell(emb_dim + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, state, enc_outs, mask):
        emb = self.dropout(self.embed(x))
        h_prev, c_prev = state
        attn_w = self.attn(h_prev, enc_outs, mask)
        ctx = torch.bmm(attn_w.unsqueeze(1), enc_outs).squeeze(1)
        rnn_input = torch.cat([emb, ctx], dim=1)
        h, c = self.rnn(rnn_input, state)
        logits = self.out(torch.cat([h, ctx], dim=1))
        return F.log_softmax(logits, dim=-1), (h, c)


# Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim, hidden_size,
                 max_len, src_pad, tgt_pad, dropout, tf_ratio, beam_size):
        super().__init__()
        self.encoder = EncoderRNN(len(src_vocab), emb_dim, hidden_size, dropout)
        self.decoder = DecoderRNN(len(tgt_vocab), emb_dim, hidden_size, dropout)
        self.src_pad = src_pad
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.tf_ratio = tf_ratio
        self.beam_size = beam_size
        self.hidden_size = hidden_size

    def init_state(self, bs, device):
        return (torch.zeros(bs, self.hidden_size, device=device),
                torch.zeros(bs, self.hidden_size, device=device))

    def forward(self, src, tgt):
        bs, ls = src.size()
        state = self.init_state(bs, src.device)
        enc_outs = []
        for i in range(ls):
            state = self.encoder(src[:, i], state)
            enc_outs.append(state[0])
        enc_outs = torch.stack(enc_outs, 1)
        mask = (src == self.src_pad)

        outputs = []
        inp = tgt[:, 0]
        for t in range(1, tgt.size(1)):
            out, state = self.decoder(inp, state, enc_outs, mask)
            outputs.append(out)
            use_tf = torch.rand(bs, device=src.device) < self.tf_ratio
            inp = torch.where(use_tf, tgt[:, t], out.argmax(-1))
        return torch.stack(outputs, 1)

    def predict(self, src, alpha=0.7):
        # 仅支持 batch=1 的 beam search
        state = self.init_state(1, src.device)
        enc_outs = []
        for i in range(src.size(1)):
            state = self.encoder(src[:, i], state)
            enc_outs.append(state[0])
        enc_outs = torch.stack(enc_outs, 1)
        mask = (src == self.src_pad)

        bos = self.tgt_vocab.index("[BOS]")
        eos = self.tgt_vocab.index("[EOS]")
        beams = [(0.0, [bos], state)]

        for _ in range(self.max_len):
            new_beams = []
            for score, seq, st in beams:
                if seq[-1] == eos:
                    new_beams.append((score, seq, st))
                    continue
                inp = torch.tensor([seq[-1]], device=src.device)
                out, new_state = self.decoder(inp, st, enc_outs, mask)
                logp, idx = torch.topk(out.squeeze(0), self.beam_size)
                for lp, i in zip(logp.tolist(), idx.tolist()):
                    length_pen = ((5 + len(seq)) / 6) ** alpha
                    new_beams.append(
                        ((score + lp) / length_pen, seq + [i], new_state)
                    )
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:self.beam_size]
        return torch.tensor(beams[0][1], device=src.device).unsqueeze(0)


# 训练 & 测试
def train_loop(model, opt, crit, loader, device):
    model.train()
    loss_sum = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        opt.zero_grad()
        out = model(src, tgt)
        loss = crit(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item()
    return loss_sum / len(loader)


def test_loop(model, loader, tgt_vocab, device):
    model.eval()
    bleu = BLEU(force=True, effective_order=True)
    hypotheses, references = [], []

    for src, tgt in tqdm(loader):
        B = len(src)
        for i in range(B):
            _src = src[i].unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model.predict(_src)

            ref = " ".join(tgt_vocab.decode(tgt[i].tolist(), strip=True))
            hypo = " ".join(tgt_vocab.decode(outputs[0].cpu().tolist(), strip=True))

            references.append(ref)
            hypotheses.append(hypo)

    score = bleu.corpus_score(hypotheses, [references]).score

    return hypotheses, references, score


# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_train', type=int, default=-1)
    args = parser.parse_args()

    # 1. 读取数据
    zh_sents, en_sents = load_data(args.num_train)

    # 2. 构建词表
    zh_vocab, en_vocab = Vocab(), Vocab()
    for zh, en in zip(zh_sents['train'], en_sents['train']):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)

    print("中文词表大小:", len(zh_vocab))
    print("英文词表大小:", len(en_vocab))

    src_pad = zh_vocab.word2idx["[PAD]"]
    tgt_pad = en_vocab.word2idx["[PAD]"]

    # 3. 构建 DataLoader
    def pad(ids, max_len, pad_id):
        ids = ids[:max_len + 2]
        res = [pad_id] * (max_len + 2)
        res[:len(ids)] = ids
        return torch.LongTensor(res)

    def collate(batch):
        src, tgt = zip(*batch)
        return torch.stack(src), torch.stack(tgt)

    loaders = {}
    for split in ['train', 'val', 'test']:
        data = [
            (pad(zh_vocab.encode(z, args.max_len), args.max_len, src_pad),
             pad(en_vocab.encode(e, args.max_len), args.max_len, tgt_pad))
            for z, e in zip(zh_sents[split], en_sents[split])
        ]
        loaders[split] = torch.utils.data.DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=(split == 'train'),
            collate_fn=collate
        )

    trainloader = loaders['train']
    validloader = loaders['val']
    testloader = loaders['test']

    # 4. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(
        zh_vocab,
        en_vocab,
        emb_dim=256,
        hidden_size=256,
        max_len=args.max_len,
        src_pad=src_pad,
        tgt_pad=tgt_pad,
        dropout=0.1,
        tf_ratio=0.9,
        beam_size=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    weights = torch.ones(len(en_vocab), device=device)
    weights[tgt_pad] = 0
    criterion = nn.NLLLoss(weight=weights)

    # 5. 训练循环
    best_bleu = 0.0
    for epoch in range(args.num_epoch):
        loss = train_loop(model, optimizer, criterion, trainloader, device)
        hypotheses, references, bleu_score = test_loop(
            model, validloader, en_vocab, device)

        print(
            f"Epoch {epoch}: "
            f"loss = {loss:.6f}, "
            f"valid bleu = {bleu_score:.6f}, "
            f"teacher forcing = {model.tf_ratio:.3f}"
        )

        print(references[0])
        print(hypotheses[0])

        if bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save(model.state_dict(), "model_lstm_att_best.pt")

    model.load_state_dict(torch.load("model_lstm_att_best.pt"))
    hypotheses, references, bleu_score = test_loop(
        model, testloader, en_vocab, device)

    print(f"Test bleu = {bleu_score}")
    print(references[0])
    print(hypotheses[0])