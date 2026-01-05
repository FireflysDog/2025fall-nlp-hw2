from sacrebleu.metrics import BLEU
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from tqdm import tqdm
import argparse, time
import numpy as np

from torch import Tensor
from torch.cuda.amp import autocast, GradScaler


# 加载数据
def load_data(num_train):
    zh_sents = {}
    en_sents = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for split in ['train', 'val', 'test']:
        zh_sents[split] = []
        en_sents[split] = []
        file_path = os.path.join(base_dir, "data", f"zh_en_{split}.txt")
        with open(file_path, encoding='utf-8') as f:
            for line in f.readlines():
                zh, en = line.strip().split("\t")
                zh = zh.split()
                en = en.split()
                zh_sents[split].append(zh)
                en_sents[split].append(en)
    num_train = len(zh_sents['train']) if num_train==-1 else num_train
    zh_sents['train'] = zh_sents['train'][:num_train]
    en_sents['train'] = en_sents['train'][:num_train]
    print("训练集 验证集 测试集大小分别为", len(zh_sents['train']), len(zh_sents['val']), len(zh_sents['test']))
    return zh_sents, en_sents


# 构建词表
class Vocab():
    def __init__(self):
        self.word2idx = {}
        self.word2cnt = {}
        self.idx2word = []
        self.add_word("[BOS]")
        self.add_word("[EOS]")
        self.add_word("[UNK]")
        self.add_word("[PAD]")
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2cnt[word] = 0
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        self.word2cnt[word] += 1
    
    def add_sent(self, sent):
        for word in sent:
            self.add_word(word)
    
    def index(self, word):
        return self.word2idx.get(word, self.word2idx["[UNK]"])
    
    def encode(self, sent, max_len):
        return [self.word2idx["[BOS]"]] + [self.index(word) for word in sent][:max_len] + [self.word2idx["[EOS]"]]
    
    def decode(self, encoded, strip_bos_eos_pad=False):
        return [self.idx2word[_] for _ in encoded if not strip_bos_eos_pad or self.idx2word[_] not in ["[BOS]", "[EOS]", "[PAD]"]]
    
    def __len__(self):
        return len(self.idx2word)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# Transformer Seq2Seq
class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=512, dropout=0.1, max_len=100, pad_id=3):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len+2)
        
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # 权重共享（重要）
        self.generator.weight = self.tgt_embed.weight
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    def create_mask(self, src, tgt):
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        src_padding_mask = (src == self.pad_id)
        tgt_padding_mask = (tgt == self.pad_id)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer(
            src_emb, tgt_emb,
            src_mask, tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask
        )
        return self.generator(out)

    def predict(self, src, max_len, start_symbol, end_symbol, beam_size=1):
        src_padding_mask = (src == self.pad_id)
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        if beam_size <= 1:
            # 贪心解码
            ys = torch.ones(1, 1).fill_(start_symbol).long().to(src.device)
            for _ in range(max_len):
                tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(src.device)
                tgt_emb = self.pos_encoder(self.tgt_embed(ys) * math.sqrt(self.d_model))
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
                prob = self.generator(out[:, -1])
                next_word = prob.argmax(dim=1).item()
                ys = torch.cat([ys, torch.tensor([[next_word]], device=src.device)], dim=1)
                if next_word == end_symbol:
                    break
            return ys

        # 束搜索 (Beam Search)
        beams = [(0.0, [start_symbol])]
        completed_beams = []
        
        for _ in range(max_len):
            new_beams = []
            for score, seq in beams:
                if seq[-1] == end_symbol:
                    completed_beams.append((score, seq))
                    continue
                
                ys = torch.tensor([seq], device=src.device).long()
                tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(src.device)
                tgt_emb = self.pos_encoder(self.tgt_embed(ys) * math.sqrt(self.d_model))
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
                
                log_probs = F.log_softmax(self.generator(out[:, -1]), dim=-1).squeeze(0)
                topk_probs, topk_ids = torch.topk(log_probs, beam_size)
                
                for lp, idx in zip(topk_probs, topk_ids):
                    new_beams.append((score + lp.item(), seq + [idx.item()]))
            
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
            if not beams: break

        all_candidates = completed_beams + beams
        if not all_candidates:
            return torch.tensor([[start_symbol, end_symbol]], device=src.device)
            
        best_seq = max(all_candidates, key=lambda x: x[0] / (len(x[1])**0.7))[1]
        return torch.tensor([best_seq], device=src.device)


# padding & dataloader
def collate(data_list):
    return torch.stack([torch.LongTensor(_[0]) for _ in data_list]), torch.stack([torch.LongTensor(_[1]) for _ in data_list])

def padding(inp_ids, max_len, pad_id):
    max_len += 2
    ids_ = np.ones(max_len, dtype=np.int32) * pad_id
    max_len = min(len(inp_ids), max_len)
    ids_[:max_len] = inp_ids
    return ids_

def create_dataloader(zh_sents, en_sents, max_len, batch_size, pad_id):
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        shuffle = split == 'train'
        datas = [(padding(zh_vocab.encode(zh, max_len), max_len, pad_id),
                  padding(en_vocab.encode(en, max_len), max_len, pad_id))
                 for zh, en in zip(zh_sents[split], en_sents[split])]
        dataloaders[split] = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return dataloaders['train'], dataloaders['val'], dataloaders['test']


# 训练
def train_loop(model, optimizer, criterion, loader, device, scaler):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(loader):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input, tgt_out = tgt[:, :-1], tgt[:, 1:]

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_out.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(loader)


def test_loop(model, loader, tgt_vocab, device, beam_size=1):
    model.eval()
    bleu = BLEU(force=True)
    hyps, refs = [], []
    for src, tgt in tqdm(loader):
        for i in range(len(src)):
            with torch.no_grad():
                out = model.predict(
                    src[i:i+1].to(device),
                    max_len=args.max_len+2,
                    start_symbol=tgt_vocab.word2idx["[BOS]"],
                    end_symbol=tgt_vocab.word2idx["[EOS]"]
                )
            refs.append(" ".join(tgt_vocab.decode(tgt[i].tolist(), True)))
            hyps.append(" ".join(tgt_vocab.decode(out[0].tolist(), True)))
    return hyps, refs, bleu.corpus_score(hyps, [refs]).score


# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train', type=int, default=-1)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--beam_size', type=int, default=4)
    args = parser.parse_args()

    zh_sents, en_sents = load_data(args.num_train)

    zh_vocab, en_vocab = Vocab(), Vocab()
    for zh, en in zip(zh_sents['train'], en_sents['train']):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)

    pad_id = zh_vocab.word2idx['[PAD]']
    trainloader, validloader, testloader = create_dataloader(zh_sents, en_sents, args.max_len, args.batch_size, pad_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerSeq2Seq(len(zh_vocab), len(en_vocab), pad_id=pad_id).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')

    weights = torch.ones(len(en_vocab)).to(device)
    weights[en_vocab.word2idx['[PAD]']] = 0
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=pad_id, label_smoothing=0.1)

    start_time = time.time()
    best_score = 0
    patience = 5
    no_improve = 0

    for epoch in range(args.num_epoch):
        loss = train_loop(model, optimizer, criterion, trainloader, device, scaler)
        hyps, refs, bleu = test_loop(model, validloader, en_vocab, device)
        
        scheduler.step(bleu)
        
        if bleu > best_score:
            best_score = bleu
            no_improve = 0
            torch.save(model.state_dict(), "model_transformer_best.pt")
        else:
            no_improve += 1

        print(f"Epoch {epoch}: loss = {loss}, valid bleu = {bleu}")
        print(f"Ref: {refs[0]}")
        print(f"Hyp: {hyps[0]}")
        
        if no_improve >= patience:
            print("Early stopping!")
            break

    model.load_state_dict(torch.load("model_transformer_best.pt"))
    hyps, refs, bleu = test_loop(model, testloader, en_vocab, device)
    print(f"Test bleu = {bleu:f}")
    print(f"Ref: {refs[0]}")
    print(f"Hyp: {hyps[0]}")
    print(f"Training time: {round((time.time() - start_time)/60, 2)}min")
