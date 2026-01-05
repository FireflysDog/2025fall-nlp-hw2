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
        """
        将单词word加入到词表中
        """
        if word not in self.word2idx:
            self.word2cnt[word] = 0
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        self.word2cnt[word] += 1
    
    def add_sent(self, sent):
        """
        将句子sent中的每一个单词加入到词表中
        sent是由单词构成的list
        """
        for word in sent:
            self.add_word(word)
    
    def index(self, word):
        """
        若word在词表中则返回其下标，否则返回[UNK]对应序号
        """
        return self.word2idx.get(word, self.word2idx["[UNK]"])
    
    def encode(self, sent, max_len):
        """
        在句子sent的首尾分别添加BOS和EOS之后编码为整数序列
        """
        encoded = [self.word2idx["[BOS]"]] + [self.index(word) for word in sent][:max_len] + [self.word2idx["[EOS]"]]
        return encoded
    
    def decode(self, encoded, strip_bos_eos_pad=False):
        """
        将整数序列解码为单词序列
        """
        return [self.idx2word[_] for _ in encoded if not strip_bos_eos_pad or self.idx2word[_] not in ["[BOS]", "[EOS]", "[PAD]"]]
    
    def __len__(self):
        """
        返回词表大小
        """
        return len(self.idx2word)


# 定义模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 为输入添加位置编码并使用 dropout
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8, 
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1, max_len=100, pad_id=3):
        super(TransformerSeq2Seq, self).__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len+2) # +2 for BOS/EOS
        
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout, 
                                          batch_first=True)
        
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        # 生成解码器自回归的布尔 mask（True 表示屏蔽，即后续位置不可见）
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return ~mask.to(torch.bool)

    def create_mask(self, src, tgt):
        # 构建 encoder/decoder 所需的各种 mask
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        # decoder 自回归 mask (bool)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        # encoder 不需要自回归 mask（全可见，使用 bool False）
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

        # padding mask 标记 True 表示 PAD
        src_padding_mask = (src == self.pad_id)
        tgt_padding_mask = (tgt == self.pad_id)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        # 输入: src [N, S], tgt [N, T]（tgt 包含 BOS）
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)

        # Embedding + 位置编码
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        # transformer 直接输出每个目标位置的表示
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, 
                                src_padding_mask, tgt_padding_mask, src_padding_mask)
        return self.generator(outs)

    def predict(self, src, max_len, start_symbol, end_symbol, beam_size=1):
        # 预测接口：支持贪心解码 (beam_size=1) 或束搜索 (beam_size > 1)
        num_tokens = src.shape[1]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(src.device)
        src_padding_mask = (src == self.pad_id).to(src.device)

        # 编码器 memory
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        
        if beam_size <= 1:
            # 贪心解码
            ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(src.device)
            for i in range(max_len):
                tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(src.device)
                tgt_emb = self.pos_encoder(self.tgt_embed(ys) * math.sqrt(self.d_model))
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask)
                prob = self.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()
                ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
                if next_word == end_symbol:
                    break
            return ys
        
        # 束搜索 (Beam Search) - 仅支持 batch_size=1
        beams = [(0.0, [start_symbol])]
        completed_beams = []
        
        for _ in range(max_len):
            new_beams = []
            for score, seq in beams:
                if seq[-1] == end_symbol:
                    completed_beams.append((score, seq))
                    continue
                
                ys = torch.tensor([seq], device=src.device).type(torch.long)
                tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(src.device)
                tgt_emb = self.pos_encoder(self.tgt_embed(ys) * math.sqrt(self.d_model))
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask)
                
                # 只取最后一个时间步的概率
                log_probs = F.log_softmax(self.generator(out[:, -1]), dim=-1).squeeze(0)
                topk_probs, topk_ids = torch.topk(log_probs, beam_size)
                
                for lp, idx in zip(topk_probs, topk_ids):
                    new_beams.append((score + lp.item(), seq + [idx.item()]))
            
            # 选取当前得分最高的 beam_size 个候选
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
            if not beams: break

        # 结合已完成和未完成的候选，取最高分
        all_candidates = completed_beams + beams
        if not all_candidates:
            return torch.tensor([[start_symbol, end_symbol]], device=src.device)
            
        # 简单的长度惩罚 (Length Penalty)
        best_seq = max(all_candidates, key=lambda x: x[0] / (len(x[1])**0.7))[1]
        return torch.tensor([best_seq], device=src.device)
    

# 构建Dataloader
def collate(data_list):
    src = torch.stack([torch.LongTensor(_[0]) for _ in data_list])
    tgt = torch.stack([torch.LongTensor(_[1]) for _ in data_list])
    return src, tgt

def padding(inp_ids, max_len, pad_id):
    max_len += 2    # [BOS] [EOS]
    ids_ = np.ones(max_len, dtype=np.int32) * pad_id
    max_len = min(len(inp_ids), max_len)
    ids_[:max_len] = inp_ids
    return ids_

def create_dataloader(zh_sents, en_sents, max_len, batch_size, pad_id):
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        shuffle = True if split=='train' else False
        datas = [(padding(zh_vocab.encode(zh, max_len), max_len, pad_id), padding(en_vocab.encode(en, max_len), max_len, pad_id)) for zh, en in zip(zh_sents[split], en_sents[split])]
        dataloaders[split] = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return dataloaders['train'], dataloaders['val'], dataloaders['test']

# 训练、测试函数
def train_loop(model, optimizer, criterion, loader, device):
    model.train()
    epoch_loss = 0.0
    for src, tgt in tqdm(loader):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Transformer input: tgt includes [BOS]...
        # Transformer output: predicts ...[EOS]
        # So we feed tgt[:, :-1] to the decoder, and expect it to predict tgt[:, 1:]
        
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        
        outputs = model(src, tgt_input)
        
        # outputs: [N, T-1, V]
        # tgt_out: [N, T-1]
        
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(loader)
    return epoch_loss

def test_loop(model, loader, tgt_vocab, device, beam_size=1):
    model.eval()
    bleu = BLEU(force=True)
    hypotheses, references = [], []
    for src, tgt in tqdm(loader):
        B = len(src)
        for _ in range(B):
            _src = src[_].unsqueeze(0).to(device)     # 1 * L
            with torch.no_grad():
                # predict 支持 beam_size
                outputs = model.predict(_src, max_len=args.max_len+2, 
                                        start_symbol=tgt_vocab.word2idx["[BOS]"], 
                                        end_symbol=tgt_vocab.word2idx["[EOS]"],
                                        beam_size=beam_size)
            
            ref = " ".join(tgt_vocab.decode(tgt[_].tolist(), strip_bos_eos_pad=True))
            hypo = " ".join(tgt_vocab.decode(outputs[0].cpu().tolist(), strip_bos_eos_pad=True))
            references.append(ref)
            hypotheses.append(hypo)
    
    score = bleu.corpus_score(hypotheses, [references]).score
    return hypotheses, references, score

# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()      
    parser.add_argument('--num_train', type=int, default=-1, help="训练集大小，等于-1时将包含全部训练数据")
    parser.add_argument('--max_len', type=int, default=20, help="句子最大长度")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--beam_size', type=int, default=4)
    args = parser.parse_args()

    zh_sents, en_sents = load_data(args.num_train)

    zh_vocab = Vocab()
    en_vocab = Vocab()
    for zh, en in zip(zh_sents['train'], en_sents['train']):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)
    print("中文词表大小为", len(zh_vocab))
    print("英语词表大小为", len(en_vocab))

    pad_id = zh_vocab.word2idx['[PAD]']
    trainloader, validloader, testloader = create_dataloader(zh_sents, en_sents, args.max_len, args.batch_size, pad_id=pad_id)

    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transformer settings from README
    # num_layer=3, num_head=8, hidden_size=256, ffn_hidden_size=512, dropout=0.1
    model = TransformerSeq2Seq(len(zh_vocab), len(en_vocab), 
                               d_model=256, nhead=8, 
                               num_encoder_layers=3, num_decoder_layers=3, 
                               dim_feedforward=512, dropout=0.1, 
                               max_len=args.max_len, pad_id=pad_id)
    model.to(device)
    
    if args.optim=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    weights = torch.ones(len(en_vocab)).to(device)
    weights[en_vocab.word2idx['[PAD]']] = 0 
    
    # 使用标签平滑 (Label Smoothing)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=pad_id, label_smoothing=args.label_smoothing)

    # 训练
    start_time = time.time()
    best_score = 0.0
    for epoch in range(args.num_epoch):
        loss = train_loop(model, optimizer, criterion, trainloader, device)
        # 验证时使用贪心解码以节省时间
        hypotheses, references, bleu_score = test_loop(model, validloader, en_vocab, device, beam_size=1)
        
        # 更新学习率
        scheduler.step(bleu_score)
        
        # 保存验证集上bleu最高的checkpoint
        if bleu_score > best_score:
            torch.save(model.state_dict(), "model_transformer_best.pt")
            best_score = bleu_score
        print(f"Epoch {epoch}: loss = {loss:.4f}, valid bleu = {bleu_score:.2f}")
        print(f"Ref: {references[0]}")
        print(f"Hyp: {hypotheses[0]}")
    end_time = time.time()

    # 测试时使用束搜索 (Beam Search)
    model.load_state_dict(torch.load("model_transformer_best.pt"))
    hypotheses, references, bleu_score = test_loop(model, testloader, en_vocab, device, beam_size=args.beam_size)
    print(f"Test bleu (Beam Search, size={args.beam_size}) = {bleu_score:.2f}")
    print(f"Ref: {references[0]}")
    print(f"Hyp: {hypotheses[0]}")
    print(f"Training time: {round((end_time - start_time)/60, 2)}min")
