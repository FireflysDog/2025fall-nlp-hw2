from sacrebleu.metrics import BLEU
import torch
import torch.nn as nn
import torch.nn.functional as F
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
class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, hidden_size)
        self.weight_hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, input, hx):
        
        igates = self.weight_ih(input)
        hgates = self.weight_hh(hx)
        ret = torch.tanh(igates + hgates)
        
        return ret

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = RNNCell(embedding_dim, hidden_size)
    
    def forward(self, input, hidden):
        """
        input: N
        hidden: N * H
        
        输出更新后的隐状态hidden（大小为N * H）
        """
        embedding = self.dropout(self.embed(input))
        hidden = self.rnn(embedding, hidden)
        return hidden

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
        energy = self.attn(encoder_outputs) 
        
        # Calculate scores: [batch_size, seq_len]
        # hidden.unsqueeze(2): [batch_size, hidden_size, 1]
        scores = torch.bmm(energy, hidden.unsqueeze(2)).squeeze(2)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        return F.softmax(scores, dim=1)

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.rnn = RNNCell(embedding_dim + hidden_size, hidden_size) # Input is embedding + context
        self.attention = Attention(hidden_size)
        # Optimization: Concatenate hidden and context for output
        self.h2o = nn.Linear(hidden_size * 2, vocab_size)
        self.output_dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input, hidden, encoder_outputs, mask=None):
        """
        input: N
        hidden: N * H
        encoder_outputs: N * L * H
        """
        embedding = self.input_dropout(self.embed(input)) # [N, Emb]
        
        # Calculate attention weights with mask
        attn_weights = self.attention(hidden, encoder_outputs, mask) # [N, L]
        
        # Calculate context vector
        # attn_weights.unsqueeze(1): [N, 1, L]
        # encoder_outputs: [N, L, H]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1) # [N, H]
        
        # Combine embedding and context for RNN input
        rnn_input = torch.cat((embedding, context), dim=1) # [N, Emb + H]
        
        hidden = self.rnn(rnn_input, hidden)
        
        # Optimization: Use both hidden and context for prediction
        output_input = self.output_dropout(torch.cat((hidden, context), dim=1))
        output = self.h2o(output_input)
        output = self.softmax(output)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len, pad_id, dropout=0.0, teacher_forcing_ratio=1.0):
        super(Seq2Seq, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.pad_id = pad_id
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = EncoderRNN(len(src_vocab), embedding_dim, hidden_size, dropout)
        self.decoder = DecoderRNN(len(tgt_vocab), embedding_dim, hidden_size, dropout)
        self.max_len = max_len
        
    def init_hidden(self, batch_size):
        """
        初始化编码器端隐状态为全0向量（大小为1 * H）
        """
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size).to(device)
    
    def init_tgt_bos(self, batch_size):
        """
        预测时，初始化解码器端输入为[BOS]（大小为batch_size）
        """
        device = next(self.parameters()).device
        return (torch.ones(batch_size)*self.tgt_vocab.index("[BOS]")).long().to(device)
    
    def forward_encoder(self, src):
        """
        src: N * L
        编码器前向传播，输出最终隐状态hidden (N * H)和隐状态序列encoder_hiddens (N * L * H)
        """
        Bs, Ls = src.size()
        hidden = self.init_hidden(batch_size=Bs)
        encoder_hiddens = []
        # 编码器端每个时间片，取出输入单词的下标，与上一个时间片的隐状态一起送入encoder，得到更新后的隐状态，存入enocder_hiddens
        for i in range(Ls):
            input = src[:, i]
            hidden = self.encoder(input, hidden)
            encoder_hiddens.append(hidden)
        encoder_hiddens = torch.stack(encoder_hiddens, dim=1)
        
        # Create mask for padding
        mask = (src == self.pad_id)
        
        return hidden, encoder_hiddens, mask
    
    def forward_decoder(self, tgt, hidden, encoder_hiddens, mask):
        """
        tgt: N * L
        hidden: N * H
        encoder_hiddens: N * L * H
        mask: N * L
        
        解码器前向传播，结合teacher forcing比率进行训练，输出预测结果outputs，大小为N * (L-1) * V，其中V为目标语言词表大小
        """
        Bs, Lt = tgt.size()
        outputs = []
        input_token = tgt[:, 0]
        for i in range(1, Lt):
            output, hidden = self.decoder(input_token, hidden, encoder_hiddens, mask)
            outputs.append(output)
            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            next_input = tgt[:, i] if teacher_force else output.argmax(-1)
            input_token = next_input
        outputs = torch.stack(outputs, dim=1)
        return outputs
        
    
    def forward(self, src, tgt):
        """
            src: 1 * Ls
            tgt: 1 * Lt
            
            训练时的前向传播
        """
        hidden, encoder_hiddens, mask = self.forward_encoder(src)
        outputs = self.forward_decoder(tgt, hidden, encoder_hiddens, mask)
        return outputs
    
    def predict(self, src):
        """
            src: 1 * Ls
            
            用于预测，解码器端初始输入为[BOS]，之后每个位置的输入为上个时间片预测概率最大的单词
            当解码长度超过self.max_len或预测出了[EOS]时解码终止
            输出预测的单词编号序列，大小为1 * L，L为预测长度
        """
        hidden, encoder_hiddens, mask = self.forward_encoder(src)
        input = self.init_tgt_bos(batch_size=src.shape[0])
        preds = [input]
        while len(preds) < self.max_len:
            output, hidden = self.decoder(input, hidden, encoder_hiddens, mask)
            input = output.argmax(-1)
            preds.append(input)
            if input == self.tgt_vocab.index("[EOS]"):
                break
        preds = torch.stack(preds, dim=-1)
        return preds
    

# 构建Dataloader
def collate(data_list):
    src = torch.stack([torch.LongTensor(_[0]) for _ in data_list])
    tgt = torch.stack([torch.LongTensor(_[1]) for _ in data_list])
    return src, tgt

def padding(inp_ids, max_len, pad_id):
    max_len += 2    # include [BOS] and [EOS]
    ids_ = np.ones(max_len, dtype=np.int32) * pad_id
    max_len = min(len(inp_ids), max_len)
    ids_[:max_len] = inp_ids
    return ids_

def create_dataloader(zh_sents, en_sents, max_len, batch_size, src_pad_id, tgt_pad_id):
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        shuffle = True if split=='train' else False
        datas = [
            (
                padding(zh_vocab.encode(zh, max_len), max_len, src_pad_id),
                padding(en_vocab.encode(en, max_len), max_len, tgt_pad_id)
            )
            for zh, en in zip(zh_sents[split], en_sents[split])
        ]
        dataloaders[split] = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return dataloaders['train'], dataloaders['val'], dataloaders['test']

# 训练、测试函数
def train_loop(model, optimizer, criterion, loader, device):
    model.train()
    epoch_loss = 0.0
    for src, tgt in tqdm(loader):
        src = src.to(device)
        tgt = tgt.to(device)
        outputs = model(src, tgt)
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tgt[:,1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)     # 裁剪梯度，将梯度范数裁剪为1，使训练更稳定
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(loader)
    return epoch_loss

def test_loop(model, loader, tgt_vocab, device):
    model.eval()
    bleu = BLEU(force=True)
    hypotheses, references = [], []
    for src, tgt in tqdm(loader):
        B = len(src)
        for _ in range(B):
            _src = src[_].unsqueeze(0).to(device)     # 1 * L
            with torch.no_grad():
                outputs = model.predict(_src)         # 1 * L
            
            # 保留预测结果，使用词表vocab解码成文本，并删去BOS与EOS
            ref = " ".join(tgt_vocab.decode(tgt[_].tolist(), strip_bos_eos_pad=True))
            hypo = " ".join(tgt_vocab.decode(outputs[0].cpu().tolist(), strip_bos_eos_pad=True))
            references.append(ref)    # 标准答案
            hypotheses.append(hypo)   # 预测结果
    
    score = bleu.corpus_score(hypotheses, [references]).score      # 计算BLEU分数
    return hypotheses, references, score

# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()      
    parser.add_argument('--num_train', type=int, default=-1, help="训练集大小，等于-1时将包含全部训练数据")
    parser.add_argument('--max_len', type=int, default=20, help="句子最大长度")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.9)
    args = parser.parse_args()

    zh_sents, en_sents = load_data(args.num_train)

    zh_vocab = Vocab()
    en_vocab = Vocab()
    for zh, en in zip(zh_sents['train'], en_sents['train']):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)
    print("中文词表大小为", len(zh_vocab))
    print("英语词表大小为", len(en_vocab))

    src_pad_id = zh_vocab.word2idx['[PAD]']
    tgt_pad_id = en_vocab.word2idx['[PAD]']
    trainloader, validloader, testloader = create_dataloader(zh_sents, en_sents, args.max_len, args.batch_size, src_pad_id=src_pad_id, tgt_pad_id=tgt_pad_id)

    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(zh_vocab, en_vocab, embedding_dim=256, hidden_size=256, max_len=args.max_len, pad_id=src_pad_id, dropout=args.dropout, teacher_forcing_ratio=args.teacher_forcing_ratio)
    model.to(device)
    if args.optim=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    weights = torch.ones(len(en_vocab)).to(device)
    weights[en_vocab.word2idx['[PAD]']] = 0 # set the loss of [PAD] to zero
    criterion = nn.NLLLoss(weight=weights)

    # 训练
    start_time = time.time()
    best_score = 0.0
    for _ in range(args.num_epoch):
        loss = train_loop(model, optimizer, criterion, trainloader, device)
        hypotheses, references, bleu_score = test_loop(model, validloader, en_vocab, device)
        # 保存验证集上bleu最高的checkpoint
        if bleu_score > best_score:
            torch.save(model.state_dict(), "model_rnn_att_best.pt")
            best_score = bleu_score
        print(f"Epoch {_}: loss = {loss}, valid bleu = {bleu_score}")
        print(references[0])
        print(hypotheses[0])
    end_time = time.time()

    #测试
    model.load_state_dict(torch.load("model_rnn_att_best.pt"))
    hypotheses, references, bleu_score = test_loop(model, testloader, en_vocab, device)
    print(f"Test bleu = {bleu_score}")
    print(references[0])
    print(hypotheses[0])
    print(f"Training time: {round((end_time - start_time)/60, 2)}min")
