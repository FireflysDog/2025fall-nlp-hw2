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
        """
        参数:
        - input: [N]
        - hidden: [N, H]
        返回:
        - 更新后的隐状态 hidden (N, H)
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
        """
        参数:
        - input: [N] 当前时间步输入 token id
        - hidden: [N, H] 当前隐状态
        - encoder_outputs: [N, L, H] 编码器所有时间步隐状态
        返回:
        - output: [N, V] log-probabilities
        - hidden: 更新后的隐状态 [N, H]
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
    def __init__(self, src_vocab, tgt_vocab, embedding_dim, hidden_size, max_len, pad_id,
                 dropout=0.0, teacher_forcing_ratio=1.0, beam_size=1):
        super(Seq2Seq, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.pad_id = pad_id
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.default_beam_size = beam_size
        self.encoder = EncoderRNN(len(src_vocab), embedding_dim, hidden_size, dropout)
        self.decoder = DecoderRNN(len(tgt_vocab), embedding_dim, hidden_size, dropout)
        self.max_len = max_len
        
    def set_teacher_forcing_ratio(self, ratio: float):
        self.teacher_forcing_ratio = max(0.0, min(1.0, ratio))

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
        """
        输入:
        - src: [N, L]
        返回:
        - hidden: 最终隐状态 [N, H]
        - encoder_hiddens: 每个时间步隐状态 [N, L, H]
        - mask: 填充掩码 [N, L]（True 表示 PAD）
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
        """
        参数:
        - tgt: [N, L] 目标序列（含 BOS/EOS）
        - hidden: [N, H] 编码器最后隐状态（用于初始化解码器）
        - encoder_hiddens: [N, L, H]
        - mask: [N, L] 填充掩码
        根据 teacher forcing 比率进行逐步解码并返回所有时间步的预测 logits: [N, L-1, V]
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
        """
        训练时前向传播。
        输入:
        - src: [N, Ls]
        - tgt: [N, Lt]
        返回:
        - outputs: [N, Lt-1, V]
        """
        hidden, encoder_hiddens, mask = self.forward_encoder(src)
        outputs = self.forward_decoder(tgt, hidden, encoder_hiddens, mask)
        return outputs
    
    def predict(self, src, beam_size=None):
        """
            src: 1 * Ls
            
            使用束搜索（默认束宽 self.default_beam_size）或贪心进行预测
        """
        """
        预测接口。
        输入:
        - src: [N, Ls]
        可选使用束搜索（beam_size>1），否则使用贪心解码。
        返回预测的 token id 序列。
        """
        hidden, encoder_hiddens, mask = self.forward_encoder(src)
        beam_size = beam_size or self.default_beam_size
        if beam_size <= 1:
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

        # Beam search assumes batch size 1
        device = src.device
        start_token = self.tgt_vocab.index("[BOS]")
        end_token = self.tgt_vocab.index("[EOS]")

        beams = [(0.0, [start_token], hidden)]
        finished = []
        for _ in range(self.max_len - 1):
            new_beams = []
            for score, seq, h in beams:
                last_token = torch.tensor([seq[-1]], device=device)
                output, new_hidden = self.decoder(last_token, h, encoder_hiddens, mask)
                log_probs = output.squeeze(0)
                topk = torch.topk(log_probs, beam_size)
                for next_score, token_id in zip(topk.values.tolist(), topk.indices.tolist()):
                    new_seq = seq + [token_id]
                    total_score = score + next_score
                    if token_id == end_token:
                        finished.append((total_score, new_seq))
                    else:
                        new_beams.append((total_score, new_seq, new_hidden.clone()))
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
            if not beams:
                break
        if not finished and beams:
            finished = [(score, seq) for score, seq, _ in beams]
        best_seq = max(finished, key=lambda x: x[0])[1]
        return torch.tensor(best_seq, device=device).unsqueeze(0)
    

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
    parser.add_argument('--teacher_forcing_start', type=float, default=0.95)
    parser.add_argument('--teacher_forcing_end', type=float, default=0.5)
    parser.add_argument('--teacher_forcing_decay_epochs', type=int, default=25)
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

    src_pad_id = zh_vocab.word2idx['[PAD]']
    tgt_pad_id = en_vocab.word2idx['[PAD]']
    trainloader, validloader, testloader = create_dataloader(zh_sents, en_sents, args.max_len, args.batch_size, src_pad_id=src_pad_id, tgt_pad_id=tgt_pad_id)

    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(
        zh_vocab,
        en_vocab,
        embedding_dim=256,
        hidden_size=256,
        max_len=args.max_len,
        pad_id=src_pad_id,
        dropout=args.dropout,
        teacher_forcing_ratio=args.teacher_forcing_start,
        beam_size=args.beam_size,
    )
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
    if args.teacher_forcing_decay_epochs <= 0:
        decay_step = 0.0
    else:
        decay_step = (args.teacher_forcing_start - args.teacher_forcing_end) / max(1, args.teacher_forcing_decay_epochs - 1)

    for epoch_idx in range(args.num_epoch):
        current_ratio = max(
            args.teacher_forcing_end,
            args.teacher_forcing_start - decay_step * epoch_idx,
        )
        model.set_teacher_forcing_ratio(current_ratio)
        loss = train_loop(model, optimizer, criterion, trainloader, device)
        hypotheses, references, bleu_score = test_loop(model, validloader, en_vocab, device)
        # 保存验证集上bleu最高的checkpoint
        if bleu_score > best_score:
            torch.save(model.state_dict(), "model_rnn_att_best.pt")
            best_score = bleu_score
        print(f"Epoch {epoch_idx}: loss = {loss}, valid bleu = {bleu_score}, teacher forcing = {current_ratio:.3f}")
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
