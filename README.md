# seq2seq_translation_example
自然语言处理23学年秋季学期作业2（基于Seq2Seq模型的机器翻译）示例代码

## 数据集
本次作业数据选自[Tatoeba](https://www.manythings.org/anki/)数据集中的中文-英文翻译数据。

数据已经下载并处理好，位于`data`文件夹中。其中包含26,187条训练集数据`zh_en_train.txt`，1,000条验证集数据`zh_en_val.txt`，以及1,000条测试集数据`zh_en_test.txt`。每一行是一组数据，形式为“中文句子\t英文句子”。

## 评测指标
本次作业采用BLEU Score为评测指标。评测代码提供在示例代码中，需要下载`sacrebleu`包。

## Baseline模型
本次作业提供的baseline模型是基于普通RNN的Seq2Seq模型，在`seq2seq-rnn.py`中。

## 已实现的模型与优化
除了基础的 RNN 外，本项目还实现了以下模型并进行了深度优化：

1.  **RNN + Attention (`seq2seq-rnn-attention.py`)**:
    *   在 RNN 基础上增加了注意力机制。
    *   支持 Dropout 和 梯度裁剪。
2.  **LSTM + Attention (`seq2seq-lstm-attention.py`)**:
    *   **手动实现 LSTMCell**（不使用 `nn.LSTM`），满足作业进阶要求。
    *   结合注意力机制，性能优于基础 RNN。
3.  **Transformer (`seq2seq-transformer.py`)**:
    *   基于 PyTorch `nn.Transformer` 实现。
    *   **优化特性**：
        *   **束搜索 (Beam Search)**：推理时使用束搜索并加入长度惩罚。
        *   **标签平滑 (Label Smoothing)**：降低过拟合。
        *   **混合精度训练 (AMP)**：加速训练并减少显存占用。
        *   **学习率调度 (ReduceLROnPlateau)**：根据验证集表现动态调整学习率。
        *   **早停机制 (Early Stopping)**：防止无效训练。

## 使用说明

### 环境配置
```bash
pip install -r requirements.txt
```

### 运行模型
你可以通过以下命令运行不同的模型：

*   **运行 RNN + Attention**:
    ```bash
    python seq2seq-rnn-attention.py
    ```
*   **运行 LSTM + Attention**:
    ```bash
    python seq2seq-lstm-attention.py
    ```
*   **运行 Transformer**:
    ```bash
    python seq2seq-transformer.py
    ```

### 主要参数
*   `--num_train`: 训练集大小（-1 表示使用全部数据）。
*   `--batch_size`: 批大小。
*   `--lr`: 学习率。
*   `--beam_size`: 束搜索大小（仅限 Transformer 和优化后的模型）。

## 实验结果

以下模型都可以在有16GB内存CPU的电脑上训练。其中RNN为提供baseline示例模型的效果。其他模型需要同学们自己实现，结果是提供参考的。
其中Transformer模型的设置为：`num_layer=3`, `num_head=8`, `hidden_size=256`, `ffn_hidden_size=512`, `dropout=0.1`。训练相关超参数和示例代码相同。
| Model |#train data |  BLEU  |  Train Time (GPU) | Train Time (CPU)| GPU Mem |
|----------|:-------------:|:-----:|:-----:|:-----:|:-----:|
| RNN         |  26,187   | 1.41 | 1.5 min | ~  50min    | 1,249 MB |
| RNN+Att     |  26,187   | 13.15 | 2.4 min | ~ 1h 10min    | 1,431 MB |
| LSTM+Att    |  26,187   | 13.52 | 3.1 min | ~ 1h 10min    | 1,449 MB |
| Transformer |  26,187   | 23.41  | 5.5 min | ~ 1h 10min | 1,501 MB  |
