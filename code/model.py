import torch
import torch.nn as nn
import torch.nn.functional as F


class SLCABG(nn.Module):
    def __init__(self, n_dim, sentence_length, word_vectors):
        super(SLCABG, self).__init__()
        # 一个嵌入层，将输入文本中的每个词映射到一个n_dim维度的向量。嵌入层用预先训练好的词向量word_vectors初始化，这些词向量作为参数传给构造函数。
        self.word_embeddings = nn.Embedding.from_pretrained(word_vectors)
        # 在训练过程中不改变预训练的词嵌入
        self.word_embeddings.weight.requires_grad = False
        # 三个具有不同核大小（3、4、5）的卷积层，每个卷积层对输入进行一维卷积运算。每个层都有一个批处理归一化层，一个ReLU激活函数和一个最大池化操作。每层的输出沿第二维串联并传递给下一层。
        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(n_dim, 128, h),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(sentence_length - h + 1)
                           ) for h in [3, 4, 5]]
        )
        # 一个具有64个隐藏单元的双向GRU层。该层的输入是卷积层的输出。该层的输出通过权重矩阵self.weight_W和投影矩阵self.weight_proj，这些都是在训练中学习的。该层的输出是每个输入标记的隐藏状态序列，用于计算注意力分数。
        self.gru = nn.GRU(128*3, 64, batch_first=True, bidirectional=True, dropout=0.4)
        self.weight_W = nn.Parameter(torch.Tensor(128, 128))
        self.weight_proj = nn.Parameter(torch.Tensor(128, 1))
        # 一个全连接层，将GRU层的最终输出映射到一个二维输出（每类一个）。正向方法接收一批输入数据x，并将其传递给模型的各个层，返回最后的线性层的输出。
        self.fc = nn.Linear(128, 2)
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, x):
        embed_x = self.word_embeddings(x)
        embed_x = embed_x.permute(0, 2, 1)
        out = [conv(embed_x) for conv in self.convs]
        out = torch.cat(out, 1)
        out = out.permute(0, 2, 1)
        out, _ = self.gru(out)
        u = torch.tanh(torch.matmul(out, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = out * att_score
        feat = torch.sum(scored_x, dim=1)
        out = self.fc(feat)
        return out
