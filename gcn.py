import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tree import Tree, head_to_tree, tree_to_adj

class GCNClassifier(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim
        self.args = args
        self.gcn_model = GCNAbsaModel(args, emb_matrix=emb_matrix)
        # in_dim, shape[50]
        self.classifier = nn.Linear(in_dim, args.num_class)

    def forward(self, inputs):
        # outputs,shape[batch_size=32,hidden_size=50]
        outputs = self.gcn_model(inputs)
        # logits是未经过softmax的原始预测分数
        # logits,shape[batch_size=32, num_class=3]
        logits = self.classifier(outputs)
        return logits, outputs

class GCNAbsaModel(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        self.emb_matrix = emb_matrix

        # create embedding layers
        # padding_idx用于在对输入序列进行嵌入（embedding）时，将指定的填充索引对应的嵌入向量设置为零向量
        # emb, shape[token_vocab, emb_dim]
        self.emb = nn.Embedding(args.tok_size, args.emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix.cuda(), requires_grad=False)

        # pos_emb, shape[pos_vocab, pos_dim]
        self.pos_emb = nn.Embedding(args.pos_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None        # POS emb
        # post_emb, shape[post_vocab, post_dim]
        self.post_emb = nn.Embedding(args.post_size, args.post_dim, padding_idx=0) if args.post_dim > 0 else None    # position emb
        embeddings = (self.emb, self.pos_emb, self.post_emb)

        # gcn layer
        self.gcn = GCN(args, embeddings, args.hidden_dim, args.num_layers)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l = inputs           # unpack inputs
        maxlen = max(l.data)

        # 获取邻接矩阵
        def inputs_to_tree_reps(head, words, l):
            trees = [head_to_tree(head[i], words[i], l[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=self.args.direct, self_loop=self.args.loop).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return adj.cuda()

        # head.data&tok.data, shape[batch_size,seq_len] ; l.data, shape[batch_size,]
        # adj, shape[batch_size,seq_len,seq_len]
        adj = inputs_to_tree_reps(head.data, tok.data, l.data)
        # input,list[8]: from tok to l
        # h,shape[batch=32,seq_len=41,hidden_dim=50]
        h = self.gcn(adj, inputs)
        
        # avg pooling asp feature   todo 平均池化
        # asp_wn,shape[batch_size=32, 1]
        asp_wn = mask.sum(dim=1).unsqueeze(-1)                        # aspect words num
        # 输入mask,shape[batch_size=32, seq_len=41]   self.args.hidden_dim=50
        # 输出mask,shape[batch_size=32, seq_len=41, hidden_dim=50]
        # repeat方法用于沿指定维度上重复张量的元素  在dim=0,1上重复1次，形状不变  在dim=2上重复50次
        # 重复操作只会复制原始张量中的元素，并按照指定的重复次数在对应的维度上进行复制
        mask = mask.unsqueeze(-1).repeat(1,1,self.args.hidden_dim)    # mask for h
        # outputs, shape[batch_size=32, hidden_num=50]
        outputs = (h*mask).sum(dim=1) / asp_wn                        # mask h
        
        return outputs

class GCN(nn.Module):
    def __init__(self, args, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim  # 图卷积层的隐藏状态维度
        self.in_dim = args.emb_dim+args.post_dim+args.pos_dim
        self.emb, self.pos_emb, self.post_emb = embeddings

        # rnn layer
        input_size = self.in_dim
        self.rnn = nn.LSTM(input_size, args.rnn_hidden, args.rnn_layers, batch_first=True, \
                dropout=args.rnn_dropout, bidirectional=args.bidirect)
        if args.bidirect:
            self.in_dim = args.rnn_hidden * 2
        else:
            self.in_dim = args.rnn_hidden

        # drop out
        self.rnn_drop = nn.Dropout(args.rnn_dropout)
        self.in_drop = nn.Dropout(args.input_dropout)
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        # gcn layer
        self.W = nn.ModuleList()    # ModuleList 是 PyTorch 中用于保存模块列表的容器，self.W用于存储图卷积层中每一层的权重矩阵
        # 迭代 self.layers 次，为每一层创建一个 nn.Linear 模块，并将其添加到 self.W 中
        # 在后续的图卷积层计算中，可以通过索引访问 self.W 中的权重矩阵
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    """利用 LSTM 对输入序列进行编码，并返回编码后的序列"""   # todo Bi-LSTM输出h
    # rnn_inputs[batch_size=32,seq_len,emb_dim+pos_dim+post_dim=360]   seq_lens[batch_size=32]
    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        # batch_size=32    rnn_hidden=50     rnn_layer=1     bidirect=True
        # h0==c0==0,shape[rnn_layer*2=2, batch_size=32, rnn_hidden=50]
        h0, c0 = rnn_zero_state(batch_size, self.args.rnn_hidden, self.args.rnn_layers, self.args.bidirect)
        # 输入 rnn_inputs,shape[batch_size=32, seq_len=41, emb_dim+pos_dim+post_dim=360]    seq_len,shape[batch_size=32,]
        # 输出 rnn_inputs, packedSequence=2  使用 pack_padded_sequence 可以在处理变长序列时，将填充部分忽略，只处理有效的序列部分，提高计算效率和减少内存消耗
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        # rnn_outputs,shape[batch_size, seq_len, rnn_hidden*2=100]
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    # adj,shape[batch_size,seq_len,seq_len]   inputs,list[8] 保存tok, asp, pos, head, deprel, post, mask, l
    def forward(self, adj, inputs):
        """head是依存关系的头部信息，deprel是依存关系，mask是掩码输入（用于只获取方面），l是length"""
        # tok&pos&head&deprel&post&mask, shape[batch_size, seq_len]
        # asp, shape[batch_size, seq_len(asp)]      l, shape[batch_size,]
        tok, asp, pos, head, deprel, post, mask, l = inputs           # unpack inputs
        # embedding
        # 输出 word_embs, shape[batch_size, seq_len, emb_dim]
        # 这里的语法 goto 34   emb是一个嵌入层
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            # pos_emb, shape[batch_size, seq_len, pos_dim]
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            # post_emb, shape[batch_size, seq_len, post_dim]
            embs += [self.post_emb(post)]   # embs,list[word_embs, pos_emb, post_emb]
        embs = torch.cat(embs, dim=2)   # 输出 embs, shape[batch_size, seq_len, emb_dim+pos_dim+post_dim]
        embs = self.in_drop(embs)

        # rnn layer
        # embs, shape[batch_size=32,seq_len=41,emb_size=360]   l, shape[batch_size=32,]   tok.size()[0]==batch_size, shape==32
        # gcn_inputs, shape[batch=32,len=41,2*rnn_hidden=100]
        gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, l, tok.size()[0]))
        
        # gcn layer
        # adj[batch_size=32,seq_len=41,seq_len=41]
        # sum(2)是对第三个维度求和，结果的维度为shape[32,41]
        # unsqueeze(2)是要在第三个维度上扩展，结果的维度为shape[32,41,1]
        # denom,shape[batch_size, seq_len, 1]
        denom = adj.sum(2).unsqueeze(2) + 1    # norm 求度数
        for l in range(self.layers):
            # input: gcn_inputs[batch=32,len=41,2*rnn_hidden=100] adj[batch_size=32,seq_len=41,seq_len=41]
            # output: Ax,shape[batch=32,len=41,2*rnn_hidden=100]
            Ax = adj.bmm(gcn_inputs)    # bmm：batch matrix multiplication   (batch_size, n, m) 和 (batch_size, m, p) 的张量，得到一个形状为 (batch_size, n, p) 的输出张量。
            # AxW,shape[batch=32,len=41,self.mem_dim=50]
            AxW = self.W[l](Ax)     # 先经过线性层还是先和邻接矩阵相乘都可以 因为邻接矩阵均是0/1 先后顺序没有影响
            AxW = AxW / denom
            # gAxW,shape[batch=32,len=41,self.mem_dim=50]
            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW   # 最后一层不丢弃神经元

        return gcn_inputs

"""生成 LSTM 的初始隐藏状态和细胞状态"""
def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    # total_layers=2    batch_size=32   hidden_dim=arg.rnn_hidden=50
    state_shape = (total_layers, batch_size, hidden_dim)
    # *state_shape，*的作用是将元组 state_shape 展开为三个单独的参数
    # 将其中的三个值分别赋给 total_layers、batch_size 和 hidden_dim 这三个变量,且均初始化为0
    # 所以h0和c0均是shape[total_layers=2, batch_size=32, hidden_dim=50]的三维零张量
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()

