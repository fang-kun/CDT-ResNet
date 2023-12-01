import json
import random
import torch
import numpy as np

class DataLoader(object):
    def __init__(self, filename, batch_size, args, vocab):
        self.batch_size = batch_size
        self.args = args
        self.vocab = vocab

        with open(filename) as infile:
            data = json.load(infile)
        # raw_data中都是json格式的数据，list[1980]，list[i]中均含有token、POS、head、deprel和aspect的原始信息
        self.raw_data = data
        
        # preprocess data 预处理数据 将json
        data = self.preprocess(data, vocab, args)

        # labels
        pol_vocab = vocab[-1]
        self.labels = [pol_vocab.itos[d[-1]] for d in data]

        # example num
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, args):
        # unpack vocab
        token_vocab, post_vocab, pos_vocab, dep_vocab, pol_vocab = vocab
        processed = []

        for d in data:
            for aspect in d['aspects']:
                # word token
                tok = list(d['token'])
                if args.lower == True:
                    tok = [t.lower() for t in tok]
                
                # aspect
                asp = list(aspect['term'])
                # label
                label = aspect['polarity']
                # pos_tag 
                pos = list(d['pos'])
                # head  依存头部信息
                head = list(d['head'])
                # deprel    依存关系
                deprel = list(d['deprel'])
                # real length
                length = len(tok)
                # position
                # 以方面词为基准，其之前的词距离为负，之后的词距离为正
                post = [i-aspect['from'] for i in range(aspect['from'])] \
                       +[0 for _ in range(aspect['from'], aspect['to'])] \
                       +[i-aspect['to']+1 for i in range(aspect['to'], length)]
                # aspect mask
                # 除了方面词坐标元素为1，其余元素均为0，用于后续平均池化
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]    # for rest16
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]

                # 通过stoi词汇表映射
                # mapping token
                tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tok]
                # mapping aspect
                asp = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in asp]
                # mapping label
                label = pol_vocab.stoi.get(label)
                # mapping pos
                pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in pos]
                # mapping head to int
                head = [int(x) for x in head]
                assert any([x == 0 for x in head])
                # mapping deprel 
                deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in deprel]
                # mapping post
                post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in post]
                
                assert len(tok) == length \
                       and len(pos) == length \
                       and len(head) == length \
                       and len(deprel) == length \
                       and len(post) == length \
                       and len(mask) == length

                # processed是包含多个元组的列表，其中每个元组包含9个值
                # 分别代表 tok, asp, pos, head, deprel, post, mask, length, label
                processed += [(tok, asp, pos, head, deprel, post, mask, length, label)]

        return processed

    def gold(self):
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError

        batch = self.data[key]
        batch_size = len(batch)
        # 将一个batch的数据按类型重组 即重组为从tok到label的9个元组 每个元组包含这个批次所有对应的数据
        batch = list(zip(*batch))
        
        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        # 将token转化为LongTensor，shape[batch_size, max_len]，下列操作均相同
        tok = get_long_tensor(batch[0], batch_size)
        # aspect
        asp = get_long_tensor(batch[1], batch_size)
        # pos
        pos = get_long_tensor(batch[2], batch_size)
        # head
        head = get_long_tensor(batch[3], batch_size)
        # deprel
        deprel = get_long_tensor(batch[4], batch_size)
        # post
        post = get_long_tensor(batch[5], batch_size)
        # mask
        mask = get_float_tensor(batch[6], batch_size)
        # length
        length = torch.LongTensor(batch[7])
        # label
        label = torch.LongTensor(batch[8])

        return (tok, asp, pos, head, deprel, post, mask, length, label)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    """得到的张量维度为shape[batch_size, max_len]"""
    token_len = max(len(x) for x in tokens_list)
    # 将所有序列扩展到相同的长度token_len，不足的部分用0填充
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    # i是索引，s是子列表，将列表s的值复制到tokens张量的第i行的前lens(s)列
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.FloatTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]
