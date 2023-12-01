import pickle
import tqdm
from collections import Counter


class Vocab(object):
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()  # counter是一个包含词频计数的词典
        self.itos = list(specials)  # itos是index to string 词汇列表
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        # counter.items()返回包含键-值对的可迭代对象，每个键-值对表示词汇表中的一个词和对应的词频，如（apple，2）
        # key=lambda tup: tup[0] 中tup是临时变量，代表元组。tup[0]也就是元组中的第一个元素，指的是单词
        # 所以第一步我们是根据单词首字母进行排序
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        # 第二步是根据tup[1]排序，也就是根据词频排序。reverse=True 代表降序排序
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # word接收元组中的第一个元素（词汇），freq接收元组中的第二个元素（词频）
        for word, freq in words_and_frequencies:
            # 将当前词汇（word）添加到self.itos列表中
            self.itos.append(word)

        # stoi is simply a reverse dict for itos    词汇索引字典string to index
        # enumerate返回的元素例如(0, 'apple')
        # stoi为（tok:i） 而for循环中用i接受索引，tok接收词汇
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)
