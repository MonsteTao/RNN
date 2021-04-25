import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
# import torchtext
import collections
from torchtext import vocab
from torch.utils.data import TensorDataset, DataLoader


# def read_imdb(folder, data_root):
#     data = []
#     for label in ['pos', 'neg']:
#         folder_name = os.path.join(data_root, folder, label)
#         for file in tqdm(os.listdir(folder_name)):
#             with open(os.path.join(folder_name, file), 'rb') as f:
#                 review = f.read().decode('utf-8').replace('\n', '').lower()
#                 data.append([review, 1 if label == 'pos' else 0])
#     random.shuffle(data)
#     return data


# data_root = "D:/workspace/data/IMDB"
# train_data, test_data = read_imdb('train', data_root),\
#     read_imdb('test', data_root)

# # 打印训练数据中的前五个sample
# for sample in train_data[:5]:
#     print(sample[1], '\t', sample[0][:50])


def get_tokenized_imdb(data):
    '''
    @params:
        data: 数据的列表，列表中的每个元素为 [文本字符串，0/1标签] 二元组
    @return: 切分词后的文本的列表，列表中的每个元素为切分后的词序列
    '''
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]


def get_vocab_imdb(data):
    '''
    @params:
        data: 同上
    @return: 数据集上的词典，Vocab 的实例（freqs, stoi, itos）
    '''
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return vocab.Vocab(counter, min_freq=5)


# v_ocab = get_vocab_imdb(data)
# print('# words in vocab:', len(v_ocab))
# words in vocab: 46152

def preprocess_imdb(data, vocabb):
    '''
    @params:
        data: 同上，原始的读入数据
        vocab: 训练集上生成的词典
    @return:
        features: 单词下标序列，形状为 (n, max_l) 的整数张量
        labels: 情感标签，形状为 (n,) 的0/1整数张量
    '''
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocabb.stoi[word] for word in words])
                            for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


data = [["I LOVE YOU", 1], ["HELLO WORLD", 1], ["I AM SAD", 0]]
vocabb = get_vocab_imdb(data)
# print(len(vocabb))
# a, b = preprocess_imdb(data, vocabb)


train_set = TensorDataset(*preprocess_imdb(data, vocabb))
test_set = TensorDataset(*preprocess_imdb(data, vocabb))

# 上面的代码等价于下面的注释代码
# train_features, train_labels = preprocess_imdb(train_data, vocab)
# test_features, test_labels = preprocess_imdb(test_data, vocab)
# train_set = Data.TensorDataset(train_features, train_labels)
# test_set = Data.TensorDataset(test_features, test_labels)

# len(train_set) = features.shape[0] or labels.shape[0]
# train_set[index] = (features[index], labels[index])

batch_size = 64
train_iter = DataLoader(train_set, batch_size, shuffle=True)
test_iter = DataLoader(test_set, batch_size)

# for X, y in train_iter:
#     print('X', X.shape, 'y', y.shape)
#     b = [t for t in X[2] if t != 0]
#     print(b)
#     break
# print('#batches:', len(train_iter))
# X torch.Size([64, 500]) y torch.Size([64])
# batches: 391


class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        '''
        @params:
            vocab: 在数据集上创建的词典，用于获取词典大小
            embed_size: 嵌入维度大小
            num_hiddens: 隐藏状态维度大小
            num_layers: 隐藏层个数
        '''
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size) 
        # encoder-decoder framework
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size, 
                               hidden_size=num_hiddens, 
                               num_layers=num_layers,
                               bidirectional=True)
        self.decoder = nn.Linear(4*num_hiddens, 2)  # 初始时间步和最终时间步的隐藏状态作为全连接层输入

    def forward(self, inputs):
        '''
        @params:
            inputs: 词语下标序列，形状为 (batch_size, seq_len) 的整数张量
        @return:
            outs: 对文本情感的预测，形状为 (batch_size, 2) 的张量
        '''
        # 因为LSTM需要将序列长度(seq_len)作为第一维，所以需要将输入转置
        embeddings = self.embedding(inputs.permute(1, 0))  # (seq_len, batch_size, d)
        # rnn.LSTM 返回输出、隐藏状态和记忆单元，格式如 outputs, (h, c)
        outputs, _ = self.encoder(embeddings)  # (seq_len, batch_size, 2*h)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)  # (batch_size, 4*h)
        outs = self.decoder(encoding)  # (batch_size, 2)
        return outs


# embed_size, num_hiddens, num_layers = 100, 100, 2
# net = BiRNN(vocab, embed_size, num_hiddens, num_layers)

cache_dir = "/home/kesci/input/GloVe6B5429"
glove_vocab = vocab.GloVe(name='6B', dim=100, cache=cache_dir)


def load_pretrained_embedding(words, pretrained_vocab):
    '''
    @params:
        words: 需要加载词向量的词语列表，以 itos (index to string) 的词典形式给出
        pretrained_vocab: 预训练词向量
    @return:
        embed: 加载到的词向量
    '''
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为0
    oov_count = 0  # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    return embed

# net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))
# net.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它