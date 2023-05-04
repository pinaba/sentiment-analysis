import jieba
import numpy as np
import collections
import torch
from bert_serving.client import BertClient

# 从大连理工大学情感词汇本体库中读取数据，去掉代表中性和褒贬两性的词，筛选出情感极性为1和2的词，如果情感极性为1，情感得分就是情感强度；如果情感极性为2，情感得分为情感强度的负数，将其放入sentiment_words.txt。
def process_sentiment_words():
    f = open('data/sentiment_words.txt', 'w', encoding='utf-8')
    with open('data/sentiment_words.csv', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines[1:]:
            line = line.strip().replace(' ', '').split(',')
            if line[1] == 'idiom':
                continue
            if line[6] == '1.0':
                f.write(line[0] + ',' + str(line[5]) + '\n')
            elif line[6] == '2.0':
                f.write(line[0] + ',' + str(-1 * float(line[5])) + '\n')
    f.close()

# 对情感词汇的情感得分进行归一化处理：新的情感得分是(情感得分-平均值)/标准差，将情感词汇和新的情感得分写入新的文件。
def normalize_sentiment_words():
    words = []
    weights = []
    with open('data/sentiment_words.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().split(',')
            words.append(line[0])
            weights.append(float(line[1]))
    weights = np.array(weights)
    mean = weights.mean()
    std = weights.std()
    weights = (weights - mean)/std
    with open('data/normal_sentiment_words.txt', 'w', encoding='utf-8') as fp:
        for i in range(len(words)):
            fp.write(words[i] + ',' + str(weights[i]) + '\n')

# 将归一化处理后的情感词汇及其对应的情感得分存入sentiment_dict字典中
def get_sentiment_dict():
    sentiment_dict = {}
    with open('data/normal_sentiment_words.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().split(',')
            sentiment_dict[line[0]] = float(line[1])
    return sentiment_dict

# 得到装有停用词的数组
def get_stopwords():
    stopwords = []
    with open('data/stop_words.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords

# 将正面数据集和负面数据集中的都放入sentences列表中，分词并去除停用词后返回分词后的句子列表rs_sentences
def get_data():
    stopwords = get_stopwords()
    sentiment_dict = get_sentiment_dict()
    sentences = []
    rs_sentences = []
    with open('data/positive.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            sentences.append(line.strip())
    with open('data/negative.txt', 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            sentences.append(line.strip())
    jieba.load_userdict(sentiment_dict.keys())
    for sentence in sentences:
        sentence = list(jieba.cut(sentence))
        split_sentence = []
        for word in sentence:
            if '\u4e00' <= word <= '\u9fff' and word not in stopwords:
                split_sentence.append(word)
        rs_sentences.append(split_sentence)
    return rs_sentences

# 将去除停用词、分词后的数据集存入word_list.txt中
def process_words_list():
    sentences = get_data()
    words_list = []
    for sentence in sentences:
        words_list.extend(sentence) # 在列表末尾添加值
    words_list = list(set(words_list))
    with open('data/word_list.txt', 'w', encoding='utf-8') as fp:
        for word in words_list:
            fp.write(word + '\n')


def get_words_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words_list = [line.strip() for line in f.readlines()]
    return words_list
# 转词向量
def get_word_vectors(words_list, bc):
    word_vectors = bc.encode(words_list)
    return np.array(word_vectors)
if __name__ == '__main__':
    file_path = 'data/word_list.txt'  
    bc = BertClient()
    words_list = get_words_list(file_path)
    word_vectors = get_word_vectors(words_list, bc)
    np.savetxt('data/word_vectors.txt', word_vectors)


# 创建一个列表words_list中的单词和其对应的列表vecs中的向量的键值对，返回word2vec字典
def get_word_vectors():
    word_list = get_words_list()
    vecs = np.loadtxt('data/vecs.txt')
    word2vec = {}
    for i in range(len(word_list)):
        word2vec[word_list[i]] = vecs[i]
    return word2vec

# 如果该单词在sentiment_dict中也存在，说明该单词具有情感得分，那么将该单词的向量乘以对应的情感得分；如果该单词不在sentiment_dict中，则将其向量保持不变。返回加权单词向量字典word2vec
def get_weighted_word_vectors():
    word2vec = get_word_vectors()
    sentiment_dict = get_sentiment_dict()
    for i in word2vec.keys():
        if i in sentiment_dict.keys():
            word2vec[i] = sentiment_dict[i] * word2vec[i]
    return word2vec


def process_data(sentence_length, words_size, embed_size):
    sentences = get_data()
    # 计算句子中每个词的频率
    frequency = collections.Counter() 
    for sentence in sentences:
        for word in sentence:
            frequency[word] += 1
    word2index = dict()
    # 循环查看频率计数器对象中最常见的words_size单词，并为每个单词分配一个从1开始的唯一索引（0是为填充保留）。
    for i, x in enumerate(frequency.most_common(words_size)):
        word2index[x[0]] = i + 1
    word2vec = get_weighted_word_vectors() 
    # 创建一个大小为(words_size + 1, embed_size)的零张量来存储词嵌入。
    word_vectors = torch.zeros(words_size + 1, embed_size)
    # 循环浏览word2index词典，并将相应的嵌入分配给word_vectors张量。
    for k, v in word2index.items():
        word_vectors[v, :] = torch.from_numpy(word2vec[k])
    rs_sentences = []
    for sentence in sentences:
        sen = []
        #  如果数据集句子中的词在word2index词典中，则将相应的索引追加到sen列表中；如果该词不在word2index词典中，则在sen列表中添加0
        for word in sentence:
            if word in word2index.keys():
                sen.append(word2index[word])
            else:
                sen.append(0)
        # 如果sen列表的长度小于sentence_length，则用0填充，使其长度为sentence_length； 如果sen列表的长度大于sentence_length，则将其截断为sentence_length长度。
        if len(sen) < sentence_length:
            sen.extend([0 for _ in range(sentence_length - len(sen))])
        else:
            sen = sen[:sentence_length]
        rs_sentences.append(sen)
    # 创建一个1和0的标签数组（各50000个）并将其转换为一个numpy数组。
    label = [1 for _ in range(50000)]
    label.extend([0 for _ in range(50000)])
    label = np.array(label)
    # 返回经过处理的句子（rs_sentences）、标签（label）和词嵌入（word_vectors）
    return rs_sentences, label, word_vectors
