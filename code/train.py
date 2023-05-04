import torch as t
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.utils.data
from data_loader import MyData
from model import SLCABG
import data_util


device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
SENTENCE_LENGTH = 12 # 一个句子中允许的最大单词数
WORD_SIZE = 35000 # 模型中包含的唯一单词数
EMBED_SIZE = 768 # 每个单词的嵌入向量的大小


if __name__ == '__main__':
    # 通过调用data_util.process_data()的SENTENCE_LENGTH、WORD_SIZE和EMBED_SIZE等超参数，获得句子、标签和单词向量
    sentences, label, word_vectors = data_util.process_data(SENTENCE_LENGTH, WORD_SIZE, EMBED_SIZE)
    # 使用scikit-learn的train_test_split()将句子和标签分成x_train、x_test、y_train和y_test，测试大小为0.2
    x_train, x_test, y_train, y_test = train_test_split(sentences, label, test_size=0.2)

    # 用训练和测试数据创建MyData实例，并传递给批量大小为32的DataLoader实例。train_data_loader在训练过程中会对数据进行洗牌，而test_data_loader不会对数据进行洗牌。
    train_data_loader = torch.utils.data.DataLoader(MyData(x_train, y_train), 32, True)
    test_data_loader = torch.utils.data.DataLoader(MyData(x_test, y_test), 32, False)

    net = SLCABG(EMBED_SIZE, SENTENCE_LENGTH, word_vectors).to(device)
    optimizer = t.optim.Adam(net.parameters(), 0.01) # 使用Adam，学习速率为0.01
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    tp = 1
    tn = 1
    fp = 1
    fn = 1
    for epoch in range(15):
        for i, (cls, sentences) in enumerate(train_data_loader):
            optimizer.zero_grad()
            sentences = sentences.type(t.LongTensor).to(device)
            cls = cls.type(t.LongTensor).to(device)
            out = net(sentences)
            _, predicted = torch.max(out.data, 1)
            predict = predicted.cpu().numpy().tolist()
            pred = cls.cpu().numpy().tolist()
            for f, n in zip(predict, pred):
                if f == 1 and n == 1:
                    tp += 1
                elif f == 1 and n == 0:
                    fp += 1
                elif f == 0 and n == 1:
                    fn += 1
                else:
                    tn += 1
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * r * p / (r + p)
            acc = (tp + tn) / (tp + tn + fp + fn)
            loss = criterion(out, cls).to(device)
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
                print('acc', acc, 'p', p, 'r', r, 'f1', f1)

    net.eval()
    print('==========================================================================================')
    with torch.no_grad():
        tp = 1
        tn = 1
        fp = 1
        fn = 1
        for cls, sentences in test_data_loader:
            sentences = sentences.type(t.LongTensor).to(device)
            cls = cls.type(t.LongTensor).to(device)
            out = net(sentences)
            _, predicted = torch.max(out.data, 1)
            predict = predicted.cpu().numpy().tolist()
            pred = cls.cpu().numpy().tolist()
            for f, n in zip(predict, pred):
                if f == 1 and n == 1:
                    tp += 1
                elif f == 1 and n == 0:
                    fp += 1
                elif f == 0 and n == 1:
                    fn += 1
                else:
                    tn += 1
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * r * p / (r + p)
        acc = (tp + tn) / (tp + tn + fp + fn)
        print('acc', acc, 'p', p, 'r', r, 'f1', f1)
