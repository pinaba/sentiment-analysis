import torch.utils.data
import numpy as np

# 用来分批加载数据以训练深度学习模型
class MyData(torch.utils.data.Dataset):
    def __init__(self, dt, lb):
        # dt和lb分别代表数据和相应的标签,这些参数被存储在类的变量，self.dt 和 self.lb中
        self.dt = dt
        self.lb = lb

    def __len__(self):
        # 返回数据集的长度
        return len(self.dt)

    def __getitem__(self, index):
        # 给定一个索引，它返回标签和相应索引的数据。标签作为一个元组的第一个元素被返回，数据作为第二个元素被返回。数据被返回为一个numpy数组，而标签则是一个整数。
        return self.lb[index], np.array(self.dt[index])

