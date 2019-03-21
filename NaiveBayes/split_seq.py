import random

def split(data_set, shuffle=False, ratio = 0.2):
    if shuffle:
        random.shuffle(data_set)
    num = len(data_set)
    testSet = []
    trainingSet = []
    trainingIndex = set(range(num))
      # 创建存储训练集的索引值的列表和测试集的索引值的列表
    slice = random.sample(trainingIndex, int(num*ratio)) # 随机挑选出1/5作为训练集,个做测试
    testSet.extend([data_set[i] for i in slice])
    trainingSet.extend([data_set[i] for i in trainingIndex-set(slice)])
    return testSet, trainingSet
