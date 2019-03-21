import numpy as np
import build_vec as bv
from data_loader import *
from split_seq import split
import model

def spamTest():
    fileName = 'email'
    files_num, docList, fullText, vocabDict = get_data(fileName)
    # 1/5的数据作为测试集
    testSet, trainingSet = split(docList)
    train_label = [trainingSet[i][-1] for i in range(len(trainingSet))]
    trainingSet = [trainingSet[i][: -1] for i in range(len(trainingSet))]
    trainMat = []
    for item in trainingSet:
        trainMat.append(bv.setOfWords2Vec(vocabDict, item))
    # 训练模型
    p0_vec, p1_vec, p_of_spam = model.trainNB0(np.array(trainMat), np.array(train_label))

    #测试
    errorCount = 0
    test_label = [testSet[i][-1] for i in range(len(testSet))]
    testSet = [testSet[i][: -1] for i in range(len(testSet))]
    for i in range(len(testSet)):
        word_vector = bv.setOfWords2Vec(vocabDict, testSet[i])
        if model.NaiveBayes(np.array(word_vector), p0_vec, p1_vec, p_of_spam) != test_label[i]:
            errorCount += 1
            print("分类错误的测试集：", docList[i])
    print("测试集数量:", len(test_label))
    print("错误预测数量:", errorCount)
    print("错误率: %.2f%%" % (float(errorCount)/len(testSet)*100))

spamTest()