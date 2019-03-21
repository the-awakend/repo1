# -*- coding: UTF-8 -*-


"""
函数说明:构建词集模型，根据vocabDict词汇表，将inputSet向量化，向量的每个元素为1或0
Parameters:
    vocabDict - createvocabDict返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词集模型
"""


def setOfWords2Vec(vocabDict, inputSet):
    one_hot_Vec = [0] * len(vocabDict)               # 创建一个其中所含元素都为0的向量
    for word in inputSet:                          # 遍历每个词条
        if word in vocabDict:                      # 如果词条存在于词汇表中，则置1
            index = vocabDict[word]
            one_hot_Vec[index] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return one_hot_Vec        # 返回文档向量


"""
函数说明:词袋模型，根据vocabDict词汇表，构建词袋模型
Parameters:
    vocabDict - createvocabDict返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词袋模型
"""


def bag0fWordsVec(vocabDict, inputSet):
    wordBag_Vec = [0] * len(vocabDict)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabDict: # # 如果词条存在于词汇表中，则计数加一
            index = vocabDict[word]
            wordBag_Vec[index] += 1
    return wordBag_Vec
