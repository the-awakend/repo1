import os
import re
"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表
"""
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
        vocabList = list(vocabSet)
    data_dic = {}
    for i in range(len(vocabList)):
        data_dic[vocabList[i]] = i
    return data_dic

"""
函数说明:接收一个大字符串并将其解析为字符串列表
"""
def textParse(bigString):  # 将字符串转换为字符列表
    listOfTokens = re.split(r'\W*', bigString)  # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 除了单个字母，例如大写的I，其
    # 它单词变成小写

'''
得到数据集
'''
def get_data(filepath):
    files_num = 0
    docList = []
    fullText = []
    spam_path = filepath + '/spam'
    ham_path = filepath + '/ham'
    print("loding data...")
    for emailName in os.listdir(spam_path):
        print(spam_path + '/' + emailName)
        files_num += 1
        with open(spam_path + '/' + emailName, 'r') as f:
            wordList = textParse(f.read())
            wordList.append(1)  # 标记垃圾邮件，1表示垃圾文件
            docList.append(wordList)
            fullText.append(wordList)
    for emailName in os.listdir(ham_path):
        files_num += 1
        with open(ham_path + '/' + emailName, 'r') as f:
            wordList = textParse(f.read())
            wordList.append(0)
            docList.append(wordList)
            fullText.append(wordList)

    vocabDict = createVocabList(docList)  # 创建词汇表，不重复
    return files_num, docList, fullText, vocabDict



