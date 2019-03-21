import numpy as np


'''
    朴素贝叶斯分类函数
    email_array: 要预测的邮件数组
    povec: 正常邮件类的条件概率数组
    p1Vec - 垃圾邮件类的条件概率数组
    p_of_spam - 文档属于垃圾邮件的概率
'''
def NaiveBayes(email_array, p0vec, p1vec, p_of_spam):
    p1 = sum(email_array * p1vec)+np.log(p_of_spam) # 这里p1vec已经是取过对数的值了，不必再取对数,
    p0 = sum(email_array*p0vec)+np.log(1.0-p_of_spam)
    if p1 > p0:
        return 1
    else:
        return 0



"""
函数说明:朴素贝叶斯分类器训练函数
Parameters:
    trainMatrix - 文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵，必须是ndarray或mat
    trainCategory - 类别标签向量
Returns:
    p0Vect - 正常邮件类的条件概率数组
    p1Vect - 垃圾邮件类的条件概率数组
    p_of_spam - 文档属于垃圾邮件类的概率
"""
def trainNB0(trainMatrix, trainCategory):
    # 因为词表长度为646，对于一封邮件e，就一共有646个特征。每个特征的值代表出现和不出现(或出现次数)
    num_docs = len(trainMatrix)  # 计算训练的文档数目
    num_col = len(trainMatrix[0]) # 计算矩阵列数，即数据维度，也是词条数
    p_of_spam = sum(trainCategory) / float(num_docs)  # 文档属于垃圾邮件类的概率，先验概率
    # 计算属于类别1和类别2的条件概率矩阵
    p0Num = np.ones(num_col)
    p1Num = np.ones(num_col)  # 创建numpy.ones数组,词条出现数初始化为1,拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0  # 分母初始化为2 ,拉普拉斯平滑
    for i in range(num_docs):
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i] # p1Num[0]是指标签为1的时候，各个特征出现的次数(显然在使用词袋模型的时候不是这样的，但是依然可以这么使用，相当于给这个特征加了权重？)
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom) # p1Vec中的单项表示p(xi|y=c1),也即条件概率
    # 即，y=c1的时候，邮件中出现xi这个特征的概率.也就是，对于一个样本X而言，p(Xj=xj|y=1)
    p0Vect = np.log(p0Num / p0Denom)   #取对数，防止下溢出
    return p0Vect, p1Vect, p_of_spam  # 先验概率
