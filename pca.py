# -*- coding: utf-8 -*-

from numpy import *

def loadDataSet(fileName, delim = '\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat = 9999999):
    meanVals = mean(dataMat, axis = 0)
    # 首先去平均值
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar =False)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    # 从小到大对N个值排序
    eigValInd = eigValInd[: -(topNfeat+1) : -1]
    redEigVects = eigVects[:, eigValInd]
    # 将数据切换到新的空间
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean():
    dataMat = loadDataSet('secom.data.txt', '')
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):
        # 计算所有非 NaN 的平均值
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i])
        # 将所有 NaN 置为平均值
        dataMat[nonzero(isnan(dataMat[:,i].A))[0], i] = meanVal
    return dataMat



