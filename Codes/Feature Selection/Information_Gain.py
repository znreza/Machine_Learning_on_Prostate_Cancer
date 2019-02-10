import pandas as pd
import pickle
import numpy as np
from sklearn import preprocessing

with open('Predict_gleason_score.pickle','rb') as f:
    data = pickle.load(f)
    
Y = np.array(data['gleason_score'])
X = np.array(data.drop(['Unnamed: 0','gleason_score'],1))

minmax_scale = preprocessing.MinMaxScaler().fit(X)
normalX = minmax_scale.transform(X)

def information_gain(X, y):

    def _calIg():
        entropy_x_set = 0
        entropy_x_not_set = 0
        for c in classCnt:
            try:
                probs = classCnt[c] / float(featureTot)
                entropy_x_set = entropy_x_set - probs * np.log(probs)
                probs = (classTotCnt[c] - classCnt[c]) / float(tot - featureTot)
                entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
            except RuntimeWarning:
                entropy_x_not_set = entropy_x_not_set - 0
        for c in classTotCnt:
            if c not in classCnt:
                try: 
                    probs = classTotCnt[c] / float(tot - featureTot)
                    entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
                except RuntimeWarning:
                    entropy_x_not_set = entropy_x_not_set - 0
        return entropy_before - ((featureTot / float(tot)) * entropy_x_set
                             +  ((tot - featureTot) / float(tot)) * entropy_x_not_set)

    tot = X.shape[0]
    classTotCnt = {}
    entropy_before = 0
    for i in y:
        if i not in classTotCnt:
            classTotCnt[i] = 1
        else:
            classTotCnt[i] = classTotCnt[i] + 1
    for c in classTotCnt:
        probs = classTotCnt[c] / float(tot)
        entropy_before = entropy_before - probs * np.log(probs)

    nz = X.T.nonzero()
    pre = 0
    classCnt = {}
    featureTot = 0
    information_gain = []
    for i in range(0, len(nz[0])):
        if (i != 0 and nz[0][i] != pre):
            for notappear in range(pre+1, nz[0][i]):
                information_gain.append(0)
            ig = _calIg()
            information_gain.append(ig)
            pre = nz[0][i]
            classCnt = {}
            featureTot = 0
        featureTot = featureTot + 1
        yclass = y[nz[1][i]]
        if yclass not in classCnt:
            classCnt[yclass] = 1
        else:
            classCnt[yclass] = classCnt[yclass] + 1
    ig = _calIg()
    information_gain.append(ig)

    return np.asarray(information_gain)

def ig(X, y):

    def get_t1(fc, c, f):
        t = np.log2(fc/(c * f))
        t[~np.isfinite(t)] = 0
        return np.multiply(fc, t)

    def get_t2(fc, c, f):
        t = np.log2((1-f-c+fc)/((1-c)*(1-f)))
        t[~np.isfinite(t)] = 0
        return np.multiply((1-f-c+fc), t)

    def get_t3(c, f, class_count, observed, total):
        nfc = (class_count - observed)/total
        t = np.log2(nfc/(c*(1-f)))
        t[~np.isfinite(t)] = 0
        return np.multiply(nfc, t)

    def get_t4(c, f, feature_count, observed, total):
        fnc = (feature_count - observed)/total
        t = np.log2(fnc/((1-c)*f))
        t[~np.isfinite(t)] = 0
        return np.multiply(fnc, t)

    # counts
    # n_classes * n_features
    observed = np.dot(Y.T,X)          
    total = observed.sum(axis=0).reshape(1, -1).sum()
    feature_count = X.sum(axis=0).reshape(1, -1)
    class_count = (X.sum(axis=1).reshape(1, -1) * Y).T

    # probs
    f = feature_count / feature_count.sum()
    c = class_count / float(class_count.sum())
    fc = observed / total

    # the feature score is averaged over classes
    scores = (get_t1(fc, c, f) +
            get_t2(fc, c, f) +
            get_t3(c, f, class_count, observed, total) +
            get_t4(c, f, feature_count, observed, total)).mean(axis=0)

    scores = np.asarray(scores).reshape(-1)

    return scores, []

subset = information_gain(normalX, Y)
subset = ig(normalX,Y)
