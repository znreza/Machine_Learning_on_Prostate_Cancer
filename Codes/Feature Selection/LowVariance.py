import pandas as pd
import pickle
import numpy as np
from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA,KernelPCA
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

data = pd.read_csv('Predict_gleasonScore.csv')

#Input X and Target Y (here gleason_score)
X = np.array(data.drop(['gleason_score','Unnamed: 0'],1))
Y = np.array(data['gleason_score'])

X = preprocessing.scale(X)

selector = VarianceThreshold()
new_features = selector.fit_transform(X)

print(X.shape)
print(new_features.shape)
#print(len(X.T)) #no. of columns

feature_indices = selector.get_support(indices=True)
remove = []
for i in range(len(X.T)):
    if i not in feature_indices:
        remove.append(i)
        print(i)

#Drop feature columns with low variance        
data = data.drop(data.columns[remove],axis=1)

new_X = np.array(data.drop(['gleason_score','Unnamed: 0'],1))
new_X = preprocessing.scale(new_X)        
print(new_X.shape)
data.to_csv('newFeatures1.csv', sep=',')
