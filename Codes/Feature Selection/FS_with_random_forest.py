import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

from sklearn.ensemble import ExtraTreesClassifier

data = pd.read_csv('Predict_gleason_score.csv')

#Input X and Target Y (here gleason_score)
X = np.array(data.drop(['Unnamed: 0','gleason_score'],1))
Y = np.array(data['gleason_score'])

minmax_scale = preprocessing.MinMaxScaler().fit(X)
normalX = minmax_scale.transform(X)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(normalX, Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)

#[::-1] is for reverse or ascending order
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(20):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

#Drop all the columns apart from the top 20 columns based on importance ranking    
index = indices[21:-1]
data.drop(data.columns[[index]], axis=1, inplace=True)
data.to_csv('top20.csv', sep=',')
