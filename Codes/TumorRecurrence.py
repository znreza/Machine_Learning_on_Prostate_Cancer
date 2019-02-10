import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA,KernelPCA
from sklearn.metrics import roc_curve, auc, mean_squared_error
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

data = pd.read_csv('Predict_tumor_rec.csv')

#Drop the rows where 't_stage' has no value (i.e. Not Available)
data = data[~data['tumor'].isin(['[Not Available]'])] 

#Drop non-required columns
X = np.array(data.drop(['Unnamed: 0','gleason_score','tumor'],1))

#Target 'tumor'
Y = np.array(data['tumor'])

minmax_scale = preprocessing.MinMaxScaler().fit(X)
normalX = minmax_scale.transform(X)

#Principal Component Analysis
pca = PCA(n_components=10)
pca_fit = pca.fit_transform(normalX)

#Linear Discriminant Analysis follwed by PCA
lda = LinearDiscriminantAnalysis(n_components = 4,solver='svd', shrinkage=None)
X_features = lda.fit_transform(pca_fit,Y)
X_features = np.array(X_features)

#10-fold cross-validation (x_test and y_test is holdout for final test)
X_features, x_test, Y, y_test = cross_validation.train_test_split(X_features,Y,test_size=0.2)
accuracy = []
kf = KFold(n_splits=10,shuffle=True)
for train_index, test_index in kf.split(X_features):
               
    X_train, X_test = X_features[train_index],X_features[test_index]
    Y_train, Y_test = Y[train_index],Y[test_index]
                
    clf = RandomForestClassifier(n_estimators=100,n_jobs=-1, random_state=None)
    #clf = svm.SVC(kernel='poly')
    clf.fit(X_train,Y_train)
    prediction = clf.predict(X_test)
    accuracy.append(clf.score(X_test,Y_test))
    print(confusion_matrix(Y_test, prediction))
    print('\n')

print("mean: ",np.mean(accuracy))
print(clf.score(x_test,y_test))

colors = ['m','r','c','b','k','m','yellow','orchid','fuchsia','lightcoral','g']
markers = ['o','x','*','^','1','p','D','8','s','P','o']

le = preprocessing.LabelEncoder()
le.fit(Y)
label = le.transform(Y) 

for i in range(len(X_features)):
    plt.scatter(X_features[i][0],X_features[i][1],c = colors[label[i]],marker= markers[label[i]])
plt.show()

#Test on holdout test set
prediction = clf.predict(x_test)
print(clf.score(x_test,y_test))
print(confusion_matrix(y_test, prediction))


