import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

filename = 'train(1).csv'

raw_data = open(filename,'r')

reader = csv.reader(raw_data, delimiter = ',', quoting = csv.QUOTE_NONE)
x = list(reader)


data = pd.read_csv(filename)

data.fillna(data.median(numeric_only=True).round(1), inplace = True)

X = data.loc[:, data.columns != 'Class']
Y = data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)


knn_results = []
max_knn =[]
max_knn.append(0)
for i in range(251):
    classifier = KNeighborsClassifier(n_neighbors = i+1)
    classifier.fit(X_train,Y_train)
    knn_score = classifier.score(X_test,Y_test)
    knn_results.append(knn_score)
    if(knn_score>max_knn[0]):
        if(len(max_knn) < 2):
            max_knn.append(i)
            max_knn[0] = knn_score
        else:
            max_knn[1] = i
            max_knn[0] = knn_score
        
    
print("Knn best k: " + str(max_knn[0]) + ' ' + str(max_knn[1]))

x_axis = range(len(knn_results))

plt.plot(x_axis,knn_results)
plt.show()


boost_results = []
max_boost= []
max_boost.append(0)
for i in range(251):
    adaboost = AdaBoostClassifier(n_estimators = i+1, learning_rate = 0.2).fit(X_train, Y_train)
    score = adaboost.score(X_test, Y_test)
    boost_results.append(score)
    if(score>max_boost[0]):
        if(len(max_boost) < 2):
            max_boost.append(i)
            max_boost[0] = score
        else:
            max_boost[1] = i
            max_boost[0] = score
        
print("Adaboost best num of estimators: " +str(max_boost[0]) + ' ' + str(max_boost[1]))

plt.plot(x_axis,boost_results)
plt.show()




fn = 'test(1).csv'

data_test = pd.read_csv(fn)

X_t = data_test.loc[:, data_test.columns != 'Class']
Y_t = data_test['Class']


#score_test = adaboost.score(X_t,Y_t)

#knn_test_score = classifier.score(X_t,Y_t)

#print(adaboost)

#print(score)
#print(score_test)

#print("Knn testna " + str(knn_score))
#print("Knn proba " + str(knn_test_score))

#print(X)
#print(X.to_numpy())
#X_embedded = TSNE().fit_transform(X.to_numpy().astype(np.float32))
#print(X_embedded.shape)
#colors = ['blue','green','red']
#for i, x in enumerate(X_embedded):
#    plt.scatter(x[0], x[1], color=colors[int(Y[i])])
#plt.show()
# Implementiran KNN in ADABOOST nekako mi je malo cudno glede vsega saj so vrjetnosti pravilnih napovedi zelo visoke

