import sklearn
# Loading requisite libraries
import pandas as pd
import numpy as np

data = pd.read_csv("data/data/train.csv")
Xtrn = data.drop(data.columns[[0]], axis=1)
Ytrn = data[data.columns[0]]

data = pd.read_csv("data/data/test.csv")
Xtest = data.drop(data.columns[[0]], axis=1)
ID = data[data.columns[0]]

data = pd.read_csv("data/data/validation_1.csv")
Xval1 = data.drop(data.columns[[0]], axis=1)
Yval1 = data[data.columns[0]]

data = pd.read_csv("data/data/validation_2.csv")
Xval2 = data.drop(data.columns[[0]], axis=1)
Yval2 = data[data.columns[0]]

data = pd.read_csv("data/data/validation_3.csv")
Xval3 = data.drop(data.columns[[0]], axis=1)
Yval3 = data[data.columns[0]]

someX = Xtrn.append(Xval3)
someX = someX.append(Xval1)
someX = someX.append(Xval2)
someY = Ytrn.append(Yval3)
someY = someY.append(Yval1)
someY = someY.append(Yval2)
print len(someX)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis(tol=1.0e-2,reg_param=0.08)
clf.fit(Xtrn, Ytrn)

target = clf.predict(Xtest)
filename = open('final.csv','w')
filename.write('ID,TARGET\n')
for i in range(len(ID)):
    filename.write(str(ID[i])+','+str(target[i])+'\n')
filename.close()
