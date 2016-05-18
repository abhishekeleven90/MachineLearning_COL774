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

def dtree():
    ##Decision Tree

    from sklearn.tree import DecisionTreeClassifier

    dtree = DecisionTreeClassifier(criterion="entropy",max_depth=20, min_samples_leaf=4, splitter = "best", min_samples_split=10)
    dtree.fit(Xtrn,Ytrn)

    print dtree.get_params()
    print dtree.score(Xtrn,Ytrn)
    print dtree.score(Xval1, Yval1)
    print dtree.score(Xval2, Yval2)
    print dtree.score(Xval3, Yval3)
    
def bayes():
    ##Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(Xtrn, Ytrn)
    print gnb.score(Xtrn, Ytrn)
    print gnb.score(Xval1, Yval1)
    print gnb.score(Xval2, Yval2)
    print gnb.score(Xval3, Yval3)
    
def linearsvm():
    ##SVM
    from sklearn import svm
    lin = svm.LinearSVC(tol=1e-6)
    lin.fit(Xtrn, Ytrn)
    print lin.score(Xtrn, Ytrn)
    print lin.score(Xval1, Yval1)
    print lin.score(Xval2, Yval2)
    print lin.score(Xval3, Yval3)
    
def random():
    from sklearn.ensemble import RandomForestClassifier
    rtree = RandomForestClassifier(n_estimators=11, criterion='gini', max_depth=8,  min_samples_leaf=5, min_samples_split=10)
    rtree.fit(Xtrn,Ytrn)
    print rtree.score(Xtrn, Ytrn)
    print rtree.score(Xval1, Yval1)
    print rtree.score(Xval2, Yval2)
    print rtree.score(Xval3, Yval3)
    
def gaussian():
    ##SVM
    from sklearn import svm
    gauss = svm.SVC(kernel='rbf', C=1, gamma=0.012)
    gauss.fit(Xtrn, Ytrn)
    print gauss.score(Xtrn, Ytrn)
    print gauss.score(Xval1, Yval1)
    print gauss.score(Xval2, Yval2)
    print gauss.score(Xval3, Yval3)
