
# coding: utf-8

# In[3]:

from collections import Counter

import pandas as pd


def orientWithMedian(dataFrame, column):
    columnMedian = dataFrame[column].median()
    dataFrame.loc[dataFrame[column] <= columnMedian, column] = 0
    dataFrame.loc[dataFrame[column] > columnMedian, column] = 1
    ##changed no need to return!
    
def orientAllMedians(dataFrame):
    ourList = ['Age','SibSp','Parch','Ticket','Fare','Cabin_b']
    for col in ourList:
        orientWithMedian(dataFrame, col)

class Tree:
    
    def __init__(self):
        self.parent = None ##type Tree
        self.children = []
        self.label = None
        self.classCounts = None ##empty dict
        self.splitFeatureValue = None
        self.splitFeature = None ##name of the feature
    
    def __str__(self): ##will help in debugging
        toreturn = ''
        toreturn = toreturn + 'Label: '+str(self.label) +'\n'
        toreturn = toreturn + 'splitFeature: '+str(self.splitFeature) +'\n'
        toreturn = toreturn + 'classCounts: '+str(self.classCounts) +'\n'
        toreturn = toreturn + 'Number of children: '+str(len(self.children)) +'\n'
        toreturn = toreturn + 'The splitFeature value of parent by using which we came to this node: '+str(self.splitFeatureValue)
        return toreturn
    
class MaxAns:
    
    def __init__(self):
        self.attribute = None ##type Tree
        self.length = 0
    
def bfs(tree):
    finalans = []
    queue=[]
    queue.append(tree)
    while len(queue)!=0:
        curr = queue.pop(0)
        finalans.append(curr)
        for child in curr.children:
            queue.append(child)
    finalans.reverse()
    print len(finalans)
    return finalans


def removeNodeFromTree(node):
    papa = node.parent
    if papa is None: ##root
        return
    papa.children.remove(node)
    return papa

def addNodeToOriginal(father, child):
    father.children.append(child)
    return father, child ##no need though


def allLabelsSame(Ytrain):
    #print len(set(Ytrain))
    return len(set(Ytrain))<=1

def bestLabelInFavor(labels, root):
    allCounts = Counter(list(labels))
    maxCount = 0
    maxLabel = None
    for key in allCounts:
        if (allCounts[key]>maxCount):
            maxLabel = key
            maxCount = allCounts[key]
    root.label = maxLabel
    root.classCounts = allCounts
    #print '-------'
    #print root
    #print '-------'
    return root

def entropy(labels): ##for a particular node
    import math
    counts = Counter(list(labels))
    n = len(labels)
    sum = 0
    for key in counts:
        count = counts[key]
        if count!=0:
            p = float(count)/float(n)
            sum = sum - (p * math.log(p,2))
    return sum

def info_gain(features, labels, splitFeature):
    ##based on the splitFeature - all the values that it can take
    ##we have to divide the features and their splitfearure accordingly
    ##and then compute the fraction * entropy add and minus from overall entropy
    current_entropy = entropy(labels)
    all_values = set(features[splitFeature])
    total_len = len(features)
    for value in all_values:
        boolList = (features[splitFeature]==value)
        subLabels = labels[boolList]
        subLen = len(subLabels)
        subFraction = float(subLen)/total_len
        subEntropy = entropy(subLabels)
        current_entropy = current_entropy - (subFraction*subEntropy)
    return current_entropy

def getBestFeature(features, labels, remainingOnes):
    bestFeature = None
    maxGain = 0
    for currFeature in remainingOnes: ##going in order!
        currGain = info_gain(features, labels, currFeature)
        #print currGain, currFeature
        if currGain>maxGain:##since going in order, ties are handled that way!
            maxGain = currGain
            bestFeature = currFeature
    return bestFeature, maxGain
    


def build(Xtrain, Ytrain, root, feature_list):
    
    global count
    
    ##return a leaf node with this label
    ##no point to go any futher!
    if allLabelsSame(Ytrain):
        leaf = root
        label_only = list(set(Ytrain))[0]
        #rint 'Only label: '+str(label_only)
        leaf.label = label_only
        leaf.classCounts = Counter(list(Ytrain))
        return leaf ##returns basically the root

    ##if all features finished, assuming no feature repeats
    ##use this!
    if len(feature_list)==0:
        return bestLabelInFavor(Ytrain, root) ##returns basically the root
    
    
    ##get bestFeature and maxGain
    bestFeature, maxGain = getBestFeature(Xtrain, Ytrain, feature_list)
    #print bestFeature, maxGain 
    
    ##or if the bestFeature is None
    if maxGain==0: ##there is no point in splitting 
        return bestLabelInFavor(Ytrain, root)
    
    
    #print bestFeature
    root.splitFeature = bestFeature ## set the splitfeature, dont include the 
    #print root.splitFeature
    
    ##also forgot this in first go
    root.classCounts = Counter(list(Ytrain))
    
    
    best_feature_values = set(Xtrain[bestFeature])  
    
    
    
    for value in best_feature_values:
        boolList = (Xtrain[bestFeature]==value)
        subLabels = Ytrain[boolList]
        subFeatures = Xtrain[boolList]
        
        ##make a new node
        newChild = Tree()
        count = count + 1
        newChild.parent = root
        newChild.splitFeatureValue = value
        
        ##append the new node to child list
        root.children.append(newChild)
        
        
        ##remove this feature from the original list of features
        ##as this feature has been covered
        newFeatureList = list(feature_list)
        newFeatureList.remove(bestFeature)
        
        ##now the recursive call
        build(subFeatures, subLabels, newChild, newFeatureList)
        
        
    return root
    
#tnaya = build(Xtrn, Ytrn, Tree(), [])
#print tnaya

## for repeating numerical features
def isNumerical(feature):
    return feature in ['Age','SibSp','Parch','Ticket','Fare','Cabin_b']

## for repeating numerical features
def info_gain2(features, labels, splitFeature):
    ##based on the splitFeature - all the values that it can take
    ##we have to divide the features and their splitfearure accordingly
    ##and then compute the fraction * entropy add and minus from overall entropy
    
    if not isNumerical(splitFeature):
        return info_gain(features, labels, splitFeature)
    else:
        ##it is a numerical attribute
        current_entropy = entropy(labels)
        curr_median = features[splitFeature].median()
        total_len = len(features)
        
        boolList = (features[splitFeature]>curr_median)
        subLabels = labels[boolList]
        subLen = len(subLabels)
        subFraction = float(subLen)/total_len
        subEntropy = entropy(subLabels)
        current_entropy = current_entropy - (subFraction*subEntropy)
        
        boolList = (features[splitFeature]<=curr_median)
        subLabels = labels[boolList]
        subLen = len(subLabels)
        subFraction = float(subLen)/total_len
        subEntropy = entropy(subLabels)
        current_entropy = current_entropy - (subFraction*subEntropy)
        
        return current_entropy

## for repeating numerical features
def getBestFeature2(features, labels, remainingOnes):
    bestFeature = None
    maxGain = 0
    for currFeature in remainingOnes: ##going in order!
        currGain = info_gain2(features, labels, currFeature)
        #print currGain, currFeature
        if currGain>maxGain:##since going in order, ties are handled that way!
            maxGain = currGain
            bestFeature = currFeature
    return bestFeature, maxGain

## for repeating numerical features
def build2(Xtrain, Ytrain, root, feature_list):
    
    global count
    
    ##return a leaf node with this label
    ##no point to go any futher!
    if allLabelsSame(Ytrain):
        leaf = root
        label_only = list(set(Ytrain))[0]
        #rint 'Only label: '+str(label_only)
        leaf.label = label_only
        leaf.classCounts = Counter(list(Ytrain))
        return leaf ##returns basically the root

    ##if all features finished, but feature can repeat here in this so will not should not reach here!
    if len(feature_list)==0:
        return bestLabelInFavor(Ytrain, root) ##returns basically the root
    
    
    ##get bestFeature and maxGain
    bestFeature, maxGain = getBestFeature2(Xtrain, Ytrain, feature_list)
    #print bestFeature, maxGain 
    
    ##or if the bestFeature is None
    if maxGain==0: ##there is no point in splitting 
        return bestLabelInFavor(Ytrain, root)
    
    
    #print bestFeature
    root.splitFeature = bestFeature ## set the splitfeature, dont include the 
    #print root.splitFeature
    
    ##also forgot this in first go
    root.classCounts = Counter(list(Ytrain))
    
    if not isNumerical(bestFeature):
    
        best_feature_values = set(Xtrain[bestFeature])  



        for value in best_feature_values:
            boolList = (Xtrain[bestFeature]==value)
            subLabels = Ytrain[boolList]
            subFeatures = Xtrain[boolList]

            ##make a new node
            newChild = Tree()
            count = count + 1
            newChild.parent = root
            newChild.splitFeatureValue = value

            ##append the new node to child list
            root.children.append(newChild)


            ##remove this feature from the original list of features
            ##as this feature has been covered
            newFeatureList = list(feature_list)
            newFeatureList.remove(bestFeature)

            ##now the recursive call
            build2(subFeatures, subLabels, newChild, newFeatureList)
    else:
        ##is a numercial attribute!
        curr_median = Xtrain[bestFeature].median()
        
        ##child 1 left <=
        boolList = (Xtrain[bestFeature]<=curr_median)
        subLabels = Ytrain[boolList]
        subFeatures = Xtrain[boolList]

        ##make a new node
        newChild = Tree()
        count = count + 1
        newChild.parent = root
        newChild.splitFeatureValue = curr_median

        ##append the new node to child list
        root.children.append(newChild)


        ##remove this feature from the original list of features
        ##as this feature has been covered
        newFeatureList = list(feature_list)
        #newFeatureList.remove(bestFeature)

        ##now the recursive call
        build2(subFeatures, subLabels, newChild, newFeatureList)
        
        
        
        ###child 2 right >
        
        boolList = (Xtrain[bestFeature]>curr_median)
        subLabels = Ytrain[boolList]
        subFeatures = Xtrain[boolList]

        ##make a new node
        newChild = Tree()
        count = count + 1
        newChild.parent = root
        newChild.splitFeatureValue = curr_median

        ##append the new node to child list
        root.children.append(newChild)


        ##remove this feature from the original list of features
        ##as this feature has been covered
        newFeatureList = list(feature_list)
        #newFeatureList.remove(bestFeature)

        ##now the recursive call
        build2(subFeatures, subLabels, newChild, newFeatureList)

        
    return root

def getMaxCountClass(someCounts):
    maxCount = 0
    maxLabel = None
    for key in someCounts:
        if (someCounts[key]>maxCount):
            maxLabel = key
            maxCount = someCounts[key]
    return maxLabel, maxCount
    
##accuracy of our prediction
def predictAccuracy(tree, dataX, dataY):
    rows = len(dataX)
    #print str(rows)
    #print str(range(rows))
    count = 0
    for row in range(rows):
        #print 'Analysing row ## '+str(row) + ' ' +str(predict(tree, dataX.iloc[row]))+ ' '+ str(dataY.iloc[row])
        if (predict(tree, dataX.iloc[row])==(dataY.iloc[row])):
            count = count + 1
    return float(count)/rows

## for repeating numerical features
def predict2(tree, onerowdf):
    ##tree is the tree built by us
    ##onerowdf is the onerowdf from pandas df one by one
    
    #print '### AT NODE #####'
    #print tree
    #print 'Number of children @@@@ ' + str(len(tree.children))
#     for child in tree.children:
#         print '->->->'
#         print child
    
#     print 'Will be Seeing: '+str(tree.splitFeature)
#     print '### AT NODE #####\n\n'   
        
    if len(tree.children)==0:
        return tree.label
    else:
        ##use tree.splitFeature
        ##all children nodes have been splitted on this splitFeature
        ##for all children whose splitfeature is this value
        whichChild = []
        
        if not isNumerical(tree.splitFeature):
            for child in tree.children:
                if child.splitFeatureValue == onerowdf[tree.splitFeature]:
                    whichChild.append(child)
        else:
            median_value = tree.children[0].splitFeatureValue
            try:
                if onerowdf[tree.splitFeature] <= median_value:
                    whichChild.append(tree.children[0])
                elif onerowdf[tree.splitFeature] > median_value:
                    whichChild.append(tree.children[1])
            except: ##why this code because when plotting, we are pruning from back sometimes the child gets lost!!!
                maxLabel, maxCount = getMaxCountClass(tree.classCounts)
                #print 'result: '+str(maxLabel)
                return maxLabel
                
        

        if (len(whichChild))==0:
            ##here we return the max of the labels!
            #print tree.classCounts
            maxLabel, maxCount = getMaxCountClass(tree.classCounts)
            #print 'result: '+str(maxLabel)
            return maxLabel
        
        return predict2(whichChild[0], onerowdf)
    
## for repeating numerical features
##accuracy of our prediction
def predictAccuracy2(tree, dataX, dataY):
    rows = len(dataX)
    #print str(rows)
    #print str(range(rows))
    count = 0
    for row in range(rows):
        #print 'Analysing row ## '+str(row) + ' ' +str(predict(tree, dataX.iloc[row]))+ ' '+ str(dataY.iloc[row])
        if (predict2(tree, dataX.iloc[row])==(dataY.iloc[row])):
            count = count + 1
    return float(count)/rows

##onerowdf should be median oriented as in question
##the data should be handled previously before calling this function
def predict(tree, onerowdf):
    ##tree is the tree built by us
    ##onerowdf is the onerowdf from pandas df one by one
    
    #print '### AT NODE #####'
    #print tree
    #print 'Number of children @@@@ ' + str(len(tree.children))
#     for child in tree.children:
#         print '->->->'
#         print child
    
#     print 'Will be Seeing: '+str(tree.splitFeature)
#     print '### AT NODE #####\n\n'   
    
        
    
    if len(tree.children)==0:
        return tree.label
    else:
        ##use tree.splitFeature
        ##all children nodes have been splitted on this splitFeature
        ##for all children whose splitfeature is this value
        whichChild = []
        for child in tree.children:
            if child.splitFeatureValue == onerowdf[tree.splitFeature]:
                whichChild.append(child)
        
        if (len(whichChild))==0:
            ##here we return the max of the labels!
            #print tree.classCounts
            maxLabel, maxCount = getMaxCountClass(tree.classCounts)
            #print 'result: '+str(maxLabel)
            return maxLabel
        
        return predict(whichChild[0], onerowdf)

def getInfoForLeaf(listgiven):
    coveredAttribute = []
    firstValueDict = {}
    repeated = {}
    maxRepeated = ''
    maxLen = 0 
    
    for (a,b) in listgiven:
        if a not in coveredAttribute:
            coveredAttribute.append(a)
            firstValueDict[a] = b
        else:
            if not repeated.get(a):
                repeated[a]=[]
                repeated[a].append(firstValueDict[a])
            repeated[a].append(b)
            if(len(repeated[a])>maxLen):
                maxRepeated = a
                maxLen = len(repeated[a])
    ##repeated constrcuted till now
    ##now print the max length of all
    return repeated, maxRepeated, maxLen
    # repeated[maxRepeated]
    
#printInfoForLeaf([('A',19),('A',20),('B',1),('C',2)])

def partCHelper(tree):
    ourAns = MaxAns()
    traverseWithList(tree,[], ourAns)
    print ourAns.attribute, ourAns.length
    
def traverseWithList(tree,listgiven, ourAns):

    if tree.children==[]:
        repeated, maxRepeated, maxLen = getInfoForLeaf(listgiven)
        #print len(repeated)
        if len(repeated)>0:
            print repeated
            #print maxLen
            if maxLen > ourAns.length :
                #print 'here'
                ourAns.length = maxLen 
                ourAns.attribute = maxRepeated
    else:
        for child in tree.children:
            listgiven.append((tree.splitFeature,child.splitFeatureValue))
            traverseWithList(child, listgiven, ourAns)
            listgiven.pop()

def plot(topmost, Xtrn, Ytrn, valX, valY, testX, testY, kind = 1): 
    
    ##kind =1 for original kind, repetion not allowed
    ##kind =2 for new kind, repetion allowed on numerical
    
    ##plot accuracies
    trainAccList = []
    valAccList = []
    testAccList = []
    nodeCountList = []
    
    traverselist = bfs(topmost) ##this bfs is the reverse traversal
    totalNodes = len(traverselist)
    
    for node in traverselist:
        print totalNodes, 
        father=removeNodeFromTree(node)
        
        
        if kind==1:
            tempAccVal = predictAccuracy(topmost, valX, valY)
            tempAccTrain = predictAccuracy(topmost, Xtrn, Ytrn)
            tempAccTest = predictAccuracy(topmost, testX, testY)
        else:
            tempAccVal = predictAccuracy2(topmost, valX, valY)
            tempAccTrain = predictAccuracy2(topmost, Xtrn, Ytrn)
            tempAccTest = predictAccuracy2(topmost, testX, testY)
        
        nodeCountList.append(totalNodes)
        trainAccList.append(tempAccTrain)
        valAccList.append(tempAccVal)
        testAccList.append(tempAccTest)
        
        totalNodes = totalNodes - 1
    
    nodeCountList.reverse()
    trainAccList.reverse()
    valAccList.reverse()
    testAccList.reverse()
    
    import matplotlib.pyplot as plt
    plt.plot(nodeCountList, trainAccList)
    plt.plot(nodeCountList, valAccList)
    plt.plot(nodeCountList, testAccList)
    plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of nodes')
    plt.show()


def partA(plotnow=False):
    
    global count
    count = 1 ## to account for the root  that we pass in argument.
    
    data = pd.read_csv("train.csv")
    Xtrn = data.drop("Survived",1)
    Ytrn = data["Survived"]

    data = pd.read_csv("validation.csv")
    valX = data.drop("Survived",1)
    valY = data["Survived"]

    data = pd.read_csv("test.csv")
    testX = data.drop("Survived",1)
    testY = data["Survived"]
    
    orientAllMedians(Xtrn)
    orientAllMedians(valX)
    orientAllMedians(testX)
    
    
    col_list = list(Xtrn.columns)
    print col_list
    
    topmost = build(Xtrn, Ytrn, Tree(),col_list)
    print topmost, count
    
    ##part a    
    print '\n\nBefore Pruning'
    print predictAccuracy(topmost, Xtrn , Ytrn)
    accVal =  predictAccuracy(topmost, valX, valY)
    print accVal
    print predictAccuracy(topmost, testX, testY)
    
    if plotnow:
        plot(topmost, Xtrn, Ytrn, valX, valY, testX, testY, 1)
    
    
    

def partB(plotnow=False):
    
    global count
    global topmost
    
    count = 1 ## to account for the root  that we pass in argument.
    
    data = pd.read_csv("train.csv")
    Xtrn = data.drop("Survived",1)
    Ytrn = data["Survived"]

    data = pd.read_csv("validation.csv")
    valX = data.drop("Survived",1)
    valY = data["Survived"]

    data = pd.read_csv("test.csv")
    testX = data.drop("Survived",1)
    testY = data["Survived"]
    
    orientAllMedians(Xtrn)
    orientAllMedians(valX)
    orientAllMedians(testX)
    
    
    col_list = list(Xtrn.columns)
    print col_list
    
    topmost = build(Xtrn, Ytrn, Tree(),col_list)
    print topmost, count
    
    
    accVal =  predictAccuracy(topmost, valX, valY)
    print accVal
    
    ##part b - post pruning
    traverselist = bfs(topmost)
    
    delCount = 0
    
    for node in traverselist:
        father=removeNodeFromTree(node)
        tempAccVal = predictAccuracy(topmost, valX, valY)
        if tempAccVal > accVal:
            delCount = delCount + 1
            accVal = tempAccVal
        else:
            if father is not None:
                addNodeToOriginal(father, node)
            
    print delCount, accVal ##number of deleted nodes and now the accuracy
    
    
    print '\n\nAfter Pruning'
    print predictAccuracy(topmost, Xtrn , Ytrn)
    accVal =  predictAccuracy(topmost, valX, valY)
    print accVal
    print predictAccuracy(topmost, testX, testY)
    
    if plotnow:
        plot(topmost, Xtrn, Ytrn, valX, valY, testX, testY, 1)


def partC(plotnow=False):
    global count
    count = 1 ## to account for the root  that we pass in argument.
    
    data = pd.read_csv("train.csv")
    Xtrn = data.drop("Survived",1)
    Ytrn = data["Survived"]

    data = pd.read_csv("validation.csv")
    valX = data.drop("Survived",1)
    valY = data["Survived"]

    data = pd.read_csv("test.csv")
    testX = data.drop("Survived",1)
    testY = data["Survived"]
    
    #orientAllMedians(Xtrn)
    #orientAllMedians(valX)
    #orientAllMedians(testX)
    
    
    col_list = list(Xtrn.columns)
    print col_list
    
    topmost = build2(Xtrn, Ytrn, Tree(),col_list)
    print topmost, count
    
    print '\n\nNo Pruning and Repeating Allowed Part C'
    print predictAccuracy2(topmost, Xtrn , Ytrn)
    accVal =  predictAccuracy2(topmost, valX, valY)
    print accVal
    print predictAccuracy2(topmost, testX, testY)
    
    
    #global maxTotalCount
    #global maxAttribute
    #maxTotalCount = 0
    #maxAttribute = 0
    
    ##all that repeated thing they asked
    partCHelper(topmost)
    
    #print maxAttribute, maxTotalCount
    
    if plotnow:
        plot(topmost, Xtrn, Ytrn, valX, valY, testX, testY, 2)
    
#partC(True)
partB(False)

